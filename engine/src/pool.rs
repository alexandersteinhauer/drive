//! Tasks run in parallel on a few threads that wait between them spinning (native builds with
//! the `parallel` feature, `DUCT_THREADS` threads): the reference simulation's pipes, each
//! stage a few hundred microseconds of work split among them, where a pool that puts its
//! threads to sleep spends as long waking them. The calling thread takes tasks too. Without
//! the feature, or with one thread, the tasks run in order on the caller.

/// Runs `f(0)` .. `f(n - 1)`, each once, in any order and in parallel; returns when all are done.
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    #[cfg(feature = "parallel")]
    if spinning::run(n, f) {
        return;
    }
    for i in 0..n {
        f(i);
    }
}

/// Threads `run` uses (the caller's included).
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    return spinning::pool().map_or(1, |p| p.threads);
    #[cfg(not(feature = "parallel"))]
    1
}

#[cfg(feature = "parallel")]
mod spinning {
    use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
    use std::sync::{Mutex, OnceLock};

    type Job = dyn Fn(usize) + Sync;

    pub struct Pool {
        pub threads: usize,
        /// The current batch (high 32 bits) and the next task in it (low 32): a task is taken
        /// by advancing it, so only while its batch is current.
        ticket: AtomicU64,
        tasks: AtomicUsize,
        done: AtomicUsize,
        /// The batch's job: a reference to it on the caller's stack, valid while the batch
        /// has tasks to take.
        job: AtomicPtr<&'static Job>,
        /// One batch at a time (a second caller runs its tasks itself).
        busy: Mutex<()>,
    }

    pub fn pool() -> Option<&'static Pool> {
        static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
        *POOL.get_or_init(|| {
            let threads = std::env::var("DUCT_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
            if threads <= 1 {
                return None;
            }
            let pool: &'static Pool = Box::leak(Box::new(Pool {
                threads,
                ticket: AtomicU64::new(0),
                tasks: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                job: AtomicPtr::new(std::ptr::null_mut()),
                busy: Mutex::new(()),
            }));
            for _ in 1..threads {
                std::thread::spawn(move || worker(pool));
            }
            Some(pool)
        })
    }

    fn take(p: &Pool, batch: u64, tasks: usize) -> Option<usize> {
        let mut t = p.ticket.load(Acquire);
        loop {
            let i = (t & 0xffff_ffff) as usize;
            if t >> 32 != batch || i >= tasks {
                return None;
            }
            match p.ticket.compare_exchange_weak(t, t + 1, AcqRel, Acquire) {
                Ok(_) => return Some(i),
                Err(now) => t = now,
            }
        }
    }

    fn worker(p: &'static Pool) {
        #[cfg(target_os = "macos")]
        unsafe {
            // Performance cores, as the interactive thread that waits on them.
            extern "C" {
                fn pthread_set_qos_class_self_np(qos: u32, relative: i32) -> i32;
            }
            pthread_set_qos_class_self_np(0x21, 0);
        }
        let mut seen = 0;
        loop {
            let mut idle = 0u32;
            let batch = loop {
                let b = p.ticket.load(Acquire) >> 32;
                if b != seen {
                    break b;
                }
                idle = idle.saturating_add(1);
                if idle < 1 << 20 {
                    std::hint::spin_loop();
                } else if idle < (1 << 20) + 1000 {
                    std::thread::yield_now();
                } else {
                    // Long idle (the simulation not stepping): sleep, checking now and then.
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            };
            seen = batch;
            let tasks = p.tasks.load(Acquire);
            while let Some(i) = take(p, batch, tasks) {
                // Taken in the current batch: its caller is still waiting, its job alive.
                let job: &Job = unsafe { *p.job.load(Acquire) };
                job(i);
                p.done.fetch_add(1, Release);
            }
        }
    }

    /// Runs the batch on the pool; false if it can't (no pool, or busy).
    pub fn run<'a>(n: usize, f: &'a (dyn Fn(usize) + Sync + 'a)) -> bool {
        let Some(p) = pool() else { return false };
        if n < 2 {
            return false;
        }
        let Ok(_one) = p.busy.try_lock() else { return false };
        // The workers see it only while this batch is current, and it is until all its tasks
        // are done, below.
        let job: &'static Job = unsafe { std::mem::transmute::<&'a (dyn Fn(usize) + Sync + 'a), &'static Job>(f) };
        let job: &&'static Job = &job;
        p.job.store(job as *const &'static Job as *mut &'static Job, Relaxed);
        p.tasks.store(n, Relaxed);
        p.done.store(0, Relaxed);
        let batch = (p.ticket.load(Relaxed) >> 32) + 1;
        p.ticket.store(batch << 32, Release);
        while let Some(i) = take(p, batch, n) {
            f(i);
            p.done.fetch_add(1, Release);
        }
        while p.done.load(Acquire) < n {
            std::hint::spin_loop();
        }
        true
    }
}
