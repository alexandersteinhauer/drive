//! Work split among a few threads that wait between batches spinning (native builds with the
//! `parallel` feature, `DUCT_THREADS` threads): the reference simulation's pipes, each stage
//! ~100 microseconds of work split among them, where a pool that puts its threads to sleep
//! spends as long waking them. Each thread runs its own share, the same every batch, so its
//! data stays in its core's cache; the calling thread is thread 0. Without the feature, or with
//! one thread, the shares run in order on the caller.

/// Runs `f(0)` .. `f(threads() - 1)`, each on its own thread, the same each time (the caller's
/// 0); returns when all are done.
pub fn run_each(f: &(dyn Fn(usize) + Sync)) {
    #[cfg(feature = "parallel")]
    if spinning::run_each(f) {
        return;
    }
    for i in 0..threads() {
        f(i);
    }
}

/// Threads `run_each` uses (the caller's included).
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
        /// The current batch: a worker seeing it change runs its share of it.
        batch: AtomicU64,
        done: AtomicUsize,
        /// The batch's job: a reference to it on the caller's stack, valid till all are done.
        job: AtomicPtr<&'static Job>,
        /// One batch at a time (a second caller runs its shares itself).
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
                batch: AtomicU64::new(0),
                done: AtomicUsize::new(0),
                job: AtomicPtr::new(std::ptr::null_mut()),
                busy: Mutex::new(()),
            }));
            for w in 1..threads {
                std::thread::spawn(move || worker(pool, w));
            }
            Some(pool)
        })
    }

    fn worker(p: &'static Pool, w: usize) {
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
            seen = loop {
                let b = p.batch.load(Acquire);
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
            // The caller waits for this share: its job is alive.
            let job: &Job = unsafe { *p.job.load(Acquire) };
            job(w);
            p.done.fetch_add(1, Release);
        }
    }

    /// Runs the batch on the pool; false if it can't (no pool, or busy).
    pub fn run_each<'a>(f: &'a (dyn Fn(usize) + Sync + 'a)) -> bool {
        let Some(p) = pool() else { return false };
        let Ok(_one) = p.busy.try_lock() else { return false };
        // The workers see it once the batch changes, and it lives till they're all done, below.
        let job: &'static Job = unsafe { std::mem::transmute::<&'a (dyn Fn(usize) + Sync + 'a), &'static Job>(f) };
        let job: &&'static Job = &job;
        p.job.store(job as *const &'static Job as *mut &'static Job, Relaxed);
        p.done.store(0, Relaxed);
        p.batch.fetch_add(1, Release);
        f(0);
        p.done.fetch_add(1, Release);
        while p.done.load(Acquire) < p.threads {
            std::hint::spin_loop();
        }
        true
    }
}
