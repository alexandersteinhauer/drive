//! Render a learned FIR model: surrogate <model.json> <flow.f32> <controls.f32> <out.f32> [--bench].
//! Raw f32 files are little-endian, frame-interleaved. The output has one frame per input frame.
//! `--bench` times warmed, repeated, complete input passes from reset state. File I/O,
//! allocation, output encoding and resets are outside those timings; input validation is included.
use engine_sim::surrogate::Surrogate;
use engine_sim::synth::BLOCK;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// This executable is single-threaded. Count allocations (including reallocations) so
// --bench verifies its no-allocation claim rather than relying on code inspection.
struct AllocationAudit;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: AllocationAudit = AllocationAudit;
unsafe impl GlobalAlloc for AllocationAudit {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{path}: incomplete f32 sample").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn length(frames: usize, channels: usize) -> Result<usize, Box<dyn std::error::Error>> {
    frames
        .checked_mul(channels)
        .ok_or_else(|| "sample count overflow".into())
}

/// The inputs have already been padded to a whole number of blocks. No allocations.
fn run_pass(
    surrogate: &mut Surrogate,
    flow: &[f32],
    controls: &[f32],
    output_block: &mut [f32],
) -> Result<(), String> {
    let cylinders = surrogate.model().cylinders;
    let controls_dim = surrogate.model().controls();
    for block in 0..flow.len() / (BLOCK * cylinders) {
        let x = block * BLOCK * cylinders;
        let c = block * BLOCK * controls_dim;
        surrogate.process_block(
            &flow[x..x + BLOCK * cylinders],
            &controls[c..c + BLOCK * controls_dim],
            output_block,
        )?;
        std::hint::black_box(&*output_block);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 || args.len() > 6 || (args.len() == 6 && args[5] != "--bench") {
        return Err(
            "usage: surrogate <model.json> <flow.f32> <controls.f32> <out.f32> [--bench]".into(),
        );
    }
    let json = std::fs::read_to_string(&args[1])?;
    let mut surrogate = Surrogate::from_json(&json)?;
    let m = surrogate.model();
    let (cylinders, controls_dim, outputs, rate) = (m.cylinders, m.controls(), m.outputs(), m.rate);
    let mut flow = read_f32(&args[2])?;
    let mut controls = read_f32(&args[3])?;
    if flow.len() % cylinders != 0 {
        return Err("flow file has partial frame".into());
    }
    let frames = flow.len() / cylinders;
    if frames == 0 {
        return Err("flow file must contain at least one frame".into());
    }
    if controls.len() != length(frames, controls_dim)? {
        return Err("control frame count does not match flow".into());
    }
    let padded_frames = frames
        .checked_add(BLOCK - 1)
        .ok_or("frame count overflow")?
        / BLOCK
        * BLOCK;
    flow.resize(length(padded_frames, cylinders)?, 0.0);
    // Hold the final controls while padding the final block; only the real frames are written.
    controls.resize(length(padded_frames, controls_dim)?, 0.0);
    for frame in frames..padded_frames {
        let (before, after) = controls.split_at_mut(frame * controls_dim);
        after[..controls_dim]
            .copy_from_slice(&before[(frames - 1) * controls_dim..frames * controls_dim]);
    }
    let mut output_block = vec![0.0; length(BLOCK, outputs)?];
    let mut rendered = vec![0.0; length(frames, outputs)?];
    let started = Instant::now();
    for offset in (0..padded_frames).step_by(BLOCK) {
        surrogate.process_block(
            &flow[offset * cylinders..(offset + BLOCK) * cylinders],
            &controls[offset * controls_dim..(offset + BLOCK) * controls_dim],
            &mut output_block,
        )?;
        let count = (frames - offset).min(BLOCK);
        rendered[offset * outputs..(offset + count) * outputs]
            .copy_from_slice(&output_block[..count * outputs]);
    }
    let render_time = started.elapsed().as_secs_f64();
    let mut bytes = vec![0; length(rendered.len(), 4)?];
    for (&sample, bytes) in rendered.iter().zip(bytes.chunks_exact_mut(4)) {
        bytes.copy_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(&args[4], bytes)?;
    eprintln!(
        "{outputs} outputs, model {} bytes, {frames} frames at {rate} Hz, render {:.2}x realtime ({render_time:.6}s)",
        json.len(), frames as f64 / rate as f64 / render_time,
    );
    if args.len() == 6 {
        for _ in 0..2 {
            surrogate.reset();
            run_pass(&mut surrogate, &flow, &controls, &mut output_block)?;
        }
        let (mut passes, mut elapsed, mut fastest, mut slowest) = (0, 0.0, f64::INFINITY, 0.0_f64);
        // Every repetition consumes the complete input, not just its first/final block.
        // Each starts from zero history, just like the render above. Only processing is timed.
        while passes < 5 || elapsed < 0.5 {
            surrogate.reset();
            let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
            let started = Instant::now();
            run_pass(&mut surrogate, &flow, &controls, &mut output_block)?;
            let seconds = started.elapsed().as_secs_f64();
            let allocations = ALLOCATIONS
                .load(Ordering::Relaxed)
                .wrapping_sub(allocations_before);
            if allocations != 0 {
                return Err(format!("benchmark processing allocated {allocations} times").into());
            }
            elapsed += seconds;
            fastest = fastest.min(seconds);
            slowest = slowest.max(seconds);
            passes += 1;
        }
        let duration = frames as f64 / rate as f64;
        eprintln!(
            "bench {:.2}x realtime: {passes} full passes, mean {:.6}s, min {fastest:.6}s, max {slowest:.6}s ({padded_frames} processed frames/pass; 2 warmups; 0 timed allocations observed; I/O and resets excluded)",
            duration * passes as f64 / elapsed, elapsed / passes as f64,
        );
    }
    Ok(())
}
