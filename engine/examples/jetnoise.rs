//! The reference's Stone-Groesbeck-Zola mixing noise, through the Rust runtime.
//!   jetnoise <states.f32> <out.f32> <diameter,diameter,...> [distance] [off-axis-deg]
//! Input: exit [velocity(m/s),density(kg/m3),temperature(K)] triples, ends interleaved,
//! at 4 kHz. Output: 48 kHz mono float pressure (Pa), with causal 250 us band interpolation.
use engine_sim::jet::Jet;
use std::io::Write;

fn main() {
    let a: Vec<_> = std::env::args().collect();
    let diameters: Vec<f64> = a[3].split(',').map(|s| s.parse().unwrap()).collect();
    let distance = a.get(4).map_or(5.0, |s| s.parse().unwrap());
    let angle = a.get(5).map_or(20.0, |s| s.parse().unwrap());
    let mut jet = Jet::new(48000.0, &diameters, distance, angle).unwrap();
    let bytes = std::fs::read(&a[1]).unwrap();
    assert_eq!(bytes.len() % (diameters.len()*3*4), 0, "incomplete state frame");
    let states: Vec<f64> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect();
    let mut exits = vec![[0.; 3]; diameters.len()];
    let count = states.len() / (3*diameters.len());
    let mut output = Vec::with_capacity(count*12*4);
    let started = std::time::Instant::now();
    for s in states.chunks_exact(3*diameters.len()) {
        for (e, s) in exits.iter_mut().zip(s.chunks_exact(3)) { e.copy_from_slice(s); }
        jet.set_exit_states(&exits).unwrap();
        for _ in 0..12 { output.extend((jet.next_sample() as f32).to_le_bytes()); }
    }
    let mut f = std::fs::File::create(&a[2]).unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    f.write_all(&output).unwrap();
    println!("{} s of jet sound in {:.4} s ({:.2}% realtime, includes output byte conversion)", count as f64/4000.0, elapsed, elapsed/(count as f64/4000.0)*100.0);
}
