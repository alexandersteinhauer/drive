//! Times the Beetle's engine: cargo run --release --example bench
use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;
use engine_sim::units::rpm;
use std::time::Instant;

fn main() {
    let spec: EngineSpec =
        serde_json::from_str(include_str!("../tests/vw1200.json")).unwrap();
    // CONSTRAINED=1: engine-sim's constrained bodies instead of the reduced crank.
    let mut e = if std::env::var("CONSTRAINED").is_ok() { Engine::constrained(&spec) } else { Engine::new(&spec) };
    e.set_crank_speed(-spec.starter_speed);
    e.set_ignition(true);
    let mut total = 0.0;
    for (r, thr) in [(900.0, 0.0), (2000.0, 1.0), (3600.0, 1.0), (4500.0, 0.2)] {
        e.set_dyno(true, true, rpm(r));
        e.set_speed_control(thr);
        let t0 = Instant::now();
        for _ in 0..std::env::var("STEPS").map_or(20000, |s| s.parse().unwrap()) {
            e.step();
        }
        let s = t0.elapsed().as_secs_f64();
        total += s;
        for a in &mut e.audio {
            a.clear();
        }
        println!("{r} rpm, throttle {thr}: {:.1} % of real time, {:.1} Nm", s / 2.0 * 100.0, e.filtered_dyno_torque());
    }
    println!("total {:.2} s for 8 s", total);
}
