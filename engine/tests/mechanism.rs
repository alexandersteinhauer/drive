//! The reduced crank mechanism (the game's) against engine-sim's constrained bodies, on the
//! reference scenario: the same engine, within tolerances rather than bit for bit.

use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;
use engine_sim::units::rpm;

fn spec() -> EngineSpec {
    let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vw1200.json")).unwrap();
    serde_json::from_str(&json).unwrap()
}

/// Per step: crank speed, filtered dyno torque, cylinder 1 pressure.
fn run(mut engine: Engine, spec: &EngineSpec, steps: usize) -> Vec<[f64; 3]> {
    engine.set_crank_speed(-spec.starter_speed);
    engine.set_ignition(true);
    engine.set_clutch(0.0);
    (0..steps)
        .map(|step| {
            let s = step as f64;
            let (dyno, r, throttle) = if step < 5000 {
                (true, 900.0, 0.0)
            } else if step < 15000 {
                (true, 900.0 + (s - 5000.0) / 10000.0 * 3100.0, 1.0)
            } else if step < 20000 {
                (true, 4000.0 - (s - 15000.0) / 5000.0 * 2000.0, 0.3)
            } else if step < 25000 {
                (false, 0.0, 0.6)
            } else {
                (false, 0.0, 0.0)
            };
            engine.set_dyno(dyno, true, rpm(r));
            engine.set_speed_control(throttle);
            engine.step();
            [engine.crank_speed(), engine.filtered_dyno_torque(), engine.cylinder_pressure(0)]
        })
        .collect()
}

#[test]
fn reduced_crank_agrees_with_engine_sims() {
    let spec = spec();
    let steps = 30000;
    let ours = run(Engine::new(&spec), &spec, steps);
    let theirs = run(Engine::constrained(&spec), &spec, steps);
    let mut worst_speed: f64 = 0.0;
    let mut worst_torque: f64 = 0.0;
    for step in (0..steps).step_by(500) {
        let (a, b) = (ours[step], theirs[step]);
        let speed = (a[0] - b[0]).abs() / b[0].abs();
        // Torque against the full-throttle peak, so the zero crossings don't blow it up.
        let torque = (a[1] - b[1]).abs() / 100.0;
        if std::env::var("TRACE").is_ok() {
            eprintln!("{step}: {:.1} vs {:.1} rpm, {:.2} vs {:.2} Nm", a[0].abs() * 30.0 / std::f64::consts::PI, b[0].abs() * 30.0 / std::f64::consts::PI, a[1], b[1]);
        }
        if step >= 1000 {
            worst_speed = worst_speed.max(speed);
            worst_torque = worst_torque.max(torque);
        }
    }
    eprintln!("worst: speed {:.3} %, torque {:.3} % of 100 Nm", worst_speed * 100.0, worst_torque * 100.0);
    assert!(worst_speed < 0.02, "speed differs by {:.2} %", worst_speed * 100.0);
    assert!(worst_torque < 0.03, "torque differs by {:.2} Nm", worst_torque * 100.0);
}
