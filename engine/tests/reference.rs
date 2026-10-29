//! Compares the port with the original engine-sim, step by step and sample by sample: run
//! reference/build.sh first; skipped when its output is missing.

use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;
use engine_sim::units::rpm;
use std::path::Path;

struct Controls {
    dyno: bool,
    rpm: f64,
    throttle: f64,
}

// reference/scenario.h
const SCENARIO_STEPS: usize = 30000;
fn scenario(step: usize) -> Controls {
    let s = step as f64;
    if step < 5000 {
        Controls { dyno: true, rpm: 900.0, throttle: 0.0 }
    } else if step < 15000 {
        Controls { dyno: true, rpm: 900.0 + (s - 5000.0) / 10000.0 * 3100.0, throttle: 1.0 }
    } else if step < 20000 {
        Controls { dyno: true, rpm: 4000.0 - (s - 15000.0) / 5000.0 * 2000.0, throttle: 0.3 }
    } else if step < 25000 {
        Controls { dyno: false, rpm: 0.0, throttle: 0.6 }
    } else {
        Controls { dyno: false, rpm: 0.0, throttle: 0.0 }
    }
}

fn ulps(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

/// The script's vehicle on that body (VehicleDragConstraint).
fn vehicle_drag(body: &engine_sim::solver::Body) -> f64 {
    use engine_sim::units::{celcius, AIR_MOLECULAR_MASS, ATM, R};
    let (mass, drag, area, rolling) = (840.0, 0.48, 1.8, 0.013 * 840.0 * 9.81);
    let air_density = AIR_MOLECULAR_MASS * (1.0 * ATM) / (R * celcius(25.0));
    let e_r = 0.5 * body.i * body.v_theta * body.v_theta;
    let v = (2.0 * e_r / mass).sqrt();
    let v_squared = v * v;
    (body.i / mass).sqrt() * (rolling + 0.5 * air_density * v_squared * drag * area)
}

#[test]
fn matches_engine_sim() {
    if !cfg!(feature = "reference") {
        eprintln!("needs --features reference (engine-sim's exact arithmetic); skipped");
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/.build");
    let (Ok(spec), Ok(trace)) = (
        std::fs::read_to_string(dir.join("beetle.json")),
        std::fs::read_to_string(dir.join("trace.txt")),
    ) else {
        eprintln!("no reference output in {}; skipped", dir.display());
        return;
    };
    // The fixture the game's TypeScript Beetle is checked against must be this engine.
    let fixture = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vw1200.json")).unwrap();
    assert_eq!(fixture.trim(), spec.trim(), "tests/vw1200.json is stale: copy reference/.build/beetle.json");
    let spec: EngineSpec = serde_json::from_str(&spec).unwrap();
    let mut engine = Engine::constrained(&spec);
    engine.set_crank_speed(-spec.starter_speed);
    engine.set_ignition(true);
    // engine-sim only limits the clutch from the transmission's update after the first
    // step: until then it is locked to the vehicle's (1 kg m^2) body.
    engine.set_clutch(f64::MAX);
    let mut synth = engine_sim::synth::Synthesizer::new(1, 10000.0, 44100.0, {
        let mut p = engine_sim::synth::AudioParameters::default();
        p.input_sample_noise = 0.0;
        p.air_noise = spec.audio.noise as f32;
        p.df_f_mix = spec.audio.hf_gain as f32;
        p
    });
    let impulses = Path::new(env!("CARGO_MANIFEST_DIR")).join("../public/audio/impulse");
    let wav = std::fs::read(impulses.join(&spec.exhausts[0].impulse_response)).unwrap();
    let impulse = engine_sim::synth::read_wav(&wav).unwrap();
    synth.set_impulse_response(0, &impulse, spec.exhausts[0].impulse_volume as f32);
    let mut pcm: Vec<i16> = Vec::new();

    let names = ["crank speed", "crank angle", "cylinder 1 pressure", "cylinder 3 exhaust", "audio"];
    let mut worst = [0u64; 5];
    for (step, line) in trace.lines().enumerate().take(SCENARIO_STEPS) {
        let c = scenario(step);
        engine.set_dyno(c.dyno, true, rpm(c.rpm));
        engine.set_speed_control(c.throttle);
        engine.set_drag(vehicle_drag(engine.load_body()));
        engine.step();
        engine.set_clutch(0.0); // in neutral
        let ours = [
            engine.crank_speed(),
            engine.crank_theta(),
            engine.cylinder_pressure(0),
            engine.exhaust_runner_pressure(2),
            engine.staging()[0],
        ];
        synth.write_input(engine.staging());
        if step % 100 == 99 {
            synth.render_pending();
            let mut out = vec![0i16; synth.output_available()];
            synth.read_output(&mut out);
            pcm.extend_from_slice(&out);
        }
        let theirs: Vec<f64> = line.split(' ').map(|v| v.parse().unwrap()).collect();
        for k in 0..5 {
            let d = ulps(ours[k], theirs[k]);
            worst[k] = worst[k].max(d);
            let rel = (ours[k] - theirs[k]).abs() / theirs[k].abs().max(1e-9);
            assert!(
                d <= 64 || rel < 1e-9,
                "step {step}: {} is {} in the port, {} in engine-sim",
                names[k],
                ours[k],
                theirs[k]
            );
        }
    }
    eprintln!("worst difference (ulps): {:?}", worst);

    let reference: Vec<i16> = std::fs::read(dir.join("trace.txt.pcm"))
        .unwrap()
        .chunks(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(pcm.len(), reference.len(), "audio length");
    let differing = pcm.iter().zip(&reference).filter(|(a, b)| a != b).count();
    eprintln!("audio: {} samples, {} differ", pcm.len(), differing);
    if let Some(i) = pcm.iter().zip(&reference).position(|(a, b)| a != b) {
        panic!("audio sample {i}: {} in the port, {} in engine-sim", pcm[i], reference[i]);
    }
}
