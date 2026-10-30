//! The engine's state step by step on the dynamometer, with the reference exhaust (duct.rs)
//! or the game's lumped one, for comparing the two on the same crank motion: the dyno holds
//! the crank to the program's speed, and with `physics.cycleVariation` each cylinder's burns
//! draw from its own random sequence, so both engines burn alike.
//!   abtrace <spec.json> <reference.json> <out prefix> <ducted|lumped> <program>
//! The reference config's keys merge into the spec as in examples/reference.rs; `lumped`
//! drops its ducts. Programs: segments `seconds:rpm0:rpm1:throttle`, comma separated.
//! env: FREQ (default 20000), FLUID (fluid substeps, default 12), DUCT_THREADS.
//! Writes <out>.steps.f32 (per step: cycle angle, rpm, dyno torque, boost (gauge Pa),
//! manifold pressure, turbo speed, wastegate, turbine power, throttle; then per cylinder
//! its pressure, temperature and exhaust valve flow (kg/s)), <out>.steps.json (the layout);
//! ducted, also <out>.ends.f32 (each open end's pressure at the listener, interleaved, at the
//! fluid rate, before directivity), <out>.f32 (their sum) and <out>.exits.f32.
use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let out = &args[3];
    let ducted = match args[4].as_str() {
        "ducted" => true,
        "lumped" => false,
        m => panic!("mode {m}: ducted or lumped"),
    };
    let program: Vec<[f64; 4]> = args[5]
        .split(',')
        .map(|s| {
            let v: Vec<f64> = s.split(':').map(|x| x.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3]]
        })
        .collect();
    let env = |k: &str, d: f64| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    merge(&mut spec, &reference);
    if !ducted {
        spec.as_object_mut().unwrap().remove("ducts");
    }
    spec["simulationFrequency"] = env("FREQ", 20000.0).into();
    spec["fluidSteps"] = (env("FLUID", 12.0) as usize).into();
    let spec: EngineSpec = serde_json::from_value(spec).unwrap();
    let rate = spec.simulation_frequency * spec.fluid_steps as f64;

    let mut e = Engine::new(&spec);
    e.collect_audio = false;
    e.set_ignition(true);
    if let Some(d) = &mut e.ducts {
        d.listening_ends = true;
    }
    let rpm = std::f64::consts::PI / 30.0;
    e.set_crank_speed(-program[0][1] * rpm);
    let n = e.cylinder_count();
    let mut steps: Vec<f32> = Vec::new();
    let mut f = std::fs::File::create(format!("{out}.steps.f32")).unwrap();
    let started = std::time::Instant::now();
    let dt = 1.0 / spec.simulation_frequency;
    let total: f64 = program.iter().map(|s| s[0]).sum();
    let mut t = 0.0;
    for &[dur, r0, r1, thr] in &program {
        let count = (dur * spec.simulation_frequency).round() as usize;
        for k in 0..count {
            let r = r0 + (r1 - r0) * (k as f64 / count as f64);
            e.set_dyno(true, true, r * rpm);
            e.set_speed_control(thr);
            e.step();
            t += dt;
            let (boost, shaft) = e.turbo_state();
            steps.extend([
                e.cycle_angle(),
                -e.crank_speed() / rpm,
                e.dyno_torque(),
                boost,
                e.manifold_pressure(),
                shaft,
                e.wastegate(),
                e.turbine_power(),
                thr,
            ].map(|v| v as f32));
            for i in 0..n {
                steps.extend([e.cylinder_pressure(i), e.cylinder_temperature(i), e.cylinder_exhaust_flow(i)].map(|v| v as f32));
            }
            if steps.len() > 1 << 20 {
                f.write_all(&bytes(&steps)).unwrap();
                steps.clear();
            }
            if ((t * 4.0) - (t * 4.0).round()).abs() < dt / 2.0 {
                println!(
                    "{t:.2}/{total} rpm {:.0} thr {thr} torque {:.0} boost {:.2} bar turbine {:.0} kW gate {:.2} wall {:.1}s",
                    -e.crank_speed() / rpm,
                    e.filtered_dyno_torque(),
                    boost / 1e5,
                    e.turbine_power() / 1e3,
                    e.wastegate(),
                    started.elapsed().as_secs_f64()
                );
                std::io::stdout().flush().unwrap();
            }
        }
    }
    f.write_all(&bytes(&steps)).unwrap();
    let mut layout = serde_json::json!({
        "rate": spec.simulation_frequency,
        "cylinders": n,
        "global": ["cycleAngle", "rpm", "dynoTorque", "boost", "manifold", "turboSpeed", "wastegate", "turbinePower", "throttle"],
        "cylinder": ["pressure", "temperature", "exhaustFlow"],
        "firingAngles": (0..n).map(|i| e.firing_angle(i)).collect::<Vec<_>>(),
        "program": program,
        "ducted": ducted,
    });
    if let Some(d) = &e.ducts {
        std::fs::write(format!("{out}.ends.f32"), bytes(&d.end_sound)).unwrap();
        std::fs::write(format!("{out}.f32"), bytes(&d.sound)).unwrap();
        std::fs::write(format!("{out}.exits.f32"), bytes(&d.exits)).unwrap();
        let diameters: Vec<f64> = d.outlets.iter().map(|&l| d.net.pipes[d.net.links[l].pipe].diameter).collect();
        layout["fluidRate"] = rate.into();
        layout["ends"] = d.outlets.len().into();
        layout["diameters"] = diameters.into();
        layout["distance"] = d.distance.into();
    }
    std::fs::write(format!("{out}.steps.json"), serde_json::to_string_pretty(&layout).unwrap()).unwrap();
    println!("{total} s in {:.1} s", started.elapsed().as_secs_f64());
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn merge(a: &mut serde_json::Value, b: &serde_json::Value) {
    if let (Some(a), Some(b)) = (a.as_object_mut(), b.as_object()) {
        for (k, v) in b {
            match a.get_mut(k) {
                Some(x) if x.is_object() && v.is_object() => merge(x, v),
                _ => {
                    a.insert(k.clone(), v.clone());
                }
            }
        }
    }
}
