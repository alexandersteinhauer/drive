//! The engine's state step by step on the dynamometer, with the reference exhaust (duct.rs)
//! or the game's lumped one, for comparing the two on the same crank motion: the dyno holds
//! the crank to the program's speed, and with `physics.cycleVariation` each cylinder's burns
//! draw from its own random sequence, so both engines burn alike.
//!   abtrace <spec.json> <reference.json> <out prefix> <ducted|lumped> <program>
//! The reference config's keys merge into the spec as in examples/reference.rs; `lumped`
//! drops its ducts. Programs: segments `seconds:rpm0:rpm1:throttle`, comma separated.
//! env: FREQ (default 20000), FLUID (fluid substeps, default 12), DUCT_THREADS,
//! SEED (burn randomness, default 1), MIN_SUBSTEPS (pipes per fluid step, default 1),
//! EXHAUST_ONLY=1 (add only the reference ducts; keep the game's upstream engine settings).
//! Writes <out>.steps.f32 (per step: cycle angle, rpm, dyno torque, boost (gauge Pa),
//! manifold pressure, turbo speed, wastegate, turbine power, throttle, flap; per cylinder
//! pressure, temperature, exhaust flow (kg/s), effective valve area, volume, port pressure), <out>.steps.json (the layout);
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
    let exhaust_only = std::env::var_os("EXHAUST_ONLY").is_some();
    configure_exhaust(&mut spec, &reference, ducted, exhaust_only);
    spec["simulationFrequency"] = env("FREQ", 20000.0).into();
    spec["fluidSteps"] = (env("FLUID", 12.0) as usize).into();
    let simulation_spec = spec.clone();
    let spec: EngineSpec = serde_json::from_value(spec).unwrap();
    let rate = spec.simulation_frequency * spec.fluid_steps as f64;

    let mut e = Engine::new(&spec);
    e.collect_audio = false;
    e.seed_burns(env("SEED", 1.0) as u32);
    e.set_ignition(true);
    if let Some(d) = &mut e.ducts {
        d.net.min_substeps = env("MIN_SUBSTEPS", 1.0) as usize;
    }
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
                e.ducts.as_ref().map_or_else(|| {
                    if spec.audio.surrogate.is_some() { e.staging()[n+4] } else { 0.0 }
                }, |d| d.flap),
            ].map(|v| v as f32));
            for i in 0..n {
                steps.extend([e.cylinder_pressure(i), e.cylinder_temperature(i), e.cylinder_exhaust_flow(i), e.cylinder_exhaust_area(i), e.cylinder_volume(i), e.exhaust_runner_pressure(i)].map(|v| v as f32));
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
        "global": ["cycleAngle", "rpm", "dynoTorque", "boost", "manifold", "turboSpeed", "wastegate", "turbinePower", "throttle", "flap"],
        "cylinder": ["pressure", "temperature", "exhaustFlow", "exhaustArea", "volume", "portPressure"],
        "firingAngles": (0..n).map(|i| e.firing_angle(i)).collect::<Vec<_>>(),
        "program": program,
        "ducted": ducted,
        "fluidSteps": spec.fluid_steps,
        "surrogateAudio": spec.audio.surrogate.is_some(),
        "upstreamMode": if exhaust_only { "game" } else { "mergedReference" },
        "simulationSpec": simulation_spec,
        "sourceRevision": std::env::var("SOURCE_REVISION").unwrap_or_default(),
        "seed": env("SEED", 1.0) as u32,
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
        std::fs::write(format!("{out}.json"), serde_json::json!({"rate": rate}).to_string()).unwrap();
        let diameters = d.outlets.iter().map(|&l| format!("{}", d.net.pipes[d.net.links[l].pipe].diameter)).collect::<Vec<_>>().join(" ");
        std::fs::write(format!("{out}.exits.txt"), format!("rate {rate}\ndiameters {diameters}\ndistance {}\n", d.distance)).unwrap();
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

/// Data acquisition can swap only the exhaust, preserving the deployed engine's
/// combustion and boost controls. Legacy diagnostic renders can still merge the full config.
fn configure_exhaust(spec: &mut serde_json::Value, reference: &serde_json::Value, ducted: bool, exhaust_only: bool) {
    if exhaust_only {
        if ducted { spec["ducts"] = reference.get("ducts").unwrap_or(reference).clone(); }
    } else {
        merge(spec, reference);
    }
    if !ducted { spec.as_object_mut().unwrap().remove("ducts"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exhaust_only_preserves_the_games_upstream_settings() {
        let original = json!({"physics":{"realGas":true},"turbo":{"boost":"stock"},"audio":{"surrogate":{"model":"pilot"}}});
        let reference = json!({"physics":{"cycleVariation":0.1},"turbo":{"boost":"reference"},"ducts":{"pipes":[]}});
        for ducted in [false,true] {
            let mut spec = original.clone();
            configure_exhaust(&mut spec,&reference,ducted,true);
            assert_eq!(spec["physics"],original["physics"]);
            assert_eq!(spec["turbo"],original["turbo"]);
            assert_eq!(spec["audio"],original["audio"]);
            assert_eq!(spec.get("ducts").is_some(),ducted);
        }
    }

    #[test]
    fn legacy_reference_merge_and_bare_ducts_remain_available() {
        let reference = json!({"physics":{"cycleVariation":0.1},"ducts":{"pipes":[]}});
        let mut spec = json!({"physics":{"realGas":true}});
        configure_exhaust(&mut spec,&reference,false,false);
        assert_eq!(spec["physics"],json!({"realGas":true,"cycleVariation":0.1}));
        assert!(spec.get("ducts").is_none());
        configure_exhaust(&mut spec,&reference["ducts"],true,true);
        assert_eq!(spec["ducts"],reference["ducts"]);
    }
}
