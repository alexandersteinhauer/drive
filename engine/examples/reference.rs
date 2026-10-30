//! The reference simulation: an engine with its exhaust as pipes (duct.rs, `spec.ducts`) on
//! the dynamometer; its sound, the pressure the open ends radiate to the listener.
//!   reference <spec.json> <ducts.json> <out prefix> [program]
//! The second file holds the ducts, or a reference config whose keys merge into the spec
//! ("ducts", "physics", ...).
//! env: FREQ (simulation frequency, default 20000), FLUID (fluid substeps, default 1).
//! Writes <out>.wav (32-bit float at the fluid rate: resample to listen) and <out>.txt (log).
use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;
use std::io::Write;

type Segment = (f64, f64, f64, f64);

fn program(name: &str) -> Vec<Segment> {
    match name {
        // Idle, full-throttle holds, overrun holds, a full-throttle sweep (as the voice's
        // renders, scripts/vrender.ts).
        "dyno" => vec![
            (4.0, 1000.0, 1000.0, 0.0),
            (3.0, 3000.0, 3000.0, 1.0),
            (3.0, 4500.0, 4500.0, 1.0),
            (3.0, 6000.0, 6000.0, 1.0),
            (3.0, 7500.0, 7500.0, 1.0),
            (3.0, 6000.0, 6000.0, 0.0),
            (3.0, 3000.0, 3000.0, 0.0),
            (2.0, 2000.0, 2000.0, 1.0),
            (16.0, 2000.0, 8000.0, 1.0),
        ],
        "short" => vec![(1.0, 1000.0, 1000.0, 0.0), (2.0, 6000.0, 6000.0, 1.0)],
        "hold" => vec![(4.0, 6000.0, 6000.0, 1.0)],
        "hold3" => vec![(4.0, 3000.0, 3000.0, 1.0)],
        "diag6" => vec![(2.0, 6000.0, 6000.0, 1.0)],
        "idle" => vec![(3.0, 1000.0, 1000.0, 0.0)],
        _ => panic!("no program {name}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let ducts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let out = &args[3];
    let program = program(args.get(4).map_or("dyno", |s| s.as_str()));
    let env = |k: &str, d: f64| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    // A reference config: its keys merged into the spec ("ducts", "physics", ...); or the
    // ducts alone.
    if ducts.get("ducts").is_some() {
        merge(&mut spec, &ducts);
    } else {
        spec["ducts"] = ducts;
    }
    spec["simulationFrequency"] = env("FREQ", 20000.0).into();
    spec["fluidSteps"] = (env("FLUID", 1.0) as usize).into();
    let spec: EngineSpec = serde_json::from_value(spec).unwrap();
    let rate = spec.simulation_frequency * spec.fluid_steps as f64;

    let mut e = Engine::new(&spec);
    e.collect_audio = false;
    e.set_ignition(true);
    let rpm = std::f64::consts::PI / 30.0;
    e.set_crank_speed(-program[0].1 * rpm);
    let frame = 1.0 / 60.0;
    let per_frame = (spec.simulation_frequency * frame).round() as usize;
    let mut log = String::new();
    let (mut t, mut substeps, mut steps) = (0.0, 0usize, 0usize);
    let mut last_good = String::new();
    // Means over each log interval: turbine inlet (scroll) and outlet pressures, turbine
    // power, wastegate opening.
    let probe = |e: &Engine, name: &str| {
        let d = e.ducts.as_ref().unwrap();
        // The turbine's inlet: its nozzle's plenum, or the scroll it is fed from.
        let alt = name.replace("scroll", "nozzle");
        d.names.get(&alt).or_else(|| d.names.get(name)).map_or(0.0, |&n| d.net.nodes[n].pressure())
    };
    let (mut p_in, mut p_out, mut power, mut gate, mut n_mean) = (0.0, 0.0, 0.0, 0.0, 0usize);
    let started = std::time::Instant::now();
    for &(dur, r0, r1, thr) in &program {
        let mut s = 0.0;
        while s < dur - 1e-9 {
            let r = r0 + (r1 - r0) * (s / dur);
            e.set_dyno(true, true, r * rpm);
            e.set_speed_control(thr);
            for _ in 0..per_frame {
                e.step();
                let d = e.ducts.as_ref().unwrap();
                if std::env::var("TRACEU").is_ok() && steps % 50 == 0 {
                    for k in [10usize, 14] {
                        let p = &d.net.pipes[k];
                        let (i, um) = p.prim.iter().enumerate().map(|(i, c)| (i, c.u.abs())).fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
                        eprint!("pipe {k} max|u| {um:.1} at {i}  ");
                    }
                    eprintln!("step {steps}");
                }
                if std::env::var("CHECK").is_ok() {
                    if let Some(w) = d.net.check() {
                        eprintln!("before: {last_good}");
                        panic!("step {steps}: {w}");
                    }
                    let p = &d.net.pipes[14];
                    let n = d.names["left turbine out"];
                    last_good = format!("orifices {:?}\nnodes {:?}\ncells {:?}\nnode p {:.0} T {:.0} links {:?}", d.net.orifices.iter().map(|o| (o.a, o.b, o.cda, o.flow, o.power, o.efficiency)).collect::<Vec<_>>(), d.net.nodes.iter().map(|n| (n.volume, n.m, n.e)).collect::<Vec<_>>(), &p.prim[..3].iter().map(|c| (c.p as i64, c.u as i64, c.t as i64)).collect::<Vec<_>>(), d.net.nodes[n].pressure(), d.net.nodes[n].temperature(), d.net.links.iter().filter(|l| l.pipe == 14).map(|l| (l.flow.m, l.flow.p as i64, l.flow.un as i64)).collect::<Vec<_>>());
                }
                substeps += d.net.substeps;
                steps += 1;
                p_in += probe(&e, "right scroll A");
                p_out += probe(&e, "right turbine out");
                power += d.turbines.iter().map(|&o| d.net.orifices[o].power).sum::<f64>();
                gate += d.wastegates.first().map_or(0.0, |&(o, a)| d.net.orifices[o].cda / a);
                n_mean += 1;
            }
            s += frame;
            t += frame;
            if ((t * 4.0) - (t * 4.0).round()).abs() < frame * 2.0 {
                let d = e.ducts.as_ref().unwrap();
                let p1 = &d.net.pipes[0];
                let (boost, _) = e.turbo_state();
                let k = n_mean.max(1) as f64;
                log += &format!(
                    "{t:.2} rpm {:.0} thr {thr} torque {:.0} air {:.3} kg/s boost {:.2} map {:.2} turbine {:.2}->{:.2} bar {:.0} kW gate {:.2} port {:.2}..{:.2} bar {:.0} K substeps {:.1} wall {:.1}s\n",
                    -e.crank_speed() / rpm,
                    e.filtered_dyno_torque(),
                    e.air_mass_flow(),
                    boost / 1e5,
                    e.manifold_pressure() / 1e5,
                    p_in / k / 1e5,
                    p_out / k / 1e5,
                    power / k / 1e3,
                    gate / k,
                    p1.prim.iter().map(|p| p.p).fold(f64::INFINITY, f64::min) / 1e5,
                    p1.prim.iter().map(|p| p.p).fold(0.0, f64::max) / 1e5,
                    p1.prim[0].t,
                    substeps as f64 / steps.max(1) as f64,
                    started.elapsed().as_secs_f64(),
                );
                (p_in, p_out, power, gate, n_mean) = (0.0, 0.0, 0.0, 0.0, 0);
                print!("{}", log.lines().last().unwrap());
                println!();
                std::io::stdout().flush().unwrap();
            }
        }
    }
    let d = e.ducts.as_ref().unwrap();
    write_wav(&format!("{out}.wav"), &d.sound, rate as u32);
    // Each open end's outward velocity, density and temperature per sample (f32 triples),
    // and the ends' diameters.
    let bytes: Vec<u8> = d.exits.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out}.exits.f32"), bytes).unwrap();
    let diameters: Vec<String> = d.outlets.iter().map(|&l| format!("{}", d.net.pipes[d.net.links[l].pipe].diameter)).collect();
    std::fs::write(format!("{out}.exits.txt"), format!("rate {rate}\ndiameters {}\ndistance {}\n", diameters.join(" "), d.distance)).unwrap();
    std::fs::write(format!("{out}.txt"), log).unwrap();
    println!("{} s of sound at {rate} Hz in {:.1} s", d.sound.len() as f64 / rate, started.elapsed().as_secs_f64());
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

fn write_wav(path: &str, data: &[f32], rate: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    let bytes = (data.len() * 4) as u32;
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36 + bytes).to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&3u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&rate.to_le_bytes()).unwrap();
    f.write_all(&(rate * 4).to_le_bytes()).unwrap();
    f.write_all(&4u16.to_le_bytes()).unwrap();
    f.write_all(&32u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&bytes.to_le_bytes()).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}
