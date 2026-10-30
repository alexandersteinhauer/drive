//! The reference simulation: an engine with its exhaust as pipes (duct.rs, `spec.ducts`) on
//! the dynamometer; its sound, the pressure the open ends radiate to the listener.
//!   reference <spec.json> <ducts.json> <out prefix> [program]
//! The second file holds the ducts, or a reference config whose keys merge into the spec
//! ("ducts", "physics", ...).
//! env: FREQ (simulation frequency, default 20000), FLUID (fluid substeps, default 1; 12 at
//! 20 kHz records the sound at 240 kHz), CHECK / DUCTCHECK (stop at a state that isn't finite).
//! Programs: dyno (as the voice's renders), short, hold (6000 rpm), hold3, diag6, idle,
//! hold:<rpm>:<seconds> (full throttle).
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
        // hold:<rpm>:<seconds>, full throttle.
        _ if name.starts_with("hold:") => {
            let v: Vec<f64> = name[5..].split(':').map(|x| x.parse().unwrap()).collect();
            vec![(v[1], v[0], v[0], 1.0)]
        }
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
    // Means over each log interval: turbine inlet (scroll) and outlet pressures, turbine
    // power, wastegate opening.
    let probe = |e: &Engine, name: &str| {
        let d = e.ducts.as_ref().unwrap();
        // The turbine's inlet: its nozzle's plenum, or the scroll it is fed from.
        let alt = name.replace("scroll", "nozzle");
        d.names.get(&alt).or_else(|| d.names.get(name)).map_or(0.0, |&n| d.net.nodes[n].pressure())
    };
    let (mut p_in, mut p_out, mut power, mut gate, mut n_mean) = (0.0, 0.0, 0.0, 0.0, 0usize);
    // PROBE=name,name,...: those nodes' mean pressures in the log too.
    let probes: Vec<String> = std::env::var("PROBE").map(|v| v.split(',').map(String::from).collect()).unwrap_or_default();
    let mut probe_sums = vec![0.0; probes.len()];
    // HFMAP=<seconds>: from then on, each pipe cell's and node's pressure fluctuation above
    // ~1 kHz (rms, sampled each step), printed at the end, strongest first.
    let hf_from = std::env::var("HFMAP").ok().map(|v| v.parse::<f64>().unwrap());
    // TRACE=pipe:cell,pipe:cell,...: those cells' pressures each step, to <out>.trace.f32
    // (frames of one value per cell, at the simulation frequency).
    let traces: Vec<(usize, usize)> = std::env::var("TRACE")
        .map(|v| v.split(',').map(|x| { let (a, b) = x.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) }).collect())
        .unwrap_or_default();
    let mut trace_out: Vec<f32> = Vec::new();
    let mut hf: Vec<Vec<(f64, f64, f64)>> = Vec::new();
    let mut hf_nodes: Vec<(f64, f64, f64)> = Vec::new();
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
                // CHECK=1: stop at the first state that isn't finite (DUCTCHECK=1 checks every
                // substep, with the state before).
                if std::env::var("CHECK").is_ok() {
                    if let Some(w) = d.net.check() {
                        panic!("step {steps}: {w}");
                    }
                }
                substeps += d.net.substeps;
                steps += 1;
                for &(k, i) in &traces {
                    trace_out.push(d.net.pipes[k].prim[i.min(d.net.pipes[k].cells() - 1)].p as f32);
                }
                if let Some(from) = hf_from {
                    if hf.is_empty() {
                        hf = d.net.pipes.iter().map(|p| vec![(p.prim[0].p, 0.0, 0.0); p.cells()]).collect();
                        hf_nodes = d.net.nodes.iter().map(|n| (n.pressure(), 0.0, 0.0)).collect();
                    }
                    let a = 1.0 - (-2.0 * std::f64::consts::PI * 1000.0 / spec.simulation_frequency).exp();
                    let on = t + s >= from;
                    let update = |st: &mut (f64, f64, f64), x: f64| {
                        st.0 += a * (x - st.0);
                        if on {
                            st.1 += (x - st.0).powi(2);
                            st.2 += 1.0;
                        }
                    };
                    for (k, p) in d.net.pipes.iter().enumerate() {
                        for (i, c) in p.prim.iter().enumerate() {
                            update(&mut hf[k][i], c.p);
                        }
                    }
                    for (k, n) in d.net.nodes.iter().enumerate() {
                        if !n.atmosphere {
                            update(&mut hf_nodes[k], n.pressure());
                        }
                    }
                }
                p_in += probe(&e, "right scroll A");
                p_out += probe(&e, "right turbine out");
                power += d.turbines.iter().map(|&o| d.net.orifices[o].power).sum::<f64>();
                gate += d.wastegates.first().map_or(0.0, |&(o, a)| d.net.orifices[o].cda / a);
                for (sum, name) in probe_sums.iter_mut().zip(&probes) {
                    *sum += d.names.get(name).map_or(f64::NAN, |&n| d.net.nodes[n].pressure());
                }
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
                    "{t:.2} rpm {:.0} thr {thr} torque {:.0} air {:.3} kg/s boost {:.2} map {:.2} turbine {:.2}->{:.2} bar {:.0} kW gate {:.2} port {:.2}..{:.2} bar {:.0} K substeps {:.1} rejected {} wall {:.1}s\n",
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
                    d.net.rejected,
                    started.elapsed().as_secs_f64(),
                );
                if !probes.is_empty() {
                    let d = e.ducts.as_ref().unwrap();
                    let parts: Vec<String> = probes.iter().zip(&probe_sums).map(|(n, s)| format!("{n} {:.3}", s / k / 1e5)).collect();
                    log += &format!("    {} | flap {:.2}\n", parts.join(", "), d.flap);
                }
                probe_sums.iter_mut().for_each(|s| *s = 0.0);
                (p_in, p_out, power, gate, n_mean) = (0.0, 0.0, 0.0, 0.0, 0);
                let lines: Vec<&str> = log.lines().collect();
                let tail = if probes.is_empty() { 1 } else { 2 };
                for l in &lines[lines.len() - tail..] {
                    println!("{l}");
                }
                std::io::stdout().flush().unwrap();
            }
        }
    }
    let d = e.ducts.as_ref().unwrap();
    if hf_from.is_some() {
        // Each pipe's strongest cell, and every volume.
        let names: std::collections::HashMap<usize, &String> = d.names.iter().map(|(n, &i)| (i, n)).collect();
        for (k, cells) in hf.iter().enumerate() {
            let (i, v) = cells.iter().enumerate().map(|(i, st)| (i, (st.1 / st.2.max(1.0)).sqrt())).fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            let mean: f64 = cells.iter().map(|st| (st.1 / st.2.max(1.0)).sqrt()).sum::<f64>() / cells.len() as f64;
            println!("hf pipe {k:2}: max {v:7.0} Pa at cell {i}/{} ({:.3} m), mean {mean:7.0} Pa", cells.len(), (i as f64 + 0.5) * d.net.pipes[k].dx);
        }
        for (k, st) in hf_nodes.iter().enumerate() {
            if st.2 > 0.0 {
                println!("hf node {:24}: {:7.0} Pa", names.get(&k).map_or("?", |s| s.as_str()), (st.1 / st.2).sqrt());
            }
        }
    }
    if !traces.is_empty() {
        let bytes: Vec<u8> = trace_out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{out}.trace.f32"), bytes).unwrap();
    }
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
