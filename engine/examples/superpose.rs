//! Whether the exhaust's response to the burns superposes: from one warmed-up state on the
//! dynamometer, the engine is run on (branches, each a clone) with burns scaled, singly and in
//! pairs, recording each open end's sound (before directivity) and each cylinder's exhaust
//! valve flow. With y the sound, y(a+b) - y(a) - y(b) + y(base) is what the pair adds beyond
//! the sum of its parts.
//!   superpose <spec.json> <reference.json> <out prefix> <rpm> <throttle> <warm-up s> <record s> <factors>
//! factors: comma separated, each a factor on the fuel a scaled burn burns. The burns: the
//! first to fire 5 ms into the record (A), the next to fire (B), the next on A's bank (C), A's
//! next (A2); branches: base, then per factor A, B, C, A2, A+B, A+C, A+A2.
//! env: FREQ (default 20000), FLUID (default 12), JOBS (branches, default 15;
//! forced to 1 when the duct solver uses its process-global parallel pool),
//! BRANCHES (optional comma-separated subset: A,B,C,A2,A+B,A+C,A+A2).
//! TRACE_STATE records p,u,T,Re for TRACE_FROM..TRACE_TO engine steps (default 200..240).
//! Writes <out>.json and per branch <out>.<branch>.ends.f32 (the ends interleaved, at the
//! fluid rate) and <out>.<branch>.flow.f32 (each step, each cylinder's exhaust valve flow).
use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;
use std::io::Write;

struct Run {
    ends: Vec<f32>,
    flow: Vec<f32>,
    /// Per step, each cylinder's burns so far.
    burns: Vec<Vec<u64>>,
    angles: Vec<f64>,
    rejections: usize,
    states: Vec<f64>,
}

fn run(mut e: Engine, steps: usize, scales: Vec<(usize, u64, f64)>) -> Run {
    e.burn_scales = scales;
    let n = e.cylinder_count();
    let mut flow = Vec::with_capacity(steps * n);
    let mut burns = Vec::with_capacity(steps);
    let mut angles = Vec::with_capacity(steps);
    let trace = std::env::var("TRACE_STATE").is_ok();
    let trace_from = std::env::var("TRACE_FROM").ok().map_or(200, |v| v.parse::<usize>().unwrap());
    let trace_to = std::env::var("TRACE_TO").ok().map_or(240, |v| v.parse::<usize>().unwrap());
    let mut states = Vec::new();
    for step in 0..steps {
        e.step();
        if trace && (trace_from..trace_to).contains(&step) {
            for p in &e.ducts.as_ref().unwrap().net.pipes {
                for i in 0..p.cells() {
                    let s = engine_sim::duct::Prim::of(&p.q[i]);
                    states.extend([s.p, s.u, s.t, s.rho * s.u.abs() * p.cell_diameter(i) / engine_sim::duct::viscosity(s.t)]);
                }
            }
        }
        angles.push(e.cycle_angle());
        for i in 0..n {
            flow.push(e.cylinder_exhaust_flow(i) as f32);
        }
        burns.push((0..n).map(|i| e.burns(i)).collect());
    }
    let ends = std::mem::take(&mut e.ducts.as_mut().unwrap().end_sound);
    Run { ends, flow, burns, angles, states, rejections: e.ducts.as_ref().unwrap().net.rejected }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let out = &args[3];
    let (rpm, throttle, warm, record): (f64, f64, f64, f64) =
        (args[4].parse().unwrap(), args[5].parse().unwrap(), args[6].parse().unwrap(), args[7].parse().unwrap());
    let factors: Vec<f64> = args[8].split(',').map(|x| x.parse().unwrap()).collect();
    let env = |k: &str, d: f64| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    merge(&mut spec, &reference);
    spec["simulationFrequency"] = env("FREQ", 20000.0).into();
    spec["fluidSteps"] = (env("FLUID", 12.0) as usize).into();
    let spec: EngineSpec = serde_json::from_value(spec).unwrap();
    let rate = spec.simulation_frequency;
    let requested_jobs = (env("JOBS", 15.0) as usize).max(1);
    // The parallel duct solver has one process-global pool. Independent Engines
    // must not enter it concurrently; a branch can still use all its workers.
    let solver_threads = engine_sim::pool::threads();
    let jobs = if solver_threads > 1 { 1 } else { requested_jobs };
    let selected = std::env::var("BRANCHES").ok().map(|s| s.split(',').map(|x| x.trim().to_owned()).collect::<Vec<_>>());
    if let Some(selected) = &selected {
        assert!(selected.iter().all(|s| ["A", "B", "C", "A2", "A+B", "A+C", "A+A2"].contains(&s.as_str())), "unknown BRANCHES member");
    }

    let started = std::time::Instant::now();
    let mut e = Engine::new(&spec);
    e.collect_audio = false;
    e.set_ignition(true);
    let w = std::f64::consts::PI / 30.0;
    e.set_crank_speed(-rpm * w);
    e.set_dyno(true, true, rpm * w);
    e.set_speed_control(throttle);
    let warm_steps = (warm * rate).round() as usize;
    // MIN_SUBSTEPS: the pipes' substeps a fluid step, at least.
    e.ducts.as_mut().unwrap().net.min_substeps = env("MIN_SUBSTEPS", 1.0) as usize;
    let mut most = 0;
    for k in 0..warm_steps {
        e.step();
        most = most.max(e.ducts.as_ref().unwrap().net.substeps);
        if k % (rate as usize / 4) == 0 {
            let d = e.ducts.as_mut().unwrap();
            (d.sound.clear(), d.exits.clear());
            println!("warm-up {:.2} s, most substeps {most}, torque {:.0}, boost {:.2} bar, wall {:.0} s", k as f64 / rate, e.filtered_dyno_torque(), e.turbo_state().0 / 1e5, started.elapsed().as_secs_f64());
            std::io::stdout().flush().unwrap();
        }
    }
    {
        let d = e.ducts.as_mut().unwrap();
        (d.sound.clear(), d.exits.clear());
        d.listening_ends = true;
    }
    let n = e.cylinder_count();
    let steps = (record * rate).round() as usize;

    // The base branch: when each cylinder fires.
    let base = run(e.clone(), steps, vec![]);
    let start: Vec<u64> = (0..n).map(|i| e.burns(i)).collect();
    // Each burn in the record: (step, cylinder, its count).
    let mut fired: Vec<(usize, usize, u64)> = Vec::new();
    for (s, b) in base.burns.iter().enumerate() {
        let prev = if s == 0 { &start } else { &base.burns[s - 1] };
        for i in 0..n {
            if b[i] > prev[i] {
                fired.push((s, i, prev[i]));
            }
        }
    }
    let lead = (0.005 * rate) as usize;
    let k = fired.iter().position(|f| f.0 >= lead).expect("a burn in the record");
    let a = fired[k];
    let b = fired[k + 1];
    let c = *fired[k + 1..].iter().find(|f| e.cylinder_bank(f.1) == e.cylinder_bank(a.1)).expect("a burn on A's bank");
    let a2 = *fired[k + 1..].iter().find(|f| f.1 == a.1).expect("A's next burn");
    println!("burns (step, cylinder, count): A {a:?} B {b:?} C {c:?} A2 {a2:?}");

    let mut branches: Vec<(String, Vec<(usize, u64, f64)>)> = Vec::new();
    for &f in &factors {
        let s = |x: (usize, usize, u64)| (x.1, x.2, f);
        for (name, set) in [
            ("A", vec![s(a)]),
            ("B", vec![s(b)]),
            ("C", vec![s(c)]),
            ("A2", vec![s(a2)]),
            ("A+B", vec![s(a), s(b)]),
            ("A+C", vec![s(a), s(c)]),
            ("A+A2", vec![s(a), s(a2)]),
        ] {
            if selected.as_ref().is_none_or(|s| s.iter().any(|x| x == name)) {
                branches.push((format!("{name}@{f}"), set));
            }
        }
    }
    let results: Vec<Run> = std::thread::scope(|scope| {
        let mut done: Vec<Option<Run>> = (0..branches.len()).map(|_| None).collect();
        for chunk in (0..branches.len()).collect::<Vec<_>>().chunks(jobs) {
            let handles: Vec<_> = chunk.iter().map(|&j| (j, scope.spawn({
                let e = e.clone();
                let set = branches[j].1.clone();
                move || run(e, steps, set)
            }))).collect();
            for (j, h) in handles {
                done[j] = Some(h.join().unwrap());
            }
            println!("{} branches done, wall {:.0} s", chunk.last().unwrap() + 1, started.elapsed().as_secs_f64());
            std::io::stdout().flush().unwrap();
        }
        done.into_iter().map(|r| r.unwrap()).collect()
    });

    let write = |name: &str, r: &Run| {
        std::fs::write(format!("{out}.{name}.ends.f32"), bytes(&r.ends)).unwrap();
        std::fs::write(format!("{out}.{name}.flow.f32"), bytes(&r.flow)).unwrap();
        if !r.states.is_empty() {
            let bytes: Vec<_> = r.states.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{out}.{name}.states.f64"), bytes).unwrap();
        }
    };
    write("base", &base);
    println!("base rejections {}", base.rejections);
    for ((name, _), r) in branches.iter().zip(&results) {
        write(name, r);
        let changes: Vec<_> = r.burns.iter().zip(&base.burns).enumerate().flat_map(|(s, (a, b))| {
            a.iter().zip(b).enumerate().filter_map(move |(i, (a, b))| (a != b).then_some((s, i)))
        }).collect();
        let angle = r.angles.iter().zip(&base.angles).map(|(a, b)| (a-b).abs()).fold(0.0, f64::max);
        println!("{name}: rejections {}, angle diff {angle:e}, burn-count mismatches {} first {:?}", r.rejections, changes.len(), &changes[..changes.len().min(20)]);

    }
    let d = e.ducts.as_ref().unwrap();
    let meta = serde_json::json!({
        "rpm": rpm, "throttle": throttle, "warm": warm, "record": record,
        "rate": rate, "fluidRate": rate * spec.fluid_steps as f64,
        "cylinders": n, "ends": d.outlets.len(), "distance": d.distance,
        "banks": (0..n).map(|i| e.cylinder_bank(i)).collect::<Vec<_>>(),
        "burns": {"A": [a.0, a.1], "B": [b.0, b.1], "C": [c.0, c.1], "A2": [a2.0, a2.1]},
        "factors": factors,
        "solverThreads": solver_threads, "branchJobs": jobs,
        "minSubsteps": d.net.min_substeps,
        "stateTrace": {"from": env("TRACE_FROM", 200.0) as usize, "to": env("TRACE_TO", 240.0) as usize, "columns": ["p", "u", "T", "Re"]},
        "cells": d.net.pipes.iter().map(|p| p.cells()).collect::<Vec<_>>(),
        "branches": branches.iter().map(|b| b.0.clone()).collect::<Vec<_>>(),
    });
    std::fs::write(format!("{out}.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("done in {:.0} s", started.elapsed().as_secs_f64());
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
