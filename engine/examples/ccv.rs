//! Cycle-to-cycle variation on the dynamometer: per cylinder and cycle, the indicated mean
//! effective pressure (the cycle's p dV over the swept volume) and the peak pressure; prints
//! their coefficients of variation.
//!   ccv <spec.json> <rpm> <throttle> <flame speed sigma> [cycles]
//! env REF=<reference config> (keys merged into the spec: ducts, ...), FREQ, FLUID.
//! Also splits cylinder 1's mean cycle into gross (compression and expansion) and pumping
//! work.
use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let rpm: f64 = a[2].parse().unwrap();
    let throttle: f64 = a[3].parse().unwrap();
    let sigma: f64 = a[4].parse().unwrap();
    let cycles: usize = a.get(5).map_or(150, |s| s.parse().unwrap());
    if sigma > 0.0 {
        spec["physics"]["cycleVariation"] = sigma.into();
    }
    if let Ok(path) = std::env::var("REF") {
        let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for (k, v) in r.as_object().unwrap() {
            if k == "physics" {
                for (pk, pv) in v.as_object().unwrap() {
                    spec["physics"][pk] = pv.clone();
                }
            } else {
                spec[k] = v.clone();
            }
        }
        if let Some(f) = std::env::var("FREQ").ok().and_then(|v| v.parse::<f64>().ok()) {
            spec["simulationFrequency"] = f.into();
        }
        if let Some(f) = std::env::var("FLUID").ok().and_then(|v| v.parse::<usize>().ok()) {
            spec["fluidSteps"] = f.into();
        }
    }
    let spec: EngineSpec = serde_json::from_value(spec).unwrap();
    let w = rpm * std::f64::consts::PI / 30.0;
    let mut e = Engine::new(&spec);
    e.collect_audio = false;
    e.set_ignition(true);
    e.set_crank_speed(-w);
    e.set_dyno(true, true, w);
    e.set_speed_control(throttle);
    for _ in 0..(2.0 * spec.simulation_frequency) as usize {
        e.step();
    }
    let n = spec.banks.iter().map(|b| b.cylinders.len()).sum::<usize>();
    let per = (spec.simulation_frequency * 120.0 / rpm).round() as usize;
    let (mut imep, mut pmax) = (vec![Vec::new(); n], vec![Vec::new(); n]);
    let vmin: Vec<f64> = (0..n).map(|i| e.cylinder_volume(i)).collect();
    let _ = vmin;
    // Swept volume from the volumes seen over a cycle.
    let (mut lo, mut hi) = (vec![f64::INFINITY; n], vec![0.0f64; n]);
    for _ in 0..per {
        e.step();
        for i in 0..n {
            lo[i] = lo[i].min(e.cylinder_volume(i));
            hi[i] = hi[i].max(e.cylinder_volume(i));
        }
    }
    let (mut collector, mut gate, mut samples) = (0.0, 0.0, 0usize);
    let (mut gross, mut pumping) = (0.0, 0.0);
    for _ in 0..cycles {
        let mut trace: Vec<(f64, f64)> = Vec::with_capacity(per + 1);
        trace.push((e.cylinder_pressure(0), e.cylinder_volume(0)));
        let mut work = vec![0.0; n];
        let mut peak = vec![0.0f64; n];
        let mut last: Vec<(f64, f64)> = (0..n).map(|i| (e.cylinder_pressure(i), e.cylinder_volume(i))).collect();
        for _ in 0..per {
            e.step();
            trace.push((e.cylinder_pressure(0), e.cylinder_volume(0)));
            if e.ducts.is_none() {
                collector += e.exhaust_pressure(0);
                gate += e.wastegate();
                samples += 1;
            }
            for i in 0..n {
                let (p, v) = (e.cylinder_pressure(i), e.cylinder_volume(i));
                work[i] += 0.5 * (p + last[i].0) * (v - last[i].1);
                peak[i] = peak[i].max(p);
                last[i] = (p, v);
            }
        }
        for i in 0..n {
            imep[i].push(work[i] / (hi[i] - lo[i]));
            pmax[i].push(peak[i]);
        }
        // Gross work: between the bottom centres either side of the peak pressure, the cycle
        // taken as periodic (its last sample is its first).
        let m = per;
        let at = |j: isize| trace[j.rem_euclid(m as isize) as usize];
        let k = (0..m).max_by(|&a, &b| trace[a].0.total_cmp(&trace[b].0)).unwrap() as isize;
        let half = (m / 2) as isize;
        let a = (k - half..k).max_by(|&x, &y| at(x).1.total_cmp(&at(y).1)).unwrap();
        let b = (k..k + half).max_by(|&x, &y| at(x).1.total_cmp(&at(y).1)).unwrap();
        let w: f64 = (a + 1..=b).map(|j| 0.5 * (at(j).0 + at(j - 1).0) * (at(j).1 - at(j - 1).1)).sum();
        gross += w / (hi[0] - lo[0]);
        pumping += (work[0] - w) / (hi[0] - lo[0]);
    }
    println!("cylinder 1: gross IMEP {:.2} bar, pumping {:.2} bar ", gross / cycles as f64 / 1e5, pumping / cycles as f64 / 1e5);
    let stats = |x: &[f64]| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let s = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        (m, 100.0 * s / m)
    };
    let all_i: Vec<f64> = imep.iter().flatten().copied().collect();
    let all_p: Vec<f64> = pmax.iter().flatten().copied().collect();
    for i in 0..n {
        let (mi, ci) = stats(&imep[i]);
        let (mp, cp) = stats(&pmax[i]);
        println!("cylinder {}: IMEP {:.2} bar COV {ci:.2} %, peak {:.1} bar COV {cp:.1} %", i + 1, mi / 1e5, mp / 1e5);
    }
    let (mi, ci) = stats(&all_i);
    let (mp, cp) = stats(&all_p);
    if samples > 0 {
        println!("collector {:.2} bar, wastegate {:.2}", collector / samples as f64 / 1e5, gate / samples as f64);
    }
    println!("all: IMEP {:.2} bar COV {ci:.2} %, peak {:.1} bar COV {cp:.1} %, torque {:.0} Nm, air {:.3} kg/s", mi / 1e5, mp / 1e5, e.filtered_dyno_torque(), e.air_mass_flow());
}
