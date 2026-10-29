//! Calibrates an engine spec's combustion against physical targets, on the dynamometer.
//!   calibrate measure <spec.json> <rpm> <boost cap bar> [throttle] [retard deg]
//!   calibrate boost   <spec.json> <rpm> <max boost bar> <torque Nm>   boost cap for the torque
//!   calibrate mbt     <spec.json> <rpm> <boost cap bar>          best-torque spark advance
//!   calibrate ca50    <spec.json> <rpm> <boost cap bar> <deg ATDC> [base|full]
//!       the spark advance putting 50 % burned at the given angle, by shifting the base
//!       (part-load) or the full-load timing curve
//! Prints the measured spark advance (deg BTDC), torque, brake efficiency (shaft work over
//! the heat the burn releases), the burn's 10/50/90 % points and peak pressure (deg ATDC),
//! boost and the mean exhaust-runner temperature.
use engine_sim::engine::Engine;
use engine_sim::spec::EngineSpec;

struct Result {
    spark: f64,
    torque: f64,
    efficiency: f64,
    b10: f64,
    b50: f64,
    b90: f64,
    peak: f64,
    boost: f64,
    exhaust_t: f64,
}

fn measure(spec: &EngineSpec, rpm: f64, boost: f64, throttle: f64) -> Result {
    measure_retarded(spec, rpm, boost, throttle, 0.0)
}

fn measure_retarded(spec: &EngineSpec, rpm: f64, boost: f64, throttle: f64, retard: f64) -> Result {
    let w = rpm * std::f64::consts::PI / 30.0;
    let mut e = Engine::new(spec);
    e.set_ignition(true);
    e.set_crank_speed(-w);
    e.set_dyno(true, true, w);
    e.set_speed_control(throttle);
    e.set_boost_limit(boost * 1e5);
    e.set_spark_retard(retard.to_radians());
    e.collect_audio = false;
    for _ in 0..20000 {
        e.step();
    }
    let deg = rpm * 6.0 / 10000.0;
    let per = (10000.0 * 120.0 / rpm).round() as usize;
    let q0 = e.combustion_energy();
    let (mut torque, mut n_t, mut steps, mut boost_sum, mut t_ex) = (0.0, 0, 0usize, 0.0, 0.0);
    let mut rows: Vec<[f64; 5]> = Vec::new();
    for _ in 0..12 {
        let (mut v, mut p, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..2 * per {
            e.step();
            v.push(e.cylinder_volume(0));
            p.push(e.cylinder_pressure(0));
            b.push(e.cylinder_burned(0));
            boost_sum += e.turbo_state().0;
            t_ex += e.exhaust_runner_temperature(0);
            steps += 1;
        }
        torque += e.filtered_dyno_torque();
        n_t += 1;
        // Firing TDC: of the volume minima with room around them, the one at highest pressure.
        let (lo_lim, hi_lim) = (per / 3, 2 * per - per / 2);
        let tdc = (lo_lim.max(1)..hi_lim)
            .filter(|&i| v[i] <= v[i - 1] && v[i] <= v[i + 1])
            .max_by(|&i, &j| p[i].total_cmp(&p[j]));
        let Some(tdc) = tdc else { continue };
        let (y0, y1, y2) = (v[tdc - 1], v[tdc], v[tdc + 1]);
        let t0 = tdc as f64 + 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2);
        let (lo, hi) = (tdc - per / 4, (tdc + per / 3).min(b.len() - 1));
        let spark = (lo + 1..hi).find(|&i| b[i] < b[i - 1] || (b[i - 1] == 0.0 && b[i] > 0.0));
        let cross = |f: f64| {
            (lo + 1..hi).find(|&i| b[i - 1] < f && b[i] >= f).map(|i| i as f64 - 1.0 + (f - b[i - 1]) / (b[i] - b[i - 1]))
        };
        let pk = (lo..hi).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
        let (q0p, q1p, q2p) = (p[pk - 1], p[pk], p[pk + 1]);
        let pkf = pk as f64 + 0.5 * (q0p - q2p) / (q0p - 2.0 * q1p + q2p);
        let ang = |x: Option<f64>| x.map_or(f64::NAN, |x| (x - t0) * deg);
        rows.push([ang(spark.map(|s| s as f64)), ang(cross(0.1)), ang(cross(0.5)), ang(cross(0.9)), ang(Some(pkf))]);
    }
    let med = |k: usize| {
        let mut x: Vec<f64> = rows.iter().map(|r| r[k]).filter(|x| x.is_finite()).collect();
        x.sort_by(|a, b| a.total_cmp(b));
        x.get(x.len() / 2).copied().unwrap_or(f64::NAN)
    };
    let torque = torque / n_t as f64;
    let heat = (e.combustion_energy() - q0) / (steps as f64 / 10000.0);
    Result {
        spark: -med(0),
        torque,
        efficiency: torque * w / heat,
        b10: med(1),
        b50: med(2),
        b90: med(3),
        peak: med(4),
        boost: boost_sum / steps as f64 / 1e5,
        exhaust_t: t_ex / steps as f64,
    }
}

fn shift(spec: &EngineSpec, deg: f64, full: bool) -> EngineSpec {
    let mut s = spec.clone();
    let f = if full { &mut s.ignition.full_load.as_mut().expect("no full-load timing").timing } else { &mut s.ignition.timing };
    for x in &mut f.samples {
        x.1 += deg.to_radians();
    }
    s
}

fn print(r: &Result) {
    println!(
        "spark {:5.1} BTDC | {:5.0} Nm | efficiency {:4.1}% | 10% {:5.1} 50% {:5.1} 90% {:5.1} (10-90 {:4.1}) | peak p {:5.1} | boost {:4.2} bar | exhaust {:4.0} K",
        r.spark, r.torque, 100.0 * r.efficiency, r.b10, r.b50, r.b90, r.b90 - r.b10, r.peak, r.boost, r.exhaust_t
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let spec: EngineSpec = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    let rpm: f64 = a[3].parse().unwrap();
    let boost: f64 = a[4].parse().unwrap();
    match a[1].as_str() {
        "measure" => print(&measure_retarded(
            &spec,
            rpm,
            boost,
            a.get(5).map_or(1.0, |t| t.parse().unwrap()),
            a.get(6).map_or(0.0, |t| t.parse().unwrap()),
        )),
        "mbt" => {
            // Golden-section search on the offset of the base curve.
            let g = (5f64.sqrt() - 1.0) / 2.0;
            let (mut lo, mut hi) = (-30.0f64, 40.0f64);
            let f = |d: f64| measure(&shift(&spec, d, false), rpm, boost, 1.0).torque;
            let (mut x1, mut x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
            let (mut f1, mut f2) = (f(x1), f(x2));
            while hi - lo > 1.0 {
                if f1 < f2 {
                    lo = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = lo + g * (hi - lo);
                    f2 = f(x2);
                } else {
                    hi = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = hi - g * (hi - lo);
                    f1 = f(x1);
                }
            }
            print(&measure(&shift(&spec, (lo + hi) / 2.0, false), rpm, boost, 1.0));
        }
        "ca50" => {
            let target: f64 = a[5].parse().unwrap();
            let full = a.get(6).is_some_and(|s| s == "full");
            // CA50 falls as the spark advances; bisect on the offset.
            let (mut lo, mut hi) = (-40.0f64, 50.0f64);
            let mut r = measure(&shift(&spec, 0.0, full), rpm, boost, 1.0);
            for _ in 0..9 {
                let mid = (lo + hi) / 2.0;
                r = measure(&shift(&spec, mid, full), rpm, boost, 1.0);
                if r.b50.is_nan() || r.b50 > target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            print(&r);
        }
        "boost" => {
            // The boost cap giving the target torque (full throttle), by bisection.
            let target: f64 = a[5].parse().unwrap();
            let (mut lo, mut hi) = (0.0f64, boost);
            let mut r = measure(&spec, rpm, hi, 1.0);
            if r.torque < target {
                print!("(short at the cap) ");
            } else {
                for _ in 0..7 {
                    let mid = (lo + hi) / 2.0;
                    r = measure(&spec, rpm, mid, 1.0);
                    if r.torque < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                r = measure(&spec, rpm, hi, 1.0);
            }
            println!("cap {hi:.3} bar:");
            print(&r);
        }
        m => panic!("unknown mode {m}"),
    }
}
