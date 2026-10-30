//! The pipe-network solver (src/duct.rs) against exact solutions and conservation.

use engine_sim::duct::*;

const ATM: f64 = 101325.0;

fn set_cell(p: &mut Pipe, i: usize, rho: f64, u: f64, pr: f64) {
    let s = Prim::new(rho, u, pr, 0.0, 0.25);
    let e = rho * (energy(s.t) + 0.5 * u * u);
    p.q[i] = [rho, rho * u, e, 0.0, 0.25 * rho];
}

/// A pipe closed at both ends (no flow through a node's zero opening).
fn closed(pipe: Pipe) -> Network {
    let mut n = Network::default();
    let a = n.add_node(Node::atmosphere(ATM, 300.0, 0.0, 0.25));
    let p = n.add_pipe(pipe);
    n.link(p, Side::A, a, 0.0);
    n.link(p, Side::B, a, 0.0);
    n
}

/// Toro's pressure function across a wave from state (rho, p) with heat capacity ratio g.
fn f_wave(p: f64, rho: f64, pk: f64, g: f64) -> f64 {
    let c = (g * pk / rho).sqrt();
    if p > pk {
        let a = 2.0 / ((g + 1.0) * rho);
        let b = (g - 1.0) / (g + 1.0) * pk;
        (p - pk) * (a / (p + b)).sqrt()
    } else {
        2.0 * c / (g - 1.0) * ((p / pk).powf((g - 1.0) / (2.0 * g)) - 1.0)
    }
}

#[test]
fn shock_tube() {
    let (rl, pl, rr, pr) = (1.0, 1e5, 0.125, 1e4);
    let mut pipe = Pipe::new(1.0, 0.05, 0.0025, ATM, 300.0, 0.0, 0.25);
    pipe.friction = false;
    let n_cells = pipe.cells();
    for i in 0..n_cells {
        if i < n_cells / 2 {
            set_cell(&mut pipe, i, rl, 0.0, pl);
        } else {
            set_cell(&mut pipe, i, rr, 0.0, pr);
        }
    }
    let mut net = closed(pipe);
    let t_end = 6e-4;
    let mut t: f64 = 0.0;
    while t < t_end - 1e-12 {
        let dt = (t_end - t).min(1e-5);
        net.step(dt);
        t += dt;
    }
    // The exact star pressure and velocity, each side's wave at its own gas's gamma.
    let (gl, gr) = (gamma(pl / (rl * RS)), gamma(pr / (rr * RS)));
    let (mut lo, mut hi) = (pr, pl);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if f_wave(m, rl, pl, gl) + f_wave(m, rr, pr, gr) > 0.0 {
            hi = m;
        } else {
            lo = m;
        }
    }
    let ps = 0.5 * (lo + hi);
    let us = 0.5 * (f_wave(ps, rr, pr, gr) - f_wave(ps, rl, pl, gl));
    // The shock's speed from the jump in mass flux.
    let k = (gr - 1.0) / (gr + 1.0);
    let rs = rr * (ps / pr + k) / (k * ps / pr + 1.0);
    let shock = rs * us / (rs - rr);
    let pipe = &net.pipes[0];
    let x = |i: usize| (i as f64 + 0.5) * pipe.dx;
    let contact = 0.5 + us * t_end;
    let front = 0.5 + shock * t_end;
    let mid = 0.5 * (contact + front);
    let cell = |x0: f64| (x0 / pipe.dx) as usize;
    let p_mid = pipe.prim[cell(mid)].p;
    let u_mid = pipe.prim[cell(mid)].u;
    println!("star p {ps:.1} sim {p_mid:.1}, u {us:.2} sim {u_mid:.2}, shock at {front:.4}");
    assert!((p_mid / ps - 1.0).abs() < 0.005, "star pressure {p_mid} vs {ps}");
    assert!((u_mid / us - 1.0).abs() < 0.01, "star velocity {u_mid} vs {us}");
    let half = 0.5 * (ps + pr);
    let found = (0..pipe.cells()).rev().find(|&i| pipe.prim[i].p > half).map(x).unwrap();
    println!("shock found at {found:.4}");
    assert!((found - front).abs() < 2.0 * pipe.dx, "shock at {found} vs {front}");
}

fn pulse_pipe(length: f64, t: f64, amplitude: f64, x0: f64, width: f64) -> Pipe {
    let mut pipe = Pipe::new(length, 0.05, 0.01, ATM, t, 0.0, 0.25);
    pipe.friction = false;
    let rho0 = ATM / (RS * t);
    let c = (gamma(t) * RS * t).sqrt();
    for i in 0..pipe.cells() {
        let x = (i as f64 + 0.5) * pipe.dx;
        let dp = amplitude * (-((x - x0) / width).powi(2)).exp();
        // A wave at rest splits in two: pressure and density alone.
        set_cell(&mut pipe, i, rho0 + dp / (c * c), 0.0, ATM + dp);
    }
    pipe
}

fn run(net: &mut Network, t_end: f64) {
    let mut t: f64 = 0.0;
    while t < t_end - 1e-12 {
        let dt = (t_end - t).min(2e-5);
        net.step(dt);
        t += dt;
    }
}

/// Peak of a pressure excursion (sign `s`) and where it is.
fn peak(pipe: &Pipe, s: f64, from: f64, to: f64) -> (f64, f64) {
    let mut best = (0.0, 0.0);
    for i in 0..pipe.cells() {
        let x = (i as f64 + 0.5) * pipe.dx;
        if x < from || x > to {
            continue;
        }
        let dp = s * (pipe.prim[i].p - ATM);
        if dp > best.0 {
            // Refine by a parabola through the neighbours.
            let (a, b, c) = (pipe.prim[i - 1].p, pipe.prim[i].p, pipe.prim[i + 1].p);
            let off = 0.5 * (a - c) / (a - 2.0 * b + c);
            best = (dp, x + off * pipe.dx);
        }
    }
    best
}

#[test]
fn sound_speed_and_closed_end() {
    let t = 600.0;
    let c = (gamma(t) * RS * t).sqrt();
    let mut net = closed(pulse_pipe(3.0, t, 200.0, 1.0, 0.05));
    let time = 1.0 / c;
    run(&mut net, time);
    let (a, x) = peak(&net.pipes[0], 1.0, 1.5, 2.5);
    println!("c {c:.1}: peak {a:.2} Pa at {x:.4} (expected 2.0)");
    assert!((x - 2.0).abs() < 0.003, "pulse at {x}");
    assert!((a / 100.0 - 1.0).abs() < 0.02, "amplitude {a}");
    // Off the closed end at 3 m and back to 2.5 m: the same sign. (The pulse is 5 cells
    // wide: its few kHz lose ~1.5 % a metre, numerical_damping's rate there.)
    run(&mut net, 1.5 / c);
    let (a, x) = peak(&net.pipes[0], 1.0, 2.0, 3.0);
    println!("after the closed end: {a:.2} Pa at {x:.4} (expected 2.5)");
    assert!((x - 2.5).abs() < 0.005 && (a / 100.0 - 1.0).abs() < 0.05);
}

#[test]
fn open_end_reflects_inverted() {
    let t = 600.0;
    let c = (gamma(t) * RS * t).sqrt();
    let mut net = Network::default();
    let atm = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
    let p = net.add_pipe(pulse_pipe(3.0, t, 200.0, 1.0, 0.05));
    net.link(p, Side::A, atm, 0.0);
    net.connect(p, Side::B, atm);
    run(&mut net, 2.5 / c);
    let (a, x) = peak(&net.pipes[0], -1.0, 2.0, 3.0);
    println!("after the open end: {a:.2} Pa below at {x:.4} (expected 2.5)");
    assert!((x - 2.5).abs() < 0.005 && (a / 100.0 - 1.0).abs() < 0.05);
}

#[test]
fn choked_orifice() {
    // A large vessel at 2.5 bar through a pipe to the atmosphere through a third of its
    // section: once steady, the outlet passes what a choked nozzle does.
    let mut net = Network::default();
    let (p0, t0) = (2.5e5, 400.0);
    let tank = net.add_node(Node::volume(100.0, p0, t0, 0.0, 0.25));
    let atm = net.add_node(Node::atmosphere(ATM, 300.0, 0.0, 0.25));
    let mut pipe = Pipe::new(0.5, 0.04, 0.01, p0, t0, 0.0, 0.25);
    pipe.friction = false;
    let area = pipe.area;
    let p = net.add_pipe(pipe);
    net.connect(p, Side::A, tank);
    let out = net.link(p, Side::B, atm, area / 3.0);
    run(&mut net, 0.05);
    let tank_now = net.nodes[tank].reservoir();
    let expected = area / 3.0 * orifice(tank_now.p, tank_now.t, gamma(tank_now.t), ATM);
    let m = net.links[out].flow.m;
    println!("outflow {m:.6} kg/s, nozzle {expected:.6}");
    assert!((m / expected - 1.0).abs() < 0.003);
}

#[test]
fn conservation() {
    // Two vessels and three pipes, closed to the outside: mass and energy stay.
    let mut net = Network::default();
    let a = net.add_node(Node::volume(1e-3, 3e5, 900.0, 0.0, 0.0));
    let b = net.add_node(Node::volume(2e-3, 1e5, 300.0, 0.05, 0.2));
    let atm = net.add_node(Node::atmosphere(ATM, 300.0, 0.0, 0.25));
    let p1 = net.add_pipe(Pipe::new(0.4, 0.03, 0.01, 1.5e5, 600.0, 0.0, 0.1));
    let p2 = net.add_pipe(Pipe::new(0.3, 0.05, 0.01, 1e5, 400.0, 0.0, 0.25));
    let p3 = net.add_pipe(Pipe::new(0.2, 0.02, 0.01, 2e5, 700.0, 0.0, 0.0));
    net.link(p1, Side::A, a, 4e-4);
    net.connect(p1, Side::B, b);
    net.connect(p2, Side::A, b);
    net.link(p2, Side::B, atm, 0.0);
    net.connect(p3, Side::A, a);
    net.link(p3, Side::B, b, 1e-4);
    net.orifice(a, b, 5e-5, 0.0);
    let (m0, e0) = net.totals();
    run(&mut net, 0.02);
    let (m1, e1) = net.totals();
    println!("mass {:.3e}, energy {:.3e}", m1 / m0 - 1.0, e1 / e0 - 1.0);
    assert!((m1 / m0 - 1.0).abs() < 1e-10);
    assert!((e1 / e0 - 1.0).abs() < 1e-10);
}

#[test]
fn wall_friction() {
    // Steady flow down a long pipe: the pressure falls as Darcy's (4 Cf L / D) dynamic head.
    let mut net = Network::default();
    let up = net.add_node(Node::volume(100.0, 1.1e5, 300.0, 0.0, 0.25));
    let atm = net.add_node(Node::atmosphere(ATM, 300.0, 0.0, 0.25));
    let p = net.add_pipe(Pipe::new(6.0, 0.03, 0.02, ATM, 300.0, 0.0, 0.25));
    net.connect(p, Side::A, up);
    net.connect(p, Side::B, atm);
    run(&mut net, 0.4);
    let pipe = &net.pipes[0];
    let (i, j) = (pipe.cells() / 4, 3 * pipe.cells() / 4);
    let (a, b) = (pipe.prim[i], pipe.prim[j]);
    let l = (j - i) as f64 * pipe.dx;
    let mid = pipe.prim[pipe.cells() / 2];
    let re = mid.rho * mid.u * pipe.diameter / viscosity(mid.t);
    let cf = friction_factor(re, pipe.roughness / pipe.diameter);
    let expected = 4.0 * cf * l / pipe.diameter * 0.5 * mid.rho * mid.u * mid.u;
    // Momentum also changes as the gas speeds up with falling density.
    let accel = b.rho * b.u * b.u - a.rho * a.u * a.u;
    let drop = a.p - b.p - accel;
    println!("Re {re:.0}, Cf {cf:.5}: drop {drop:.1} Pa, Darcy {expected:.1}");
    assert!((drop / expected - 1.0).abs() < 0.02);
}

/// Numerical damping of a sine wave in 1 cm cells, by cells to its wavelength: at 30 (what
/// the reference sizes cells for, duct::CELLS_PER_WAVELENGTH) far below an exhaust's
/// physical losses (its walls' alone are ~1 dB/m at 1 kHz in a hot 36 mm pipe). Fifth-order
/// upwinding damps as its sixth power: 12 cells take 1 dB/m.
#[test]
fn numerical_damping() {
    let t = 900.0;
    let c = (gamma(t) * RS * t).sqrt();
    for (f, limit) in [(500.0, 0.001), (1000.0, 0.003), (1950.0, 0.03), (5000.0, 1.5)] {
        // A packet of 6 cycles, measured after 2 m.
        let lambda = c / f;
        let half = 3.0 * lambda;
        let x0 = half + 0.1;
        let length = 2.0 * half + 2.5;
        let mut pipe = Pipe::new(length, 0.05, 0.01, ATM, t, 0.0, 0.25);
        pipe.friction = false;
        let rho0 = ATM / (RS * t);
        for i in 0..pipe.cells() {
            let x = (i as f64 + 0.5) * pipe.dx;
            let env = if (x - x0).abs() < half { (std::f64::consts::PI * (x - x0) / (2.0 * half)).cos().powi(2) } else { 0.0 };
            let dp = 20.0 * env * (2.0 * std::f64::consts::PI * (x - x0) / lambda).sin();
            // A right-going wave: u = p / (rho c).
            set_cell(&mut pipe, i, rho0 + dp / (c * c), dp / (rho0 * c), ATM + dp);
        }
        let energy_of = |p: &Pipe| p.q.iter().map(|q| (Prim::of(q).p - ATM).powi(2)).sum::<f64>();
        let e0 = energy_of(&pipe);
        let mut net = closed(pipe);
        run(&mut net, 2.0 / c);
        let e1 = energy_of(&net.pipes[0]);
        let db_per_m = -10.0 * (e1 / e0).log10() / 2.0;
        println!("{f} Hz ({:.0} cells per wavelength): {db_per_m:.4} dB/m", lambda / 0.01);
        assert!(db_per_m < limit, "{f} Hz: {db_per_m} dB/m");
    }
}

/// A sine packet's energy lost on reflecting off a closed end and an open one.
#[test]
fn reflection_loss() {
    let t = 900.0;
    let c = (gamma(t) * RS * t).sqrt();
    for f in [500.0, 1000.0, 2000.0, 3000.0] {
        for open in [false, true] {
            let lambda = c / f;
            let length = 6.0 * lambda + 1.0;
            let mut pipe = Pipe::new(length, 0.05, 0.01, ATM, t, 0.0, 0.25);
            pipe.friction = false;
            let rho0 = ATM / (RS * t);
            let (half, x0) = (3.0 * lambda, 3.0 * lambda + 0.1);
            for i in 0..pipe.cells() {
                let x = (i as f64 + 0.5) * pipe.dx;
                let env = if (x - x0).abs() < half { (std::f64::consts::PI * (x - x0) / (2.0 * half)).cos().powi(2) } else { 0.0 };
                let dp = 20.0 * env * (2.0 * std::f64::consts::PI * (x - x0) / lambda).sin();
                set_cell(&mut pipe, i, rho0 + dp / (c * c), dp / (rho0 * c), ATM + dp);
            }
            let sum = |p: &Pipe| p.q.iter().map(|q| (Prim::of(q).p - ATM).powi(2)).sum::<f64>();
            let e0 = sum(&pipe);
            let mut net = Network::default();
            let atm = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
            let p = net.add_pipe(pipe);
            net.link(p, Side::A, atm, 0.0);
            if open { net.connect(p, Side::B, atm); } else { net.link(p, Side::B, atm, 0.0); }
            // Out to the end at 3 m and back to where it started.
            let mut reference = Network::default();
            let atm2 = reference.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
            let mut far = Pipe::new(length, 0.05, 0.01, ATM, t, 0.0, 0.25);
            far.friction = false;
            far.q.clone_from(&net.pipes[0].q);
            let p2 = reference.add_pipe(far);
            reference.link(p2, Side::A, atm2, 0.0);
            reference.link(p2, Side::B, atm2, 0.0);
            let _ = (e0, p2);
            run(&mut net, 2.0 * (length - x0) / c);
            let e1 = sum(&net.pipes[0]);
            println!("{f} Hz, {}: {:.4} dB in {:.2} m with the reflection", if open { "open" } else { "closed" }, -10.0 * (e1 / e0).log10(), 2.0 * (length - x0));
        }
    }
}

#[test]
fn perforation_conserves_and_rings() {
    // A closed core perforated into a closed chamber (a node), and into an annulus (a pipe):
    // mass and energy stay. Without the annulus, the holes' gas rings against the chamber at
    // the Helmholtz frequency of its mass and the two volumes' compliances in series.
    for with_annulus in [true, false] {
        let t = 600.0;
        let mut net = Network::default();
        let atm = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
        let chamber_v = 2e-3;
        let chamber = net.add_node(Node::volume(chamber_v, ATM, t, 0.0, 0.25));
        // A compact core: 6 cm against a wavelength of ~1 m at the ring.
        let mut core = Pipe::new(0.06, 0.1, 0.0025, 1.002 * ATM, t, 0.0, 0.25);
        core.friction = false;
        let core_v = core.area * core.length;
        let c = net.add_pipe(core);
        net.link(c, Side::A, atm, 0.0);
        net.link(c, Side::B, atm, 0.0);
        let f = Perforation::new(&net.pipes, c, 0.02, 0.04, Outer::Node(chamber), 0.01, 0.004, 0.001).unwrap();
        let hole_area: f64 = f.hole_area * f.v.len() as f64;
        let length = f.length;
        net.perforations.push(f);
        if with_annulus {
            let mut annulus = Pipe::new(0.06, 0.05, 0.0025, 1.01 * ATM, t, 0.0, 0.25);
            annulus.friction = false;
            let a = net.add_pipe(annulus);
            net.link(a, Side::A, atm, 0.0);
            net.link(a, Side::B, atm, 0.0);
            let g = Perforation::new(&net.pipes, a, 0.0, 0.06, Outer::Pipe(c, 0.0), 0.02, 0.004, 0.001).unwrap();
            net.perforations.push(g);
        }
        let (m0, e0) = net.totals();
        let mut v = Vec::new();
        let dt = 5e-6;
        for _ in 0..4000 {
            net.step(dt);
            v.push(net.perforations[0].v[0]);
        }
        let (m1, e1) = net.totals();
        if with_annulus {
            println!("mass {:.2e}, energy {:.2e}", m1 / m0 - 1.0, e1 / e0 - 1.0);
            assert!((m1 / m0 - 1.0).abs() < 1e-10 && (e1 / e0 - 1.0).abs() < 1e-10);
            continue;
        }
        let crossings: Vec<f64> = (1..v.len()).filter(|&i| v[i - 1] < 0.0 && v[i] >= 0.0).map(|i| i as f64 * dt).collect();
        let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
        let rho = ATM / (RS * t);
        let csq = gamma(t) * RS * t;
        let k = rho * csq * (1.0 / chamber_v + 1.0 / core_v);
        let expected = 2.0 * std::f64::consts::PI / (k * hole_area / (rho * length)).sqrt();
        println!("period {:.3} ms, Helmholtz {:.3} ms", period * 1e3, expected * 1e3);
        assert!((period / expected - 1.0).abs() < 0.03);
    }
}

#[test]
fn cone() {
    let t = 600.0;
    let c = (gamma(t) * RS * t).sqrt();
    // At rest, it stays at rest: the walls' push balances the fluxes where the area changes.
    let mut pipe = Pipe::new(2.4, 0.05, 0.01, ATM, t, 0.0, 0.25);
    pipe.friction = false;
    pipe.taper(0.05, 0.1);
    let mut net = closed(pipe);
    run(&mut net, 0.01);
    let umax = net.pipes[0].prim.iter().map(|p| p.u.abs()).fold(0.0, f64::max);
    println!("at rest: |u| < {umax:.2e} m/s");
    assert!(umax < 1e-9);
    // A short pulse through a gentle cone (2:1 in diameter over 2.4 m): its pressure falls
    // as the area's square root, so its energy flux, p^2 A / rho c, holds (the WKB limit of
    // Webster's horn equation). Against the same pulse down a uniform pipe, so the scheme's
    // own damping of the pulse cancels.
    let run_pulse = |taper: bool| {
        let mut pipe = Pipe::new(2.4, 0.05, 0.005, ATM, t, 0.0, 0.25);
        pipe.friction = false;
        if taper {
            pipe.taper(0.05, 0.1);
        }
        let rho0 = ATM / (RS * t);
        for i in 0..pipe.cells() {
            let x = (i as f64 + 0.5) * pipe.dx;
            let dp = 100.0 * (-((x - 0.25) / 0.06).powi(2)).exp();
            set_cell(&mut pipe, i, rho0 + dp / (c * c), dp / (rho0 * c), ATM + dp);
        }
        let mut net = closed(pipe);
        run(&mut net, 1.8 / c);
        peak(&net.pipes[0], 1.0, 1.5, 2.4)
    };
    let (uniform, _) = run_pulse(false);
    let (a, x) = run_pulse(true);
    let d = |x: f64| 0.05 + 0.05 * x / 2.4;
    let expected = d(0.25) / d(x);
    println!("pulse at {x:.3} m: {:.4} of the uniform pipe's ({uniform:.2} Pa), A^-1/2 gives {expected:.4}", a / uniform);
    assert!((a / uniform / expected - 1.0).abs() < 0.02);
}

#[test]
fn sudden_expansion() {
    // Steady flow out of a vessel through a 40 mm pipe stepping to 80 mm: the step loses
    // Borda and Carnot's (u1 - u2)^2 / 2 of total pressure; back the other way, a sudden
    // contraction, next to nothing (no vena contracta here).
    for forward in [true, false] {
        let t = 300.0;
        let mut net = Network::default();
        let tank = net.add_node(Node::volume(100.0, 1.03 * ATM, t, 0.0, 0.25));
        let atm = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
        let (d1, d2) = if forward { (0.04, 0.08) } else { (0.08, 0.04) };
        let mut pipe = Pipe::with_sections(&[Section::round(0.6, d1), Section::round(0.6, d2)], 0.01, ATM, t, 0.0, 0.25);
        pipe.friction = false;
        let p = net.add_pipe(pipe);
        net.connect(p, Side::A, tank);
        net.connect(p, Side::B, atm);
        run(&mut net, 0.3);
        let pipe = &net.pipes[0];
        let total = |q: &Prim| q.p + 0.5 * q.rho * q.u * q.u;
        let (a, b) = (pipe.prim[40], pipe.prim[80]);
        let loss = total(&a) - total(&b);
        let borda = 0.5 * a.rho * (a.u - b.u).powi(2);
        println!("{}: u {:.1} -> {:.1} m/s, total pressure lost {loss:.1} Pa, Borda {borda:.1}", if forward { "expansion" } else { "contraction" }, a.u, b.u);
        if forward {
            assert!((loss / borda - 1.0).abs() < 0.05);
        } else {
            assert!(loss.abs() < 0.1 * 0.5 * b.rho * b.u * b.u);
        }
    }
}

#[test]
fn step_sweep() {
    // solve_step over many states and area ratios: finite, the same mass and energy both
    // sides, and continuous with an unchanged section (the plain Riemann problem's flow).
    let mk = |p: f64, u: f64, t: f64| {
        let pr = Prim::new(p / (RS * t), u, p, 0.0, 0.25);
        EndGas { rho: pr.rho, un: u, p, t, g: pr.g, c: pr.c, yf: 0.0, yo: 0.25 }
    };
    let mut bad = 0;
    for &ratio in &[0.5, 0.86, 1.0, 1.16, 2.0] {
        for &pa in &[0.8e5, 1.0e5, 1.2e5, 2.0e5] {
            for &pb in &[0.8e5, 1.0e5, 1.2e5, 2.0e5] {
                for &ua in &[-150.0, -60.0, -1.0, 0.0, 0.05, 60.0, 150.0] {
                    for &ub in &[-150.0, -60.0, -0.05, 0.0, 1.0, 60.0, 150.0] {
                        for &(ta, tb) in &[(600.0, 600.0), (1000.0, 700.0)] {
                            let (a, b) = (mk(pa, ua, ta), mk(pb, ub, tb));
                            for hint in [0.0, 0.9 * pa, -0.9 * pb] {
                                let (fa, fb, _) = solve_step(&a, 0.005, &b, 0.005 * ratio, hint);
                                // Each side's flow is its own state's, rho u A, and that state lies on
                                // its gas's wave curve (or the end is choked).
                                let own = |f: &EndFlow, area: f64| (f.m - f.rho * f.un * area).abs() <= 1e-6 * f.m.abs() + 1e-8;
                                let ok = [fa.m, fa.energy, fa.p, fa.un, fb.p, fb.un].iter().all(|v| v.is_finite())
                                    && (fa.m + fb.m).abs() <= 1e-9 * (1.0 + fa.m.abs())
                                    && fa.p > 0.0 && fb.p > 0.0
                                    && own(&fa, 0.005) && own(&fb, 0.005 * ratio);
                                if !ok {
                                    bad += 1;
                                    if bad < 6 {
                                        println!("ratio {ratio} a {:?} b {:?} hint {hint}: {:?} {:?}", (pa, ua, ta), (pb, ub, tb), fa, fb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{bad} bad");
    assert_eq!(bad, 0);
}

/// A catalyst between open ends, its walls heating the gas: it stays bounded.
#[test]
fn monolith_alone() {
    // A catalyst between open ends: gap, monolith (0.96 mm square channels, 86 % open),
    // gap; its walls hotter than the gas. It must stay bounded.
    for (walls, steps) in [(false, true), (true, false), (true, true)] {
        let t = 800.0;
        let mono = Section { area: Some(0.86 * std::f64::consts::PI * 0.11 * 0.11 / 4.0), square: true, roughness: 0.0, wall: walls.then_some(1000.0), ..Section::round(0.12, 0.96e-3) };
        let gap = |w: Option<f64>| Section { wall: w, ..Section::round(0.1, 0.11) };
        let mut sections = vec![gap(None), mono.clone(), gap(None)];
        if !steps {
            // The same open area all along.
            sections[0].area = mono.area;
            sections[2].area = mono.area;
        }
        let pipe = Pipe::with_sections(&sections, 0.00615, ATM, t, 0.0, 0.25);
        let mut net = Network::default();
        let a1 = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
        let p = net.add_pipe(pipe);
        net.connect(p, Side::A, a1);
        net.connect(p, Side::B, a1);
        let mut umax = 0.0f64;
        for _ in 0..2000 {
            net.step(2e-5);
            if net.check().is_some() {
                umax = f64::NAN;
                break;
            }
            umax = umax.max(net.pipes[0].prim.iter().map(|c| c.u.abs()).fold(0.0, f64::max));
        }
        println!("walls {walls} steps {steps}: max |u| {umax:.2}");
        // The walls heat the gas 200 K in ~0.4 ms: it rushes out at tens of m/s, no more.
        assert!(umax < 100.0);
    }
}

/// A shock tube across a step of section, from either side: it stays bounded.
#[test]
fn step_shock_tube() {
    for (pl, pr, ratio) in [(1.5e5, 1e5, 0.86), (1e5, 1.5e5, 0.86), (1.5e5, 1e5, 1.16), (1e5, 1.5e5, 1.16), (3e5, 1e5, 0.5), (1e5, 3e5, 0.5)] {
        let t = 800.0;
        let a = std::f64::consts::PI * 0.11 * 0.11 / 4.0;
        let s1 = Section::round(0.5, 0.11);
        let s2 = Section { area: Some(a * ratio), ..Section::round(0.5, 0.11) };
        let mut pipe = Pipe::with_sections(&[s1, s2], 0.00615, ATM, t, 0.0, 0.25);
        pipe.friction = false;
        let n = pipe.cells();
        for i in 0..n {
            let p = if i < n / 2 { pl } else { pr };
            set_cell(&mut pipe, i, p / (RS * t), 0.0, p);
        }
        let mut net = closed(pipe);
        let mut umax = 0.0f64;
        let mut fail = None;
        for k in 0..400 {
            net.step(1e-5);
            if let Some(w) = net.check() {
                fail = Some((k, w));
                break;
            }
            umax = umax.max(net.pipes[0].prim.iter().map(|c| c.u.abs()).fold(0.0, f64::max));
        }
        println!("{pl} | {pr}, ratio {ratio}: max |u| {umax:.1} {:?}", fail);
        // A pressure ratio of 3 drives ~300 m/s through the step; nothing runs away.
        assert!(fail.is_none() && umax < 400.0);
    }
}


/// A blowdown: a vessel at 7 bar opening through half the pipe's section into it. The jet
/// mixes out into the wider pipe (the end subsonic), not on as in a nozzle's diverging part.
#[test]
fn blowdown_into_wider_pipe() {
    let mut net = Network::default();
    let tank = net.add_node(Node::volume(5e-4, 7e5, 1500.0, 0.0, 0.0));
    let atm = net.add_node(Node::atmosphere(ATM, 600.0, 0.0, 0.25));
    let pipe = Pipe::new(1.0, 0.036, 0.005, ATM, 600.0, 0.0, 0.25);
    let area = pipe.area;
    let p = net.add_pipe(pipe);
    let valve = net.link(p, Side::A, tank, 0.5 * area);
    net.connect(p, Side::B, atm);
    let mut mach: f64 = 0.0;
    for _ in 0..400 {
        net.step(1e-5);
        assert!(net.check().is_none());
        let f = &net.links[valve].flow;
        let c_end = (gamma(f.p / (f.rho * RS)) * RS * f.p / (f.rho * RS)).sqrt();
        mach = mach.max(-f.un / c_end);
    }
    println!("highest Mach number at the valve end: {mach:.3}");
    assert!(mach <= 1.0 + 1e-6);
}

#[test]
fn nozzles_conserve() {
    // A closed, pressurised pipe discharging along its length into a closed volume through
    // nozzles: mass and energy stay, and the two settle at one pressure.
    let mut net = Network::default();
    let atm = net.add_node(Node::atmosphere(ATM, 600.0, 0.0, 0.25));
    let plenum = net.add_node(Node::volume(5e-5, ATM, 600.0, 0.0, 0.25));
    let mut pipe = Pipe::with_sections(&[Section { diameter_b: 0.015, ..Section::round(0.3, 0.037) }], 0.006, 2.0 * ATM, 1000.0, 0.0, 0.0);
    pipe.friction = false;
    let p = net.add_pipe(pipe);
    net.link(p, Side::A, atm, 0.0);
    net.link(p, Side::B, atm, 0.0);
    let n = net.pipes[p].cells();
    net.nozzles.push(Nozzles::new(p, 0, plenum, vec![4.8e-4 / n as f64; n], 0.02));
    let (m0, e0) = net.totals();
    for (k, span) in [0.01, 0.04].iter().enumerate() {
        run(&mut net, *span);
        let (m1, e1) = net.totals();
        println!("after {:.2} s: mass {:.1e}, energy {:.1e}", [0.01, 0.05][k], m1 / m0 - 1.0, e1 / e0 - 1.0);
    }
    run(&mut net, 0.05);
    let (m1, e1) = net.totals();
    let pp = net.nodes[plenum].pressure();
    let pmean = net.pipes[p].prim.iter().map(|c| c.p).sum::<f64>() / n as f64;
    println!("mass {:.1e}, energy {:.1e}; plenum {:.0} Pa, pipe {:.0} Pa", m1 / m0 - 1.0, e1 / e0 - 1.0, pp, pmean);
    assert!((m1 / m0 - 1.0).abs() < 1e-10 && (e1 / e0 - 1.0).abs() < 1e-10);
    assert!((pp / pmean - 1.0).abs() < 0.01);
}

#[test]
fn slug_steady_is_orifice() {
    // A slug's flow is steady exactly at the compressible orifice's, subsonic and choked,
    // either way round.
    let (p0, t0) = (2.0e5, 900.0);
    let (cda, l) = (3e-4, 0.03);
    for r in [0.999, 0.95, 0.8, 0.6, 0.5, 0.3] {
        let g = gamma(t0);
        let (p, t) = (r * p0, 600.0);
        let up = (p0, t0, g, p0 / (RS * t0));
        let down = (p, t, gamma(t), p / (RS * t));
        let m = cda * orifice(p0, t0, g, p);
        let scale = cda / l * (p0 - p);
        let forward = slug_rate(cda, l, m, up, down) / scale;
        let backward = slug_rate(cda, l, -m, down, up) / scale;
        println!("r {r}: m {m:.5} kg/s, rates {forward:.2e} {backward:.2e}");
        assert!(forward.abs() < 1e-9 && backward.abs() < 1e-9);
        // Less flow is accelerated, more held back.
        assert!(slug_rate(cda, l, 0.9 * m, up, down) > 0.0 && slug_rate(cda, l, 1.1 * m, up, down) < 0.0);
    }
}

#[test]
fn orifice_helmholtz() {
    // A litre opening to the air through a hole whose gas slug has length l rings at the
    // Helmholtz frequency, sqrt(cda c^2 / (l V)) / 2 pi.
    let (v, cda, l, t0) = (1e-3, 1e-4, 0.02, 300.0);
    let mut net = Network::default();
    let vessel = net.add_node(Node::volume(v, ATM + 10.0, t0, 0.0, 0.25));
    let atm = net.add_node(Node::atmosphere(ATM, t0, 0.0, 0.25));
    let o = net.orifice(vessel, atm, cda, 0.0);
    net.orifices[o].length = l;
    let c2 = gamma(t0) * RS * t0;
    let expected = (cda * c2 / (l * v)).sqrt() / (2.0 * std::f64::consts::PI);
    let dt = 1e-6;
    let (mut last, mut crossings) = (net.nodes[vessel].pressure() - ATM, vec![]);
    for k in 1..=60_000 {
        net.step(dt);
        let x = net.nodes[vessel].pressure() - ATM;
        if last > 0.0 && x <= 0.0 {
            crossings.push(k as f64 * dt - dt * x / (x - last));
        }
        last = x;
    }
    let n = crossings.len() - 1;
    let f = n as f64 / (crossings[n] - crossings[0]);
    println!("{f:.2} Hz over {n} periods, Helmholtz {expected:.2} Hz");
    assert!(n >= 5 && (f / expected - 1.0).abs() < 0.01);
}

#[test]
fn closing_orifice() {
    // A valve between two vessels closing on its flow (a wastegate), gradually and at once:
    // the gas in it passes at most what the narrowed opening chokes at, and none once shut.
    let mut net = Network::default();
    let a = net.add_node(Node::volume(2e-3, 2.0e5, 900.0, 0.0, 0.25));
    let b = net.add_node(Node::volume(2e-3, 1.0e5, 900.0, 0.0, 0.25));
    let o = net.orifice(a, b, 5e-4, 0.0);
    let dt = 5e-6;
    for _ in 0..200 {
        net.step(dt);
    }
    for (k, cda) in [3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 0.0, 5e-4, 1e-7, 5e-4, 0.0].into_iter().enumerate() {
        net.orifices[o].cda = cda;
        for _ in 0..20 {
            net.step(dt);
            let m = net.orifices[o].flow;
            assert!(m.is_finite() && net.nodes[a].pressure().is_finite(), "step {k}: cda {cda} m {m}");
        }
        let (ra, rb) = (net.nodes[a].reservoir(), net.nodes[b].reservoir());
        let steady = cda * orifice(ra.p, ra.t, gamma(ra.t), rb.p);
        println!("cda {cda:.0e}: flow {:.6} kg/s, quasi-steady {steady:.6}", net.orifices[o].flow);
        assert!(net.orifices[o].flow <= 1.0001 * cda * orifice(ra.p, ra.t, gamma(ra.t), 0.0) + 1e-9);
    }
}

/// A narrow inlet with initially sonic gas inside: the boundary can choke without
/// creating or losing mass/energy, even while the downstream closed end sends a shock back.
#[test]
fn sonic_inlet_conservation() {
    let mut net = Network::default();
    let tank = net.add_node(Node::volume(5e-4, 7.9e5, 1530.0, 0.0, 0.0));
    let wall = net.add_node(Node::atmosphere(ATM, 600.0, 0.0, 0.25));
    let mut pipe = Pipe::new(0.4, 0.034, 0.005, 2.7e5, 1330.0, 0.0, 0.0);
    pipe.friction = false;
    let area = pipe.area;
    let rho = 2.7e5 / (RS * 1330.0);
    let speed = (gamma(1330.0) * RS * 1330.0).sqrt();
    for i in 0..pipe.cells() {
        set_cell(&mut pipe, i, rho, speed, 2.7e5);
    }
    let p = net.add_pipe(pipe);
    let valve = net.link(p, Side::A, tank, 0.65 * area);
    net.link(p, Side::B, wall, 0.0);
    let (m0, e0) = net.totals();
    let mut choked = false;
    for _ in 0..500 {
        net.step(2e-6);
        assert!(net.check().is_none());
        let f = net.links[valve].flow;
        if f.m < 0.0 {
            let t = f.p / (RS * f.rho);
            let mach = -f.un / (gamma(t) * RS * t).sqrt();
            assert!(mach <= 1.0 + 1e-10);
            choked |= mach > 1.0 - 1e-8;
        }
    }
    let (m1, e1) = net.totals();
    assert!(choked, "the test must exercise sonic inflow");
    assert_eq!(net.rejected, 0);
    assert!((m1 / m0 - 1.0).abs() < 1e-11);
    assert!((e1 / e0 - 1.0).abs() < 1e-11);
}

/// An acoustic packet imposed through a reservoir inlet, compared with linear
/// acoustics before it reaches the far end. This exercises the lower-order incoming
/// ghost extension, rather than only interior propagation or reflection at rest.
#[test]
fn prescribed_inlet_packet_converges_with_mesh_refinement() {
    let t = 900.0;
    let c = (gamma(t) * RS * t).sqrt();
    let final_time = 1.2e-3;
    let packet = |time: f64| 10.0 * (-((time - 4e-4) / 8e-5).powi(2)).exp();
    let mut errors = Vec::new();
    for dx in [0.01, 0.005, 0.0025, 0.00125] {
        // Refine time with space so frozen-midpoint source data does not set the
        // error floor while measuring the inlet's spatial approximation.
        let steps = (final_time / (0.1 * dx / c)).ceil() as usize;
        let dt = final_time / steps as f64;
        let mut net = Network::default();
        let source = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
        let outlet = net.add_node(Node::atmosphere(ATM, t, 0.0, 0.25));
        let mut pipe = Pipe::new(1.0, 0.05, dx, ATM, t, 0.0, 0.25);
        pipe.friction = false;
        let p = net.add_pipe(pipe);
        net.connect(p, Side::A, source);
        net.connect(p, Side::B, outlet);
        for k in 0..steps {
            net.nodes[source].p_fixed = ATM + packet((k as f64 + 0.5) * dt);
            net.step(dt);
        }
        let pipe = &net.pipes[p];
        let (mut error, mut exact) = (0.0, 0.0);
        for i in 0..pipe.cells() {
            let x = (i as f64 + 0.5) * pipe.dx;
            let expected = (packet(final_time - (x - 0.5 * pipe.dx) / c)
                + 4.0 * packet(final_time - x / c)
                + packet(final_time - (x + 0.5 * pipe.dx) / c)) / 6.0;
            let found = Prim::of(&pipe.q[i]).p - ATM;
            error += (found - expected).powi(2);
            exact += expected.powi(2);
        }
        errors.push((error / exact).sqrt());
        assert_eq!(net.rejected, 0);
    }
    eprintln!("inlet packet relative L2 errors: {errors:?}");
    assert!(errors[1] < 0.75 * errors[0], "{errors:?}");
    assert!(errors[2] < 0.75 * errors[1], "{errors:?}");
    assert!(errors[3] < errors[2], "{errors:?}");
    assert!(errors[3] < 0.01, "{errors:?}");
}

/// Both inlet ghost regions overlap in the shortest pipes. Reordering link registration
/// must not let whichever inlet was processed last change the conservative solution.
#[test]
fn overlapping_inlet_stencils_are_link_order_independent() {
    for cells in [2, 3, 4, 10] {
        let make = |reverse: bool| {
            let mut net = Network::default();
            let left = net.add_node(Node::atmosphere(1.3e5, 1000.0, 0.0, 0.25));
            let right = net.add_node(Node::atmosphere(1.5e5, 1100.0, 0.0, 0.25));
            let dx = 1.0 / 128.0;
            let mut pipe = Pipe::new(cells as f64 * dx, 0.034, dx, 1e5, 900.0, 0.0, 0.25);
            pipe.friction = false;
            assert_eq!(pipe.cells(), cells);
            let p = net.add_pipe(pipe);
            if reverse {
                net.connect(p, Side::B, right);
                net.connect(p, Side::A, left);
            } else {
                net.connect(p, Side::A, left);
                net.connect(p, Side::B, right);
            }
            net
        };
        let (mut a, mut b) = (make(false), make(true));
        for k in 0..16 {
            a.step(2e-7);
            b.step(2e-7);
            if k == 0 {
                assert!(a.links.iter().all(|l| l.flow.m < 0.0), "both ends must supply incoming gas");
            }
            assert_eq!(a.pipes[0].q, b.pipes[0].q, "{cells} cells, step {k}");
            assert!(a.check().is_none() && b.check().is_none());
        }
        assert_eq!((a.rejected, b.rejected), (0, 0));
    }
}
