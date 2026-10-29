//! The crankshaft, rods and pistons, and what acts on the crank: bearing friction, the clutch
//! to the driven side (with its drag), the dynamometer and the starter.
//!
//! The game uses `Crank`, which has one coordinate, the crank angle. Each piston and rod
//! follows from it exactly, and the crank's motion comes from the Lagrangian of the whole
//! mechanism: its inertia, which varies with the angle, and the generalised force of the gas
//! on the pistons. The limited torques (friction, clutch, drag, dyno, starter) are solved
//! exactly. `Linkage` is engine-sim's version, a rigid body each for the crank, rods and
//! pistons, joined by constraints that Gauss-Seidel solves approximately; it is kept for the
//! comparison with engine-sim (tests/reference.rs).

use crate::solver::{Body, Constraint, Kind, State, System};
use crate::spec::EngineSpec;
use crate::units::PI;

/// The limited torques on the crank and the driven side.
#[derive(Clone, Copy, Debug, Default)]
pub struct Controls {
    pub friction_torque: f64,
    pub clutch_capacity: f64,
    pub drag: f64,
    pub dyno_speed: f64,
    pub dyno_max_torque: f64,
    pub dyno_hold: bool,
    pub dyno_enabled: bool,
    /// Signed: the crank's direction.
    pub starter_speed: f64,
    pub starter_max_torque: f64,
    pub starter_enabled: bool,
}

/// One bank's cylinder line: the bore's origin and direction (towards the head).
#[derive(Clone, Copy, Debug)]
struct Bore {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

fn bores(spec: &EngineSpec) -> Vec<Bore> {
    spec.banks
        .iter()
        .flat_map(|b| {
            let bore = Bore { x: b.x, y: b.y, dx: (b.angle + PI / 2.0).cos(), dy: (b.angle + PI / 2.0).sin() };
            b.cylinders.iter().map(move |_| bore)
        })
        .collect()
}

/// The mechanism, either way.
pub struct Mechanism {
    pub controls: Controls,
    model: Model,
}

enum Model {
    Reduced(Crank),
    Constrained(Linkage),
}

impl Mechanism {
    /// `reduced` false: engine-sim's constrained bodies.
    pub fn new(spec: &EngineSpec, reduced: bool) -> Mechanism {
        let controls = Controls {
            friction_torque: spec.crankshaft.friction_torque,
            dyno_max_torque: 10000.0 * crate::units::FT_LB,
            starter_speed: -spec.starter_speed,
            starter_max_torque: spec.starter_torque,
            ..Controls::default()
        };
        let model = if reduced { Model::Reduced(Crank::new(spec)) } else { Model::Constrained(Linkage::new(spec)) };
        Mechanism { controls, model }
    }

    pub fn theta(&self) -> f64 {
        match &self.model {
            Model::Reduced(c) => c.theta,
            Model::Constrained(l) => l.system.bodies[l.crank].theta,
        }
    }

    /// Keeps the crank angle within one cycle.
    pub fn wrap_theta(&mut self) {
        match &mut self.model {
            Model::Reduced(c) => c.theta %= 4.0 * PI,
            Model::Constrained(l) => l.system.bodies[l.crank].theta %= 4.0 * PI,
        }
    }

    pub fn omega(&self) -> f64 {
        match &self.model {
            Model::Reduced(c) => c.omega(),
            Model::Constrained(l) => l.system.bodies[l.crank].v_theta,
        }
    }

    pub fn set_omega(&mut self, v: f64) {
        match &mut self.model {
            Model::Reduced(c) => c.set_omega(v),
            Model::Constrained(l) => l.system.bodies[l.crank].v_theta = v,
        }
    }

    /// The driven side (only its inertia and spin matter).
    pub fn load(&self) -> &Body {
        match &self.model {
            Model::Reduced(c) => &c.load,
            Model::Constrained(l) => &l.system.bodies[l.load],
        }
    }

    pub fn load_mut(&mut self) -> &mut Body {
        match &mut self.model {
            Model::Reduced(c) => &mut c.load,
            Model::Constrained(l) => &mut l.system.bodies[l.load],
        }
    }

    /// Cylinder `i`'s piston: its centre's position along the bore from the bank's origin,
    /// and its speed.
    pub fn piston(&self, i: usize) -> (f64, f64) {
        match &self.model {
            Model::Reduced(c) => {
                let k = &c.kinematics[i];
                (k.s + c.throws[i].piston_offset, k.s1 * c.omega())
            }
            Model::Constrained(l) => {
                let (b, bore) = (&l.system.bodies[l.pistons[i].0], &l.bores[i]);
                ((b.p_x - bore.x) * bore.dx + (b.p_y - bore.y) * bore.dy, b.v_x * bore.dx + b.v_y * bore.dy)
            }
        }
    }

    /// The cylinder wall's force on piston `i` in the last step.
    pub fn wall_force(&self, i: usize) -> f64 {
        match &self.model {
            Model::Reduced(c) => c.wall_force[i],
            Model::Constrained(l) => {
                let wall = &l.system.constraints[l.pistons[i].1];
                (wall.f_x[0][0] * wall.f_x[0][0] + wall.f_y[0][0] * wall.f_y[0][0]).sqrt()
            }
        }
    }

    /// The dyno's torque on the crank in the last step.
    pub fn dyno_torque(&self) -> f64 {
        match &self.model {
            Model::Reduced(c) => c.dyno_torque,
            Model::Constrained(l) => l.system.constraints[l.dyno].f_t[0][0],
        }
    }

    /// The clutch's torque on the driven side in the last step.
    pub fn clutch_torque(&self) -> f64 {
        match &self.model {
            Model::Reduced(c) => c.clutch_torque,
            Model::Constrained(l) => l.system.constraints[l.clutch].f_t[0][1],
        }
    }

    /// One step of `dt`, with `forces` along each bore on the pistons (towards the head
    /// positive) and `load_torque` on the driven side.
    pub fn step(&mut self, dt: f64, forces: &[f64], load_torque: f64) {
        match &mut self.model {
            Model::Reduced(c) => c.step(dt, &self.controls, forces, load_torque),
            Model::Constrained(l) => l.step(dt, &self.controls, forces, load_torque),
        }
    }
}

/// A cylinder's crank throw, rod and piston.
#[derive(Clone, Copy, Debug)]
struct Throw {
    /// The rod journal, on the crank.
    journal_x: f64,
    journal_y: f64,
    bore: Bore,
    /// Big end to little end.
    rod_length: f64,
    /// The rod's centre of mass, as a fraction of the way from its big end.
    rod_com: f64,
    rod_mass: f64,
    rod_inertia: f64,
    piston_mass: f64,
    /// The piston's centre beyond its wrist pin, along the bore.
    piston_offset: f64,
}

/// A throw's position and its first two derivatives by the crank angle.
#[derive(Clone, Copy, Debug, Default)]
struct Kinematics {
    /// Wrist pin along the bore.
    s: f64,
    s1: f64,
    s2: f64,
    /// Rod centre of mass.
    r1: (f64, f64),
    r2: (f64, f64),
    /// Rod angle.
    th1: f64,
    th2: f64,
    /// The rod, big end to little end: along the bore (q) and across it (-e).
    q: f64,
    e: f64,
    w: (f64, f64),
}

/// The mechanism in reduced coordinates.
pub struct Crank {
    pub theta: f64,
    /// The crank's speed over the last step.
    speed: f64,
    /// Generalised momentum (the angular momentum about the crank).
    momentum: f64,
    /// The mechanism's inertia about the crank and half its derivative, at `theta`.
    m: f64,
    h: f64,
    inertia: f64,
    x: f64,
    y: f64,
    throws: Vec<Throw>,
    kinematics: Vec<Kinematics>,
    pub load: Body,
    wall_force: Vec<f64>,
    dyno_torque: f64,
    clutch_torque: f64,
}

/// A torque that holds a speed `t` within `[lo, hi]`: `lo` while faster, `hi` while slower.
#[derive(Clone, Copy, Debug)]
struct Limited {
    t: f64,
    lo: f64,
    hi: f64,
}

/// Solves `w = w0 + c * sum(torques(w))` for the speed `w` (c > 0), and the torques.
fn solve_limited(w0: f64, c: f64, torques: &[Limited], out: &mut [f64]) -> f64 {
    let n = torques.len();
    let mut faster = [false; 8];
    for k in 0..n {
        let t = torques[k].t;
        // The sum just below and just above t.
        let (mut below, mut above) = (0.0, 0.0);
        for l in &torques[..n] {
            if l.t < t {
                below += l.lo;
                above += l.lo;
            } else if l.t > t {
                below += l.hi;
                above += l.hi;
            } else {
                below += l.hi;
                above += l.lo;
            }
        }
        if t - w0 - c * below <= 0.0 && t - w0 - c * above >= 0.0 {
            // Held at t by the torques targeting it, the others at their limits.
            let mut rest = (t - w0) / c;
            for (i, l) in torques.iter().enumerate() {
                out[i] = if l.t < t {
                    l.lo
                } else if l.t > t {
                    l.hi
                } else {
                    l.lo
                };
                rest -= out[i];
            }
            for (i, l) in torques.iter().enumerate() {
                if l.t == t && rest > 0.0 {
                    let add = rest.min(l.hi - l.lo);
                    out[i] += add;
                    rest -= add;
                }
            }
            return t;
        }
        faster[k] = t - w0 - c * above < 0.0;
    }
    let mut sum = 0.0;
    for (i, l) in torques.iter().enumerate() {
        out[i] = if faster[i] { l.lo } else { l.hi };
        sum += out[i];
    }
    w0 + c * sum
}

impl Crank {
    pub fn new(spec: &EngineSpec) -> Crank {
        let cs = &spec.crankshaft;
        let bores = bores(spec);
        let throws: Vec<Throw> = spec
            .banks
            .iter()
            .flat_map(|b| b.cylinders.iter())
            .zip(&bores)
            .map(|(c, &bore)| {
                let angle = cs.journals[c.journal];
                // Where engine-sim's constraints join the rod (linkage below).
                let little_end = (c.rod.length / 2.0) - c.rod.center_of_mass;
                let big_end = -(c.rod.length / 2.0) + c.rod.center_of_mass;
                let rod_length = little_end - big_end;
                Throw {
                    journal_x: angle.cos() * cs.throw,
                    journal_y: angle.sin() * cs.throw,
                    bore,
                    rod_length,
                    rod_com: -big_end / rod_length,
                    rod_mass: c.rod.mass,
                    rod_inertia: c.rod.moment_of_inertia,
                    piston_mass: c.piston.mass,
                    piston_offset: c.piston.wrist_pin_position,
                }
            })
            .collect();
        let n = throws.len();
        let mut crank = Crank {
            theta: 0.0,
            speed: 0.0,
            momentum: 0.0,
            m: 1.0,
            h: 0.0,
            inertia: cs.moment_of_inertia,
            x: cs.x,
            y: cs.y,
            throws,
            kinematics: vec![Kinematics::default(); n],
            load: Body { m: 1.0, i: 1.0, ..Body::default() },
            wall_force: vec![0.0; n],
            dyno_torque: 0.0,
            clutch_torque: 0.0,
        };
        crank.update_kinematics();
        crank
    }

    pub fn omega(&self) -> f64 {
        self.speed
    }

    pub fn set_omega(&mut self, v: f64) {
        self.speed = v;
        self.momentum = self.m * v;
    }

    fn update_kinematics(&mut self) {
        let (sin, cos) = self.theta.sin_cos();
        for (t, k) in self.throws.iter().zip(self.kinematics.iter_mut()) {
            let Bore { x, y, dx, dy } = t.bore;
            let (nx, ny) = (-dy, dx);
            // Journal from the crank's centre, and its derivatives.
            let rx = cos * t.journal_x - sin * t.journal_y;
            let ry = sin * t.journal_x + cos * t.journal_y;
            let (j1x, j1y) = (-ry, rx);
            let (j2x, j2y) = (-rx, -ry);
            // Journal from the bore's origin: along the bore (a) and across it (e).
            let (px, py) = (self.x + rx - x, self.y + ry - y);
            let (a, e) = (px * dx + py * dy, px * nx + py * ny);
            let (a1, e1) = (j1x * dx + j1y * dy, j1x * nx + j1y * ny);
            let (a2, e2) = (j2x * dx + j2y * dy, j2x * nx + j2y * ny);
            let l = t.rod_length;
            let q = (l * l - e * e).max(1e-12 * l * l).sqrt();
            let q1 = -e * e1 / q;
            let q2 = -(e1 * e1 + e * e2 + q1 * q1) / q;
            let (s1, s2) = (a1 + q1, a2 + q2);
            let f = t.rod_com;
            *k = Kinematics {
                s: a + q,
                s1,
                s2,
                r1: ((1.0 - f) * j1x + f * s1 * dx, (1.0 - f) * j1y + f * s1 * dy),
                r2: ((1.0 - f) * j2x + f * s2 * dx, (1.0 - f) * j2y + f * s2 * dy),
                th1: -e1 / q,
                th2: -e2 / q + e1 * q1 / (q * q),
                q,
                e,
                w: (q * dx - e * nx, q * dy - e * ny),
            };
        }
        (self.m, self.h) = self.inertia();
    }

    /// The mechanism's inertia about the crank, and half its derivative by the angle.
    fn inertia(&self) -> (f64, f64) {
        let (mut m, mut h) = (self.inertia, 0.0);
        for (t, k) in self.throws.iter().zip(&self.kinematics) {
            m += t.piston_mass * k.s1 * k.s1
                + t.rod_mass * (k.r1.0 * k.r1.0 + k.r1.1 * k.r1.1)
                + t.rod_inertia * k.th1 * k.th1;
            h += t.piston_mass * k.s1 * k.s2
                + t.rod_mass * (k.r1.0 * k.r2.0 + k.r1.1 * k.r2.1)
                + t.rod_inertia * k.th1 * k.th2;
        }
        (m, h)
    }

    /// Kinetic energy of the mechanism (without the driven side).
    pub fn energy(&self) -> f64 {
        0.5 * self.momentum * self.momentum / self.m
    }

    fn step(&mut self, dt: f64, c: &Controls, forces: &[f64], load_torque: f64) {
        let (m, h) = (self.m, self.h);
        let mut q = 0.0;
        for (k, f) in self.kinematics.iter().zip(forces) {
            q += f * k.s1;
        }
        let w = self.omega();
        // Unconstrained, symplectic Euler on H = p^2 / 2M(theta): the momentum's change
        // dt (Q + (M'/2) v^2) with v, the new speed, p / M.
        let b = self.momentum + dt * q;
        let crank_free = 2.0 * b / (m + (m * m - 4.0 * dt * h * b).max(0.0).sqrt());
        let load_free = self.load.v_theta + dt * load_torque / self.load.i;

        // The limited torques, in engine-sim's terms (its constraints' limits and targets).
        let mut crank = [Limited { t: 0.0, lo: -c.friction_torque, hi: c.friction_torque }; 4];
        let on = if c.dyno_enabled { c.dyno_max_torque } else { 0.0 };
        let held = c.dyno_hold && c.dyno_enabled;
        crank[1] = if w < 0.0 {
            Limited { t: -c.dyno_speed, lo: if held { -c.dyno_max_torque } else { 0.0 }, hi: on }
        } else {
            Limited { t: c.dyno_speed, lo: -on, hi: if held { c.dyno_max_torque } else { 0.0 } }
        };
        let on = if c.starter_enabled { c.starter_max_torque } else { 0.0 };
        crank[2] = Limited {
            t: c.starter_speed,
            lo: if c.starter_speed < 0.0 { -on } else { 0.0 },
            hi: if c.starter_speed < 0.0 { 0.0 } else { on },
        };
        // Drag on the driven side: only against its forward (negative) spin.
        crank[3] = Limited { t: 0.0, lo: 0.0, hi: c.drag };
        let drag = [crank[3]];
        let crank = &crank[..3];

        // The clutch: slipping one way, the other, or locked.
        let capacity = c.clutch_capacity.min(1e30);
        let (il, mut torques, mut drag_torque) = (self.load.i, [0.0; 4], [0.0]);
        let slip = |torque: f64, torques: &mut [f64], drag_torque: &mut [f64]| {
            let wc = solve_limited(crank_free - dt * torque / m, dt / m, crank, torques);
            let wl = solve_limited(load_free + dt * torque / il, dt / il, &drag, drag_torque);
            (wc, wl)
        };
        let (mut wc, mut wl) = slip(-capacity, &mut torques, &mut drag_torque);
        let mut clutch = -capacity;
        if wl <= wc && capacity > 0.0 {
            (wc, wl) = slip(capacity, &mut torques, &mut drag_torque);
            clutch = capacity;
            if wl >= wc {
                let all = [crank[0], crank[1], crank[2], drag[0]];
                let w0 = (m * crank_free + il * load_free) / (m + il);
                wc = solve_limited(w0, dt / (m + il), &all, &mut torques);
                wl = wc;
                drag_torque[0] = torques[3];
                clutch = ((wl - load_free) * il / dt - drag_torque[0]).clamp(-capacity, capacity);
            }
        }
        self.dyno_torque = torques[1];
        self.clutch_torque = clutch;

        // The wall's reaction on each piston, from the rod's and the piston's equations of
        // motion (for the piston's friction next step).
        let a = (wc - w) / dt;
        for ((t, k), (f, wall)) in
            self.throws.iter().zip(&self.kinematics).zip(forces.iter().zip(self.wall_force.iter_mut()))
        {
            let w2 = w * w;
            let s_dd = k.s1 * a + k.s2 * w2;
            let r_dd = (k.r1.0 * a + k.r2.0 * w2, k.r1.1 * a + k.r2.1 * w2);
            let th_dd = k.th1 * a + k.th2 * w2;
            // The rod's force on the piston, along the bore and across it.
            let along = f - t.piston_mass * s_dd;
            let moment = -t.rod_mass * t.rod_com * (k.w.0 * r_dd.1 - k.w.1 * r_dd.0) - t.rod_inertia * th_dd;
            let across = (moment + k.e * along) / k.q;
            *wall = across.abs();
        }

        self.speed = wc;
        self.momentum = m * wc;
        self.theta += wc * dt;
        self.load.v_theta = wl;
        self.load.theta += wl * dt;
        self.update_kinematics();
    }
}

/// engine-sim's mechanism: constrained rigid bodies.
pub struct Linkage {
    pub system: System,
    crank: usize,
    load: usize,
    clutch: usize,
    drag: usize,
    dyno: usize,
    starter: usize,
    /// Per cylinder: piston body, wall constraint.
    pistons: Vec<(usize, usize)>,
    bores: Vec<Bore>,
}

impl Linkage {
    pub fn new(spec: &EngineSpec) -> Linkage {
        let mut system = System::default();
        let cs = &spec.crankshaft;

        // Crankshaft: fixed in place, with bearing friction.
        let crank = system.add_body(Body {
            p_x: cs.x,
            p_y: cs.y,
            m: cs.mass + cs.flywheel_mass,
            i: cs.moment_of_inertia,
            ..Body::default()
        });
        system.add_constraint(Constraint::new(
            Kind::FixedPosition { local_x: 0.0, local_y: 0.0, world_x: cs.x, world_y: cs.y },
            &[crank],
        ));
        system.add_constraint(Constraint::new(
            Kind::RotationFriction { min_torque: -cs.friction_torque, max_torque: cs.friction_torque },
            &[crank],
        ));

        // What the engine drives, through the clutch: one rotating body.
        let load = 1;
        let clutch =
            system.add_constraint(Constraint::new(Kind::Clutch { min_torque: 0.0, max_torque: 0.0 }, &[crank, load]));
        let drag = system.add_constraint(Constraint::new(Kind::Drag { max_torque: 0.0 }, &[load]));
        let lb = system.add_body(Body { m: 1.0, i: 1.0, ..Body::default() });
        debug_assert_eq!(lb, load);

        // Pistons and rods.
        let bores = bores(spec);
        let mut pistons = Vec::new();
        for (c, bank) in spec.banks.iter().flat_map(|b| b.cylinders.iter().map(move |c| (c, b))).zip(&bores) {
            let (c, bank_spec) = c;
            let wrist_pin = c.piston.wrist_pin_position;
            let little_end = (c.rod.length / 2.0) - c.rod.center_of_mass;
            let big_end = -(c.rod.length / 2.0) + c.rod.center_of_mass;
            let journal_angle = cs.journals[c.journal];
            let (journal_x, journal_y) = (journal_angle.cos() * cs.throw, journal_angle.sin() * cs.throw);

            let wall = Kind::Line { local_x: 0.0, local_y: wrist_pin, p0_x: bank.x, p0_y: bank.y, dx: bank.dx, dy: bank.dy };
            let piston_body = system.add_body(Body { m: c.piston.mass, i: 1.0, ..Body::default() });
            let rod_body = system.add_body(Body { m: c.rod.mass, i: c.rod.moment_of_inertia, ..Body::default() });
            system.add_constraint(Constraint::new(
                Kind::Link { local_x_1: 0.0, local_y_1: little_end, local_x_2: 0.0, local_y_2: wrist_pin },
                &[rod_body, piston_body],
            ));
            system.add_constraint(Constraint::new(
                Kind::Link { local_x_1: 0.0, local_y_1: big_end, local_x_2: journal_x, local_y_2: journal_y },
                &[rod_body, crank],
            ));
            let wall_constraint = system.add_constraint(Constraint::new(wall, &[piston_body]));
            pistons.push((piston_body, wall_constraint));

            // placeCylinder
            let crank_body = system.bodies[crank];
            let (p_x, p_y) = (journal_x + crank_body.p_x, journal_y + crank_body.p_y);
            let a = bank.dx * bank.dx + bank.dy * bank.dy;
            let bb = -2.0 * bank.dx * (p_x - bank.x) - 2.0 * bank.dy * (p_y - bank.y);
            let cc = (p_x - bank.x) * (p_x - bank.x) + (p_y - bank.y) * (p_y - bank.y) - c.rod.length * c.rod.length;
            let det = bb * bb - 4.0 * a * cc;
            if det >= 0.0 {
                let sqrt_det = det.sqrt();
                let s0 = (-bb + sqrt_det) / (2.0 * a);
                let s1 = (-bb - sqrt_det) / (2.0 * a);
                let s = s0.max(s1);
                if s >= 0.0 {
                    let e_x = s * bank.dx + bank.x;
                    let e_y = s * bank.dy + bank.y;
                    let theta = if (e_y - p_y) > 0.0 {
                        ((e_x - p_x) / c.rod.length).acos()
                    } else {
                        2.0 * PI - ((e_x - p_x) / c.rod.length).acos()
                    };
                    let rod = &mut system.bodies[rod_body];
                    rod.theta = theta - PI / 2.0;
                    let (cl_x, cl_y) = rod.local_to_world(0.0, big_end);
                    rod.p_x += p_x - cl_x;
                    rod.p_y += p_y - cl_y;
                    let piston = &mut system.bodies[piston_body];
                    piston.p_x = e_x;
                    piston.p_y = e_y;
                    piston.theta = bank_spec.angle + PI;
                }
            }
        }

        let dyno = system.add_constraint(Constraint::new(
            Kind::Dynamometer { speed: 0.0, max_torque: 10000.0 * crate::units::FT_LB, hold: false, enabled: false },
            &[crank],
        ));
        let starter = system.add_constraint(Constraint::new(
            Kind::Starter { speed: -spec.starter_speed, max_torque: spec.starter_torque, enabled: false },
            &[crank],
        ));
        Linkage { system, crank, load, clutch, drag, dyno, starter, pistons, bores }
    }

    fn step(&mut self, dt: f64, c: &Controls, forces: &[f64], load_torque: f64) {
        let k = &mut self.system.constraints;
        k[self.clutch].kind = Kind::Clutch { min_torque: -c.clutch_capacity, max_torque: c.clutch_capacity };
        k[self.drag].kind = Kind::Drag { max_torque: c.drag };
        k[self.dyno].kind = Kind::Dynamometer {
            speed: c.dyno_speed,
            max_torque: c.dyno_max_torque,
            hold: c.dyno_hold,
            enabled: c.dyno_enabled,
        };
        k[self.starter].kind =
            Kind::Starter { speed: c.starter_speed, max_torque: c.starter_max_torque, enabled: c.starter_enabled };
        let (pistons, bores, load) = (&self.pistons, &self.bores, self.load);
        self.system.process(dt, |state: &mut State| {
            state.t[load] += load_torque;
            for ((&(body, _), bore), &f) in pistons.iter().zip(bores).zip(forces) {
                state.apply_force(f * bore.dx, f * bore.dy, body);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> EngineSpec {
        let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vw1200.json")).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let mut c = Crank::new(&spec());
        let h = 1e-5;
        for i in 0..40 {
            let theta = i as f64 * 0.3;
            let at = |c: &mut Crank, th: f64| {
                c.theta = th;
                c.update_kinematics();
                c.kinematics.clone()
            };
            let (k0, kp, km) = (at(&mut c, theta), at(&mut c, theta + h), at(&mut c, theta - h));
            for j in 0..k0.len() {
                let d = |p: f64, m: f64| (p - m) / (2.0 * h);
                let close = |a: f64, b: f64, what: &str| {
                    assert!((a - b).abs() < 1e-5 * (1.0 + b.abs()), "{what} at {theta}: {a} vs {b}");
                };
                close(k0[j].s1, d(kp[j].s, km[j].s), "s'");
                close(k0[j].s2, d(kp[j].s1, km[j].s1), "s''");
                close(k0[j].r2.0, d(kp[j].r1.0, km[j].r1.0), "r''x");
                close(k0[j].r2.1, d(kp[j].r1.1, km[j].r1.1), "r''y");
                close(k0[j].th2, d(kp[j].th1, km[j].th1), "th''");
                let (p, m) = (kp[j].w, km[j].w);
                let turn = (m.0 * p.1 - m.1 * p.0).atan2(m.0 * p.0 + m.1 * p.1);
                close(k0[j].th1, turn / (2.0 * h), "th'");
            }
        }
    }

    #[test]
    fn rod_centre_moves_with_the_crank() {
        // r1 from finite differences of the centre of mass itself.
        let s = spec();
        let mut c = Crank::new(&s);
        let h = 1e-6;
        let com = |c: &mut Crank, th: f64| {
            c.theta = th;
            c.update_kinematics();
            let (sin, cos) = th.sin_cos();
            let t = c.throws[0];
            let (jx, jy) = (c.x + cos * t.journal_x - sin * t.journal_y, c.y + sin * t.journal_x + cos * t.journal_y);
            let k = c.kinematics[0];
            (jx + t.rod_com * k.w.0, jy + t.rod_com * k.w.1, k.r1)
        };
        for i in 0..20 {
            let th = i as f64 * 0.5;
            let (_, _, r1) = com(&mut c, th);
            let (xp, yp, _) = com(&mut c, th + h);
            let (xm, ym, _) = com(&mut c, th - h);
            assert!((r1.0 - (xp - xm) / (2.0 * h)).abs() < 1e-6);
            assert!((r1.1 - (yp - ym) / (2.0 * h)).abs() < 1e-6);
        }
    }

    #[test]
    fn conserves_energy() {
        let mut c = Crank::new(&spec());
        let controls = Controls::default();
        c.set_omega(-300.0);
        let e0 = c.energy();
        let forces = vec![0.0; c.throws.len()];
        let mut worst: f64 = 0.0;
        for _ in 0..100_000 {
            c.step(1e-4, &controls, &forces, 0.0);
            worst = worst.max((c.energy() - e0).abs() / e0);
        }
        assert!(worst < 1e-3, "energy off by {worst}");
    }

    #[test]
    fn limited_torques() {
        let l = |t, lo, hi| Limited { t, lo, hi };
        let mut out = [0.0; 4];
        // Friction holds a slow crank.
        let w = solve_limited(0.5, 0.1, &[l(0.0, -10.0, 10.0)], &mut out);
        assert_eq!((w, out[0]), (0.0, -5.0));
        // ... and slows a fast one.
        let w = solve_limited(5.0, 0.1, &[l(0.0, -10.0, 10.0)], &mut out);
        assert!((w - 4.0).abs() < 1e-12 && out[0] == -10.0);
        // A dyno at 3 against friction.
        let w = solve_limited(5.0, 0.1, &[l(0.0, -1.0, 1.0), l(3.0, -100.0, 100.0)], &mut out);
        assert!((w - 3.0).abs() < 1e-12 && out[0] == -1.0 && (out[1] + 19.0).abs() < 1e-9);
    }
}
