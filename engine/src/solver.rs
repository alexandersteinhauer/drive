//! Planar rigid bodies with velocity-level constraints, solved by projected Gauss-Seidel and
//! integrated semi-implicitly (simple-2d-constraint-solver's OptimizedNsvRigidBodySystem with
//! GaussSeidelSleSolver, as engine-sim uses it).

#[derive(Clone, Copy, Debug, Default)]
pub struct Body {
    pub p_x: f64,
    pub p_y: f64,
    pub v_x: f64,
    pub v_y: f64,
    pub theta: f64,
    pub v_theta: f64,
    pub m: f64,
    pub i: f64,
}

impl Body {
    pub fn local_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let (s, c) = (self.theta.sin(), self.theta.cos());
        (c * x - s * y + self.p_x, s * x + c * y + self.p_y)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// Pins a point on the body to a world position.
    FixedPosition { local_x: f64, local_y: f64, world_x: f64, world_y: f64 },
    /// Holds the body's spin with a torque between the limits.
    RotationFriction { min_torque: f64, max_torque: f64 },
    /// Couples two bodies' spins with a torque between the limits.
    Clutch { min_torque: f64, max_torque: f64 },
    /// Keeps a point on the body on a line.
    Line { local_x: f64, local_y: f64, p0_x: f64, p0_y: f64, dx: f64, dy: f64 },
    /// Joins a point on each of two bodies.
    Link { local_x_1: f64, local_y_1: f64, local_x_2: f64, local_y_2: f64 },
    /// Spins the body at `speed` with up to `max_torque`; `hold` also brakes it to that speed.
    Dynamometer { speed: f64, max_torque: f64, hold: bool, enabled: bool },
    /// Cranks the body at `speed` (signed) with up to `max_torque`.
    Starter { speed: f64, max_torque: f64, enabled: bool },
    /// Resists the body's spin with up to `max_torque` (a vehicle's drag).
    Drag { max_torque: f64 },
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub kind: Kind,
    pub bodies: [usize; 2],
    pub body_count: usize,
    /// Reaction forces from the last solve, [row][body].
    pub f_x: [[f64; 2]; 3],
    pub f_y: [[f64; 2]; 3],
    pub f_t: [[f64; 2]; 3],
}

impl Constraint {
    pub fn new(kind: Kind, bodies: &[usize]) -> Self {
        let mut b = [0; 2];
        b[..bodies.len()].copy_from_slice(bodies);
        Constraint {
            kind,
            bodies: b,
            body_count: bodies.len(),
            f_x: [[0.0; 2]; 3],
            f_y: [[0.0; 2]; 3],
            f_t: [[0.0; 2]; 3],
        }
    }

    pub fn rows(&self) -> usize {
        match self.kind {
            Kind::FixedPosition { .. } | Kind::Link { .. } => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Default)]
struct Output {
    c: [f64; 3],
    j: [[f64; 6]; 3],
    v_bias: [f64; 3],
    limits: [[f64; 2]; 3],
}

const NO_LIMITS: [f64; 2] = [-f64::MAX, f64::MAX];

fn calculate(constraint: &Constraint, s: &State, out: &mut Output) {
    match constraint.kind {
        Kind::FixedPosition { local_x, local_y, world_x, world_y } => {
            let b = constraint.bodies[0];
            let (q1, q2, q3) = (s.p_x[b], s.p_y[b], s.theta[b]);
            let (cos_q3, sin_q3) = (q3.cos(), q3.sin());
            let current_x = q1 + cos_q3 * local_x - sin_q3 * local_y;
            let current_y = q2 + sin_q3 * local_x + cos_q3 * local_y;
            out.j[0][0] = 1.0;
            out.j[0][1] = 0.0;
            out.j[0][2] = -sin_q3 * local_x - cos_q3 * local_y;
            out.j[1][0] = 0.0;
            out.j[1][1] = 1.0;
            out.j[1][2] = cos_q3 * local_x - sin_q3 * local_y;
            out.c[0] = current_x - world_x;
            out.c[1] = current_y - world_y;
            out.v_bias[0] = 0.0;
            out.v_bias[1] = 0.0;
            out.limits = [NO_LIMITS; 3];
        }
        Kind::RotationFriction { min_torque, max_torque } => {
            out.c[0] = 0.0;
            out.j[0][0] = 0.0;
            out.j[0][1] = 0.0;
            out.j[0][2] = 1.0;
            out.v_bias[0] = 0.0;
            out.limits[0] = [min_torque, max_torque];
        }
        Kind::Clutch { min_torque, max_torque } => {
            out.c[0] = 0.0;
            out.j[0] = [0.0, 0.0, -1.0, 0.0, 0.0, 1.0];
            out.v_bias[0] = 0.0;
            out.limits[0] = [min_torque, max_torque];
        }
        Kind::Line { local_x, local_y, p0_x, p0_y, dx, dy } => {
            let b = constraint.bodies[0];
            let (q1, q2, q3) = (s.p_x[b], s.p_y[b], s.theta[b]);
            let (cos_q3, sin_q3) = (q3.cos(), q3.sin());
            let body_x = q1 + cos_q3 * local_x - sin_q3 * local_y;
            let body_y = q2 + sin_q3 * local_x + cos_q3 * local_y;
            let perp_x = -dy;
            let perp_y = dx;
            let delta_x = body_x - p0_x;
            let delta_y = body_y - p0_y;
            out.j[0][0] = 1.0 * perp_x;
            out.j[0][1] = 1.0 * perp_y;
            out.j[0][2] = (-sin_q3 * local_x - cos_q3 * local_y) * perp_x
                + (cos_q3 * local_x - sin_q3 * local_y) * perp_y;
            out.c[0] = delta_x * perp_x + delta_y * perp_y;
            out.v_bias[0] = 0.0;
            out.limits = [NO_LIMITS; 3];
        }
        Kind::Link { local_x_1, local_y_1, local_x_2, local_y_2 } => {
            let (b, l) = (constraint.bodies[0], constraint.bodies[1]);
            let (q1, q2, q3) = (s.p_x[b], s.p_y[b], s.theta[b]);
            let (q4, q5, q6) = (s.p_x[l], s.p_y[l], s.theta[l]);
            let (cos_q3, sin_q3) = (q3.cos(), q3.sin());
            let (cos_q6, sin_q6) = (q6.cos(), q6.sin());
            let body_x = q1 + cos_q3 * local_x_1 - sin_q3 * local_y_1;
            let body_y = q2 + sin_q3 * local_x_1 + cos_q3 * local_y_1;
            let linked_x = q4 + cos_q6 * local_x_2 - sin_q6 * local_y_2;
            let linked_y = q5 + sin_q6 * local_x_2 + cos_q6 * local_y_2;
            out.j[0][0] = 1.0;
            out.j[0][1] = 0.0;
            out.j[0][2] = -sin_q3 * local_x_1 - cos_q3 * local_y_1;
            out.j[1][0] = 0.0;
            out.j[1][1] = 1.0;
            out.j[1][2] = cos_q3 * local_x_1 - sin_q3 * local_y_1;
            out.j[0][3] = -1.0;
            out.j[0][4] = 0.0;
            out.j[0][5] = sin_q6 * local_x_2 + cos_q6 * local_y_2;
            out.j[1][3] = 0.0;
            out.j[1][4] = -1.0;
            out.j[1][5] = -cos_q6 * local_x_2 + sin_q6 * local_y_2;
            out.c[0] = body_x - linked_x;
            out.c[1] = body_y - linked_y;
            out.v_bias[0] = 0.0;
            out.v_bias[1] = 0.0;
            out.limits = [NO_LIMITS; 3];
        }
        Kind::Dynamometer { speed, max_torque, hold, enabled } => {
            let b = constraint.bodies[0];
            out.j[0][0] = 0.0;
            out.j[0][1] = 0.0;
            out.j[0][2] = 1.0;
            out.c[0] = 0.0;
            let held = hold && enabled;
            let on = if enabled { max_torque } else { 0.0 };
            if s.v_theta[b] < 0.0 {
                out.v_bias[0] = speed;
                out.limits[0] = [if held { -max_torque } else { 0.0 }, on];
            } else {
                out.v_bias[0] = -speed;
                out.limits[0] = [-on, if held { max_torque } else { 0.0 }];
            }
        }
        Kind::Starter { speed, max_torque, enabled } => {
            out.j[0][0] = 0.0;
            out.j[0][1] = 0.0;
            out.j[0][2] = 1.0;
            out.c[0] = 0.0;
            out.v_bias[0] = -speed;
            let on = if enabled { max_torque } else { 0.0 };
            out.limits[0] = if speed < 0.0 { [-on, 0.0] } else { [0.0, on] };
        }
        Kind::Drag { max_torque } => {
            out.c[0] = 0.0;
            out.j[0] = [0.0, 0.0, -1.0, 0.0, 0.0, 1.0];
            out.v_bias[0] = 0.0;
            out.limits[0] = [-max_torque, 0.0];
        }
    }
}

/// Per-step state of all bodies (SystemState).
#[derive(Clone, Default)]
pub struct State {
    pub p_x: Vec<f64>,
    pub p_y: Vec<f64>,
    pub theta: Vec<f64>,
    pub v_x: Vec<f64>,
    pub v_y: Vec<f64>,
    pub v_theta: Vec<f64>,
    a_x: Vec<f64>,
    a_y: Vec<f64>,
    a_theta: Vec<f64>,
    pub f_x: Vec<f64>,
    pub f_y: Vec<f64>,
    pub t: Vec<f64>,
    r_x: Vec<f64>,
    r_y: Vec<f64>,
    r_t: Vec<f64>,
}

impl State {
    /// Adds a force through the body's centre.
    pub fn apply_force(&mut self, f_x: f64, f_y: f64, body: usize) {
        self.f_x[body] += f_x;
        self.f_y[body] += f_y;
    }
}

#[derive(Clone, Default)]
pub struct System {
    pub bodies: Vec<Body>,
    pub constraints: Vec<Constraint>,
    state: State,
    // Sparse Jacobian: per constraint row, two blocks of three (body index or NONE).
    j: Vec<[f64; 6]>,
    j_block: Vec<[usize; 2]>,
    v_bias: Vec<f64>,
    c: Vec<f64>,
    m_inv: Vec<f64>,
    q_dot_prime: Vec<f64>,
    /// Per row, the rows sharing a body with it (ascending), and their coupling values.
    neighbours: Vec<Vec<usize>>,
    coupling: Vec<Vec<f64>>,
    inv_diagonal: Vec<f64>,
    /// Warm start: the impulses of the previous step.
    pub lambda: Vec<f64>,
    pub right: Vec<f64>,
    pub limits: Vec<[f64; 2]>,
    /// Gauss-Seidel iterations so far (for profiling).
    pub iterations: u64,
}

const NONE: usize = usize::MAX;
const MAX_ITERATIONS: usize = 128;
const MIN_DELTA: f64 = 1e-1;

impl System {
    pub fn add_body(&mut self, body: Body) -> usize {
        self.bodies.push(body);
        self.bodies.len() - 1
    }

    pub fn add_constraint(&mut self, c: Constraint) -> usize {
        self.constraints.push(c);
        self.constraints.len() - 1
    }

    /// One step of `dt`; `forces` adds the external forces to the state.
    pub fn process(&mut self, dt: f64, forces: impl FnOnce(&mut State)) {
        let n = self.bodies.len();
        let m_f: usize = self.constraints.iter().map(|c| c.rows()).sum();

        // populateSystemState / populateMassMatrices
        let s = &mut self.state;
        for v in [
            &mut s.p_x, &mut s.p_y, &mut s.theta, &mut s.v_x, &mut s.v_y, &mut s.v_theta,
            &mut s.a_x, &mut s.a_y, &mut s.a_theta, &mut s.f_x, &mut s.f_y, &mut s.t,
        ] {
            v.resize(n, 0.0);
        }
        for v in [&mut s.r_x, &mut s.r_y, &mut s.r_t] {
            v.resize(m_f * 2, 0.0);
        }
        self.m_inv.resize(3 * n, 0.0);
        for (i, b) in self.bodies.iter().enumerate() {
            s.a_x[i] = 0.0;
            s.a_y[i] = 0.0;
            s.a_theta[i] = 0.0;
            s.v_x[i] = b.v_x;
            s.v_y[i] = b.v_y;
            s.p_x[i] = b.p_x;
            s.p_y[i] = b.p_y;
            s.v_theta[i] = b.v_theta;
            s.theta[i] = b.theta;
            self.m_inv[i * 3] = 1.0 / b.m;
            self.m_inv[i * 3 + 1] = 1.0 / b.m;
            self.m_inv[i * 3 + 2] = 1.0 / b.i;
        }

        // processForces
        for i in 0..n {
            s.f_x[i] = 0.0;
            s.f_y[i] = 0.0;
            s.t[i] = 0.0;
        }
        forces(s);

        self.process_constraints(dt, m_f);

        // NsvOdeSolver::solve
        let s = &mut self.state;
        for i in 0..n {
            s.v_x[i] += s.a_x[i] * dt;
            s.v_y[i] += s.a_y[i] * dt;
            s.v_theta[i] += s.a_theta[i] * dt;
            s.p_x[i] += s.v_x[i] * dt;
            s.p_y[i] += s.v_y[i] * dt;
            s.theta[i] += s.v_theta[i] * dt;
        }

        // propagateResults
        for (i, b) in self.bodies.iter_mut().enumerate() {
            b.v_x = s.v_x[i];
            b.v_y = s.v_y[i];
            b.p_x = s.p_x[i];
            b.p_y = s.p_y[i];
            b.v_theta = s.v_theta[i];
            b.theta = s.theta[i];
        }
        let mut i_f = 0;
        for c in self.constraints.iter_mut() {
            for j in 0..c.rows() {
                for k in 0..c.body_count {
                    c.f_x[j][k] = s.r_x[i_f * 2 + k];
                    c.f_y[j][k] = s.r_y[i_f * 2 + k];
                    c.f_t[j][k] = s.r_t[i_f * 2 + k];
                }
                i_f += 1;
            }
        }
    }

    fn process_constraints(&mut self, dt: f64, m_f: usize) {
        let n = self.bodies.len();
        self.j.resize(m_f, [0.0; 6]);
        self.j_block.resize(m_f, [NONE; 2]);
        self.v_bias.resize(m_f, 0.0);
        self.c.resize(m_f, 0.0);
        self.limits.resize(m_f, [0.0; 2]);

        let mut out = Output::default();
        let mut j_f = 0;
        for c in &self.constraints {
            calculate(c, &self.state, &mut out);
            for k in 0..c.rows() {
                self.j_block[j_f] = [NONE; 2];
                for i in 0..c.body_count {
                    self.j_block[j_f][i] = c.bodies[i];
                }
                for i in 0..c.body_count * 3 {
                    self.j[j_f][i] = out.j[k][i];
                }
                self.v_bias[j_f] = out.v_bias[k];
                self.c[j_f] = out.c[k];
                self.limits[j_f] = [out.limits[k][0] * dt, out.limits[k][1] * dt];
                j_f += 1;
            }
        }

        let s = &self.state;
        // q_dot_prime = q_dot + M_inv * F_ext * dt
        self.q_dot_prime.resize(3 * n, 0.0);
        for i in 0..n {
            let f = [s.f_x[i], s.f_y[i], s.t[i]];
            let q = [s.v_x[i], s.v_y[i], s.v_theta[i]];
            for k in 0..3 {
                self.q_dot_prime[i * 3 + k] = self.m_inv[i * 3 + k] * (dt * f[k]) + q[k];
            }
        }

        // right = -(J * q_dot_prime + v_bias + C * bias / dt)
        self.right.resize(m_f, 0.0);
        let bias = 1.0 / dt;
        for r in 0..m_f {
            let mut v = 0.0;
            for e in 0..2 {
                let block = self.j_block[r][e];
                if block == NONE {
                    continue;
                }
                for l in 0..3 {
                    v += self.j[r][e * 3 + l] * self.q_dot_prime[block * 3 + l];
                }
            }
            let b_err = bias * self.c[r];
            self.right[r] = -((v + self.v_bias[r]) + b_err);
        }

        self.solve_with_limits(m_f);

        // R = J^T * lambda / dt
        let s = &mut self.state;
        let inv_dt = 1.0 / dt;
        for r in 0..m_f {
            let scale = inv_dt * self.lambda[r];
            for e in 0..2 {
                if self.j_block[r][e] == NONE {
                    continue;
                }
                s.r_x[r * 2 + e] = scale * self.j[r][e * 3];
                s.r_y[r * 2 + e] = scale * self.j[r][e * 3 + 1];
                s.r_t[r * 2 + e] = scale * self.j[r][e * 3 + 2];
            }
        }

        for i in 0..n {
            s.a_x[i] = s.f_x[i];
            s.a_y[i] = s.f_y[i];
            s.a_theta[i] = s.t[i];
        }
        let mut j_f = 0;
        for c in &self.constraints {
            for _ in 0..c.rows() {
                for k in 0..c.body_count {
                    let body = c.bodies[k];
                    s.a_x[body] += s.r_x[j_f * 2 + k];
                    s.a_y[body] += s.r_y[j_f * 2 + k];
                    s.a_theta[body] += s.r_t[j_f * 2 + k];
                }
                j_f += 1;
            }
        }
        for i in 0..n {
            let inv_mass = self.m_inv[i * 3];
            let inv_inertia = self.m_inv[i * 3 + 2];
            s.a_x[i] *= inv_mass;
            s.a_y[i] *= inv_mass;
            s.a_theta[i] *= inv_inertia;
        }
    }

    /// Projected Gauss-Seidel on (J W J^T) lambda = right, lambda within the limits.
    fn solve_with_limits(&mut self, m_f: usize) {
        if self.lambda.len() != m_f {
            self.lambda.clear();
            self.lambda.resize(m_f, 0.0);
        }
        // a[i][j] = (J_i W) . J_j over shared bodies; zero for rows that share none, which
        // are left out (skipping their exact zeros leaves every sum unchanged).
        if self.neighbours.len() != m_f {
            self.neighbours = (0..m_f)
                .map(|i| {
                    (0..m_f)
                        .filter(|&j| {
                            self.j_block[i].iter().any(|&b| {
                                b != NONE && self.j_block[j].contains(&b)
                            })
                        })
                        .collect()
                })
                .collect();
            self.coupling = self.neighbours.iter().map(|n| vec![0.0; n.len()]).collect();
            self.inv_diagonal = vec![0.0; m_f];
        }
        for i in 0..m_f {
            for (slot, &jj) in self.neighbours[i].iter().enumerate() {
                let mut dot = 0.0;
                for k in 0..2 {
                    let block0 = self.j_block[i][k];
                    if block0 == NONE {
                        continue;
                    }
                    for l in 0..2 {
                        if block0 == self.j_block[jj][l] {
                            for m in 0..3 {
                                let reg = self.m_inv[block0 * 3 + m] * self.j[i][k * 3 + m];
                                dot += reg * self.j[jj][l * 3 + m];
                            }
                        }
                    }
                }
                self.coupling[i][slot] = dot;
                if jj == i {
                    self.inv_diagonal[i] = 1.0 / dot;
                }
            }
        }

        for _ in 0..MAX_ITERATIONS {
            self.iterations += 1;
            let mut max_difference = 0.0;
            for i in 0..m_f {
                let mut s0 = 0.0;
                let mut s1 = 0.0;
                for (&j, &a) in self.neighbours[i].iter().zip(&self.coupling[i]) {
                    if j < i {
                        s0 += a * self.lambda[j];
                    } else if j > i {
                        s1 += a * self.lambda[j];
                    }
                }
                let k_next_i = self.inv_diagonal[i] * (self.right[i] - s0 - s1);
                let [limit_min, limit_max] = self.limits[i];
                let x = limit_min.max(limit_max.min(k_next_i));
                let k_i = self.lambda[i];
                let min_k = 1e-3f64.max(k_i.abs());
                let delta = (x - k_i).abs() / min_k;
                if delta > max_difference {
                    max_difference = delta;
                }
                self.lambda[i] = x;
            }
            if max_difference < MIN_DELTA {
                return;
            }
        }
    }
}
