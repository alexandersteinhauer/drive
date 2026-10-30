//! One-dimensional unsteady gas dynamics in networks of pipes: the exhaust as a reference
//! (offline) simulation, whose sound is the waves that leave its tailpipes.
//!
//! Each pipe is cut into cells, finite volumes of the one-dimensional Euler equations (mass,
//! momentum, energy, and the gas's composition carried along), fifth order in space (WENO-Z)
//! and third in time (strong-stability-preserving Runge-Kutta), with HLLC fluxes (Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd ed., 10.4).
//! The gas is the cylinders' (gas.rs): thermally perfect, its heat capacity ratio falling
//! with temperature, so energy passes between them exactly. The walls drag on the gas and
//! exchange heat with it by the steady-flow correlations at the local Reynolds number
//! (Haaland's friction factor; Colburn's analogy, or laminar flow's Nusselt number).
//!
//! A pipe's end meets a node, a volume of gas at rest (a cylinder, a junction, a chamber)
//! or the atmosphere, through an opening: a valve, an orifice, or the pipe's own section.
//! The state at the end is solved as a half Riemann problem: it lies on the wave curve
//! (shock or rarefaction) of the gas next to it, and passes the mass the opening does,
//! quasi-steady and compressible from the upstream stagnation state to the downstream static
//! pressure, which is how a flow bench defines a port's discharge coefficient. Out of the
//! pipe, the jet's dynamic head is lost in the node; into it, the flow is isentropic up to
//! the opening (Benson, "The Thermodynamics and Gas Dynamics of Internal Combustion
//! Engines", 1982; Winterbone and Pearson, "Theory of Engine Manifold Design", 2000).
//! Nodes exchange gas through orifices, a turbine's nozzle among them, which takes its
//! work from the gas passing.

use crate::gas::{GAMMA_0, GAMMA_SLOPE};
use crate::units::{AIR_MOLECULAR_MASS, R};

/// The gas constant per kg.
pub const RS: f64 = R / AIR_MOLECULAR_MASS;
const NQ: usize = 5;
/// Prandtl number of air and of exhaust gas (both ~0.7 over the temperatures here).
const PRANDTL: f64 = 0.7;

/// Heat capacity ratio at `t` (gas.rs).
#[inline]
pub fn gamma(t: f64) -> f64 {
    GAMMA_0 - GAMMA_SLOPE * t
}

/// Internal energy per kg at `t`: the integral of cv = RS / (gamma - 1), as gas.rs's per mole.
#[inline]
pub fn energy(t: f64) -> f64 {
    -(RS / GAMMA_SLOPE) * (1.0 - GAMMA_SLOPE * t / (GAMMA_0 - 1.0)).ln()
}

/// Temperature at internal energy `e` per kg (energy's inverse).
#[inline]
pub fn temperature(e: f64) -> f64 {
    (GAMMA_0 - 1.0) / GAMMA_SLOPE * (1.0 - (-GAMMA_SLOPE * e / RS).exp())
}

#[inline]
pub fn enthalpy(t: f64) -> f64 {
    energy(t) + RS * t
}

#[inline]
pub fn cp(t: f64) -> f64 {
    let g = gamma(t);
    RS * g / (g - 1.0)
}

/// Temperature at enthalpy `h` per kg (Newton's method from `guess`).
pub fn temperature_of_enthalpy(h: f64, guess: f64) -> f64 {
    let mut t = guess;
    for _ in 0..8 {
        let dt = (enthalpy(t) - h) / cp(t);
        t -= dt;
        if dt.abs() < 1e-9 * t {
            break;
        }
    }
    t
}

/// Dynamic viscosity (Pa s): Sutherland's law for air, which exhaust gas follows closely.
#[inline]
pub fn viscosity(t: f64) -> f64 {
    1.458e-6 * t * t.sqrt() / (t + 110.4)
}

/// Fanning friction factor at Reynolds number `re` and relative roughness `roughness`:
/// laminar, or Haaland's explicit form of Colebrook's.
pub fn friction_factor(re: f64, roughness: f64) -> f64 {
    friction_factor_with(re, (roughness / 3.7).powf(1.11), 16.0)
}

/// friction_factor with Haaland's roughness term, (roughness / 3.7)^1.11, worked out.
#[inline]
fn friction_factor_with(re: f64, rough_term: f64, laminar: f64) -> f64 {
    if re < 1e-9 {
        return 0.0;
    }
    if re < 2300.0 {
        return laminar / re;
    }
    let x = -1.8 * std::f64::consts::LOG10_E * (rough_term + 6.9 / re).ln();
    0.25 / (x * x)
}

/// Mass flow per unit effective area (kg/s m^2) through an opening from stagnation `p0`,
/// `t0` to static `p` (choked below the critical ratio), at heat capacity ratio `g`.
#[inline]
pub fn orifice(p0: f64, t0: f64, g: f64, p: f64) -> f64 {
    let critical = (2.0 / (g + 1.0)).powf(g / (g - 1.0));
    // (max and min, not clamp: a NaN passes through to be caught, rather than panicking.)
    let r = (p / p0).max(critical).min(1.0);
    let lr = r.ln();
    let psi = 2.0 * g / (g - 1.0) * ((2.0 / g * lr).exp() - ((g + 1.0) / g * lr).exp());
    p0 / (RS * t0).sqrt() * psi.max(0.0).sqrt()
}

/// The most an opening of discharge area 1 passes from gas at rest at `p0`, `t0` (kg/s m^2):
/// choked, sonic at its vena contracta. A slug's flow is held to it (it can't be faster).
pub fn choked(p0: f64, t0: f64, g: f64) -> f64 {
    p0 / (RS * t0).sqrt() * (g * (2.0 / (g + 1.0)).powf((g + 1.0) / (g - 1.0))).sqrt()
}

/// The rate of a gas slug's flow `m` (kg/s) through an opening of discharge area `cda` and
/// slug length `l`, between gas at `a` and `b` (pressure, temperature, ratio of specific
/// heats, density): d m/dt = cda/l (p_a - p_b - its jet's loss), the loss m|m| / (2 rho
/// cda^2 Y^2) of the gas it comes from, with the compressible expansion factor Y of its
/// pressure ratio. Once steady the flow is so exactly `orifice`'s, choked included.
pub fn slug_rate(cda: f64, l: f64, m: f64, a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    let (src, dst) = if m > 0.0 || (m == 0.0 && a.0 >= b.0) { (a, b) } else { (b, a) };
    let y2 = expansion2(src.0, src.2, dst.0);
    cda / l * (a.0 - b.0 - m * m.abs() / (2.0 * src.3 * cda * cda * y2))
}

/// The square of the compressible expansion factor Y of `orifice`'s flow from p0 to p, by
/// which it falls short of the incompressible flow, cda sqrt(2 rho0 (p0 - p)): psi / (2 (1 - r)).
pub fn expansion2(p0: f64, g: f64, p: f64) -> f64 {
    let e = 1.0 - p / p0;
    if !(e > 1e-6) {
        return 1.0;
    }
    let critical = (2.0 / (g + 1.0)).powf(g / (g - 1.0));
    let r = (1.0 - e).max(critical);
    let psi = 2.0 * g / (g - 1.0) * (r.powf(2.0 / g) - r.powf((g + 1.0) / g));
    psi.max(0.0) / (2.0 * e)
}

/// The gas in a cell: primitive variables, and what follows from them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prim {
    pub rho: f64,
    pub u: f64,
    pub p: f64,
    pub t: f64,
    pub g: f64,
    pub c: f64,
    /// Mass fractions of fuel and oxygen (engine-sim's mix, its species all of one molar mass).
    pub yf: f64,
    pub yo: f64,
    /// Internal energy per kg.
    pub e: f64,
}

impl Prim {
    pub fn of(q: &[f64; NQ]) -> Prim {
        let rho = q[0];
        let u = q[1] / rho;
        let e = q[2] / rho - 0.5 * u * u;
        Prim::with(rho, u, temperature(e), e, q[3] / rho, q[4] / rho)
    }

    fn with(rho: f64, u: f64, t: f64, e: f64, yf: f64, yo: f64) -> Prim {
        let g = gamma(t);
        Prim { rho, u, p: rho * RS * t, t, g, c: (g * RS * t).sqrt(), yf, yo, e }
    }

    pub fn new(rho: f64, u: f64, p: f64, yf: f64, yo: f64) -> Prim {
        let t = p / (rho * RS);
        Prim::with(rho, u, t, energy(t), yf, yo)
    }

    fn conserved(&self) -> [f64; NQ] {
        let e = self.rho * (self.e + 0.5 * self.u * self.u);
        [self.rho, self.rho * self.u, e, self.rho * self.yf, self.rho * self.yo]
    }

    fn flux(&self) -> [f64; NQ] {
        let m = self.rho * self.u;
        let h0 = self.e + RS * self.t + 0.5 * self.u * self.u;
        [m, m * self.u + self.p, m * h0, m * self.yf, m * self.yo]
    }
}

/// HLLC flux between two states (Toro 10.4, Davis's wave speed estimates).
fn hllc(l: &Prim, r: &Prim) -> [f64; NQ] {
    let sl = (l.u - l.c).min(r.u - r.c);
    let sr = (l.u + l.c).max(r.u + r.c);
    if sl >= 0.0 {
        return l.flux();
    }
    if sr <= 0.0 {
        return r.flux();
    }
    let ml = l.rho * (sl - l.u);
    let mr = r.rho * (sr - r.u);
    let ss = (r.p - l.p + l.u * ml - r.u * mr) / (ml - mr);
    let star = |s: &Prim, sk: f64, m: f64| {
        let f = s.flux();
        let q = s.conserved();
        let k = m / (sk - ss);
        let qs = [k, k * ss, k * (q[2] / s.rho + (ss - s.u) * (ss + s.p / m)), k * s.yf, k * s.yo];
        let mut out = [0.0; NQ];
        for i in 0..NQ {
            out[i] = f[i] + sk * (qs[i] - q[i]);
        }
        out
    };
    if ss >= 0.0 {
        star(l, sl, ml)
    } else {
        star(r, sr, mr)
    }
}

/// The values at the faces either side of cell `c` from the cells `a` to `e` around it:
/// fifth-order WENO-Z (Borges, Carmona, Costa and Don, J. Comput. Phys. 227, 2008), at the
/// face between `b` and `c`, and at the face between `c` and `d`. Both take the same three
/// stencils, mirrored, so their smoothness indicators are worked out once.
#[inline]
fn weno5z(a: f64, b: f64, c: f64, d: f64, e: f64) -> (f64, f64) {
    let s = |x: f64, y: f64, z: f64| 13.0 / 12.0 * (x - 2.0 * y + z).powi(2);
    let b0 = s(a, b, c) + 0.25 * (a - 4.0 * b + 3.0 * c).powi(2);
    let b1 = s(b, c, d) + 0.25 * (b - d).powi(2);
    let b2 = s(c, d, e) + 0.25 * (3.0 * c - 4.0 * d + e).powi(2);
    let tau = (b0 - b2).abs();
    // Small against the variable's own scale: the weights are ideal where it is smooth.
    let eps = 1e-30 * (c * c + 1e-30);
    let r = |b: f64| 1.0 + (tau / (b + eps)).powi(2);
    let (r0, r1, r2) = (r(b0), r(b1), r(b2));
    // Towards d: the stencils ending at c, centred on it, starting from it (ideal weights
    // 0.1, 0.6, 0.3); towards b, mirrored.
    let (w0, w1, w2) = (0.1 * r0, 0.6 * r1, 0.3 * r2);
    let right = (w0 * (2.0 * a - 7.0 * b + 11.0 * c) + w1 * (-b + 5.0 * c + 2.0 * d) + w2 * (2.0 * c + 5.0 * d - e)) / (6.0 * (w0 + w1 + w2));
    let (w0, w1, w2) = (0.1 * r2, 0.6 * r1, 0.3 * r0);
    let left = (w0 * (2.0 * e - 7.0 * d + 11.0 * c) + w1 * (-d + 5.0 * c + 2.0 * b) + w2 * (2.0 * c + 5.0 * b - a)) / (6.0 * (w0 + w1 + w2));
    (left, right)
}

/// A pipe's ends: A at x = 0, B at x = length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

impl Side {
    /// The end's outward normal along the pipe.
    fn sign(self) -> f64 {
        match self {
            Side::A => -1.0,
            Side::B => 1.0,
        }
    }
}

pub struct Pipe {
    pub length: f64,
    /// Hydraulic diameter (4 area / perimeter) at end A: a round pipe's own.
    pub diameter: f64,
    /// Flow area at end A: a round pipe's, or a monolith's open channels' or an annulus's.
    pub area: f64,
    /// Along it (a cone's taper): the flow area at each face between cells (and the ends),
    /// as the cell before it and the cell after it see it (they differ where the section
    /// steps), each cell's mean area, and its hydraulic diameter.
    face_l: Vec<f64>,
    face_r: Vec<f64>,
    cell_area: Vec<f64>,
    cell_d: Vec<f64>,
    pub dx: f64,
    /// Wall roughness (m), at end A.
    pub roughness: f64,
    /// Wall friction (off only for tests against inviscid solutions).
    pub friction: bool,
    /// Per cell, from its section: the wall's temperature (NaN: adiabatic); laminar flow's
    /// Fanning friction factor times Reynolds number and Nusselt number (16 and 3.66 round,
    /// 14.23 and 2.98 square: Shah and London); a packing's flow resistivity (Pa s/m^2, Darcy's
    /// drag); Haaland's roughness term, (roughness / diameter / 3.7)^1.11.
    wall: Vec<f64>,
    laminar: Vec<f64>,
    nusselt: Vec<f64>,
    resistivity: Vec<f64>,
    rough_term: Vec<f64>,
    /// Per cell: density, momentum, total energy per volume, and fuel and oxygen per volume.
    pub q: Vec<[f64; NQ]>,
    q0: Vec<[f64; NQ]>,
    rate: Vec<[f64; NQ]>,
    pub prim: Vec<Prim>,
    padded: Vec<[f64; NQ]>,
    /// `padded` by variable, for WENO along the cells.
    columns: Vec<f64>,
    /// WENO's values at the cells' faces: by variable, at their left faces and at their right.
    faces: Vec<f64>,
    /// Per face where the section steps, where its solution starts (`solve_step`'s hint).
    step_flow: Vec<f64>,
    /// The cells' own values at the faces (first order), for a substep done again.
    first_order: bool,
    /// Flux (x-ward, per area) through end A and end B, set by what they meet.
    face: [[f64; NQ]; 2],
    /// What last passed outward through end A and end B where a link solved it, and the end
    /// pressure solved (where to start next time; 0: none).
    end_flow: [EndFlow; 2],
    end_p: [f64; 2],
}

/// Commercial steel and stainless tube (m).
pub const STEEL_ROUGHNESS: f64 = 0.045e-3;

/// A stretch of pipe: round, its diameter running linearly from `diameter` to `diameter_b`
/// (a cone), or of a given flow `area` with `diameter` its hydraulic diameter (a monolith's
/// channels, an annulus); its wall and any packing.
#[derive(Clone, Debug)]
pub struct Section {
    pub length: f64,
    pub diameter: f64,
    pub diameter_b: f64,
    pub area: Option<f64>,
    pub square: bool,
    pub wall: Option<f64>,
    pub resistivity: f64,
    pub roughness: f64,
}

impl Section {
    pub fn round(length: f64, diameter: f64) -> Section {
        Section {
            length,
            diameter,
            diameter_b: diameter,
            area: None,
            square: false,
            wall: None,
            resistivity: 0.0,
            roughness: STEEL_ROUGHNESS,
        }
    }

    /// Hydraulic diameter at `u` (0..1 along it).
    fn diameter_at(&self, u: f64) -> f64 {
        self.diameter + (self.diameter_b - self.diameter) * u
    }

    fn area_at(&self, u: f64) -> f64 {
        match self.area {
            Some(a) => a,
            None => {
                let d = self.diameter_at(u);
                std::f64::consts::PI * d * d / 4.0
            }
        }
    }
}

impl Pipe {
    /// A pipe of `length` and `diameter` in cells of about `dx`, full of gas at `p`, `t`.
    pub fn new(length: f64, diameter: f64, dx: f64, p: f64, t: f64, yf: f64, yo: f64) -> Pipe {
        Pipe::with_sections(&[Section::round(length, diameter)], dx, p, t, yf, yo)
    }

    /// A pipe of `sections` end to end, in cells of about `dx` (each section's ends moved to
    /// the nearest face between cells), full of gas at `p`, `t`. Where the area steps, the
    /// face between takes the narrower: a jet out of it meets the step's wall at its own
    /// pressure, Borda's sudden expansion, losing (u1 - u2)^2 / 2 of its head.
    pub fn with_sections(sections: &[Section], dx: f64, p: f64, t: f64, yf: f64, yo: f64) -> Pipe {
        let length: f64 = sections.iter().map(|s| s.length).sum();
        let n = ((length / dx).round() as usize).max(2);
        let dx = length / n as f64;
        // Each section's first face.
        let mut starts = Vec::with_capacity(sections.len() + 1);
        let mut x = 0.0;
        for s in sections {
            starts.push(((x / dx).round() as usize).min(n));
            x += s.length;
        }
        starts.push(n);
        let mut face_l: Vec<f64> = vec![0.0; n + 1];
        let mut face_r: Vec<f64> = vec![0.0; n + 1];
        let mut cell_area = vec![0.0; n];
        let mut cell_d = vec![0.0; n];
        let (mut wall, mut laminar, mut nusselt, mut resistivity, mut rough_term) =
            (vec![f64::NAN; n], vec![16.0; n], vec![3.66; n], vec![0.0; n], vec![0.0; n]);
        for (j, sec) in sections.iter().enumerate() {
            let (k0, k1) = (starts[j], starts[j + 1]);
            let cells = (k1 - k0).max(1) as f64;
            // Local position 0..1 along the section.
            let at = |u: f64| sec.area_at(u);
            for k in k0..=k1 {
                let a = at((k - k0) as f64 / cells);
                if k > k0 || j == 0 {
                    face_l[k] = a;
                }
                if k < k1 || j + 1 == sections.len() {
                    face_r[k] = a;
                }
            }
            for i in k0..k1 {
                let (u0, u1) = ((i - k0) as f64 / cells, (i + 1 - k0) as f64 / cells);
                cell_area[i] = (at(u0) + 4.0 * at(0.5 * (u0 + u1)) + at(u1)) / 6.0;
                cell_d[i] = sec.diameter_at(0.5 * (u0 + u1));
                wall[i] = sec.wall.unwrap_or(f64::NAN);
                (laminar[i], nusselt[i]) = if sec.square { (14.23, 2.98) } else { (16.0, 3.66) };
                resistivity[i] = sec.resistivity;
                rough_term[i] = (sec.roughness / cell_d[i] / 3.7).powf(1.11);
            }
        }
        let state = Prim::new(p / (RS * t), 0.0, p, yf, yo);
        Pipe {
            length,
            diameter: cell_d[0],
            area: face_r[0],
            face_l,
            face_r,
            cell_area,
            cell_d,
            dx,
            roughness: sections[0].roughness,
            friction: true,
            wall,
            laminar,
            nusselt,
            resistivity,
            rough_term,
            q: vec![state.conserved(); n],
            q0: vec![[0.0; NQ]; n],
            rate: vec![[0.0; NQ]; n],
            prim: vec![state; n],
            padded: Vec::with_capacity(n + 6),
            columns: vec![0.0; NQ * (n + 6)],
            faces: vec![0.0; 2 * NQ * n],
            step_flow: vec![0.0; n + 1],
            first_order: false,
            face: [[0.0; NQ]; 2],
            end_flow: [EndFlow::default(); 2],
            end_p: [0.0; 2],
        }
    }

    pub fn cells(&self) -> usize {
        self.q.len()
    }

    /// A round pipe whose diameter runs linearly from `a` at end A to `b` at end B (a cone),
    /// its gas as it was per unit volume.
    pub fn taper(&mut self, a: f64, b: f64) {
        let n = self.cells();
        let d = |x: f64| a + (b - a) * x / self.length;
        let area = |x: f64| std::f64::consts::PI * d(x) * d(x) / 4.0;
        for k in 0..=n {
            self.face_l[k] = area(k as f64 * self.dx);
            self.face_r[k] = self.face_l[k];
        }
        for i in 0..n {
            let (x0, x1) = (i as f64 * self.dx, (i + 1) as f64 * self.dx);
            // Simpson's rule is exact for the quadratic.
            self.cell_area[i] = (area(x0) + 4.0 * area(0.5 * (x0 + x1)) + area(x1)) / 6.0;
            self.cell_d[i] = d(0.5 * (x0 + x1));
        }
        self.diameter = a;
        self.area = self.face_r[0];
    }

    /// The wall's temperature all along (None: adiabatic).
    pub fn set_wall(&mut self, t: Option<f64>) {
        self.wall.iter_mut().for_each(|w| *w = t.unwrap_or(f64::NAN));
    }

    /// A flow area other than the hydraulic diameter's circle (a monolith's open channels,
    /// an annulus), alike all along.
    pub fn set_area(&mut self, area: f64) {
        self.area = area;
        self.face_l.iter_mut().for_each(|f| *f = area);
        self.face_r.iter_mut().for_each(|f| *f = area);
        self.cell_area.iter_mut().for_each(|f| *f = area);
    }

    /// The flow area at an end.
    pub fn end_area(&self, side: Side) -> f64 {
        match side {
            Side::A => self.face_r[0],
            Side::B => self.face_l[self.face_l.len() - 1],
        }
    }

    /// A cell's volume and hydraulic diameter.
    pub fn cell_volume(&self, i: usize) -> f64 {
        self.cell_area[i] * self.dx
    }

    pub fn cell_diameter(&self, i: usize) -> f64 {
        self.cell_d[i]
    }

    fn update_prim(&mut self) {
        for (p, q) in self.prim.iter_mut().zip(&self.q) {
            *p = Prim::of(q);
        }
    }

    /// The cells' primitive variables (density, velocity, pressure, fuel, oxygen), padded
    /// with three ghost cells at each end extrapolated from the four nearest (cubic), for the
    /// stencils near the ends: where the ends' gas isn't smooth, WENO takes the stencils
    /// inside. Ghosts that would hold a vacuum copy the end cell instead.
    fn pad(&mut self) {
        let n = self.prim.len();
        let w = |p: &Prim| [p.rho, p.u, p.p, p.yf, p.yo];
        self.padded.clear();
        self.padded.extend([[0.0; NQ]; 3]);
        self.padded.extend(self.prim.iter().map(w));
        self.padded.extend([[0.0; NQ]; 3]);
        let v = &mut self.padded;
        let cubic = n >= 4;
        let extrapolate = |p: [f64; NQ], q: [f64; NQ], r: [f64; NQ], s: [f64; NQ]| -> [f64; NQ] {
            std::array::from_fn(|k| 4.0 * p[k] - 6.0 * q[k] + 4.0 * r[k] - s[k])
        };
        for g in 0..3 {
            // A: ghost index 2 - g from the four cells beyond it; B: n + 3 + g.
            let (a, b) = (2 - g, n + 3 + g);
            v[a] = if cubic { extrapolate(v[a + 1], v[a + 2], v[a + 3], v[a + 4]) } else { v[3] };
            v[b] = if cubic { extrapolate(v[b - 1], v[b - 2], v[b - 3], v[b - 4]) } else { v[n + 2] };
            for i in [a, b] {
                if v[i][0] <= 0.0 || v[i][2] <= 0.0 {
                    v[i] = if i == a { v[3] } else { v[n + 2] };
                }
            }
        }
        let m = n + 6;
        for (i, x) in v.iter().enumerate() {
            for k in 0..NQ {
                self.columns[k * m + i] = x[k];
            }
        }
        let (left, right) = self.faces.split_at_mut(NQ * n);
        for k in 0..NQ {
            let x = &self.columns[k * m..(k + 1) * m];
            let (l, r) = (&mut left[k * n..(k + 1) * n], &mut right[k * n..(k + 1) * n]);
            // Cell i is x[i + 3].
            for i in 0..n {
                (l[i], r[i]) = weno5z(x[i + 1], x[i + 2], x[i + 3], x[i + 4], x[i + 5]);
            }
        }
    }

    /// WENO's values at cell i's left face (`side` 0) or right face (1).
    #[inline]
    fn face_value(&self, i: usize, side: usize) -> [f64; NQ] {
        let n = self.prim.len();
        std::array::from_fn(|k| self.faces[(side * NQ + k) * n + i])
    }

    /// The gas at an end (reconstructed at the end itself, from inside), its velocity along
    /// the outward normal.
    fn end(&self, side: Side) -> EndGas {
        let n = self.prim.len();
        let x = match side {
            Side::A => self.face_value(0, 0),
            Side::B => self.face_value(n - 1, 1),
        };
        let cell = match side {
            Side::A => self.prim[0],
            Side::B => self.prim[n - 1],
        };
        let p = if x[0] > 0.0 && x[2] > 0.0 && !self.first_order { Prim::new(x[0], x[1], x[2], x[3], x[4]) } else { cell };
        EndGas { rho: p.rho, un: side.sign() * p.u, p: p.p, t: p.t, g: p.g, c: p.c, yf: p.yf, yo: p.yo }
    }

    /// Sets the flux through an end from what passes outward there.
    fn set_end(&mut self, side: Side, f: &EndFlow) {
        self.face[(side == Side::B) as usize] = end_flux(side.sign(), self.end_area(side), f);
    }

    /// The cells' rates of change (padded, its ends' faces set).
    fn rates(&mut self) {
        let n = self.prim.len();
        for r in self.rate.iter_mut() {
            *r = [0.0; NQ];
        }
        let inv_dx = 1.0 / self.dx;
        // Fluxes per unit area times the face's area, over the cell's volume; and the walls'
        // push on the gas where the area changes, p dA.
        let (first, last) = (self.face_r[0] / self.cell_area[0], self.face_l[n] / self.cell_area[n - 1]);
        for k in 0..NQ {
            self.rate[0][k] += self.face[0][k] * first * inv_dx;
            self.rate[n - 1][k] -= self.face[1][k] * last * inv_dx;
        }
        for i in 0..n {
            let da = self.face_l[i + 1] - self.face_r[i];
            if da != 0.0 {
                self.rate[i][1] += self.prim[i].p * da / self.cell_area[i] * inv_dx;
            }
        }
        for i in 0..n - 1 {
            // Face i + 1/2: cell i is padded[i + 3].
            let v = &self.padded[i + 1..i + 7];
            let mut l = self.face_value(i, 1);
            let mut r = self.face_value(i + 1, 0);
            // Where the reconstruction would leave a vacuum (never in smooth flow), the
            // cells' own values.
            if l[0] <= 0.0 || l[2] <= 0.0 || self.first_order {
                l = v[2];
            }
            if r[0] <= 0.0 || r[2] <= 0.0 || self.first_order {
                r = v[3];
            }
            let l = Prim::new(l[0], l[1], l[2], l[3], l[4]);
            let r = Prim::new(r[0], r[1], r[2], r[3], r[4]);
            let (al, ar) = (self.face_l[i + 1], self.face_r[i + 1]);
            let (fl, fr) = if al == ar {
                let f = hllc(&l, &r);
                (f, f)
            } else {
                // A step: each side its own flux, the step's wall taking the difference.
                let gas = |p: &Prim, sign: f64| EndGas { rho: p.rho, un: sign * p.u, p: p.p, t: p.t, g: p.g, c: p.c, yf: p.yf, yo: p.yo };
                let (a, b, hint) = solve_step(&gas(&l, 1.0), al, &gas(&r, -1.0), ar, self.step_flow[i + 1]);
                self.step_flow[i + 1] = hint;
                (end_flux(1.0, al, &a), end_flux(-1.0, ar, &b))
            };
            let (a, b) = (al / self.cell_area[i], ar / self.cell_area[i + 1]);
            for k in 0..NQ {
                self.rate[i][k] -= fl[k] * a * inv_dx;
                self.rate[i + 1][k] += fr[k] * b * inv_dx;
            }
        }
        // The walls: friction, and heat.
        // The walls: a packing's drag, friction, and heat.
        let any_wall = self.wall.iter().any(|w| !w.is_nan());
        if !self.friction && !any_wall {
            return;
        }
        // Pr^(-2/3) at 0.7.
        const COLBURN: f64 = 1.268_434;
        for (i, (r, p)) in self.rate.iter_mut().zip(&self.prim).enumerate() {
            let d = self.cell_d[i];
            if self.resistivity[i] > 0.0 {
                r[1] -= self.resistivity[i] * p.u;
            }
            let mu = viscosity(p.t);
            let re = p.rho * p.u.abs() * d / mu;
            let cf = friction_factor_with(re, self.rough_term[i], self.laminar[i]);
            if self.friction {
                r[1] -= 4.0 / d * 0.5 * p.rho * cf * p.u * p.u.abs();
            }
            let tw = self.wall[i];
            if !tw.is_nan() {
                let cpt = cp(p.t);
                let laminar = self.nusselt[i] * mu * cpt / PRANDTL / d;
                let turbulent = 0.5 * cf * COLBURN * p.rho * cpt * p.u.abs();
                r[2] += 4.0 / d * laminar.max(turbulent) * (tw - p.t);
            }
        }
    }

    fn stable_dt(&self) -> f64 {
        let mut s = 0.0f64;
        for p in &self.prim {
            s = s.max(p.u.abs() + p.c);
        }
        self.dx / s
    }

    /// Mass and total energy in the pipe.
    pub fn totals(&self) -> (f64, f64) {
        self.q.iter().enumerate().fold((0.0, 0.0), |(m, e), (i, q)| {
            let v = self.cell_volume(i);
            (m + q[0] * v, e + q[2] * v)
        })
    }
}

/// The flux along the pipe (per unit area) through an end of `area` whose outward normal
/// points along `sign`, from what passes outward through it.
fn end_flux(sign: f64, area: f64, f: &EndFlow) -> [f64; NQ] {
    let a = area;
    [sign * f.m / a, f.m * f.un / a + f.p, sign * f.energy / a, sign * f.m * f.yf / a, sign * f.m * f.yo / a]
}

/// The gas at a pipe's end, its velocity along the end's outward normal.
#[derive(Clone, Copy, Debug)]
pub struct EndGas {
    pub rho: f64,
    pub un: f64,
    pub p: f64,
    pub t: f64,
    pub g: f64,
    pub c: f64,
    pub yf: f64,
    pub yo: f64,
}

/// What a node offers an end: its pressure (its stagnation pressure, at rest), temperature,
/// and composition.
#[derive(Clone, Copy, Debug)]
pub struct Reservoir {
    pub p: f64,
    pub t: f64,
    pub yf: f64,
    pub yo: f64,
}

/// What passes outward through an end: mass (kg/s), its total enthalpy flow (W) and
/// composition, and the end's static pressure, outward velocity and density.
#[derive(Clone, Copy, Debug, Default)]
pub struct EndFlow {
    pub m: f64,
    pub energy: f64,
    pub yf: f64,
    pub yo: f64,
    pub p: f64,
    pub un: f64,
    pub rho: f64,
}

/// The state reached from `s` by the wave it sends into the pipe at end pressure `p`: the
/// outward velocity and the density on the pipe's side of the contact (Toro 4.2).
fn wave(s: &EndGas, p: f64) -> (f64, f64) {
    let g = s.g;
    if p > s.p {
        let a = 2.0 / ((g + 1.0) * s.rho);
        let b = (g - 1.0) / (g + 1.0) * s.p;
        let f = (p - s.p) * (a / (p + b)).sqrt();
        let r = p / s.p;
        let k = (g - 1.0) / (g + 1.0);
        (s.un - f, s.rho * (r + k) / (k * r + 1.0))
    } else {
        let lr = (p / s.p).ln();
        let f = 2.0 * s.c / (g - 1.0) * (lr * (g - 1.0) / (2.0 * g)).exp_m1();
        (s.un - f, s.rho * (lr / g).exp())
    }
}

/// The flow through an end with gas `s`, of `area`, into a node `res` through an opening of
/// effective area `cda`. `z`: the node's pressure rises by z per unit of outward velocity
/// (an open end's radiation; 0 for a node).
/// `hint`: the end pressure solved last time (0: none), to start from. Returns the flow and
/// the end pressure solved (0 where none was).
fn solve_end(s: &EndGas, area: f64, cda: f64, res: &Reservoir, z: f64, hint: f64) -> (EndFlow, f64) {
    let res_g = gamma(res.t);
    let res_h = enthalpy(res.t);
    // The node's gas made sonic (the same at every end pressure): once, if needed.
    let sonic = std::cell::Cell::new(None);
    let sonic = || {
        sonic.get().unwrap_or_else(|| {
            let v = sonic_state(res_h, res.t);
            sonic.set(Some(v));
            v
        })
    };
    let eval = |p: f64| -> (f64, EndFlow) {
        let (un, rho_w) = wave(s, p);
        let pn = res.p + z * un;
        if un >= 0.0 {
            let t = p / (rho_w * RS);
            let t0 = t + un * un / (2.0 * cp(t));
            let g = gamma(t);
            let p0 = p * (t0 / t).powf(g / (g - 1.0));
            let through = if p0 > pn {
                cda * orifice(p0, t0, g, pn)
            } else if p < pn {
                -cda * orifice(pn, res.t, res_g, p)
            } else {
                0.0
            };
            let m = rho_w * un * area;
            let f = EndFlow { m, energy: m * (enthalpy(t) + 0.5 * un * un), yf: s.yf, yo: s.yo, p, un, rho: rho_w };
            (m - through, f)
        } else {
            // Into the pipe: what the opening passes at this end pressure, carried at the
            // end by the subsonic state with that mass flux and the node's total enthalpy
            // (a jet into a pipe wider than it mixes out; it doesn't expand on supersonically
            // as in a nozzle's diverging part), at most sonic. The residual: how far the
            // end's velocity from the wave curve is from that, as a mass flow.
            // The face's own pressure: below the critical pressure a plain entry chokes, its
            // face sonic at the critical pressure (the pipe expanding on beyond it).
            let (v, rho, p_face) = if p >= pn {
                // Nothing drives gas in.
                (0.0, p / (RS * res.t), p)
            } else if cda >= area {
                // Through the pipe's own section (or wider): isentropic from the node's
                // stagnation state to the end's pressure, at most sonic. (Finding the state
                // from its mass flux instead is ill-conditioned near sonic, where the flux
                // hardly changes with the velocity.)
                isentropic_inflow(p, pn, res.t, res_h)
            } else {
                let g = if p < pn { cda * orifice(pn, res.t, res_g, p) / area } else { 0.0 };
                let (v, rho) = inflow_state_with(g, p, res_h, res.t, sonic());
                (v, rho, p)
            };
            let m = -rho * v * area;
            let f = EndFlow { m, energy: m * res_h, yf: res.yf, yo: res.yo, p: p_face, un: -v, rho };
            ((un + v) * rho * area, f)
        }
    };
    if cda == 0.0 && z == 0.0 {
        // A wall: nothing passes, the gas brought to rest against it.
        let p = rest_pressure(s);
        return (EndFlow { m: 0.0, energy: 0.0, yf: s.yf, yo: s.yo, p, un: 0.0, rho: wave(s, p).1 }, 0.0);
    }
    if s.un <= -s.c && res.p + z * s.un > s.p {
        // Supersonic inflow: nothing from inside reaches the end, the node sets it, choked
        // through the opening (sonic in the pipe's own section, or mixed out from a narrower
        // opening's choked jet at the pipe's pressure).
        let pn = res.p + z * s.un;
        let (v, rho, p) = if cda >= area {
            let g = gamma(res.t);
            let critical = pn * (2.0 / (g + 1.0)).powf(g / (g - 1.0));
            isentropic_inflow(critical, pn, res.t, res_h)
        } else {
            let (v, rho) = inflow_state(cda * orifice(pn, res.t, res_g, 0.0) / area, s.p, res_h, res.t);
            (v, rho, s.p)
        };
        let m = -rho * v * area;
        return (EndFlow { m, energy: m * res_h, yf: res.yf, yo: res.yo, p, un: -v, rho }, 0.0);
    }
    if s.un >= s.c {
        // Supersonic outflow: nothing reaches the end from the node.
        let m = s.rho * s.un * area;
        let h0 = enthalpy(s.t) + 0.5 * s.un * s.un;
        return (EndFlow { m, energy: m * h0, yf: s.yf, yo: s.yo, p: s.p, un: s.un, rho: s.rho }, 0.0);
    }
    let g = s.g;
    let x = (g - 1.0) / (g + 1.0) * (s.un / s.c) + 2.0 / (g + 1.0);
    let p_sonic = s.p * x.powf(2.0 * g / (g - 1.0));
    // The residual falls as the end pressure rises: a bracket near last time's, which also
    // shows the end isn't choked; else from the sonic pressure up.
    let near = if hint > p_sonic { bracket_near(|p| eval(p).0, hint, p_sonic, f64::INFINITY, false) } else { None };
    let (a, ra, b, rb) = match near {
        Some(x) => x,
        None => {
            let (r_lo, f_lo) = eval(p_sonic);
            if r_lo <= 0.0 {
                // The end chokes: more would pass the opening than the pipe can bring.
                return (f_lo, p_sonic);
            }
            let mut hi = s.p.max(res.p) * 1.5;
            let mut r_hi = eval(hi).0;
            while r_hi > 0.0 {
                hi *= 2.0;
                r_hi = eval(hi).0;
            }
            (p_sonic, r_lo, hi, r_hi)
        }
    };
    // (Illinois takes a rising function.)
    let (c, f) = illinois(
        |p| {
            let (r, f) = eval(p);
            (-r, f)
        },
        a,
        -ra,
        b,
        -rb,
        1e-10,
    );
    (f, c)
}

/// Gas accelerated isentropically from rest at `p0`, `t0` (enthalpy `h0`) to static pressure
/// `p`: its velocity, density and pressure; below the critical pressure, the sonic state.
fn isentropic_inflow(p: f64, p0: f64, t0: f64, h0: f64) -> (f64, f64, f64) {
    let g = gamma(t0);
    let critical = p0 * (2.0 / (g + 1.0)).powf(g / (g - 1.0));
    let ps = p.max(critical);
    let t = t0 * (ps / p0).powf((g - 1.0) / g);
    let v = (2.0 * (h0 - enthalpy(t))).max(0.0).sqrt();
    (v, ps / (RS * t), ps)
}

/// The sonic state of gas with total enthalpy `h0`: its temperature and velocity, h(T) +
/// gamma R T / 2 = h0 (Newton's method from `t0`, the stagnation temperature).
fn sonic_state(h0: f64, t0: f64) -> (f64, f64) {
    let mut ts = t0 * 0.85;
    for _ in 0..20 {
        let r = enthalpy(ts) + 0.5 * gamma(ts) * RS * ts - h0;
        let dr = cp(ts) + 0.5 * RS * (gamma(ts) - GAMMA_SLOPE * ts);
        ts -= r / dr;
        if r.abs() < 1e-9 * h0 {
            break;
        }
    }
    (ts, (gamma(ts) * RS * ts).sqrt())
}

/// The subsonic state carrying mass flux `g` (kg/s m^2) at static pressure `p` with total
/// enthalpy `h0`: its velocity and density; the sonic state if `g` is more than the most it
/// can carry. `t0` is a guess at the temperature (the stagnation temperature).
fn inflow_state(g: f64, p: f64, h0: f64, t0: f64) -> (f64, f64) {
    inflow_state_with(g, p, h0, t0, sonic_state(h0, t0))
}

/// inflow_state, given the sonic state (`sonic_state`, the same for every pressure).
fn inflow_state_with(g: f64, p: f64, h0: f64, t0: f64, (ts, vs): (f64, f64)) -> (f64, f64) {
    // At rest the gas is at its stagnation temperature.
    let rho0 = p / (RS * t0);
    if g <= 0.0 {
        return (0.0, rho0);
    }
    let rho_s = p / (RS * ts);
    if g >= rho_s * vs {
        return (vs, rho_s);
    }
    // Newton's method for the mass flux rho v = g over the velocity. The flux rises with the
    // velocity up to the sonic point, concave: from below the root (v = g / rho0, rho falling
    // as v rises), each step lands below it again, closer. d(rho v)/dv = rho (1 - v^2 / (cp
    // T)), as dT/dv = -v / cp. The temperature from the last one's (Newton's method again).
    // (Kept to a bracket, halved should a step leave it, as round-off near the sonic point
    // could make it.)
    let (mut lo, mut hi) = (0.0, vs);
    let (mut v, mut t, mut rho) = (g / rho0, t0, rho0);
    for _ in 0..60 {
        t = temperature_of_enthalpy((h0 - 0.5 * v * v).max(1e-3 * h0), t);
        rho = p / (RS * t);
        let r = rho * v - g;
        if r.abs() <= 1e-12 * g {
            break;
        }
        if r > 0.0 {
            hi = v;
        } else {
            lo = v;
        }
        v -= r / (rho * (1.0 - v * v / (cp(t) * t)));
        if !(v > lo && v < hi) {
            v = 0.5 * (lo + hi);
        }
    }
    (v, rho)
}

/// A volute's stator channels' flows (`Nozzles`) from `pipe`'s cells into the node at `res`:
/// their rates added to the cells, what they pass into the node kept (`gain`).
fn discharge(z: &mut Nozzles, pipe: &mut Pipe, res: Reservoir) {
    let res_h = enthalpy(res.t);
    let res_g = gamma(res.t);
    // (The most the node can pass back through a unit area.)
    let back = choked(res.p, res.t, res_g);
    let mut total = 0.0;
    let (mut m_sum, mut e_sum, mut f_sum, mut o_sum) = (0.0, 0.0, 0.0, 0.0);
    for (c, &cda) in z.cda.iter().enumerate() {
        let i = z.first + c;
        let cell = pipe.prim[i];
        z.m[c] = z.m[c].clamp(-cda * back, cda * choked(cell.p, cell.t, cell.g));
        let m = z.m[c];
        z.dm[c] = slug_rate(cda, z.length, m, (cell.p, cell.t, cell.g, cell.rho), (res.p, res.t, res_g, res.p / (RS * res.t)));
        let (h0, yf, yo, u) = if m >= 0.0 {
            (cell.e + RS * cell.t + 0.5 * cell.u * cell.u, cell.yf, cell.yo, cell.u)
        } else {
            (res_h, res.yf, res.yo, 0.0)
        };
        let inv_v = 1.0 / pipe.cell_volume(i);
        let r = &mut pipe.rate[i];
        r[0] -= m * inv_v;
        r[1] -= m * u * inv_v;
        r[2] -= m * h0 * inv_v;
        r[3] -= m * yf * inv_v;
        r[4] -= m * yo * inv_v;
        m_sum += m;
        e_sum += m * h0;
        f_sum += m * yf;
        o_sum += m * yo;
        total += m;
    }
    z.flow = total;
    z.gain = [m_sum, e_sum, f_sum, o_sum];
}

/// DUCTCHECK set: every stage's flows are checked, the first not finite reported with what
/// made it.
fn debug_checks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DUCTCHECK").is_ok())
}

/// The pressure at which the gas at an end comes to rest against it (no flow through it):
/// where its wave curve crosses zero velocity, a rarefaction's or a shock's in closed form.
fn rest_pressure(s: &EndGas) -> f64 {
    let g = s.g;
    if s.un >= 0.0 {
        // Coming on: stopped by a shock. un^2 = x^2 a / (x + p + b), x = p* - p.
        let a = 2.0 / ((g + 1.0) * s.rho);
        let b = (g - 1.0) / (g + 1.0) * s.p;
        let u2 = s.un * s.un;
        s.p + (u2 + (u2 * u2 + 4.0 * a * u2 * (s.p + b)).sqrt()) / (2.0 * a)
    } else {
        // Drawing away: a rarefaction (to a vacuum, past the escape speed).
        let x = (1.0 + (g - 1.0) * s.un / (2.0 * s.c)).max(0.0);
        s.p * x.powf(2.0 * g / (g - 1.0))
    }
}

/// A bracket [a, b] of the root of `f` near `x0`, within [lo, hi], `f` rising through it if
/// `rising` (else falling): 0.1 % either side of x0, widened fourfold at a time towards it;
/// with f there. None if not found so (the caller brackets it from scratch).
fn bracket_near(f: impl Fn(f64) -> f64, x0: f64, lo: f64, hi: f64, rising: bool) -> Option<(f64, f64, f64, f64)> {
    // Below the root: f negative if rising, positive if falling.
    let below = |v: f64| if rising { v < 0.0 } else { v > 0.0 };
    let mut d = 1e-3;
    let (mut a, mut b) = ((x0 * (1.0 - d)).max(lo), (x0 * (1.0 + d)).min(hi));
    if !(a < b) {
        return None;
    }
    let (mut fa, mut fb) = (f(a), f(b));
    for _ in 0..8 {
        if below(fa) && !below(fb) {
            return Some((a, fa, b, fb));
        }
        d *= 4.0;
        if !below(fa) {
            if a <= lo {
                return None;
            }
            (b, fb) = (a, fa);
            a = (x0 * (1.0 - d)).max(lo);
            fa = f(a);
        } else {
            if b >= hi {
                return None;
            }
            (a, fa) = (b, fb);
            b = (x0 * (1.0 + d)).min(hi);
            fb = f(b);
        }
    }
    None
}

/// Illinois's regula falsi for `f` rising through zero on [a, b] (f(a) < 0 < f(b) given):
/// the root, and what `f` gave with it there. To `tol` of it: till a step moves it less (as
/// the steps converge superlinearly, its error is then far less), or the bracket is so small.
fn illinois<T>(f: impl Fn(f64) -> (f64, T), mut a: f64, mut fa: f64, mut b: f64, mut fb: f64, tol: f64) -> (f64, T) {
    let mut side = 0;
    let mut last = f64::NAN;
    let mut i = 0;
    loop {
        let mut c = (a * fb - b * fa) / (fb - fa);
        // Where the secant is undefined or leaves the bracket (residuals alike, as near no
        // flow), halve it.
        if !(c > a.min(b) && c < a.max(b)) {
            c = 0.5 * (a + b);
        }
        let (fc, out) = f(c);
        i += 1;
        if fc == 0.0 || (c - last).abs() <= tol * c.abs() || (b - a).abs() <= tol * (a.abs() + b.abs()) || i == 100 {
            return (c, out);
        }
        last = c;
        if fc < 0.0 {
            a = c;
            fa = fc;
            if side == 1 {
                fb *= 0.5;
            }
            side = 1;
        } else {
            b = c;
            fb = fc;
            if side == -1 {
                fa *= 0.5;
            }
            side = -1;
        }
    }
}

/// Flow through a sudden change of section between two pipes' ends (or within a pipe): gas
/// `a` at an end of area `area_a`, gas `b` at one of `area_b`, each with its velocity along
/// its own outward normal (towards the other). Mass and total enthalpy pass; out of the
/// narrower into the wider, the jet meets the step's wall at its own pressure (Borda and
/// Carnot: the momentum balance over the wider section); from the wider into the narrower,
/// isentropically (the total pressure kept). Each side's state lies on its gas's wave curve.
/// Returns what passes outward through each end.
/// `hint`: where to start, as returned last time: the upstream end's pressure, positive with
/// the flow from `a` to `b`, negative from `b` to `a` (0: none).
pub fn solve_step(a: &EndGas, area_a: f64, b: &EndGas, area_b: f64, hint: f64) -> (EndFlow, EndFlow, f64) {
    let rest_a = rest_pressure(a);
    if wave(b, rest_a).0 < 0.0 {
        let (fa, fb, p) = step_from(a, area_a, b, area_b, rest_a, hint.max(0.0));
        (fa, fb, p)
    } else {
        let rest_b = rest_pressure(b);
        if wave(a, rest_b).0 < 0.0 {
            let (fb, fa, p) = step_from(b, area_b, a, area_a, rest_b, (-hint).max(0.0));
            (fa, fb, -p)
        } else {
            // Both at rest at the same pressure.
            let fa = EndFlow { m: 0.0, energy: 0.0, yf: a.yf, yo: a.yo, p: rest_a, un: 0.0, rho: wave(a, rest_a).1 };
            let fb = EndFlow { m: 0.0, energy: 0.0, yf: b.yf, yo: b.yo, p: rest_a, un: 0.0, rho: wave(b, rest_a).1 };
            (fa, fb, 0.0)
        }
    }
}

/// solve_step with the flow from `up` into `down`; `rest` is where up's gas is at rest;
/// `hint`, up's end pressure last time (0: none). Returns also up's end pressure.
fn step_from(up: &EndGas, area_up: f64, down: &EndGas, area_down: f64, rest: f64, hint: f64) -> (EndFlow, EndFlow, f64) {
    let g = up.g;
    // The most `up` can pass: its end sonic (or its gas's own flux, already supersonic).
    let p_sonic = if up.un < up.c {
        let x = (g - 1.0) / (g + 1.0) * (up.un / up.c) + 2.0 / (g + 1.0);
        up.p * x.powf(2.0 * g / (g - 1.0))
    } else {
        up.p
    };
    // Given up's end pressure, from its sonic point (passing all it can) to rest (nothing):
    // the mass flow its wave curve brings there, down's end, and how far down's wave curve is
    // from it (rising with the flow, so falling with up's end pressure).
    let solve = |p_u: f64| -> (f64, EndFlow, EndFlow) {
        let (un_u, rho_u) = wave(up, p_u);
        let m = if p_u >= rest { 0.0 } else { (rho_u * un_u * area_up).max(0.0) };
        let t_u = p_u / (rho_u * RS);
        let h0 = enthalpy(t_u) + 0.5 * un_u * un_u;
        let g_down = m / area_down;
        // Down's end: its static pressure and inward velocity.
        let (p_d, v_d, rho_d) = if area_down >= area_up {
            // Borda: p_d = p_u + (m / A_d) (u_u - v_d), with v_d = m / (rho_d A_d).
            let mut p_d = p_u;
            let mut v_d = 0.0;
            let mut rho_d = rho_u;
            let mut t_d = t_u;
            for _ in 0..30 {
                t_d = temperature_of_enthalpy(h0 - 0.5 * v_d * v_d, t_d);
                rho_d = p_d / (RS * t_d);
                v_d = g_down / rho_d;
                let next = p_u + g_down * (un_u - v_d);
                if (next - p_d).abs() < 1e-10 * p_d {
                    p_d = next;
                    break;
                }
                p_d = next;
            }
            (p_d, v_d, rho_d)
        } else {
            // Isentropic from up's stagnation state to the flux m / A_d.
            let t0 = temperature_of_enthalpy(h0, t_u);
            let g0 = gamma(0.5 * (t0 + t_u));
            let p0 = p_u * (t0 / t_u).powf(g0 / (g0 - 1.0));
            let state = |p: f64| {
                let t = t0 * (p / p0).powf((g0 - 1.0) / g0);
                let v = (2.0 * (h0 - enthalpy(t))).max(0.0).sqrt();
                (p / (RS * t), v)
            };
            let critical = p0 * (2.0 / (g0 + 1.0)).powf(g0 / (g0 - 1.0));
            let (rc, vc) = state(critical);
            if g_down >= rc * vc {
                (critical, vc, rc)
            } else if g_down <= 0.0 {
                (p0, 0.0, p0 / (RS * t0))
            } else {
                // Newton's method over the velocity, as inflow_state_with: the flux rises with
                // it, concave, to the critical point. Along the isentrope rho ~ T^(1 / (g0 -
                // 1)), so d(rho v)/dv = rho (1 - v^2 / ((g0 - 1) cp T)).
                let rho0 = p0 / (RS * t0);
                let (mut lo, mut hi) = (0.0, vc);
                let (mut v, mut t, mut rho) = (g_down / rho0, t0, rho0);
                for _ in 0..60 {
                    t = temperature_of_enthalpy(h0 - 0.5 * v * v, t);
                    rho = rho0 * (t / t0).powf(1.0 / (g0 - 1.0));
                    let r = rho * v - g_down;
                    if r.abs() <= 1e-12 * g_down {
                        break;
                    }
                    if r > 0.0 {
                        hi = v;
                    } else {
                        lo = v;
                    }
                    v -= r / (rho * (1.0 - v * v / ((g0 - 1.0) * cp(t) * t)));
                    if !(v > lo && v < hi) {
                        v = 0.5 * (lo + hi);
                    }
                }
                // (The velocity carrying the flux exactly.)
                (rho * RS * t, g_down / rho, rho)
            }
        };
        let residual = wave(down, p_d).0 + v_d;
        let fu = EndFlow { m, energy: m * h0, yf: up.yf, yo: up.yo, p: p_u, un: un_u, rho: rho_u };
        let fd = EndFlow { m: -m, energy: -m * h0, yf: up.yf, yo: up.yo, p: p_d, un: -v_d, rho: rho_d };
        (residual, fu, fd)
    };
    // Bracket the root near last time's if it is still there (which also shows up isn't
    // choked), else over all of it.
    let near = if hint > p_sonic && hint < rest { bracket_near(|p| solve(p).0, hint, p_sonic, rest, false) } else { None };
    let (lo, r_lo, hi, r_hi) = match near {
        Some(x) => x,
        None => {
            let (r_max, fu, fd) = solve(p_sonic);
            if r_max <= 0.0 {
                // Choked: up passes all it can.
                return (fu, fd, p_sonic);
            }
            (p_sonic, r_max, rest, solve(rest).0)
        }
    };
    // (Illinois takes a rising function.)
    // (To 1e-10 of the pressure, as solve_end: ~1e-5 Pa.)
    let (p, (fu, fd)) = illinois(
        |p| {
            let (r, fu, fd) = solve(p);
            (-r, (fu, fd))
        },
        lo,
        -r_lo,
        hi,
        -r_hi,
        1e-10,
    );
    (fu, fd, p)
}



/// A lumped volume of gas at rest, or the atmosphere (fixed, without bounds).
pub struct Node {
    pub volume: f64,
    pub atmosphere: bool,
    /// Mass, internal energy, and fuel and oxygen masses (kg, J).
    pub m: f64,
    pub e: f64,
    pub mf: f64,
    pub mo: f64,
    /// The atmosphere's pressure and temperature.
    pub p_fixed: f64,
    pub t_fixed: f64,
    s0: [f64; 4],
    rate: [f64; 4],
}

impl Node {
    pub fn volume(v: f64, p: f64, t: f64, yf: f64, yo: f64) -> Node {
        let m = p * v / (RS * t);
        Node {
            volume: v,
            atmosphere: false,
            m,
            e: m * energy(t),
            mf: m * yf,
            mo: m * yo,
            p_fixed: 0.0,
            t_fixed: 0.0,
            s0: [0.0; 4],
            rate: [0.0; 4],
        }
    }

    pub fn atmosphere(p: f64, t: f64, yf: f64, yo: f64) -> Node {
        Node {
            volume: 0.0,
            atmosphere: true,
            m: 1.0,
            e: energy(t),
            mf: yf,
            mo: yo,
            p_fixed: p,
            t_fixed: t,
            s0: [0.0; 4],
            rate: [0.0; 4],
        }
    }

    pub fn temperature(&self) -> f64 {
        if self.atmosphere {
            self.t_fixed
        } else {
            temperature(self.e / self.m)
        }
    }

    pub fn pressure(&self) -> f64 {
        if self.atmosphere {
            self.p_fixed
        } else {
            self.m * RS * self.temperature() / self.volume
        }
    }

    pub fn reservoir(&self) -> Reservoir {
        Reservoir { p: self.pressure(), t: self.temperature(), yf: self.mf / self.m, yo: self.mo / self.m }
    }

    fn gain(&mut self, m: f64, energy: f64, yf: f64, yo: f64) {
        self.rate[0] += m;
        self.rate[1] += energy;
        self.rate[2] += m * yf;
        self.rate[3] += m * yo;
    }

    fn state(&self) -> [f64; 4] {
        [self.m, self.e, self.mf, self.mo]
    }

    fn set_state(&mut self, s: [f64; 4]) {
        [self.m, self.e, self.mf, self.mo] = s;
    }
}

/// An open end's radiation into the air outside: a piston's impedance at the end of an
/// unflanged pipe, as a resistance in parallel with the end correction's mass (below).
pub struct Radiation {
    /// The resistance (Pa s/m^3) and its ratio to the mass (1/s).
    pub resistance: f64,
    rate_ql: f64,
    /// The volume flow through the mass (m^3/s), and its saved value and rate.
    ql: f64,
    ql0: f64,
    dql: f64,
    /// Far-field pressure per unit of volume flow through the resistance, at 1 m.
    coefficient: f64,
    /// The radiated pressure at 1 m (Pa), after the last step.
    pub pressure: f64,
}

/// Density and speed of sound of the air the exhaust radiates into.
pub const AIR_DENSITY: f64 = 1.184;
pub const AIR_SOUND_SPEED: f64 = 346.1;

impl Radiation {
    /// For a pipe end of `area`: Levine and Schwinger's end correction (0.6133 radius), its
    /// mass; and the resistance in parallel with it that radiates a monopole's power at low
    /// frequencies (rho omega^2 / 4 pi c per unit of volume flow squared), 1.5 rho c / A,
    /// near the piston's high-frequency limit (rho c / A).
    pub fn new(area: f64) -> Radiation {
        let a = (area / std::f64::consts::PI).sqrt();
        let delta = 0.6133 * a;
        let mass = AIR_DENSITY * delta / area;
        let resistance = 4.0 * (delta / a).powi(2) * AIR_DENSITY * AIR_SOUND_SPEED / area;
        Radiation {
            resistance,
            rate_ql: resistance / mass,
            ql: 0.0,
            ql0: 0.0,
            dql: 0.0,
            // The radiated power R Q_R^2 spread over a sphere: p^2 / rho c = R Q_R^2 / 4 pi.
            coefficient: (AIR_DENSITY * AIR_SOUND_SPEED * resistance / (4.0 * std::f64::consts::PI)).sqrt(),
            pressure: 0.0,
        }
    }
}

/// A pipe's end meeting a node through an opening of effective area `cda` (a valve's,
/// set as it moves; the pipe's own section for a plain connection).
pub struct Link {
    pub pipe: usize,
    pub side: Side,
    pub node: usize,
    pub cda: f64,
    pub radiation: Option<Radiation>,
    /// The last solution: what passed outward of the pipe.
    pub flow: EndFlow,
}

/// What a perforated stretch of a pipe's wall opens onto: a node (a chamber), or another
/// pipe alongside (a chamber's annulus), its position `offset` (m) along it where this pipe
/// is at 0: each cell's holes open into the cell there.
#[derive(Clone, Copy, Debug)]
pub enum Outer {
    Node(usize),
    Pipe(usize, f64),
}

/// A perforated stretch of a pipe's wall (a silencer's core): in each of its cells, the holes'
/// gas moves as a slug, driven by the pressure across them against its jet's loss (the
/// dynamic head at the vena contracta, `discharge`) and viscous drag (Poiseuille's, over the
/// wall's thickness); its mass is the holes' gas over the wall's thickness and their end
/// corrections (0.85 diameters, reduced by the neighbouring holes as Ingard and Fok give,
/// 1 - 1.47 sqrt(porosity) + 0.47 porosity^1.5). The gas leaving a side takes its momentum
/// along the pipe; entering, radially, it brings none.
pub struct Perforation {
    pub pipe: usize,
    pub first: usize,
    pub outer: Outer,
    /// With a pipe outside, the cell each of this one's opens into.
    outer_cells: Vec<usize>,
    /// Hole area per cell (m^2), the slug's length (m), the hole diameter and wall thickness.
    pub hole_area: f64,
    pub length: f64,
    pub hole_diameter: f64,
    pub thickness: f64,
    pub discharge: f64,
    /// Velocity through the holes in each cell (outward), saved and its rate.
    pub v: Vec<f64>,
    v0: Vec<f64>,
    dv: Vec<f64>,
}

impl Perforation {
    /// Holes of `hole_diameter` at `porosity` (open fraction of the wall) through a wall of
    /// `thickness`, over `from` to `to` (m) along `pipe`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(pipes: &[Pipe], pipe_index: usize, from: f64, to: f64, outer: Outer, porosity: f64, hole_diameter: f64, thickness: f64) -> Result<Perforation, String> {
        let pipe = &pipes[pipe_index];
        let first = (from / pipe.dx).round() as usize;
        let last = ((to / pipe.dx).round() as usize).min(pipe.cells());
        let cells = last.saturating_sub(first);
        let mut outer_cells = Vec::new();
        if let Outer::Pipe(q, offset) = outer {
            let o = &pipes[q];
            for i in first..last {
                let x = (i as f64 + 0.5) * pipe.dx + offset;
                if x < 0.0 || x >= o.length {
                    return Err(format!("a perforation reaches past its outer pipe's end ({x:.3} m)"));
                }
                outer_cells.push((x / o.dx) as usize);
            }
        }
        let s = porosity.sqrt();
        let fok = 1.0 - 1.47 * s + 0.47 * s * s * s;
        Ok(Perforation {
            pipe: pipe_index,
            first,
            outer,
            outer_cells,
            hole_area: porosity * std::f64::consts::PI * pipe.cell_diameter(first.min(pipe.cells() - 1)) * pipe.dx,
            length: thickness + 0.85 * hole_diameter * fok,
            hole_diameter,
            thickness,
            discharge: 0.61,
            v: vec![0.0; cells],
            v0: vec![0.0; cells],
            dv: vec![0.0; cells],
        })
    }
}

/// A stretch of a pipe's wall opening, cell by cell, through quasi-steady nozzles into a node:
/// a turbine volute's outflow into its stator channels, each channel fed by its own stretch of
/// the volute (Galindo, Tiseira, Fajardo and García-Cuevas, Energy Convers. Manage. 85, 2014:
/// the quasi-two-dimensional volute, which the volute's averaging over its 360 degrees makes
/// right up to ~2 kHz where a lumped one isn't). Out of the pipe, each cell's static state
/// feeds its nozzle, and the gas takes its momentum along the pipe; into it, the node's.
pub struct Nozzles {
    pub pipe: usize,
    pub first: usize,
    pub node: usize,
    /// Effective area per cell (m^2).
    pub cda: Vec<f64>,
    /// The channels' gas slug length (m), and each cell's flow into the node with its saved
    /// value and rate (the slugs' inertia, as an Orifice's).
    pub length: f64,
    pub m: Vec<f64>,
    m0: Vec<f64>,
    dm: Vec<f64>,
    /// The last total flow into the node (kg/s).
    pub flow: f64,
    /// What the stage passed into the node: mass, energy, fuel and oxygen (kg/s, W).
    gain: [f64; 4],
}

impl Nozzles {
    /// Nozzles from `pipe`'s cells from `first` into `node`, of effective area `cda` per cell
    /// and gas slug length `length`.
    pub fn new(pipe: usize, first: usize, node: usize, cda: Vec<f64>, length: f64) -> Nozzles {
        let n = cda.len();
        Nozzles { pipe, first, node, cda, length, m: vec![0.0; n], m0: vec![0.0; n], dm: vec![0.0; n], flow: 0.0, gain: [0.0; 4] }
    }
}

/// Two nodes joined through an opening (from `a` to `b`, positive), optionally a turbine's
/// nozzle taking `efficiency` of the isentropic work of what passes forward.
pub struct Orifice {
    pub a: usize,
    pub b: usize,
    pub cda: f64,
    /// The length of the gas slug in the opening (its passage and end corrections), whose
    /// inertia the flow has; and that flow (a to b), saved and its rate.
    pub length: f64,
    pub m: f64,
    m0: f64,
    dm: f64,
    pub efficiency: f64,
    /// A turbine's stage: the nodes whose mean pressure is its inlet's (the volutes' feeds),
    /// the work being the stage's isentropic work from there to `b`, taken from the gas
    /// passing (the rotor nozzle of a two-nozzle turbine); none: this opening's own drop.
    pub stage: Vec<usize>,
    /// The last mass flow (kg/s) and the work taken (W).
    pub flow: f64,
    pub power: f64,
}

pub struct Network {
    pub pipes: Vec<Pipe>,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub orifices: Vec<Orifice>,
    pub perforations: Vec<Perforation>,
    pub nozzles: Vec<Nozzles>,
    /// Courant number of each step.
    pub cfl: f64,
    /// Work taken by the orifices over the last call to `step` (J), and substeps taken.
    pub work: Vec<f64>,
    pub substeps: usize,
    /// Substeps rejected (and done again in halves) so far.
    pub rejected: usize,
    debug_last: String,
    /// Each pipe's links, in order, and the pipes split among the threads (src/pool.rs's
    /// `run_each`), each the same every stage: built as the network is first stepped.
    pipe_links: Vec<Vec<usize>>,
    pipe_nozzles: Vec<Vec<usize>>,
    pipe_parts: Vec<Vec<usize>>,
}

/// The pipes (or their nozzles), for tasks each taking some of them (src/pool.rs).
struct Each<T>(*mut T);
unsafe impl<T> Sync for Each<T> {}
unsafe impl<T> Send for Each<T> {}
impl<T> Each<T> {
    /// Item `k`: the caller's alone while it has it (no two tasks given the same `k`).
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self, k: usize) -> &mut T {
        &mut *self.0.add(k)
    }
}

impl Default for Network {
    fn default() -> Self {
        Network {
            pipes: vec![],
            nodes: vec![],
            links: vec![],
            orifices: vec![],
            perforations: vec![],
            nozzles: vec![],
            cfl: 0.6,
            work: vec![],
            substeps: 0,
            rejected: 0,
            debug_last: String::new(),
            pipe_links: vec![],
            pipe_nozzles: vec![],
            pipe_parts: vec![],
        }
    }
}

impl Network {
    pub fn add_pipe(&mut self, p: Pipe) -> usize {
        self.pipes.push(p);
        self.pipes.len() - 1
    }

    pub fn add_node(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    /// Joins a pipe's end to a node through its own section.
    pub fn connect(&mut self, pipe: usize, side: Side, node: usize) -> usize {
        let cda = self.pipes[pipe].end_area(side);
        self.link(pipe, side, node, cda)
    }

    pub fn link(&mut self, pipe: usize, side: Side, node: usize, cda: f64) -> usize {
        self.links.push(Link { pipe, side, node, cda, radiation: None, flow: EndFlow::default() });
        self.links.len() - 1
    }

    /// An open end radiating into the atmosphere `node`.
    pub fn open_end(&mut self, pipe: usize, side: Side, node: usize) -> usize {
        let i = self.connect(pipe, side, node);
        self.links[i].radiation = Some(Radiation::new(self.pipes[pipe].end_area(side)));
        i
    }

    pub fn orifice(&mut self, a: usize, b: usize, cda: f64, efficiency: f64) -> usize {
        // A thin plate's end corrections, both sides of a round hole of the area: 0.85 d.
        let length = 0.85 * (4.0 * cda.max(1e-6) / std::f64::consts::PI).sqrt();
        self.orifices.push(Orifice { a, b, cda, length, m: 0.0, m0: 0.0, dm: 0.0, efficiency, stage: vec![], flow: 0.0, power: 0.0 });
        self.work.push(0.0);
        self.orifices.len() - 1
    }

    /// Largest stable step.
    pub fn stable_dt(&mut self) -> f64 {
        self.index();
        let dts: Vec<std::sync::atomic::AtomicU64> = (0..self.pipes.len()).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
        let pipes = Each(self.pipes.as_mut_ptr());
        crate::pool::run_each(&|t| {
            for &k in &self.pipe_parts[t] {
                let p = unsafe { pipes.get(k) };
                p.update_prim();
                dts[k].store(p.stable_dt().to_bits(), std::sync::atomic::Ordering::Relaxed);
            }
        });
        self.cfl * dts.iter().map(|d| f64::from_bits(d.load(std::sync::atomic::Ordering::Relaxed))).fold(f64::INFINITY, f64::min)
    }

    /// Each pipe's links and the pipes' split among the threads (again if pipes or links were
    /// added).
    fn index(&mut self) {
        if self.pipe_parts.len() == crate::pool::threads()
            && self.pipe_parts.iter().map(Vec::len).sum::<usize>() == self.pipes.len()
            && self.pipe_links.iter().map(Vec::len).sum::<usize>() == self.links.len()
            && self.pipe_nozzles.iter().map(Vec::len).sum::<usize>() == self.nozzles.len()
        {
            return;
        }
        self.pipe_links = vec![vec![]; self.pipes.len()];
        for (i, l) in self.links.iter().enumerate() {
            self.pipe_links[l.pipe].push(i);
        }
        self.pipe_nozzles = vec![vec![]; self.pipes.len()];
        for (i, z) in self.nozzles.iter().enumerate() {
            self.pipe_nozzles[z.pipe].push(i);
        }
        // Longest processing time first: each pipe, costliest first, to the thread with least
        // so far. A pipe's stage costs about its cells, as much as 25 cells' its ends', and as
        // much as one each its nozzles' channels.
        let cost = |k: usize| self.pipes[k].cells() + 25 + self.pipe_nozzles[k].iter().map(|&z| self.nozzles[z].cda.len()).sum::<usize>();
        let mut order: Vec<usize> = (0..self.pipes.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(cost(k)));
        let threads = crate::pool::threads();
        let mut parts = vec![vec![]; threads];
        let mut load = vec![0; threads];
        for k in order {
            let t = (0..threads).min_by_key(|&t| load[t]).unwrap();
            parts[t].push(k);
            load[t] += cost(k);
        }
        self.pipe_parts = parts;
    }

    /// Advances by `dt`, in as many substeps as stability takes.
    pub fn step(&mut self, dt: f64) {
        let n = (dt / self.stable_dt()).ceil().max(1.0) as usize;
        self.work.fill(0.0);
        let debug = std::env::var("DUCTCHECK").is_ok();
        if debug {
            if let Some(w) = self.check() {
                panic!("on entry: {w}\nbefore:\n{}", self.debug_last);
            }
            self.debug_last = format!("on entry:\n{}", self.snapshot());
        }
        for _ in 0..n {
            self.checked_substep(dt / n as f64, 0);
            if debug {
                if let Some(w) = self.check() {
                    panic!("{w}\nbefore:\n{}", self.debug_last);
                }
                self.debug_last = self.snapshot();
            }
        }
        self.substeps = n;
    }

    /// A substep, rejected where it would leave a gas without mass or internal energy (a
    /// strong rarefaction outrunning the high-order scheme's positivity) and done again in
    /// halves; at the last, with first-order fluxes, which HLLC keeps positive.
    fn checked_substep(&mut self, dt: f64, depth: u32) {
        let work = self.work.clone();
        let hints: Vec<Vec<f64>> = self.pipes.iter().map(|p| p.step_flow.clone()).collect();
        if self.substep(dt) {
            return;
        }
        // Back to the start of the substep (substep saved it).
        for (p, h) in self.pipes.iter_mut().zip(hints) {
            p.q.clone_from(&p.q0);
            p.step_flow = h;
        }
        for n in &mut self.nodes {
            if !n.atmosphere {
                n.set_state(n.s0);
            }
        }
        for l in &mut self.links {
            if let Some(r) = &mut l.radiation {
                r.ql = r.ql0;
            }
        }
        for f in &mut self.perforations {
            f.v.clone_from(&f.v0);
        }
        for o in &mut self.orifices {
            o.m = o.m0;
        }
        for z in &mut self.nozzles {
            z.m.clone_from(&z.m0);
        }
        self.work = work;
        self.rejected += 1;
        if depth < 4 {
            self.checked_substep(0.5 * dt, depth + 1);
            self.checked_substep(0.5 * dt, depth + 1);
        } else {
            for p in &mut self.pipes {
                p.first_order = true;
            }
            let _ = self.substep(dt);
            for p in &mut self.pipes {
                p.first_order = false;
            }
        }
    }

    fn substep(&mut self, dt: f64) -> bool {
        for p in &mut self.pipes {
            p.q0.clone_from(&p.q);
        }
        for n in &mut self.nodes {
            n.s0 = n.state();
        }
        for l in &mut self.links {
            if let Some(r) = &mut l.radiation {
                r.ql0 = r.ql;
            }
        }
        for f in &mut self.perforations {
            f.v0.clone_from(&f.v);
        }
        for o in &mut self.orifices {
            o.m0 = o.m;
        }
        for z in &mut self.nozzles {
            z.m0.clone_from(&z.m);
        }
        // Shu and Osher's third-order strong-stability-preserving Runge-Kutta: the stages'
        // rates weigh 1/6, 1/6 and 2/3 in the step.
        // A stage leaving a gas without mass or internal energy stops it (checked_substep
        // does it again in halves).
        for (weight, keep) in [(1.0 / 6.0, 0.0), (1.0 / 6.0, 0.75), (2.0 / 3.0, 1.0 / 3.0)] {
            self.rates(dt, weight);
            if !self.advance(dt, keep) {
                return false;
            }
        }
        for l in &mut self.links {
            if let Some(r) = &mut l.radiation {
                r.pressure = r.coefficient * (l.flow.un * self.pipes[l.pipe].end_area(l.side) - r.ql);
            }
        }
        true
    }

    /// The rates of change of everything; `weight` of the stage's work and flows is kept.
    fn rates(&mut self, dt: f64, weight: f64) {
        for n in &mut self.nodes {
            n.rate = [0.0; 4];
        }
        self.index();
        // Each pipe on its own (in parallel): its cells' state, its ends against the nodes
        // they meet (which stand still through the stage), its cells' rates.
        {
            let (links, nodes, pipe_links, pipe_nozzles) = (&self.links, &self.nodes, &self.pipe_links, &self.pipe_nozzles);
            let nozzles = Each(self.nozzles.as_mut_ptr());
            let stage = |k: usize, pipe: &mut Pipe| {
                pipe.update_prim();
                pipe.pad();
                for &i in &pipe_links[k] {
                    let l = &links[i];
                    let s = pipe.end(l.side);
                    let area = pipe.end_area(l.side);
                    let mut res = nodes[l.node].reservoir();
                    let mut z = 0.0;
                    if let Some(r) = &l.radiation {
                        res.p -= r.resistance * r.ql;
                        z = r.resistance * area;
                    }
                    let side = (l.side == Side::B) as usize;
                    let (f, p_end) = solve_end(&s, area, l.cda, &res, z, pipe.end_p[side]);
                    pipe.end_p[side] = p_end;
                    if debug_checks() && !(f.m.is_finite() && f.energy.is_finite() && f.p.is_finite() && f.un.is_finite()) {
                        panic!("link pipe {} {:?} node {}: end {:?} reservoir {:?} area {area} cda {} z {z} -> {:?}", l.pipe, l.side, l.node, s, res, l.cda, f);
                    }
                    pipe.set_end(l.side, &f);
                    pipe.end_flow[(l.side == Side::B) as usize] = f;
                }
                pipe.rates();
                // Its volute's stator channels (into nodes that stand still through the stage).
                for &z in &pipe_nozzles[k] {
                    let z = unsafe { nozzles.get(z) };
                    discharge(z, pipe, nodes[z.node].reservoir());
                }
            };
            let pipes = Each(self.pipes.as_mut_ptr());
            let parts = &self.pipe_parts;
            crate::pool::run_each(&|t| {
                // Each pipe in one thread's part alone.
                for &k in &parts[t] {
                    stage(k, unsafe { pipes.get(k) });
                }
            });
        }
        // What the ends passed, into their nodes in order.
        for l in &mut self.links {
            let pipe = &self.pipes[l.pipe];
            let f = pipe.end_flow[(l.side == Side::B) as usize];
            let node = &mut self.nodes[l.node];
            if !node.atmosphere {
                node.gain(f.m, f.energy, f.yf, f.yo);
            }
            if let Some(r) = &mut l.radiation {
                r.dql = r.rate_ql * (f.un * pipe.end_area(l.side) - r.ql);
            }
            l.flow = f;
        }
        for (o, work) in self.orifices.iter_mut().zip(&mut self.work) {
            let (a, b) = (self.nodes[o.a].reservoir(), self.nodes[o.b].reservoir());
            // The flow has the slug's inertia (`slug_rate`): smooth through no flow, where the
            // quasi-steady flow (~ sqrt dp, of infinite conductance) chatters between small
            // volumes, and the compressible quasi-steady flow once steady.
            // At most what the opening chokes at (none, shut): a valve closing on its flow
            // stops its gas as fast.
            o.m = o.m.clamp(-o.cda * choked(b.p, b.t, gamma(b.t)), o.cda * choked(a.p, a.t, gamma(a.t)));
            let m = o.m;
            let forward = m >= 0.0;
            let (src, dst) = if forward { (a, b) } else { (b, a) };
            o.dm = if o.cda > 0.0 {
                let gas = |r: &Reservoir| (r.p, r.t, gamma(r.t), r.p / (RS * r.t));
                slug_rate(o.cda, o.length, m, gas(&a), gas(&b))
            } else {
                0.0
            };
            let flow = m.abs();
            let h0 = enthalpy(src.t);
            let mut w = 0.0;
            if forward && o.efficiency > 0.0 && flow > 0.0 {
                let g = gamma(src.t);
                let p_in = if o.stage.is_empty() {
                    src.p
                } else {
                    o.stage.iter().map(|&n| self.nodes[n].pressure()).sum::<f64>() / o.stage.len() as f64
                };
                let ts = src.t * (dst.p / p_in.max(dst.p)).powf((g - 1.0) / g);
                w = o.efficiency * (h0 - enthalpy(ts));
            }
            let (from, to) = if forward { (o.a, o.b) } else { (o.b, o.a) };
            if !self.nodes[from].atmosphere {
                self.nodes[from].gain(-flow, -flow * h0, src.yf, src.yo);
            }
            if !self.nodes[to].atmosphere {
                self.nodes[to].gain(flow, flow * (h0 - w), src.yf, src.yo);
            }
            if debug_checks() && !(m.is_finite() && w.is_finite()) {
                panic!("orifice {} -> {}: {:?} {:?} cda {} m {m} w {w}", o.a, o.b, a, b, o.cda);
            }
            o.flow = m;
            o.power = flow * w;
            *work += weight * dt * o.power;
        }
        for k in 0..self.perforations.len() {
            self.perforate(k);
        }
        for z in &self.nozzles {
            let node = &mut self.nodes[z.node];
            let [m, e, f, o] = z.gain;
            if !node.atmosphere && m != 0.0 {
                node.gain(m, e, f / m, o / m);
            }
        }
        if debug_checks() {
            for (k, p) in self.pipes.iter().enumerate() {
                if let Some(i) = p.rate.iter().position(|r| r.iter().any(|v| !v.is_finite())) {
                    let lo = i.saturating_sub(2);
                    panic!("pipe {k} rate at cell {i}: {:?}; cells {:?}; ends {:?}", p.rate[i], &p.prim[lo..(i + 3).min(p.cells())], p.face);
                }
            }
        }
    }

    /// A perforation's flows: their rates added to its pipe's cells and what they open onto.
    fn perforate(&mut self, k: usize) {
        let f = &mut self.perforations[k];
        for c in 0..f.v.len() {
            let i = f.first + c;
            let core = self.pipes[f.pipe].prim[i];
            // The other side: (pressure, density, temperature, axial velocity, fuel, oxygen).
            let other = match f.outer {
                Outer::Node(n) => {
                    let r = self.nodes[n].reservoir();
                    (r.p, r.p / (RS * r.t), r.t, 0.0, r.yf, r.yo)
                }
                Outer::Pipe(q, _) => {
                    let o = self.pipes[q].prim[f.outer_cells[c]];
                    (o.p, o.rho, o.t, o.u, o.yf, o.yo)
                }
            };
            let v = f.v[c];
            let out = v >= 0.0;
            let rho_up = if out { core.rho } else { other.1 };
            let rho_mean = 0.5 * (core.rho + other.1);
            let mu = viscosity(0.5 * (core.t + other.2));
            let viscous = 32.0 * mu * f.thickness / (f.hole_diameter * f.hole_diameter);
            let jet = 0.5 * rho_up * v * v.abs() / (f.discharge * f.discharge);
            f.dv[c] = (core.p - other.0 - jet - viscous * v) / (rho_mean * f.length);
            let m = rho_up * v * f.hole_area;
            // The upstream side's total enthalpy, composition and momentum along its pipe.
            let (h0, yf, yo, u_up) = if out {
                (core.e + RS * core.t + 0.5 * core.u * core.u, core.yf, core.yo, core.u)
            } else {
                (enthalpy(other.2) + 0.5 * other.3 * other.3, other.4, other.5, other.3)
            };
            let pipe = &mut self.pipes[f.pipe];
            let inv_v = 1.0 / pipe.cell_volume(i);
            let r = &mut pipe.rate[i];
            r[0] -= m * inv_v;
            r[2] -= m * h0 * inv_v;
            r[3] -= m * yf * inv_v;
            r[4] -= m * yo * inv_v;
            if out {
                r[1] -= m * u_up * inv_v;
            }
            match f.outer {
                Outer::Node(n) => {
                    if !self.nodes[n].atmosphere {
                        self.nodes[n].gain(m, m * h0, yf, yo);
                    }
                }
                Outer::Pipe(q, _) => {
                    let pipe = &mut self.pipes[q];
                    let inv_v = 1.0 / pipe.cell_volume(f.outer_cells[c]);
                    let r = &mut pipe.rate[f.outer_cells[c]];
                    r[0] += m * inv_v;
                    r[2] += m * h0 * inv_v;
                    r[3] += m * yf * inv_v;
                    r[4] += m * yo * inv_v;
                    if !out {
                        r[1] += m * u_up * inv_v;
                    }
                }
            }
        }
    }

    /// From the saved state: `keep` of it, and the rest the current state stepped by `dt`.
    /// Advances a stage; false if it left a gas without mass or internal energy.
    fn advance(&mut self, dt: f64, keep: f64) -> bool {
        let step = 1.0 - keep;
        let pipe = |p: &mut Pipe| -> bool {
            for i in 0..p.q.len() {
                for k in 0..NQ {
                    p.q[i][k] = keep * p.q0[i][k] + step * (p.q[i][k] + dt * p.rate[i][k]);
                }
            }
            p.q.iter().all(|q| q[0] > 0.0 && q[2] - 0.5 * q[1] * q[1] / q[0] > 0.0 && q[2].is_finite())
        };
        let bad = std::sync::atomic::AtomicBool::new(false);
        let pipes = Each(self.pipes.as_mut_ptr());
        crate::pool::run_each(&|t| {
            for &k in &self.pipe_parts[t] {
                if !pipe(unsafe { pipes.get(k) }) {
                    bad.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
        let cells = !bad.into_inner();
        for n in &mut self.nodes {
            if n.atmosphere {
                continue;
            }
            let s = n.state();
            let mut out = [0.0; 4];
            for k in 0..4 {
                out[k] = keep * n.s0[k] + step * (s[k] + dt * n.rate[k]);
            }
            n.set_state(out);
        }
        for l in &mut self.links {
            if let Some(r) = &mut l.radiation {
                r.ql = keep * r.ql0 + step * (r.ql + dt * r.dql);
            }
        }
        for f in &mut self.perforations {
            for i in 0..f.v.len() {
                f.v[i] = keep * f.v0[i] + step * (f.v[i] + dt * f.dv[i]);
            }
        }
        for o in &mut self.orifices {
            o.m = keep * o.m0 + step * (o.m + dt * o.dm);
        }
        for z in &mut self.nozzles {
            for i in 0..z.m.len() {
                z.m[i] = keep * z.m0[i] + step * (z.m[i] + dt * z.dm[i]);
            }
        }
        cells && self.nodes.iter().all(|n| n.atmosphere || (n.m > 0.0 && n.e > 0.0))
    }

    /// The nodes' and the links' states (for debugging).
    pub fn snapshot(&self) -> String {
        let mut out = String::new();
        for (k, n) in self.nodes.iter().enumerate().filter(|(_, n)| !n.atmosphere) {
            out += &format!("node {k}: V {:.2e} p {:.0} T {:.0}\n", n.volume, n.pressure(), n.temperature());
        }
        for l in &self.links {
            let p = &self.pipes[l.pipe];
            let c = match l.side { Side::A => p.prim[0], Side::B => p.prim[p.cells() - 1] };
            out += &format!("link pipe {} {:?} node {} cda {:.2e}: cell p {:.0} u {:.1} T {:.0}; flow m {:.4} p {:.0} un {:.1}\n", l.pipe, l.side, l.node, l.cda, c.p, c.u, c.t, l.flow.m, l.flow.p, l.flow.un);
        }
        for o in &self.orifices {
            out += &format!("orifice {}->{} cda {:.2e} flow {:.4}\n", o.a, o.b, o.cda, o.flow);
        }
        out
    }

    /// Every pipe and node whose state isn't finite (for debugging).
    pub fn check(&self) -> Option<String> {
        let mut out = Vec::new();
        for (k, n) in self.nodes.iter().enumerate() {
            if !(n.m.is_finite() && n.e.is_finite()) {
                out.push(format!("node {k}"));
            }
        }
        for (k, p) in self.pipes.iter().enumerate() {
            let bad: Vec<usize> = (0..p.cells()).filter(|&i| p.q[i].iter().any(|v| !v.is_finite())).collect();
            if !bad.is_empty() {
                out.push(format!("pipe {k} cells {:?} of {}", &bad[..bad.len().min(8)], p.cells()));
            }
        }
        (!out.is_empty()).then(|| out.join("; "))
    }

    /// Mass and energy in the pipes and volumes.
    pub fn totals(&self) -> (f64, f64) {
        let mut t = (0.0, 0.0);
        for p in &self.pipes {
            let (m, e) = p.totals();
            t.0 += m;
            t.1 += e;
        }
        for n in self.nodes.iter().filter(|n| !n.atmosphere) {
            t.0 += n.m;
            t.1 += n.e;
        }
        t
    }
}

/// An engine's exhaust as a network (spec::DuctSpec), with what the engine sets and reads.
pub struct Ducts {
    pub net: Network,
    /// Per cylinder: its node (its gas, set from the engine's each step) and its valve's link.
    pub cylinders: Vec<(usize, usize)>,
    /// Turbines' nozzles, and their wastegates with their fully open areas.
    pub turbines: Vec<usize>,
    pub wastegates: Vec<(usize, f64)>,
    /// Flaps: their links, full areas and schedules (crank speed, pedal), and how far open.
    pub flaps: Vec<(usize, f64, [f64; 2], [f64; 2])>,
    pub flap: f64,
    /// Open ends, and the listener's distance from them.
    pub outlets: Vec<usize>,
    pub distance: f64,
    /// The nodes by name.
    pub names: std::collections::HashMap<String, usize>,
    /// The pressure at the listener after each step (Pa), kept while `listening`; and each
    /// open end's outward velocity, density and temperature (for the jets' noise).
    pub sound: Vec<f32>,
    pub exits: Vec<f32>,
    /// Each open end's share of `sound`, the ends interleaved, kept while `listening_ends`.
    pub end_sound: Vec<f32>,
    pub listening: bool,
    pub listening_ends: bool,
}

/// The atmosphere: 25 C, engine-sim's air (a quarter oxygen by moles).
pub const AMBIENT: f64 = 298.15;
pub const AIR_OXYGEN: f64 = 0.25;
/// Cells to the shortest wavelength resolved (numerical damping ~0.02 dB/m: tests/duct.rs).
pub const CELLS_PER_WAVELENGTH: f64 = 30.0;

impl Ducts {
    pub fn new(spec: &crate::spec::DuctSpec, cylinders: usize) -> Result<Ducts, String> {
        let atm = crate::units::ATM;
        let mut net = Network::default();
        let mut names: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let air = net.add_node(Node::atmosphere(atm, AMBIENT, 0.0, AIR_OXYGEN));
        names.insert("atmosphere".into(), air);
        let mut cylinder_nodes = Vec::new();
        for i in 0..cylinders {
            let n = net.add_node(Node::volume(1e-4, atm, AMBIENT, 0.0, AIR_OXYGEN));
            names.insert(format!("cylinder {}", i + 1), n);
            cylinder_nodes.push(n);
        }
        for v in &spec.volumes {
            let n = net.add_node(Node::volume(v.volume, atm, v.temperature, 0.0, AIR_OXYGEN));
            if names.insert(v.node.clone(), n).is_some() {
                return Err(format!("node {} twice", v.node));
            }
        }
        let node = |name: &str| names.get(name).copied().ok_or_else(|| format!("no node {name}"));
        let mut valves = vec![None; cylinders];
        let mut outlets = Vec::new();
        let mut pipe_names: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut end_links: std::collections::HashMap<(usize, bool), usize> = std::collections::HashMap::new();
        for p in &spec.pipes {
            let c = (gamma(p.temperature) * RS * p.temperature).sqrt();
            let dx = c / (spec.resolution * CELLS_PER_WAVELENGTH);
            let section = |s: &crate::spec::DuctSectionSpec| Section {
                length: s.length,
                diameter: s.diameter,
                diameter_b: s.diameter_b.unwrap_or(s.diameter),
                area: s.area,
                square: s.square,
                wall: s.wall_temperature.or(p.wall_temperature),
                resistivity: s.flow_resistivity,
                roughness: s.roughness.or(p.roughness).unwrap_or(STEEL_ROUGHNESS),
            };
            let sections: Vec<Section> = if p.sections.is_empty() {
                vec![section(&crate::spec::DuctSectionSpec {
                    length: p.length,
                    diameter: p.diameter,
                    diameter_b: p.diameter_b,
                    area: p.area,
                    square: p.square,
                    wall_temperature: p.wall_temperature,
                    flow_resistivity: p.flow_resistivity,
                    roughness: p.roughness,
                })]
            } else {
                p.sections.iter().map(section).collect()
            };
            if let Some(s) = sections.iter().find(|s| s.length < dx) {
                return Err(format!("a section of {} m is shorter than a cell ({dx:.4} m)", s.length));
            }
            let pipe = Pipe::with_sections(&sections, dx, atm, p.temperature, 0.0, AIR_OXYGEN);
            let i = net.add_pipe(pipe);
            if let Some(name) = &p.name {
                if pipe_names.insert(name.clone(), i).is_some() {
                    return Err(format!("pipe {name} twice"));
                }
            }
            for (end, side) in [(&p.a, Side::A), (&p.b, Side::B)] {
                let l = if end == "closed" {
                    net.link(i, side, air, 0.0)
                } else {
                    let n = node(end)?;
                    if n == air {
                        let l = net.open_end(i, side, n);
                        outlets.push(l);
                        l
                    } else if let Some(k) = cylinder_nodes.iter().position(|&c| c == n) {
                        if valves[k].is_some() {
                            return Err(format!("{end} has two pipes"));
                        }
                        let l = net.link(i, side, n, 0.0);
                        valves[k] = Some(l);
                        l
                    } else {
                        net.connect(i, side, n)
                    }
                };
                end_links.insert((i, side == Side::B), l);
            }
        }
        for f in &spec.perforations {
            let pipe = *pipe_names.get(&f.pipe).ok_or_else(|| format!("no pipe {}", f.pipe))?;
            let outer = if let Some(&q) = pipe_names.get(&f.outer) {
                Outer::Pipe(q, f.outer_from - f.from)
            } else {
                Outer::Node(node(&f.outer)?)
            };
            let perf = Perforation::new(&net.pipes, pipe, f.from, f.to, outer, f.porosity, f.hole_diameter, f.thickness)?;
            net.perforations.push(perf);
        }
        for z in &spec.nozzles {
            let pipe = *pipe_names.get(&z.pipe).ok_or_else(|| format!("no pipe {}", z.pipe))?;
            let p = &net.pipes[pipe];
            let first = (z.from / p.dx).round() as usize;
            let last = ((z.to / p.dx).round() as usize).min(p.cells());
            if last <= first {
                return Err(format!("nozzles on {}: no cells", z.pipe));
            }
            let n = (last - first) as f64;
            let cda = vec![z.area / n; last - first];
            net.nozzles.push(Nozzles::new(pipe, first, node(&z.node)?, cda, z.length));
        }
        let mut flaps = Vec::new();
        for f in &spec.flaps {
            let pipe = *pipe_names.get(&f.pipe).ok_or_else(|| format!("no pipe {}", f.pipe))?;
            let b = match f.end.as_str() {
                "a" => false,
                "b" => true,
                e => return Err(format!("flap end {e}")),
            };
            let l = end_links[&(pipe, b)];
            let area = net.links[l].cda;
            net.links[l].cda = 0.0;
            flaps.push((l, area, f.speed, f.throttle));
        }
        for o in &spec.orifices {
            net.orifice(node(&o.a)?, node(&o.b)?, o.area * o.discharge, 0.0);
        }
        let (mut turbines, mut wastegates) = (Vec::new(), Vec::new());
        for t in &spec.turbines {
            let (a, b) = (node(&t.a)?, node(&t.b)?);
            // The efficiency is the turbo's, set by the engine.
            let o = net.orifice(a, b, t.area, 1.0);
            if let Some(l) = t.length {
                net.orifices[o].length = l;
            }
            net.orifices[o].stage = t.stage.iter().map(|n| node(n)).collect::<Result<_, _>>()?;
            turbines.push(o);
            if t.wastegate_from.is_empty() {
                wastegates.push((net.orifice(a, b, 0.0, 0.0), t.wastegate));
            } else {
                let share = t.wastegate / t.wastegate_from.len() as f64;
                for n in &t.wastegate_from {
                    wastegates.push((net.orifice(node(n)?, b, 0.0, 0.0), share));
                }
            }
        }
        let cylinders = cylinder_nodes
            .iter()
            .zip(&valves)
            .enumerate()
            .map(|(k, (&n, v))| v.map(|v| (n, v)).ok_or_else(|| format!("cylinder {} has no pipe", k + 1)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Ducts {
            net,
            cylinders,
            turbines,
            wastegates,
            flaps,
            flap: 0.0,
            outlets,
            distance: spec.listener_distance,
            names,
            sound: Vec::new(),
            exits: Vec::new(),
            end_sound: Vec::new(),
            listening_ends: false,
            listening: true,
        })
    }

    /// Moves the flaps over `dt` towards how far the ECU's logic opens them at crank speed
    /// `speed` (rad/s) and pedal `pedal`: whichever of the two opens them more, each
    /// smoothly between its thresholds, the actuators taking ~0.15 s.
    pub fn set_flaps(&mut self, speed: f64, pedal: f64, dt: f64) {
        let Some(&(_, _, s, t)) = self.flaps.first() else { return };
        let step = |x: f64, [from, at]: [f64; 2]| {
            let x = ((x - from) / (at - from).max(1e-9)).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let open = step(speed, s).max(step(pedal, t));
        self.flap += (open - self.flap) * (dt / 0.15).min(1.0);
        for &(l, area, _, _) in &self.flaps {
            self.net.links[l].cda = self.flap * area;
        }
    }

    /// Records the pressure at the listener: each open end's far field, summed.
    pub fn listen(&mut self) {
        if !self.listening {
            return;
        }
        let p: f64 = self.outlets.iter().filter_map(|&l| self.net.links[l].radiation.as_ref()).map(|r| r.pressure).sum();
        self.sound.push((p / self.distance) as f32);
        for &l in &self.outlets {
            let f = &self.net.links[l].flow;
            self.exits.extend([f.un as f32, f.rho as f32, (f.p / (f.rho * RS)) as f32]);
            if self.listening_ends {
                let p = self.net.links[l].radiation.as_ref().map_or(0.0, |r| r.pressure);
                self.end_sound.push((p / self.distance) as f32);
            }
        }
    }
}
