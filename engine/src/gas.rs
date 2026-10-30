//! A volume of ideal gas with bulk momentum, and flow between two of them
//! (engine-sim's gas_system.h/.cpp).
//!
//! engine-sim's gas has a constant heat capacity ratio (1.4, five degrees of freedom), and
//! what flows between two volumes carries only its internal energy. A system set `real`
//! (opt-in per engine) is closer to an engine's gases instead:
//!
//! - its heat capacity ratio falls with temperature, linearly, as fitted to equilibrium
//!   burned gas by Gatowski et al. (SAE 841359, 1984; values from Klein, "Single-zone
//!   cylinder pressure modeling", Linköping 2004): gamma(T) = 1.3678 - 8.13e-5 (T - 300).
//!   Its internal energy is the integral of cv = R / (gamma - 1), so energy is conserved
//!   exactly and pressure stays n R T / V;
//! - what flows carries its enthalpy, internal energy plus flow work (R T per mole): a
//!   cylinder blowing down cools as its gas pushes the rest out, and a filling one warms.

use crate::fastmath;
use crate::units::{AIR_MOLECULAR_MASS, R};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mix {
    pub p_fuel: f64,
    pub p_inert: f64,
    pub p_o2: f64,
}

impl Default for Mix {
    fn default() -> Self {
        Mix { p_fuel: 0.0, p_inert: 1.0, p_o2: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct State {
    n_mol: f64,
    e_k: f64,
    v: f64,
    momentum: [f64; 2],
    mix: Mix,
}

#[derive(Clone, Debug)]
pub struct GasSystem {
    state: State,
    degrees_of_freedom: i32,
    hcr: f64,
    choked_flow_limit: f64,
    choked_flow_factor_cached: f64,
    width: f64,
    height: f64,
    dx: f64,
    dy: f64,
    /// Real-gas properties (above) instead of engine-sim's.
    real: bool,
    /// The real gas's properties, for the state they were worked out from (see Derived).
    derived: Cell<Derived>,
}

/// What a real gas's state implies, worked out once per change of state: the temperature
/// takes an exp, and the pressure, heat capacity ratio and speed of sound follow from it.
/// Each is wanted several times between changes (both ends of a flow, the four faces in
/// update_velocity, the choked test, the speed limits).
#[derive(Clone, Copy, Debug)]
struct Derived {
    /// The internal energy, moles and volume these are for.
    e: f64,
    n: f64,
    v: f64,
    t: f64,
    hcr: f64,
    p: f64,
    /// Speed of sound squared, hcr R T / M (= hcr P / rho).
    c_squared: f64,
}

impl Derived {
    const STALE: Derived =
        Derived { e: f64::NAN, n: f64::NAN, v: f64::NAN, t: 0.0, hcr: 0.0, p: 0.0, c_squared: 0.0 };

    fn of(e: f64, n: f64, v: f64) -> Derived {
        let t = if n == 0.0 { 0.0 } else { real_temperature(e / n) };
        let hcr = real_gamma(t);
        Derived {
            e,
            n,
            v,
            t,
            hcr,
            p: if v != 0.0 { n * R * t / v } else { 0.0 },
            c_squared: hcr * t * (R / AIR_MOLECULAR_MASS),
        }
    }
}

/// What dynamic_pressure works out regardless of direction.
#[derive(Clone, Copy)]
struct DynamicTerms {
    inverse_mass: f64,
    static_pressure: f64,
    c_squared: f64,
    hcr: f64,
}

/// gamma(T) = GAMMA_0 - GAMMA_SLOPE T (Gatowski's line, extended to 0 K).
pub const GAMMA_SLOPE: f64 = 8.13e-5;
pub const GAMMA_0: f64 = 1.3678 + GAMMA_SLOPE * 300.0;

/// Internal energy per mole of the real gas at `t`: the integral of R / (gamma(T) - 1) dT.
#[inline]
pub fn real_energy_per_mol(t: f64) -> f64 {
    -(R / GAMMA_SLOPE) * fastmath::ln(1.0 - GAMMA_SLOPE * t / (GAMMA_0 - 1.0))
}

/// Temperature of the real gas with internal energy `u` per mole (real_energy_per_mol's
/// inverse).
#[inline]
pub fn real_temperature_exact(u: f64) -> f64 {
    (GAMMA_0 - 1.0) / GAMMA_SLOPE * (1.0 - fastmath::exp(-GAMMA_SLOPE * u / R))
}

/// real_temperature_exact, tabulated: every change of a gas's state needs its temperature,
/// and the exp was a large share of the gas dynamics. Linear interpolation over 0 to
/// 200 kJ/mol (about 4300 K) is within 2e-4 K; beyond, the formula.
const TEMPERATURE_U_MAX: f64 = 200e3;
const TEMPERATURE_STEPS: usize = 4096;
fn temperature_table() -> &'static [f64; TEMPERATURE_STEPS + 1] {
    static TABLE: std::sync::OnceLock<[f64; TEMPERATURE_STEPS + 1]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0; TEMPERATURE_STEPS + 1];
        for (i, e) in t.iter_mut().enumerate() {
            *e = real_temperature_exact(TEMPERATURE_U_MAX * i as f64 / TEMPERATURE_STEPS as f64);
        }
        t
    })
}

#[inline]
pub fn real_temperature(u: f64) -> f64 {
    let x = u * (TEMPERATURE_STEPS as f64 / TEMPERATURE_U_MAX);
    if (0.0..TEMPERATURE_STEPS as f64).contains(&x) {
        let t = temperature_table();
        let i = x as usize;
        return t[i] + (x - i as f64) * (t[i + 1] - t[i]);
    }
    real_temperature_exact(u)
}

/// The subsonic flow function, sqrt(2 g / (g - 1) s (s - r)) with s = r^(1/g), the mass flow
/// through an orifice per p0 / sqrt(R T0) at pressure ratio r, divided by sqrt(1 - r): what
/// is left is smooth up to r = 1 (where it is sqrt 2), so it tabulates well over the real
/// gas's ratios. Bilinear over 64 x 128 cells: within 4e-6 of the exact flow. Replaces a pow
/// per flow.
const SUBSONIC_G_MIN: f64 = 1.05;
const SUBSONIC_G_MAX: f64 = 1.42;
const SUBSONIC_G_STEPS: usize = 64;
/// Below every choked ratio of the range (0.525 at 1.42).
const SUBSONIC_R_MIN: f64 = 0.5;
const SUBSONIC_R_STEPS: usize = 128;
type SubsonicTable = [[f32; SUBSONIC_R_STEPS + 1]; SUBSONIC_G_STEPS + 1];

fn subsonic_exact(r: f64, g: f64) -> f64 {
    if r >= 1.0 {
        return std::f64::consts::SQRT_2;
    }
    let s = r.powf(1.0 / g);
    (2.0 * g / (g - 1.0) * s * (s - r) / (1.0 - r)).max(0.0).sqrt()
}

fn subsonic_table() -> &'static SubsonicTable {
    static TABLE: std::sync::OnceLock<SubsonicTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0.0f32; SUBSONIC_R_STEPS + 1]; SUBSONIC_G_STEPS + 1];
        for (i, row) in t.iter_mut().enumerate() {
            let g = SUBSONIC_G_MIN + (SUBSONIC_G_MAX - SUBSONIC_G_MIN) * i as f64 / SUBSONIC_G_STEPS as f64;
            for (j, e) in row.iter_mut().enumerate() {
                let r = SUBSONIC_R_MIN + (1.0 - SUBSONIC_R_MIN) * j as f64 / SUBSONIC_R_STEPS as f64;
                *e = subsonic_exact(r, g) as f32;
            }
        }
        t
    })
}

#[inline]
fn subsonic(r: f64, g: f64) -> f64 {
    if r < SUBSONIC_R_MIN {
        return subsonic_exact(r, g);
    }
    let t = subsonic_table();
    let x = ((g - SUBSONIC_G_MIN) * (SUBSONIC_G_STEPS as f64 / (SUBSONIC_G_MAX - SUBSONIC_G_MIN)))
        .clamp(0.0, SUBSONIC_G_STEPS as f64 - 1e-9);
    let y = ((r - SUBSONIC_R_MIN) * (SUBSONIC_R_STEPS as f64 / (1.0 - SUBSONIC_R_MIN)))
        .min(SUBSONIC_R_STEPS as f64 - 1e-9);
    let (i, j) = (x as usize, y as usize);
    let (fx, fy) = (x - i as f64, y - j as f64);
    let (a0, a1) = (t[i][j] as f64, t[i][j + 1] as f64);
    let (b0, b1) = (t[i + 1][j] as f64, t[i + 1][j + 1] as f64);
    let a = a0 + fy * (a1 - a0);
    let b = b0 + fy * (b1 - b0);
    a + fx * (b - a)
}

/// The real gas's choked-flow constants by heat capacity ratio: the pressure ratio below
/// which flow chokes, (2 / (g + 1))^(g / (g - 1)), and the choked flow factor,
/// sqrt(g) (2 / (g + 1))^((g + 1) / (2 (g - 1))), tabulated over the ratios the real gas
/// takes (smooth: linear interpolation is within ~1e-8).
const CHOKED_MIN: f64 = 1.05;
const CHOKED_MAX: f64 = 1.42;
const CHOKED_STEPS: usize = 1024;
fn choked_table() -> &'static [(f64, f64); CHOKED_STEPS + 1] {
    static TABLE: std::sync::OnceLock<[(f64, f64); CHOKED_STEPS + 1]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [(0.0, 0.0); CHOKED_STEPS + 1];
        for (i, e) in t.iter_mut().enumerate() {
            let g = CHOKED_MIN + (CHOKED_MAX - CHOKED_MIN) * i as f64 / CHOKED_STEPS as f64;
            *e = (
                (2.0 / (g + 1.0)).powf(g / (g - 1.0)),
                g.sqrt() * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0))),
            );
        }
        t
    })
}

#[inline]
fn choked_constants(hcr: f64) -> (f64, f64) {
    let t = choked_table();
    let x = ((hcr - CHOKED_MIN) / (CHOKED_MAX - CHOKED_MIN) * CHOKED_STEPS as f64).clamp(0.0, CHOKED_STEPS as f64 - 1e-9);
    let i = x as usize;
    let f = x - i as f64;
    let (a, b) = (t[i], t[i + 1]);
    (a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1))
}

#[inline]
pub fn real_gamma(t: f64) -> f64 {
    GAMMA_0 - GAMMA_SLOPE * t
}

impl Default for GasSystem {
    fn default() -> Self {
        GasSystem {
            state: State::default(),
            degrees_of_freedom: 5,
            hcr: heat_capacity_ratio(5),
            choked_flow_limit: 0.0,
            choked_flow_factor_cached: 0.0,
            width: 0.0,
            height: 0.0,
            dx: 0.0,
            dy: 0.0,
            real: false,
            derived: Cell::new(Derived::STALE),
        }
    }
}

pub struct FlowParameters {
    pub k_flow: f64,
    pub dt: f64,
    pub direction_x: f64,
    pub direction_y: f64,
    pub cross_section_area_0: f64,
    pub cross_section_area_1: f64,
}

#[inline]
fn clamp(x: f64, x0: f64, x1: f64) -> f64 {
    if x <= x0 {
        x0
    } else if x >= x1 {
        x1
    } else {
        x
    }
}

impl GasSystem {
    pub fn set_geometry(&mut self, width: f64, height: f64, dx: f64, dy: f64) {
        self.width = width;
        self.height = height;
        self.dx = dx;
        self.dy = dy;
    }

    pub fn initialize(&mut self, p: f64, v: f64, t: f64) {
        self.initialize_with(p, v, t, Mix::default(), 5);
    }

    pub fn initialize_with(&mut self, p: f64, v: f64, t: f64, mix: Mix, dof: i32) {
        self.degrees_of_freedom = dof;
        self.hcr = heat_capacity_ratio(dof);
        self.state.n_mol = p * v / (R * t);
        self.state.v = v;
        self.state.e_k = if self.real {
            self.state.n_mol * real_energy_per_mol(t)
        } else {
            t * (0.5 * dof as f64 * self.state.n_mol * R)
        };
        self.state.mix = mix;
        self.state.momentum = [0.0, 0.0];
        self.choked_flow_limit = choked_flow_limit(dof);
        self.choked_flow_factor_cached = choked_flow_rate(dof);
    }

    /// Real-gas properties (see the module), keeping the current pressure and temperature.
    pub fn set_real(&mut self, real: bool) {
        let t = self.temperature();
        self.real = real;
        self.state.e_k = self.state.n_mol * self.energy_per_mol_at(t);
    }

    /// Internal energy per mole at temperature `t`.
    #[inline]
    fn energy_per_mol_at(&self, t: f64) -> f64 {
        if self.real {
            real_energy_per_mol(t)
        } else {
            0.5 * t * R * self.degrees_of_freedom as f64
        }
    }

    /// Energy a mole flowing out carries: its internal energy, plus its flow work for a real
    /// gas.
    #[inline]
    fn outflow_energy_per_mol(&self) -> f64 {
        if self.real {
            self.kinetic_energy_per_mol() + R * self.temperature()
        } else {
            self.kinetic_energy_per_mol()
        }
    }

    pub fn reset(&mut self, p: f64, t: f64, mix: Mix) {
        self.state.n_mol = p * self.volume() / (R * t);
        if self.real {
            self.state.e_k = self.state.n_mol * real_energy_per_mol(t);
            self.state.mix = mix;
            self.state.momentum = [0.0, 0.0];
            return;
        }
        self.state.e_k = t * (0.5 * self.degrees_of_freedom as f64 * self.state.n_mol * R);
        self.state.mix = mix;
        self.state.momentum = [0.0, 0.0];
    }

    pub fn set_volume(&mut self, v: f64) {
        self.change_volume(v - self.state.v)
    }

    pub fn change_volume(&mut self, dv: f64) {
        let v = self.volume();
        let l = (v + dv).powf(1.0 / 3.0);
        let surface_area = l * l;
        let dl = -dv / surface_area;
        let w = dl * self.pressure() * surface_area;
        self.state.v += dv;
        self.state.e_k += w;
    }

    pub fn change_energy(&mut self, de: f64) {
        self.state.e_k += de;
    }

    pub fn react(&mut self, n: f64, mix: &Mix) -> f64 {
        let l_n_fuel = mix.p_fuel * n;
        let l_n_o2 = mix.p_o2 * n;
        let system_n_fuel = self.n_fuel();
        let system_n_o2 = self.n_o2();
        let system_n_inert = self.n_inert();
        let system_n = self.n();

        // 25[O2] + 2[C8H16] -> 16[CO2] + 18[H2O]
        const IDEAL_O2_RATIO: f64 = 25.0 / 2.0;
        const IDEAL_FUEL_RATIO: f64 = 2.0 / 25.0;
        const OUTPUT_INPUT_RATIO: f64 = (16.0 + 18.0) / (25.0 + 2.0);

        let ideal_fuel_n = IDEAL_FUEL_RATIO * l_n_o2;
        let ideal_o2_n = IDEAL_O2_RATIO * l_n_fuel;
        let a_n_fuel = system_n_fuel.min(l_n_fuel).min(ideal_fuel_n);
        let a_n_o2 = system_n_o2.min(l_n_o2).min(ideal_o2_n);

        let reactants_n = a_n_fuel + a_n_o2;
        let products_n = OUTPUT_INPUT_RATIO * reactants_n;
        let dn = products_n - reactants_n;
        self.state.n_mol += dn;

        let new_system_n_fuel = system_n_fuel - a_n_fuel;
        let new_system_n_o2 = system_n_o2 - a_n_o2;
        let new_system_n_inert = system_n_inert + products_n;
        let new_system_n = system_n + dn;
        if new_system_n != 0.0 {
            self.state.mix.p_fuel = new_system_n_fuel / new_system_n;
            self.state.mix.p_inert = new_system_n_inert / new_system_n;
            self.state.mix.p_o2 = new_system_n_o2 / new_system_n;
        } else {
            self.state.mix = Mix { p_fuel: 0.0, p_inert: 0.0, p_o2: 0.0 };
        }
        a_n_fuel
    }

    pub fn flow_constant(target_flow_rate: f64, p: f64, pressure_drop: f64, t: f64, hcr: f64) -> f64 {
        let t_0 = t;
        let p_0 = p;
        let p_t = p - pressure_drop;
        let choked = (2.0 / (hcr + 1.0)).powf(hcr / (hcr - 1.0));
        let p_ratio = p_t / p_0;
        let mut flow_rate;
        if p_ratio <= choked {
            flow_rate = hcr.sqrt();
            flow_rate *= (2.0 / (hcr + 1.0)).powf((hcr + 1.0) / (2.0 * (hcr - 1.0)));
        } else {
            flow_rate = (2.0 * hcr) / (hcr - 1.0);
            flow_rate *= 1.0 - p_ratio.powf((hcr - 1.0) / hcr);
            flow_rate = flow_rate.sqrt();
            flow_rate *= p_ratio.powf(1.0 / hcr);
        }
        flow_rate *= p_0 / (R * t_0).sqrt();
        target_flow_rate / flow_rate
    }

    #[allow(clippy::too_many_arguments)]
    fn flow_rate(
        k_flow: f64,
        p0: f64,
        p1: f64,
        t0: f64,
        t1: f64,
        hcr: f64,
        choked_flow_limit: f64,
        choked_flow_rate_cached: f64,
        real: bool,
    ) -> f64 {
        if k_flow == 0.0 {
            return 0.0;
        }
        let (direction, t_0, p_0, p_t) = if p0 > p1 { (1.0, t0, p0, p1) } else { (-1.0, t1, p1, p0) };
        let p_ratio = p_t / p_0;
        let mut flow_rate;
        if p_ratio <= choked_flow_limit {
            flow_rate = choked_flow_rate_cached;
            flow_rate /= (R * t_0).sqrt();
        } else if real {
            flow_rate = subsonic(p_ratio, hcr) * ((1.0 - p_ratio) / (R * t_0)).sqrt();
        } else {
            let s = pow_inverse_hcr(p_ratio, hcr);
            flow_rate = (2.0 * hcr) / (hcr - 1.0);
            flow_rate *= s * (s - p_ratio);
            flow_rate = (flow_rate.max(0.0) / (R * t_0)).sqrt();
        }
        flow_rate *= direction * p_0;
        flow_rate * k_flow
    }

    fn lose_n(&mut self, dn: f64, e_k_per_mol: f64) -> f64 {
        self.state.e_k -= e_k_per_mol * dn;
        self.state.n_mol -= dn;
        if self.state.n_mol < 0.0 {
            self.state.n_mol = 0.0;
        }
        dn
    }

    fn gain_n(&mut self, dn: f64, e_k_per_mol: f64, mix: &Mix) -> f64 {
        let next_n = self.state.n_mol + dn;
        let current_n = self.state.n_mol;
        self.state.e_k += dn * e_k_per_mol;
        self.state.n_mol = next_n;
        if next_n != 0.0 {
            let m = &mut self.state.mix;
            m.p_fuel = (m.p_fuel * current_n + dn * mix.p_fuel) / next_n;
            m.p_inert = (m.p_inert * current_n + dn * mix.p_inert) / next_n;
            m.p_o2 = (m.p_o2 * current_n + dn * mix.p_o2) / next_n;
        } else {
            self.state.mix = Mix { p_fuel: 0.0, p_inert: 0.0, p_o2: 0.0 };
        }
        -dn
    }

    pub fn dissipate_excess_velocity(&mut self) {
        // Nearly always below the speed of sound: nothing to do.
        let [m_x, m_y] = self.state.momentum;
        if (m_x == 0.0 && m_y == 0.0) || self.certainly_subsonic(m_x * m_x + m_y * m_y) {
            return;
        }
        let v_x = self.velocity_x();
        let v_y = self.velocity_y();
        let v_squared = v_x * v_x + v_y * v_y;
        let c = self.c();
        let c_squared = c * c;
        if c_squared >= v_squared || v_squared == 0.0 {
            return;
        }
        let k_squared = c_squared / v_squared;
        let k = k_squared.sqrt();
        self.state.momentum[0] *= k;
        self.state.momentum[1] *= k;
        self.state.e_k += 0.5 * self.mass() * (v_squared - c_squared);
        if self.state.e_k < 0.0 {
            self.state.e_k = 0.0;
        }
    }

    pub fn update_velocity(&mut self, dt: f64, beta: f64) {
        if self.n() == 0.0 {
            return;
        }
        let depth = self.volume() / (self.width * self.height);
        let mut d_momentum_x = 0.0;
        let mut d_momentum_y = 0.0;
        let (dx, dy) = (self.dx, self.dy);
        let terms = self.dynamic_pressure_terms();
        let p0 = self.dynamic_pressure_with(&terms, dx, dy);
        let p1 = self.dynamic_pressure_with(&terms, -dx, -dy);
        let p2 = self.dynamic_pressure_with(&terms, dy, dx);
        let p3 = self.dynamic_pressure_with(&terms, -dy, -dx);

        let p_sa_0 = p0 * (self.height * depth);
        let p_sa_1 = p1 * (self.height * depth);
        let p_sa_2 = p2 * (self.width * depth);
        let p_sa_3 = p3 * (self.width * depth);

        d_momentum_x += p_sa_0 * dx;
        d_momentum_y += p_sa_0 * dy;
        d_momentum_x -= p_sa_1 * dx;
        d_momentum_y -= p_sa_1 * dy;
        d_momentum_x += p_sa_2 * dy;
        d_momentum_y += p_sa_2 * dx;
        d_momentum_x -= p_sa_3 * dy;
        d_momentum_y -= p_sa_3 * dx;

        let m = self.mass();
        let inv_m = 1.0 / m;
        let v0_x = self.state.momentum[0] * inv_m;
        let v0_y = self.state.momentum[1] * inv_m;

        self.state.momentum[0] -= d_momentum_x * dt * beta;
        self.state.momentum[1] -= d_momentum_y * dt * beta;

        let v1_x = self.state.momentum[0] * inv_m;
        let v1_y = self.state.momentum[1] * inv_m;

        self.state.e_k -= 0.5 * m * (v1_x * v1_x - v0_x * v0_x);
        self.state.e_k -= 0.5 * m * (v1_y * v1_y - v0_y * v0_y);
        if self.state.e_k < 0.0 {
            self.state.e_k = 0.0;
        }
    }

    /// Flow between two systems; returns the moles moved from `s0` to `s1` (negative the
    /// other way).
    pub fn flow(params: &FlowParameters, s0: &mut GasSystem, s1: &mut GasSystem) -> f64 {
        // A closed valve (or an absent idle circuit) moves nothing: engine-sim works through
        // the whole exchange anyway, which leaves both systems as they were but for clamping
        // their energy at zero.
        if params.k_flow == 0.0 {
            for s in [s0, s1] {
                if s.state.e_k < 0.0 {
                    s.state.e_k = 0.0;
                }
            }
            return 0.0;
        }
        let p_0 = s0.total_pressure(params.direction_x, params.direction_y);
        let p_1 = s1.total_pressure(-params.direction_x, -params.direction_y);

        let (source, sink, dx, dy, source_pressure, sink_pressure, source_cs, sink_cs, direction) =
            if p_0 > p_1 {
                (
                    s0,
                    s1,
                    params.direction_x,
                    params.direction_y,
                    p_0,
                    p_1,
                    params.cross_section_area_0,
                    params.cross_section_area_1,
                    1.0,
                )
            } else {
                (
                    s1,
                    s0,
                    -params.direction_x,
                    -params.direction_y,
                    p_1,
                    p_0,
                    params.cross_section_area_1,
                    params.cross_section_area_0,
                    -1.0,
                )
            };

        // flow_rate uses the upstream temperature: the source's unless the pressures are equal.
        let (t_source, t_sink) = if source_pressure > sink_pressure {
            (source.temperature(), 0.0)
        } else {
            (0.0, sink.temperature())
        };
        let (limit, factor) = source.choked();
        let mut flow = params.dt
            * Self::flow_rate(
                params.k_flow,
                source_pressure,
                sink_pressure,
                t_source,
                t_sink,
                source.heat_capacity_ratio(),
                limit,
                factor,
                source.real,
            );

        flow = clamp(flow, 0.0, 0.9 * source.n());

        let fraction = flow / source.n();
        let fraction_volume = fraction * source.volume();
        let fraction_mass = fraction * source.mass();

        if flow != 0.0 {
            let e_k_bulk_src0 = source.bulk_kinetic_energy();
            let e_k_bulk_sink0 = sink.bulk_kinetic_energy();

            let e_k_per_mol = source.outflow_energy_per_mol();
            let source_mix = source.state.mix;
            sink.gain_n(flow, e_k_per_mol, &source_mix);
            source.lose_n(flow, e_k_per_mol);

            let dp_x = source.state.momentum[0] * fraction;
            let dp_y = source.state.momentum[1] * fraction;

            source.state.momentum[0] -= dp_x;
            source.state.momentum[1] -= dp_y;
            sink.state.momentum[0] += dp_x;
            sink.state.momentum[1] += dp_y;

            let e_k_bulk_src1 = source.bulk_kinetic_energy();
            let e_k_bulk_sink1 = sink.bulk_kinetic_energy();

            sink.state.e_k -= (e_k_bulk_src1 + e_k_bulk_sink1) - (e_k_bulk_src0 + e_k_bulk_sink0);
        }

        let source_mass = source.mass();
        let inv_source_mass = 1.0 / source_mass;
        let sink_mass = sink.mass();
        let inv_sink_mass = 1.0 / sink_mass;


        let source_initial_momentum_x = source.state.momentum[0];
        let source_initial_momentum_y = source.state.momentum[1];
        let sink_initial_momentum_x = sink.state.momentum[0];
        let sink_initial_momentum_y = sink.state.momentum[1];

        if sink_cs != 0.0 {
            let v = sink.clamp_to_c((fraction_volume / sink_cs) / params.dt);
            sink.state.momentum[0] += v * dx * fraction_mass;
            sink.state.momentum[1] += v * dy * fraction_mass;
        }

        if source_cs != 0.0 && source_mass != 0.0 {
            let v = source.clamp_to_c((fraction_volume / source_cs) / params.dt);
            source.state.momentum[0] += v * dx * fraction_mass;
            source.state.momentum[1] += v * dy * fraction_mass;
        }

        if source_mass != 0.0 {
            let v0_x = source_initial_momentum_x * inv_source_mass;
            let v0_y = source_initial_momentum_y * inv_source_mass;
            let v1_x = source.state.momentum[0] * inv_source_mass;
            let v1_y = source.state.momentum[1] * inv_source_mass;
            source.state.e_k -= 0.5 * source_mass * (v1_x * v1_x - v0_x * v0_x);
            source.state.e_k -= 0.5 * source_mass * (v1_y * v1_y - v0_y * v0_y);
        }

        if sink_mass > 0.0 {
            let v0_x = sink_initial_momentum_x * inv_sink_mass;
            let v0_y = sink_initial_momentum_y * inv_sink_mass;
            let v1_x = sink.state.momentum[0] * inv_sink_mass;
            let v1_y = sink.state.momentum[1] * inv_sink_mass;
            sink.state.e_k -= 0.5 * sink_mass * (v1_x * v1_x - v0_x * v0_x);
            sink.state.e_k -= 0.5 * sink_mass * (v1_y * v1_y - v0_y * v0_y);
        }

        if sink.state.e_k < 0.0 {
            sink.state.e_k = 0.0;
        }
        if source.state.e_k < 0.0 {
            source.state.e_k = 0.0;
        }

        flow * direction
    }

    /// Flow to or from a reservoir at `p_env`, `t_env` (blow-by).
    pub fn flow_env(&mut self, k_flow: f64, dt: f64, p_env: f64, t_env: f64, mix: &Mix) -> f64 {
        let max_flow = self.pressure_equilibrium_max_flow_env(p_env, t_env);
        let (limit, factor) = self.choked();
        let mut flow = dt
            * Self::flow_rate(
                k_flow,
                self.pressure(),
                p_env,
                self.temperature(),
                t_env,
                self.heat_capacity_ratio(),
                limit,
                factor,
                self.real,
            );

        if flow.abs() > max_flow.abs() {
            flow = max_flow;
        }

        if flow < 0.0 {
            let bulk_e_k_0 = self.bulk_kinetic_energy();
            let e = if self.real {
                real_energy_per_mol(t_env) + R * t_env
            } else {
                kinetic_energy_per_mol_at(t_env, self.degrees_of_freedom)
            };
            self.gain_n(-flow, e, mix);
            let bulk_e_k_1 = self.bulk_kinetic_energy();
            self.state.e_k += bulk_e_k_1 - bulk_e_k_0;
        } else {
            let starting_n = self.n();
            self.lose_n(flow, self.outflow_energy_per_mol());
            self.state.momentum[0] -= (flow / starting_n) * self.state.momentum[0];
            self.state.momentum[1] -= (flow / starting_n) * self.state.momentum[1];
        }
        flow
    }

    fn pressure_equilibrium_max_flow_env(&self, p_env: f64, t_env: f64) -> f64 {
        if self.real {
            // The moles that would bring the pressure to p_env at the current temperature:
            // a bound on the step, as engine-sim's is.
            return self.n() - p_env * self.volume() / (R * self.temperature().max(1.0));
        }
        let dof = self.degrees_of_freedom as f64;
        if self.pressure() > p_env {
            -(p_env * (0.5 * dof * self.volume()) - self.kinetic_energy()) / self.kinetic_energy_per_mol()
        } else {
            let e_k_per_mol_env = 0.5 * t_env * R * dof;
            -(p_env * (0.5 * dof * self.volume()) - self.kinetic_energy()) / e_k_per_mol_env
        }
    }

    /// The real gas's derived properties, worked out again only when its state has changed.
    #[inline]
    fn derived(&self) -> Derived {
        let (e, n, v) = (self.state.e_k, self.state.n_mol, self.state.v);
        let d = self.derived.get();
        if d.e == e && d.n == n && d.v == v {
            return d;
        }
        let d = Derived::of(e, n, v);
        self.derived.set(d);
        d
    }

    #[inline]
    pub fn heat_capacity_ratio(&self) -> f64 {
        if self.real {
            self.derived().hcr
        } else {
            self.hcr
        }
    }

    /// The pressure ratio below which flow out of this gas chokes, and the choked flow
    /// factor (engine-sim caches both for its constant ratio).
    #[inline]
    fn choked(&self) -> (f64, f64) {
        if !self.real {
            return (self.choked_flow_limit, self.choked_flow_factor_cached);
        }
        choked_constants(self.heat_capacity_ratio())
    }

    #[inline]
    fn approximate_density(&self) -> f64 {
        (AIR_MOLECULAR_MASS * self.n()) / self.volume()
    }

    #[inline]
    pub fn n(&self) -> f64 {
        self.state.n_mol
    }

    #[inline]
    pub fn kinetic_energy(&self) -> f64 {
        self.state.e_k
    }

    #[inline]
    fn kinetic_energy_n(&self, n: f64) -> f64 {
        (self.kinetic_energy() / self.n()) * n
    }

    #[inline]
    pub fn kinetic_energy_per_mol(&self) -> f64 {
        self.kinetic_energy_n(1.0)
    }

    #[inline]
    pub fn c(&self) -> f64 {
        if self.n() == 0.0 || self.kinetic_energy() == 0.0 {
            return 0.0;
        }
        if self.real {
            return self.derived().c_squared.sqrt();
        }
        let hcr = self.heat_capacity_ratio();
        let static_pressure = self.pressure();
        let density = self.approximate_density();
        (static_pressure * hcr / density).sqrt()
    }

    #[inline]
    fn bulk_kinetic_energy(&self) -> f64 {
        let m = self.mass();
        if m == 0.0 {
            return 0.0;
        }
        let v_x = self.state.momentum[0] / m;
        let v_y = self.state.momentum[1] / m;
        let v_squared = v_x * v_x + v_y * v_y;
        0.5 * m * v_squared
    }

    /// pressure() + dynamic_pressure(dx, dy), with the static pressure worked out once.
    #[inline]
    fn total_pressure(&self, dx: f64, dy: f64) -> f64 {
        if self.real {
            let terms = self.dynamic_pressure_terms();
            return match terms {
                Some(t) => t.static_pressure + self.dynamic_pressure_with(&terms, dx, dy),
                None => self.pressure() + 0.0,
            };
        }
        let p = self.pressure();
        if self.n() == 0.0 || self.kinetic_energy() == 0.0 {
            return p + 0.0;
        }
        let inverse_mass = 1.0 / self.mass();
        let density = self.approximate_density();
        let c_squared = p * self.heat_capacity_ratio() / density;
        let terms = Some(DynamicTerms { inverse_mass, static_pressure: p, c_squared, hcr: self.hcr });
        p + self.dynamic_pressure_with(&terms, dx, dy)
    }

    /// Whether the bulk speed |momentum| / mass is certainly below the speed of sound, without
    /// computing either: v^2 < c^2 = P hcr / rho rearranged free of divisions, with a margin
    /// far beyond their rounding (so it never says yes where c() would say no).
    #[inline]
    fn certainly_subsonic(&self, momentum_squared: f64) -> bool {
        let n = self.n();
        let e = self.kinetic_energy();
        if self.real {
            // v^2 < c^2 = hcr n R T / mass, times mass^2, without the temperature: n R T is at
            // least (hcr - 1) E, as cv rises with temperature (so E <= n cv(T) T), and
            // hcr (hcr - 1) is smallest for the hottest gas, taken as 4000 K. Below roughly
            // Mach 0.4 this decides; above it, the exact test.
            const BOUND: f64 = {
                let g = GAMMA_0 - GAMMA_SLOPE * 4000.0;
                g * (g - 1.0)
            };
            if n <= 0.0 || e <= 0.0 || self.volume() == 0.0 {
                return false;
            }
            let m = self.mass();
            return momentum_squared < BOUND * e * m * (1.0 - 1e-9)
                || momentum_squared < self.derived().c_squared * m * m * (1.0 - 1e-9);
        }
        n > 0.0
            && e > 0.0
            && self.volume() != 0.0
            && momentum_squared * (0.5 * self.degrees_of_freedom as f64)
                < e * self.hcr * AIR_MOLECULAR_MASS * n * (1.0 - 1e-9)
    }

    /// clamp(v, 0, c()), without the square root when v is certainly below c.
    #[inline]
    fn clamp_to_c(&self, v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let m = self.mass();
        if self.certainly_subsonic(v * v * m * m) {
            return v;
        }
        clamp(v, 0.0, self.c())
    }

    #[inline]
    pub fn dynamic_pressure(&self, dx: f64, dy: f64) -> f64 {
        let terms = self.dynamic_pressure_terms();
        self.dynamic_pressure_with(&terms, dx, dy)
    }

    /// What dynamic_pressure computes regardless of direction: None when the gas is empty
    /// or cold.
    #[inline]
    fn dynamic_pressure_terms(&self) -> Option<DynamicTerms> {
        if self.n() == 0.0 || self.kinetic_energy() == 0.0 {
            return None;
        }
        let inverse_mass = 1.0 / self.mass();
        if self.real {
            let d = self.derived();
            return Some(DynamicTerms { inverse_mass, static_pressure: d.p, c_squared: d.c_squared, hcr: d.hcr });
        }
        let static_pressure = self.pressure();
        let density = self.approximate_density();
        let c_squared = static_pressure * self.heat_capacity_ratio() / density;
        Some(DynamicTerms { inverse_mass, static_pressure, c_squared, hcr: self.hcr })
    }

    #[inline]
    fn dynamic_pressure_with(&self, terms: &Option<DynamicTerms>, dx: f64, dy: f64) -> f64 {
        let Some(DynamicTerms { inverse_mass, static_pressure, c_squared, hcr }) = *terms else {
            return 0.0;
        };
        let v = inverse_mass * (dx * self.state.momentum[0] + dy * self.state.momentum[1]);
        if v <= 0.0 {
            return 0.0;
        }
        let mach_number_squared = v * v / c_squared;

        let x = 1.0 + ((hcr - 1.0) / 2.0) * mach_number_squared;
        if self.real {
            // x^k - 1 for k = hcr / (hcr - 1): by the binomial series while x is close to 1
            // (Mach ~0.35 and below, exact to ~1e-12), which is nearly always.
            let k = hcr / (hcr - 1.0);
            let z = x - 1.0;
            if z < 0.02 {
                let series = 1.0
                    + (k - 1.0) * z / 2.0
                        * (1.0
                            + (k - 2.0) * z / 3.0
                                * (1.0 + (k - 3.0) * z / 4.0 * (1.0 + (k - 4.0) * z / 5.0 * (1.0 + (k - 5.0) * z / 6.0))));
                return static_pressure * k * z * series;
            }
            return static_pressure * (fastmath::pow(x, k) - 1.0);
        }
        let x_d = match self.degrees_of_freedom {
            3 => x * x * x * x * x,
            5 => {
                let x_2 = x * x;
                let x_3 = x_2 * x;
                x_3 * x_3 * x
            }
            _ => x,
        };
        static_pressure * (x_d.sqrt() - 1.0)
    }

    #[inline]
    pub fn mass(&self) -> f64 {
        AIR_MOLECULAR_MASS * self.n()
    }

    #[inline]
    pub fn pressure(&self) -> f64 {
        let volume = self.volume();
        if self.real {
            return self.derived().p;
        }
        if volume != 0.0 {
            self.kinetic_energy() / (0.5 * self.degrees_of_freedom as f64 * volume)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn temperature(&self) -> f64 {
        if self.n() == 0.0 {
            0.0
        } else if self.real {
            self.derived().t
        } else {
            self.kinetic_energy() / (0.5 * self.degrees_of_freedom as f64 * self.n() * R)
        }
    }

    #[inline]
    fn velocity_x(&self) -> f64 {
        if self.n() == 0.0 {
            0.0
        } else {
            self.state.momentum[0] / self.mass()
        }
    }

    #[inline]
    fn velocity_y(&self) -> f64 {
        if self.n() == 0.0 {
            0.0
        } else {
            self.state.momentum[1] / self.mass()
        }
    }

    #[inline]
    pub fn volume(&self) -> f64 {
        self.state.v
    }

    #[inline]
    fn n_fuel(&self) -> f64 {
        self.state.mix.p_fuel * self.n()
    }

    #[inline]
    fn n_inert(&self) -> f64 {
        self.state.mix.p_inert * self.n()
    }

    #[inline]
    fn n_o2(&self) -> f64 {
        self.state.mix.p_o2 * self.n()
    }

    #[inline]
    /// Moles and internal energy (J): what another model of the gas exchanges with.
    pub fn totals(&self) -> (f64, f64) {
        (self.state.n_mol, self.state.e_k)
    }

    /// Sets moles, internal energy and composition, after another model's exchange.
    pub fn set_totals(&mut self, n: f64, e: f64, mix: Mix) {
        self.state.n_mol = n;
        self.state.e_k = e;
        self.state.mix = mix;
    }

    pub fn mix(&self) -> Mix {
        self.state.mix
    }
}

#[inline]
pub fn kinetic_energy_per_mol_at(t: f64, dof: i32) -> f64 {
    0.5 * t * R * dof as f64
}

#[inline]
pub fn heat_capacity_ratio(dof: i32) -> f64 {
    1.0 + (2.0 / dof as f64)
}

fn choked_flow_limit(dof: i32) -> f64 {
    let hcr = heat_capacity_ratio(dof);
    (2.0 / (hcr + 1.0)).powf(hcr / (hcr - 1.0))
}

fn choked_flow_rate(dof: i32) -> f64 {
    let hcr = heat_capacity_ratio(dof);
    hcr.sqrt() * (2.0 / (hcr + 1.0)).powf((hcr + 1.0) / (2.0 * (hcr - 1.0)))
}

/// `x^(1/hcr)`, the hot power in subsonic flow. For air (hcr 1.4, x in 0.5..=1, as subsonic
/// flow has it) a cubic estimate refined by three Newton steps on y^7 = x^5: within an ulp or
/// two of powf and several times faster where powf is computed in software (WebAssembly).
/// The `reference` feature keeps powf, to compare with engine-sim exactly.
#[inline]
fn pow_inverse_hcr(x: f64, hcr: f64) -> f64 {
    if cfg!(feature = "reference") || hcr != 1.4 || !(0.5..=1.0).contains(&x) {
        return x.powf(1.0 / hcr);
    }
    const C: [f64; 4] = [
        0.08856258257572591,
        -0.35058324062335194,
        1.1518453510510853,
        0.1102301526234537,
    ];
    let mut y = ((C[0] * x + C[1]) * x + C[2]) * x + C[3];
    let x2 = x * x;
    let x5_7 = x2 * x2 * x / 7.0;
    for _ in 0..3 {
        let y2 = y * y;
        let y6 = y2 * y2 * y2;
        y = y * (6.0 / 7.0) + x5_7 / y6;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real(p: f64, v: f64, t: f64) -> GasSystem {
        let mut g = GasSystem::default();
        g.set_real(true);
        g.initialize(p, v, t);
        g.set_geometry(v.sqrt(), v.sqrt(), 1.0, 0.0);
        g
    }

    #[test]
    fn real_gas_state() {
        for t in [250.0, 300.0, 900.0, 2500.0, 3200.0, 4500.0] {
            let exact = real_temperature_exact(real_energy_per_mol(t));
            assert!((exact - t).abs() < 1e-9 * t, "{t}: {exact}");
            // The tabulated temperature (and what follows from it) within 2e-4 K.
            let g = real(2e5, 1e-3, t);
            assert!((g.temperature() - t).abs() < 2e-4, "{t}: {}", g.temperature());
            assert!((g.pressure() - 2e5).abs() < 2e5 * 2e-4 / t);
            assert!((g.heat_capacity_ratio() - (1.3678 - 8.13e-5 * (t - 300.0))).abs() < 8.13e-5 * 2e-4);
        }
        // cv is the slope of the internal energy.
        let (t, h) = (1500.0, 1e-3);
        let cv = (real_energy_per_mol(t + h) - real_energy_per_mol(t - h)) / (2.0 * h);
        assert!((cv - R / (real_gamma(t) - 1.0)).abs() < 1e-5);
    }

    /// Slow compression follows the isentrope of the varying gamma: dT/T = -(gamma - 1) dV/V.
    #[test]
    fn real_gas_isentropic_compression() {
        let mut g = real(1e5, 1e-3, 300.0);
        let mut t = 300.0;
        let steps = 200_000;
        let ratio: f64 = 10.0;
        let mut v = 1e-3;
        for i in 1..=steps {
            let v1 = 1e-3 / ratio.powf(i as f64 / steps as f64);
            t -= (real_gamma(t) - 1.0) * t * (v1 - v) / v;
            g.set_volume(v1);
            v = v1;
        }
        assert!((g.temperature() - t).abs() < 0.01 * t, "{} vs {t}", g.temperature());
    }

    /// Flow conserves the total energy and carries flow work: gas let into an empty vessel
    /// ends up hotter than its source (gamma times, for a constant gamma).
    #[test]
    fn real_gas_flow_carries_enthalpy() {
        let mut src = real(1e6, 1.0, 300.0);
        let mut dst = real(1e3, 1e-3, 300.0);
        let e0 = src.kinetic_energy() + dst.kinetic_energy();
        let params = FlowParameters {
            k_flow: 1e-5,
            dt: 1e-5,
            direction_x: 1.0,
            direction_y: 0.0,
            cross_section_area_0: 0.0,
            cross_section_area_1: 0.0,
        };
        for _ in 0..20000 {
            GasSystem::flow(&params, &mut src, &mut dst);
        }
        let e1 = src.kinetic_energy() + dst.kinetic_energy();
        assert!((e1 - e0).abs() < 1e-9 * e0);
        let t = dst.temperature();
        assert!(t > 1.3 * 300.0 && t < 1.38 * 300.0, "{t}");
    }

    #[test]
    fn real_gas_shortcuts() {
        let mut worst = 0.0f64;
        for i in 0..=10_000 {
            let g = 1.06 + 0.35 * i as f64 / 10_000.0;
            let (l, f) = choked_constants(g);
            let l0 = (2.0 / (g + 1.0)).powf(g / (g - 1.0));
            let f0 = g.sqrt() * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0)));
            worst = worst.max(((l - l0) / l0).abs()).max(((f - f0) / f0).abs());
        }
        assert!(worst < 1e-7, "choked table {worst:e}");
        // Dynamic pressure's series against the power, for subsonic speeds.
        let mut g = real(2e5, 1e-3, 900.0);
        let hcr = g.heat_capacity_ratio();
        let mut worst = 0.0f64;
        for i in 1..=200 {
            let mach = i as f64 / 200.0;
            let c = g.c();
            g.state.momentum = [mach * c * g.mass(), 0.0];
            let p = g.pressure();
            let x: f64 = 1.0 + (hcr - 1.0) / 2.0 * mach * mach;
            let exact = p * (x.powf(hcr / (hcr - 1.0)) - 1.0);
            let ours = g.dynamic_pressure(1.0, 0.0);
            worst = worst.max(((ours - exact) / exact).abs());
        }
        assert!(worst < 1e-10, "dynamic pressure {worst:e}");
    }

    #[test]
    fn temperature_table_accuracy() {
        let mut worst = 0.0f64;
        let mut u = 0.0;
        while u < 2.5e5 {
            worst = worst.max((real_temperature(u) - real_temperature_exact(u)).abs());
            u += 7.31;
        }
        assert!(worst < 2e-4, "{worst} K");
    }

    #[test]
    fn subsonic_table_accuracy() {
        // Against the flow function as flow_rate worked it out before (a pow per flow), over
        // the real gas's ratios (250 to 4000 K) and every subsonic pressure ratio.
        let mut worst = 0.0f64;
        for i in 0..=200 {
            let g = real_gamma(250.0 + 3750.0 * i as f64 / 200.0);
            let (limit, _) = choked_constants(g);
            for j in 1..=1000 {
                let r = limit + (1.0 - limit) * j as f64 / 1000.0;
                let s = r.powf(1.0 / g);
                let exact = (2.0 * g / (g - 1.0) * s * (s - r)).max(0.0).sqrt();
                let table = subsonic(r, g) * (1.0 - r).sqrt();
                if exact > 0.0 {
                    worst = worst.max((table / exact - 1.0).abs());
                }
            }
        }
        assert!(worst < 4e-6, "{worst}");
    }

    #[test]
    fn pow_inverse_hcr_matches_powf() {
        let mut worst = 0.0f64;
        for i in 0..=100_000 {
            let x = 0.5 + 0.5 * i as f64 / 100_000.0;
            let exact = x.powf(1.0 / 1.4);
            let fast = super::pow_inverse_hcr(x, 1.4);
            worst = worst.max(((fast - exact) / exact).abs());
        }
        assert!(worst < 1e-15, "worst relative error {worst:e}");
    }
}
