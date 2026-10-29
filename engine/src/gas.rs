//! A volume of ideal gas with bulk momentum, and flow between two of them
//! (engine-sim's gas_system.h/.cpp).

use crate::units::{AIR_MOLECULAR_MASS, R};

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
        self.state.e_k = t * (0.5 * dof as f64 * self.state.n_mol * R);
        self.state.mix = mix;
        self.state.momentum = [0.0, 0.0];
        self.choked_flow_limit = choked_flow_limit(dof);
        self.choked_flow_factor_cached = choked_flow_rate(dof);
    }

    pub fn reset(&mut self, p: f64, t: f64, mix: Mix) {
        self.state.n_mol = p * self.volume() / (R * t);
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
        } else {
            let s = p_ratio.powf(1.0 / hcr);
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
        let p_0 = s0.pressure() + s0.dynamic_pressure(params.direction_x, params.direction_y);
        let p_1 = s1.pressure() + s1.dynamic_pressure(-params.direction_x, -params.direction_y);

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

        let mut flow = params.dt
            * Self::flow_rate(
                params.k_flow,
                source_pressure,
                sink_pressure,
                source.temperature(),
                sink.temperature(),
                source.heat_capacity_ratio(),
                source.choked_flow_limit,
                source.choked_flow_factor_cached,
            );

        flow = clamp(flow, 0.0, 0.9 * source.n());

        let fraction = flow / source.n();
        let fraction_volume = fraction * source.volume();
        let fraction_mass = fraction * source.mass();

        if flow != 0.0 {
            let e_k_bulk_src0 = source.bulk_kinetic_energy();
            let e_k_bulk_sink0 = sink.bulk_kinetic_energy();

            let e_k_per_mol = source.kinetic_energy_per_mol();
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

        let c_source = source.c();
        let c_sink = sink.c();

        let source_initial_momentum_x = source.state.momentum[0];
        let source_initial_momentum_y = source.state.momentum[1];
        let sink_initial_momentum_x = sink.state.momentum[0];
        let sink_initial_momentum_y = sink.state.momentum[1];

        if sink_cs != 0.0 {
            let v = clamp((fraction_volume / sink_cs) / params.dt, 0.0, c_sink);
            sink.state.momentum[0] += v * dx * fraction_mass;
            sink.state.momentum[1] += v * dy * fraction_mass;
        }

        if source_cs != 0.0 && source_mass != 0.0 {
            let v = clamp((fraction_volume / source_cs) / params.dt, 0.0, c_source);
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
        let mut flow = dt
            * Self::flow_rate(
                k_flow,
                self.pressure(),
                p_env,
                self.temperature(),
                t_env,
                self.heat_capacity_ratio(),
                self.choked_flow_limit,
                self.choked_flow_factor_cached,
            );

        if flow.abs() > max_flow.abs() {
            flow = max_flow;
        }

        if flow < 0.0 {
            let bulk_e_k_0 = self.bulk_kinetic_energy();
            self.gain_n(-flow, kinetic_energy_per_mol_at(t_env, self.degrees_of_freedom), mix);
            let bulk_e_k_1 = self.bulk_kinetic_energy();
            self.state.e_k += bulk_e_k_1 - bulk_e_k_0;
        } else {
            let starting_n = self.n();
            self.lose_n(flow, self.kinetic_energy_per_mol());
            self.state.momentum[0] -= (flow / starting_n) * self.state.momentum[0];
            self.state.momentum[1] -= (flow / starting_n) * self.state.momentum[1];
        }
        flow
    }

    fn pressure_equilibrium_max_flow_env(&self, p_env: f64, t_env: f64) -> f64 {
        let dof = self.degrees_of_freedom as f64;
        if self.pressure() > p_env {
            -(p_env * (0.5 * dof * self.volume()) - self.kinetic_energy()) / self.kinetic_energy_per_mol()
        } else {
            let e_k_per_mol_env = 0.5 * t_env * R * dof;
            -(p_env * (0.5 * dof * self.volume()) - self.kinetic_energy()) / e_k_per_mol_env
        }
    }

    #[inline]
    pub fn heat_capacity_ratio(&self) -> f64 {
        self.hcr
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

    #[inline]
    pub fn dynamic_pressure(&self, dx: f64, dy: f64) -> f64 {
        let terms = self.dynamic_pressure_terms();
        self.dynamic_pressure_with(&terms, dx, dy)
    }

    /// What dynamic_pressure computes regardless of direction: None when the gas is empty
    /// or cold, else (1 / mass, static pressure, speed of sound squared).
    #[inline]
    fn dynamic_pressure_terms(&self) -> Option<(f64, f64, f64)> {
        if self.n() == 0.0 || self.kinetic_energy() == 0.0 {
            return None;
        }
        let inverse_mass = 1.0 / self.mass();
        let static_pressure = self.pressure();
        let density = self.approximate_density();
        let c_squared = static_pressure * self.heat_capacity_ratio() / density;
        Some((inverse_mass, static_pressure, c_squared))
    }

    #[inline]
    fn dynamic_pressure_with(&self, terms: &Option<(f64, f64, f64)>, dx: f64, dy: f64) -> f64 {
        let Some((inverse_mass, static_pressure, c_squared)) = *terms else {
            return 0.0;
        };
        let v = inverse_mass * (dx * self.state.momentum[0] + dy * self.state.momentum[1]);
        if v <= 0.0 {
            return 0.0;
        }
        let hcr = self.heat_capacity_ratio();
        let mach_number_squared = v * v / c_squared;

        let x = 1.0 + ((hcr - 1.0) / 2.0) * mach_number_squared;
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
