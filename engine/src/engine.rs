//! The piston engine and its simulation step (engine-sim's Engine, PistonEngineSimulator and
//! the parts they own: crankshaft, pistons, rods, heads and camshafts, combustion chambers,
//! intakes, exhausts, ignition, fuel).

use crate::function::Function;
use crate::gas::{FlowParameters, GasSystem, Mix};
use crate::rng::Rng;
use crate::crank::Mechanism;
use crate::solver::Body;
use crate::spec::{CamSpec, EngineSpec, FuelSpec};
use crate::fastmath;
use crate::turbo::{Turbo, TurboInputs};
use crate::units::{celcius, to_rpm, AIR_MOLECULAR_MASS, ATM, E, PI, ROOT_2};
use std::collections::VecDeque;

const STATE_SAMPLES: usize = 256;
const CRANKCASE_PRESSURE: f64 = ATM;

struct Bank {
    bore: f64,
    deck_height: f64,
    dx: f64,
    dy: f64,
    x: f64,
    y: f64,
}

impl Bank {
    fn bore_surface_area(&self) -> f64 {
        PI * self.bore * self.bore / 4.0
    }
}

struct Camshaft {
    advance: f64,
    profile: Function,
    /// Lobe centrelines in cam angle.
    lobes: Vec<f64>,
}

impl Camshaft {
    fn new(spec: &CamSpec) -> Self {
        Camshaft {
            advance: spec.advance,
            profile: spec.profile.clone(),
            lobes: spec.lobes.iter().map(|l| l / 2.0).collect(),
        }
    }

    fn valve_lift(&self, lobe: usize, crank_angle: f64) -> f64 {
        self.sample_lobe(self.angle(crank_angle) + self.lobes[lobe])
    }

    fn sample_lobe(&self, theta: f64) -> f64 {
        let mut t = theta % (2.0 * PI);
        if t < 0.0 {
            t += 2.0 * PI;
        }
        if t >= PI {
            t -= 2.0 * PI;
        }
        self.profile.sample_triangle(t)
    }

    fn angle(&self, crank_angle: f64) -> f64 {
        let angle = ((crank_angle + self.advance) * 0.5) % (2.0 * PI);
        if angle < 0.0 {
            angle + 2.0 * PI
        } else {
            angle
        }
    }
}

struct Head {
    chamber_volume: f64,
    intake_runner_volume: f64,
    intake_runner_area: f64,
    exhaust_runner_volume: f64,
    exhaust_runner_area: f64,
    intake_port_flow: Function,
    exhaust_port_flow: Function,
    intake_cam: Camshaft,
    exhaust_cam: Camshaft,
}

struct Intake {
    system: GasSystem,
    atmosphere: GasSystem,
    throttle: f64,
    flow: f64,
    cross_section_area: f64,
    input_flow_k: f64,
    idle_flow_k: f64,
    runner_flow_rate: f64,
    molecular_afr: f64,
    idle_throttle_plate_position: f64,
    runner_length: f64,
    velocity_decay: f64,
    /// What the throttle draws from: the atmosphere, or the turbocharger's charge.
    charge_pressure: f64,
    charge_temperature: f64,
}

impl Intake {
    fn throttle_plate_position(&self) -> f64 {
        self.idle_throttle_plate_position * self.throttle
    }

    fn process(&mut self, dt: f64) {
        let ideal_afr = 0.8 * self.molecular_afr * 4.0;
        let p_air = ideal_afr / (1.0 + ideal_afr);
        let fuel_air_mix = Mix { p_fuel: 1.0 - p_air, p_inert: p_air * 0.75, p_o2: p_air * 0.25 };

        let idle_afr = 2.0;
        let p_idle_air = idle_afr / (1.0 + idle_afr);
        let fuel_mix =
            Mix { p_fuel: 1.0 - p_idle_air, p_inert: p_idle_air * 0.75, p_o2: p_idle_air * 0.25 };

        let throttle = self.throttle_plate_position();
        let flow_attenuation = (throttle * PI / 2.0).cos();

        let mut params = FlowParameters {
            k_flow: flow_attenuation * self.input_flow_k,
            dt,
            direction_x: 0.0,
            direction_y: -1.0,
            cross_section_area_0: 10.0,
            cross_section_area_1: self.cross_section_area,
        };
        self.atmosphere.reset(self.charge_pressure, self.charge_temperature, fuel_air_mix);
        self.flow = GasSystem::flow(&params, &mut self.atmosphere, &mut self.system);

        self.atmosphere.reset(self.charge_pressure, self.charge_temperature, fuel_mix);
        params.k_flow = self.idle_flow_k;
        GasSystem::flow(&params, &mut self.atmosphere, &mut self.system);

        self.system.dissipate_excess_velocity();
        self.system.update_velocity(dt, self.velocity_decay);
    }
}

struct Exhaust {
    system: GasSystem,
    atmosphere: GasSystem,
    length: f64,
    primary_tube_length: f64,
    collector_cross_section_area: f64,
    primary_flow_rate: f64,
    outlet_flow_rate: f64,
    /// The turbine's opening (the spec's outlet), beside which a wastegate may open.
    turbine_flow_rate: f64,
    audio_volume: f64,
    velocity_decay: f64,
    /// Moles let out to the atmosphere in the last substep.
    outflow: f64,
}

impl Exhaust {
    fn process(&mut self, dt: f64) {
        let air_mix = Mix { p_fuel: 0.0, p_inert: 1.0, p_o2: 0.0 };
        self.atmosphere.reset(ATM, celcius(25.0), air_mix);
        let params = FlowParameters {
            k_flow: self.outlet_flow_rate,
            dt,
            direction_x: 1.0,
            direction_y: 0.0,
            cross_section_area_0: self.collector_cross_section_area,
            cross_section_area_1: 10.0,
        };
        self.outflow = -GasSystem::flow(&params, &mut self.atmosphere, &mut self.system);
        self.system.dissipate_excess_velocity();
        self.system.update_velocity(dt, self.velocity_decay);
    }
}

struct Fuel {
    molecular_mass: f64,
    energy_density: f64,
    molecular_afr: f64,
    max_burning_efficiency: f64,
    burning_efficiency_randomness: f64,
    low_efficiency_attenuation: f64,
    max_turbulence_effect: f64,
    max_dilution_effect: f64,
    turbulence_to_flame_speed_ratio: Function,
}

impl Fuel {
    fn new(s: &FuelSpec) -> Self {
        Fuel {
            molecular_mass: s.molecular_mass,
            energy_density: s.energy_density,
            molecular_afr: s.molecular_afr,
            max_burning_efficiency: s.max_burning_efficiency,
            burning_efficiency_randomness: s.burning_efficiency_randomness,
            low_efficiency_attenuation: s.low_efficiency_attenuation,
            max_turbulence_effect: s.max_turbulence_effect,
            max_dilution_effect: s.max_dilution_effect,
            turbulence_to_flame_speed_ratio: s.turbulence_to_flame_speed_ratio.clone(),
        }
    }

    fn flame_speed(&self, turbulence: f64, molecular_afr: f64, t: f64, p: f64) -> f64 {
        let s_l = self.laminar_burning_velocity(molecular_afr, t, p);
        self.turbulence_to_flame_speed_ratio.sample_triangle((turbulence / s_l) * 1.0) * s_l
    }

    fn laminar_burning_velocity(&self, molecular_afr: f64, t: f64, p: f64) -> f64 {
        const ER_M: f64 = 1.21;
        const B_M: f64 = 30.5 * crate::units::CM / 1.0;
        const B_ER: f64 = -(54.9 * crate::units::CM) / 1.0;
        let er = molecular_afr / self.molecular_afr;
        let alpha = 2.4 - 0.271 * er.powf(3.51);
        let beta = -0.357 + 0.14 * er.powf(2.77);
        let s_l_0 = B_M + B_ER * (er - ER_M) * (er - ER_M);
        let t_ratio = t / 298.0;
        let p_ratio = p / ATM;
        s_l_0 * t_ratio.powf(alpha) * p_ratio.powf(beta)
    }
}

#[derive(Default, Clone, Copy)]
struct FlameEvent {
    lit_n: f64,
    total_n: f64,
    percentage_lit: f64,
    efficiency: f64,
    flame_speed: f64,
    last_volume: f64,
    travel_x: f64,
    travel_y: f64,
    global_mix: Mix,
}

struct DelayFilter {
    latency: usize,
    history: VecDeque<f64>,
}

impl DelayFilter {
    fn new(delay: f64, frequency: f64) -> Self {
        let samples = (delay * frequency).round() as usize;
        DelayFilter { latency: samples, history: VecDeque::with_capacity(samples + 32) }
    }

    fn f(&mut self, sample: f64) -> f64 {
        self.history.push_back(sample);
        if self.history.len() <= self.latency {
            0.0
        } else {
            self.history.pop_front().unwrap()
        }
    }
}

/// A cylinder: piston, rod and combustion chamber.
struct Cylinder {
    bank: usize,
    /// Index within the bank (camshaft lobe, head port).
    index: usize,
    /// The piston's position along the bore and speed (from the mechanism).
    piston_s: f64,
    piston_v: f64,
    compression_height: f64,
    displacement: f64,
    blowby_k: f64,
    intake: usize,
    exhaust: usize,
    primary_length: f64,
    sound_attenuation: f64,
    firing_angle: f64,
    ignition_event: bool,
    delay: DelayFilter,

    system: GasSystem,
    intake_runner: GasSystem,
    exhaust_runner: GasSystem,
    flame: FlameEvent,
    lit: bool,
    intake_flow_rate: f64,
    exhaust_flow_rate: f64,
    manifold_to_runner_flow_rate: f64,
    primary_to_collector_flow_rate: f64,
    cross_section_area: f64,
    exhaust_flow: f64,
    /// The pressure of the wave the exhaust valve's flow launches into the runner, c m' / A,
    /// integrated since the last audio sample (Pa s; negative for flow back into the
    /// cylinder).
    exhaust_wave: f64,
    /// Moles through the exhaust valve since the last audio sample (out positive).
    exhaust_moles: f64,
    pressure_history: [f64; STATE_SAMPLES],
    piston_speed_history: [f64; STATE_SAMPLES],
    /// Combustion heat released so far (J).
    released: f64,
    /// Swept volume (m^3).
    swept: f64,
    /// The charge at intake valve closing: pressure, volume, temperature (for Woschni's
    /// motored pressure); and whether the intake valve was open in the last step.
    ivc: (f64, f64, f64),
    intake_open: bool,
    /// (V_ivc / V)^1.32, Woschni's motored compression, for the step (the piston holds
    /// still over its substeps).
    motored_ratio: f64,
    /// 0.013 bore^-0.2, Woschni's constant part.
    woschni_bore: f64,
}

/// Inputs read by the cylinders during a step that live elsewhere in the engine.
struct Context<'a> {
    /// Woschni's heat transfer to walls at this temperature (K); none: engine-sim's.
    wall_temperature: Option<f64>,
    /// With physics beyond engine-sim's, the crank speed (rad/s) the mean piston speed comes
    /// from; none: engine-sim's average of the cycle's sampled piston speeds.
    crank_speed: Option<f64>,
    banks: &'a [Bank],
    heads: &'a [Head],
    fuel: &'a Fuel,
    turbulence: &'a Function,
    crank_angle: f64,
    cycle_angle: f64,
    /// The exhaust valves open into duct.rs's pipes (the engine steps them), not the runners.
    ducted: bool,
    /// The flame speed's cycle-to-cycle variation (standard deviation), if any.
    cycle_variation: Option<f64>,
}

impl Cylinder {
    fn volume(&self, ctx: &Context) -> f64 {
        let head = &ctx.heads[self.bank];
        let bank = &ctx.banks[self.bank];
        let area = bank.bore_surface_area();
        let s = self.piston_s;
        let sweep = area * (bank.deck_height - s - self.compression_height);
        sweep + head.chamber_volume - self.displacement
    }

    fn piston_speed(&self) -> f64 {
        self.piston_v
    }

    /// The mean piston speed: 2 x stroke x revolutions per second. engine-sim averages the
    /// piston speed sampled into 256 slots over the cycle, which reads low when a cycle
    /// takes a whole number of steps short of 256 (160 at 7500 rpm): the slots it never
    /// lands on keep stale values, and the burn, which scales with it, slows.
    fn piston_speed_mean(&self, ctx: &Context) -> f64 {
        match ctx.crank_speed {
            Some(omega) => 2.0 * (self.swept / self.cross_section_area) * omega.abs() / (2.0 * PI),
            None => self.mean_piston_speed(),
        }
    }

    fn mean_piston_speed(&self) -> f64 {
        let mut avg = 0.0;
        for v in &self.piston_speed_history {
            avg += v;
        }
        avg / STATE_SAMPLES as f64
    }

    fn ignite(&mut self, ctx: &Context, rng: &mut Rng) {
        if self.lit {
            return;
        }
        let mix = self.system.mix();
        if mix.p_fuel == 0.0 {
            return;
        }
        let afr = mix.p_o2 / mix.p_fuel;
        let equivalence_ratio = afr / ctx.fuel.molecular_afr;
        if !(0.5..=1.9).contains(&equivalence_ratio) {
            return;
        }
        let ideal_inert = mix.p_o2 / 0.7;
        let dilution = (mix.p_inert / ideal_inert) - 1.0;

        self.flame.last_volume = self.volume(ctx);
        self.flame.travel_x = 0.0;
        self.flame.travel_y = 0.0;
        self.flame.lit_n = 0.0;
        self.flame.total_n = self.system.n();
        self.flame.percentage_lit = 0.0;
        self.flame.global_mix = mix;
        self.lit = true;

        let fuel = ctx.fuel;
        let turbulence = ctx.turbulence.sample_triangle(self.piston_speed_mean(ctx));
        let mixing_factor = 1.0
            - (clamp01(turbulence / fuel.max_turbulence_effect)
                * clamp01(1.0 - dilution / fuel.max_dilution_effect));
        let randomness = fuel.burning_efficiency_randomness;
        let rand_s = fuel.low_efficiency_attenuation
            * ((1.0 - randomness) + randomness * rng.unit());
        let efficiency_attenuation = mixing_factor * rand_s + (1.0 - mixing_factor);
        self.flame.efficiency = efficiency_attenuation * fuel.max_burning_efficiency;
        self.flame.flame_speed =
            fuel.flame_speed(turbulence, afr, self.system.temperature(), self.system.pressure());
        if let Some(sigma) = ctx.cycle_variation {
            // Box and Muller's normal deviate.
            let u1 = rng.unit().max(1e-12);
            let u2 = rng.unit();
            let n = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            self.flame.flame_speed *= (1.0 + sigma * n).max(0.2);
        }
    }

    /// Woschni's heat transfer coefficient (W/m^2 K; Woschni, SAE 670931, 1967, in SI units
    /// as Klein, Linköping 2004, gives it): from the gas pressure and temperature and a gas
    /// speed from the piston's mean speed plus, once the valves are shut, the pressure the
    /// burn adds over compression alone.
    fn woschni(&self, piston: f64) -> f64 {
        let p = self.system.pressure();
        let t = self.system.temperature();
        if p <= 0.0 || t <= 0.0 {
            return 0.0;
        }
        let closed = self.intake_flow_rate == 0.0 && self.exhaust_flow_rate == 0.0;
        let w = if closed {
            let (p_ivc, v_ivc, t_ivc) = self.ivc;
            // The motored pressure: the charge compressed polytropically from IVC.
            let motored = p_ivc * self.motored_ratio;
            2.28 * piston + 3.24e-3 * self.swept * t_ivc / (p_ivc * v_ivc) * (p - motored).max(0.0)
        } else {
            6.18 * piston
        };
        if w <= 0.0 {
            return 0.0;
        }
        self.woschni_bore * fastmath::exp(0.8 * fastmath::ln(p * w) - 0.55 * fastmath::ln(t))
    }

    fn update(&mut self, ctx: &Context) {
        let v = self.volume(ctx);
        self.system.set_volume(v);

        // updateCycleStates
        let mut crank_angle = ctx.cycle_angle;
        if !crank_angle.is_finite() {
            crank_angle = 0.0;
        }
        let i = ((crank_angle / (4.0 * PI)) * (STATE_SAMPLES as f64 - 1.0)).round() as usize;
        self.piston_speed_history[i] = self.piston_speed().abs();
        self.pressure_history[i] = self.system.pressure();

        let head = &ctx.heads[self.bank];
        self.intake_flow_rate = head
            .intake_port_flow
            .sample_triangle(head.intake_cam.valve_lift(self.index, ctx.crank_angle));
        self.exhaust_flow_rate = head
            .exhaust_port_flow
            .sample_triangle(head.exhaust_cam.valve_lift(self.index, ctx.crank_angle));
        let intake_open = self.intake_flow_rate > 0.0;
        if self.intake_open && !intake_open {
            self.ivc = (self.system.pressure(), v, self.system.temperature());
        }
        self.intake_open = intake_open;
        if ctx.wall_temperature.is_some() && !intake_open && self.exhaust_flow_rate == 0.0 {
            self.motored_ratio = fastmath::pow(self.ivc.1 / v, 1.32);
        }
    }

    fn flow(&mut self, dt: f64, ctx: &Context, intake: &mut Intake, exhaust: &mut Exhaust) {
        let head = &ctx.heads[self.bank];
        let bank = &ctx.banks[self.bank];

        let volume = self.volume(ctx);
        let cylinder_height = volume / self.cross_section_area;
        let cylinder_surface_area =
            cylinder_height * PI * bank.bore + self.cross_section_area * 2.0;

        match ctx.wall_temperature {
            None => {
                let dt_temp = celcius(90.0) - self.system.temperature();
                self.system.change_energy(dt_temp * cylinder_surface_area * 100.0 * dt);
            }
            Some(wall) => {
                let h = self.woschni(self.piston_speed_mean(ctx));
                let t = self.system.temperature();
                self.system.change_energy(-h * cylinder_surface_area * (t - wall) * dt);
            }
        }

        self.system.flow_env(self.blowby_k, dt, CRANKCASE_PRESSURE, celcius(25.0), &Mix::default());

        let mut params = FlowParameters {
            k_flow: self.manifold_to_runner_flow_rate,
            dt,
            direction_x: 1.0,
            direction_y: 0.0,
            cross_section_area_0: intake.cross_section_area,
            cross_section_area_1: head.intake_runner_area,
        };
        GasSystem::flow(&params, &mut intake.system, &mut self.intake_runner);
        self.intake_runner.dissipate_excess_velocity();

        params.k_flow = self.intake_flow_rate;
        params.cross_section_area_0 = head.intake_runner_area;
        params.cross_section_area_1 = volume / cylinder_height;
        let intake_flow = GasSystem::flow(&params, &mut self.intake_runner, &mut self.system);
        self.intake_runner.dissipate_excess_velocity();
        self.system.dissipate_excess_velocity();

        // With ducts, the engine exchanges the exhaust valve's flow with them after this.
        if !ctx.ducted {
            params.k_flow = self.exhaust_flow_rate;
            params.cross_section_area_0 = volume / cylinder_height;
            params.cross_section_area_1 = head.exhaust_runner_area;
            let exhaust_flow = GasSystem::flow(&params, &mut self.system, &mut self.exhaust_runner);
            self.system.dissipate_excess_velocity();
            self.exhaust_runner.dissipate_excess_velocity();

            params.k_flow = self.primary_to_collector_flow_rate;
            params.cross_section_area_0 = head.exhaust_runner_area;
            params.cross_section_area_1 = exhaust.collector_cross_section_area;
            GasSystem::flow(&params, &mut self.exhaust_runner, &mut exhaust.system);
            self.exhaust_runner.update_velocity(dt, exhaust.velocity_decay);

            self.exhaust_flow = exhaust_flow;
            self.exhaust_moles += exhaust_flow;
            // A plane wave's pressure is rho c times its particle velocity: for a mass flow m'
            // into a pipe of area A, c m' / A.
            self.exhaust_wave +=
                exhaust_flow * AIR_MOLECULAR_MASS * self.exhaust_runner.c() / head.exhaust_runner_area;
        }

        self.intake_runner.update_velocity(dt, intake.velocity_decay);
        self.system.update_velocity(dt, 0.5);

        if intake_flow.abs() > 1e-9 && self.lit {
            self.lit = false;
        }

        if self.lit {
            let total_travel_x = bank.bore / 2.0;
            let total_travel_y = volume / bank.bore_surface_area();
            let expansion = volume / self.flame.last_volume;
            let last_travel_x = self.flame.travel_x;
            let last_travel_y = self.flame.travel_y * expansion;
            let flame_speed = self.flame.flame_speed;

            self.flame.travel_x = (last_travel_x + dt * flame_speed).min(total_travel_x);
            self.flame.travel_y = (last_travel_y + dt * flame_speed).min(total_travel_y);

            if last_travel_x < self.flame.travel_x || last_travel_y < self.flame.travel_y {
                let burned_volume =
                    self.flame.travel_x * self.flame.travel_x * PI * self.flame.travel_y;
                let prev_burned_volume = last_travel_x * last_travel_x * PI * last_travel_y;
                let lit_volume = burned_volume - prev_burned_volume;
                let n = (lit_volume / volume) * self.system.n();
                let global_mix = self.flame.global_mix;
                let fuel_burned = self.system.react(n * self.flame.efficiency, &global_mix);
                let mass_fuel_burned = fuel_burned * ctx.fuel.molecular_mass;
                self.system.change_energy(mass_fuel_burned * ctx.fuel.energy_density);
                self.released += mass_fuel_burned * ctx.fuel.energy_density;
                self.flame.lit_n += n;
                self.flame.percentage_lit += lit_volume / volume;
            } else {
                self.lit = false;
            }
            self.flame.last_volume = volume;
        }
    }
}

/// The piston's friction against the cylinder wall (engine-sim's calculateFrictionForce).
fn friction_force(v_s: f64, wall_force: f64) -> f64 {
    const FRICTION_COEFF: f64 = 0.06;
    const BREAKAWAY_FRICTION: f64 = 50.0;
    const BREAKAWAY_VELOCITY: f64 = 0.1;
    const VISCOUS: f64 = 20.0;
    let f_coul = FRICTION_COEFF * wall_force;
    let v_st = BREAKAWAY_VELOCITY * ROOT_2;
    let v_coul = BREAKAWAY_VELOCITY / 10.0;
    let f_brk = BREAKAWAY_FRICTION;
    let v = v_s.abs();
    let f_0 = ROOT_2 * E * (f_brk - f_coul);
    let f_1 = v / v_st;
    let f_2 = (-f_1 * f_1).exp() * f_1;
    let f_3 = f_coul * (v / v_coul).tanh();
    let f_4 = VISCOUS * v;
    f_0 * f_2 + f_3 + f_4
}

#[inline]
fn clamp01(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x
    }
}

fn positive_mod(mut x: f64, m: f64) -> f64 {
    if x < 0.0 {
        x = (-x / m).ceil() * m + x;
    }
    x % m
}

struct Ignition {
    timing: Function,
    full_load: Option<crate::spec::FullLoadTiming>,
    /// Manifold pressure as the ECU reads it (a ~10 ms average of the plenum's), for the
    /// full-load blend.
    map: f64,
    /// Retard requested on top of the map (rad): the ECU's fast torque path.
    retard: f64,
    rev_limit: f64,
    limiter_duration: f64,
    rev_limit_timer: f64,
    last_cycle_angle: f64,
    enabled: bool,
    /// Fraction of sparks skipped, spread evenly (traction control's torque cut).
    cut: f64,
    cut_debt: f64,
}

/// The engine and everything it drives, stepped at the simulation frequency.
pub struct Engine {
    pub mech: Mechanism,
    banks: Vec<Bank>,
    heads: Vec<Head>,
    cylinders: Vec<Cylinder>,
    intakes: Vec<Intake>,
    exhausts: Vec<Exhaust>,
    fuel: Fuel,
    turbulence: Function,
    ignition: Ignition,
    rng: Rng,
    crank_tdc: f64,
    throttle_gamma: f64,
    throttle_position: f64,
    speed_control: f64,
    timestep: f64,
    fluid_steps: usize,
    filtered_rpm: f64,
    /// Exhaust signal per exhaust system, one value per step (the synthesizer's input), kept
    /// while `collect_audio` is set.
    pub audio: Vec<Vec<f32>>,
    pub collect_audio: bool,
    /// The sound's source is the exhaust valves' volume velocity (AudioSource::Flow).
    audio_flow: bool,
    /// The audio is voice.rs's control stream (voice_channels).
    voice: bool,
    /// Each exhaust's temperature averaged over MEAN_FLOW_TIME (for the voice).
    back_temperature: Vec<f64>,
    /// Moles drawn through the throttles over the step, and that flow (kg/s) averaged.
    air_moles: f64,
    air_mass_flow: f64,
    /// Torque on the driven side (the load body), e.g. from the tyres.
    load_torque: f64,
    /// Sum of the clutch torque (forwards, on the load) over the steps since `set_load`.
    clutch_torque_sum: f64,
    clutch_steps: u32,
    staging: Vec<f64>,
    /// Moles out of the exhausts' outlets over the step, and that flow (kg/s) averaged over
    /// MEAN_FLOW_TIME (for the voice).
    exhaust_moles: f64,
    exhaust_mass_flow: f64,
    dyno_torque_samples: Vec<f64>,
    last_dyno_torque_sample: usize,
    pub impulse_responses: Vec<(String, f64)>,
    pub audio_params: (f64, f64, f64),
    turbo: Option<Turbo>,
    /// Physics beyond engine-sim's in use (any of the spec's `physics`).
    physical: bool,
    /// Each exhaust's wastegate, fully open (flow constant).
    wastegate_flow_k: f64,
    wall_temperature: Option<f64>,
    /// The exhaust as duct.rs's pipes (spec.ducts), in place of the runners and collectors.
    pub ducts: Option<crate::duct::Ducts>,
    cycle_variation: Option<f64>,
}

impl Engine {
    /// With the crank mechanism in reduced coordinates (crank.rs).
    pub fn new(spec: &EngineSpec) -> Engine {
        Engine::with_mechanism(spec, true)
    }

    /// With engine-sim's constrained rigid bodies, as engine-sim computes it.
    pub fn constrained(spec: &EngineSpec) -> Engine {
        Engine::with_mechanism(spec, false)
    }

    fn with_mechanism(spec: &EngineSpec, reduced: bool) -> Engine {
        let cs = &spec.crankshaft;
        let mech = Mechanism::new(spec, reduced);

        let banks: Vec<Bank> = spec
            .banks
            .iter()
            .map(|b| Bank {
                bore: b.bore,
                deck_height: b.deck_height,
                dx: (b.angle + PI / 2.0).cos(),
                dy: (b.angle + PI / 2.0).sin(),
                x: b.x,
                y: b.y,
            })
            .collect();
        let heads: Vec<Head> = spec
            .banks
            .iter()
            .map(|b| Head {
                chamber_volume: b.head.chamber_volume,
                intake_runner_volume: b.head.intake_runner_volume,
                intake_runner_area: b.head.intake_runner_area,
                exhaust_runner_volume: b.head.exhaust_runner_volume,
                exhaust_runner_area: b.head.exhaust_runner_area,
                intake_port_flow: b.head.intake_port_flow.clone(),
                exhaust_port_flow: b.head.exhaust_port_flow.clone(),
                intake_cam: Camshaft::new(&b.head.intake_cam),
                exhaust_cam: Camshaft::new(&b.head.exhaust_cam),
            })
            .collect();

        let intakes: Vec<Intake> = spec
            .intakes
            .iter()
            .map(|s| {
                let mut system = GasSystem::default();
                system.initialize(ATM, s.plenum_volume, celcius(25.0));
                system.set_geometry(s.plenum_area.sqrt(), s.plenum_volume / s.plenum_area, 1.0, 0.0);
                let mut atmosphere = GasSystem::default();
                atmosphere.initialize(ATM, 1000.0, celcius(25.0));
                atmosphere.set_geometry(100.0, 100.0, 1.0, 0.0);
                Intake {
                    system,
                    atmosphere,
                    throttle: 1.0,
                    flow: 0.0,
                    cross_section_area: s.plenum_area,
                    input_flow_k: s.flow_k,
                    idle_flow_k: s.idle_flow_k,
                    runner_flow_rate: s.runner_flow_k,
                    molecular_afr: s.molecular_afr,
                    idle_throttle_plate_position: s.idle_throttle_plate,
                    runner_length: s.runner_length,
                    velocity_decay: s.velocity_decay,
                    charge_pressure: ATM,
                    charge_temperature: celcius(25.0),
                }
            })
            .collect();

        let exhausts: Vec<Exhaust> = spec
            .exhausts
            .iter()
            .map(|s| {
                let mut system = GasSystem::default();
                system.initialize(ATM, s.collector_area * s.length, celcius(25.0));
                system.set_geometry(s.length, s.collector_area.sqrt(), 1.0, 0.0);
                let mut atmosphere = GasSystem::default();
                atmosphere.initialize(ATM, 1000.0, celcius(25.0));
                atmosphere.set_geometry(10.0, 10.0, 1.0, 0.0);
                Exhaust {
                    system,
                    atmosphere,
                    length: s.length,
                    primary_tube_length: s.primary_tube_length,
                    collector_cross_section_area: s.collector_area,
                    primary_flow_rate: s.primary_flow_k,
                    outlet_flow_rate: s.outlet_flow_k,
                    turbine_flow_rate: s.outlet_flow_k,
                    audio_volume: s.audio_volume,
                    velocity_decay: s.velocity_decay,
                    outflow: 0.0,
                }
            })
            .collect();

        // Cylinders, as engine-sim's script builds them: pistons and chambers before placing.
        let mut cylinders = Vec::new();
        let mut firing = spec.ignition.firing.iter();
        for (bank_index, b) in spec.banks.iter().enumerate() {
            let bank = &banks[bank_index];
            let head = &heads[bank_index];
            for (index, c) in b.cylinders.iter().enumerate() {
                let intake = &intakes[c.intake];
                let exhaust = &exhausts[c.exhaust];
                let bore_r = bank.bore / 2.0;
                let cross_section_area = PI * bore_r * bore_r;
                let width = cross_section_area.sqrt();
                // The piston is still at the origin: engine-sim sizes the chamber from there.
                let s0 = (0.0 - bank.x) * bank.dx + (0.0 - bank.y) * bank.dy;
                let initial_volume = bank.bore_surface_area()
                    * (bank.deck_height - s0 - c.piston.compression_height)
                    + head.chamber_volume
                    - c.piston.displacement;
                let mut chamber = GasSystem::default();
                chamber.set_geometry(width, initial_volume / cross_section_area, 1.0, 0.0);

                let intake_runner_width = head.intake_runner_area.sqrt();
                let manifold_runner_volume = head.intake_runner_area * intake.runner_length;
                let total_intake_runner_volume = head.intake_runner_volume + manifold_runner_volume;
                let overall_intake_runner_length = total_intake_runner_volume / head.intake_runner_area;
                let mut intake_runner = GasSystem::default();
                intake_runner.initialize(ATM, total_intake_runner_volume, celcius(25.0));
                intake_runner.set_geometry(overall_intake_runner_length, intake_runner_width, 1.0, 0.0);

                let exhaust_runner_width = head.exhaust_runner_area.sqrt();
                let exhaust_tube_length = exhaust.primary_tube_length + c.primary_length;
                let exhaust_tube_volume = head.exhaust_runner_area * exhaust_tube_length;
                let total_exhaust_runner_volume = head.exhaust_runner_volume + exhaust_tube_volume;
                let overall_exhaust_runner_length = total_exhaust_runner_volume / head.exhaust_runner_area;
                let mut exhaust_runner = GasSystem::default();
                exhaust_runner.initialize(ATM, total_exhaust_runner_volume, celcius(25.0));
                exhaust_runner.set_geometry(overall_exhaust_runner_length, exhaust_runner_width, 1.0, 0.0);

                let exhaust_length = c.primary_length + exhaust.length;
                cylinders.push(Cylinder {
                    bank: bank_index,
                    index,
                    piston_s: 0.0,
                    piston_v: 0.0,
                    compression_height: c.piston.compression_height,
                    displacement: c.piston.displacement,
                    blowby_k: c.piston.blowby,
                    intake: c.intake,
                    exhaust: c.exhaust,
                    primary_length: c.primary_length,
                    sound_attenuation: c.sound_attenuation,
                    firing_angle: *firing.next().expect("a firing angle per cylinder"),
                    ignition_event: false,
                    delay: DelayFilter::new(exhaust_length / 343.0, 10000.0),
                    system: chamber,
                    intake_runner,
                    exhaust_runner,
                    flame: FlameEvent { efficiency: 1.0, ..FlameEvent::default() },
                    lit: false,
                    intake_flow_rate: 0.0,
                    exhaust_flow_rate: 0.0,
                    manifold_to_runner_flow_rate: intake.runner_flow_rate,
                    primary_to_collector_flow_rate: exhaust.primary_flow_rate,
                    cross_section_area,
                    exhaust_flow: 0.0,
                    exhaust_wave: 0.0,
                    exhaust_moles: 0.0,
                    pressure_history: [0.0; STATE_SAMPLES],
                    piston_speed_history: [0.0; STATE_SAMPLES],
                    released: 0.0,
                    swept: cross_section_area * 2.0 * cs.throw,
                    ivc: (ATM, 1.0, celcius(25.0)),
                    intake_open: false,
                    motored_ratio: 1.0,
                    woschni_bore: 0.013 * bank.bore.powf(-0.2),
                });
            }
        }

        let mut turbulence = Function::new(1.0);
        for i in 0..30 {
            let s = i as f64;
            turbulence.add_sample(s, s * 0.5);
        }

        let mut engine = Engine {
            mech,
            banks,
            heads,
            cylinders,
            intakes,
            exhausts,
            fuel: Fuel::new(&spec.fuel),
            turbulence,
            ignition: Ignition {
                timing: spec.ignition.timing.clone(),
                full_load: spec.ignition.full_load.clone(),
                map: ATM,
                retard: 0.0,
                rev_limit: spec.ignition.rev_limit,
                limiter_duration: spec.ignition.limiter_duration,
                rev_limit_timer: 0.0,
                last_cycle_angle: 0.0,
                enabled: false,
                cut: 0.0,
                cut_debt: 0.0,
            },
            rng: Rng::new(1),
            crank_tdc: cs.tdc,
            throttle_gamma: spec.throttle_gamma,
            throttle_position: 1.0,
            speed_control: 0.0,
            timestep: 1.0 / spec.simulation_frequency,
            fluid_steps: spec.fluid_steps,
            filtered_rpm: 0.0,
            audio: vec![Vec::new(); audio_channels(spec)],
            collect_audio: true,
            audio_flow: spec.audio.source == crate::spec::AudioSource::Flow,
            voice: spec.audio.voice.is_some(),
            back_temperature: vec![celcius(25.0); spec.exhausts.len()],
            air_moles: 0.0,
            air_mass_flow: 0.0,
            load_torque: 0.0,
            clutch_torque_sum: 0.0,
            clutch_steps: 0,
            staging: vec![0.0; audio_channels(spec)],
            exhaust_moles: 0.0,
            exhaust_mass_flow: 0.0,
            dyno_torque_samples: vec![0.0; 512],
            last_dyno_torque_sample: 0,
            impulse_responses: spec
                .exhausts
                .iter()
                .map(|e| (e.impulse_response.clone(), e.impulse_volume))
                .collect(),
            audio_params: (spec.audio.hf_gain, spec.audio.noise, spec.audio.jitter),
            turbo: spec.turbo.as_ref().map(|t| Turbo::new(t, spec.physics.real_gas)),
            wastegate_flow_k: spec.turbo.as_ref().map_or(0.0, |t| t.wastegate_flow_k),
            wall_temperature: spec.physics.wall_temperature,
            physical: spec.physics.real_gas || spec.physics.wall_temperature.is_some(),
            cycle_variation: spec.physics.cycle_variation,
            ducts: spec.ducts.as_ref().map(|d| {
                assert!(spec.physics.real_gas, "ducts need physics.realGas (the gases must agree)");
                let n = spec.banks.iter().map(|b| b.cylinders.len()).sum();
                crate::duct::Ducts::new(d, n).unwrap_or_else(|e| panic!("ducts: {e}"))
            }),
        };

        // Chambers filled with air at the placed pistons.
        engine.update_pistons();
        for i in 0..engine.cylinders.len() {
            let v = {
                let ctx = engine.context();
                engine.cylinders[i].volume(&ctx)
            };
            engine.cylinders[i].system.initialize(ATM, v, celcius(25.0));
        }
        if spec.physics.real_gas {
            for i in &mut engine.intakes {
                i.system.set_real(true);
                i.atmosphere.set_real(true);
            }
            for e in &mut engine.exhausts {
                e.system.set_real(true);
                e.atmosphere.set_real(true);
            }
            for c in &mut engine.cylinders {
                for s in [&mut c.system, &mut c.intake_runner, &mut c.exhaust_runner] {
                    s.set_real(true);
                }
            }
        }
        engine.ignition.last_cycle_angle = engine.cycle_angle();
        engine
    }

    fn context(&self) -> Context<'_> {
        Context {
            wall_temperature: self.wall_temperature,
                crank_speed: self.physical.then(|| self.mech.omega()),
            banks: &self.banks,
            heads: &self.heads,
            fuel: &self.fuel,
            turbulence: &self.turbulence,
            crank_angle: self.crank_angle(),
            cycle_angle: self.cycle_angle(),
            ducted: self.ducts.is_some(),
            cycle_variation: self.cycle_variation,
        }
    }

    fn update_pistons(&mut self) {
        for (i, c) in self.cylinders.iter_mut().enumerate() {
            (c.piston_s, c.piston_v) = self.mech.piston(i);
        }
    }

    fn crank_angle(&self) -> f64 {
        self.mech.theta() - self.crank_tdc
    }

    /// The crank's angle (engine-sim's crank body's).
    pub fn crank_theta(&self) -> f64 {
        self.mech.theta()
    }

    /// Position in the four-stroke cycle, 0..4 pi.
    fn cycle_angle(&self) -> f64 {
        let wrapped = (-self.crank_angle() + 0.0) % (4.0 * PI);
        if wrapped < 0.0 {
            wrapped + 4.0 * PI
        } else {
            wrapped
        }
    }

    /// Crank speed (rad/s; engine-sim's engines turn negative).
    pub fn crank_speed(&self) -> f64 {
        self.mech.omega()
    }

    pub fn set_crank_speed(&mut self, v: f64) {
        self.mech.set_omega(v);
        self.update_pistons();
    }

    pub fn rpm(&self) -> f64 {
        to_rpm(self.crank_speed()).abs()
    }

    /// Pedal 0..1.
    pub fn set_speed_control(&mut self, s: f64) {
        self.speed_control = s;
        self.throttle_position = 1.0 - s.powf(self.throttle_gamma);
    }

    pub fn speed_control(&self) -> f64 {
        self.speed_control
    }

    pub fn set_ignition(&mut self, on: bool) {
        self.ignition.enabled = on;
    }

    /// Skips this fraction of the sparks (0..1), spread evenly over the cylinders: a torque
    /// cut that leaves the throttle, and a turbo's boost, as they are.
    pub fn set_spark_cut(&mut self, cut: f64) {
        self.ignition.cut = cut.clamp(0.0, 1.0);
    }

    /// Retards the spark from the map by `retard` (rad): the ECU's fast torque reduction.
    pub fn set_spark_retard(&mut self, retard: f64) {
        self.ignition.retard = retard.max(0.0);
    }

    /// Caps the turbocharger's boost target (gauge Pa), as an ECU limiting torque does.
    pub fn set_boost_limit(&mut self, limit: f64) {
        if let Some(t) = &mut self.turbo {
            t.limit = limit;
        }
    }

    pub fn set_dyno(&mut self, enabled: bool, hold: bool, speed: f64) {
        let c = &mut self.mech.controls;
        (c.dyno_enabled, c.dyno_hold, c.dyno_speed) = (enabled, hold, speed);
    }

    pub fn set_starter(&mut self, enabled: bool) {
        self.mech.controls.starter_enabled = enabled;
    }

    /// Torque the dyno applies to the crank (from the last step).
    pub fn dyno_torque(&self) -> f64 {
        let t = self.mech.dyno_torque();
        if self.crank_speed() > 0.0 {
            -t
        } else {
            t
        }
    }

    /// Dyno torque averaged over the last engine cycle.
    pub fn filtered_dyno_torque(&self) -> f64 {
        self.dyno_torque_samples.iter().sum::<f64>() / self.dyno_torque_samples.len() as f64
    }

    /// The driven side, seen from the crank, for the steps that follow: its inertia, its
    /// speed and the torque on it (forwards positive), and the torque the clutch can pass.
    /// Starts a new average of the clutch torque.
    pub fn set_load(&mut self, inertia: f64, speed: f64, torque: f64, clutch_capacity: f64) {
        let b = self.mech.load_mut();
        b.i = inertia;
        b.m = 1.0;
        b.v_theta = -speed;
        self.load_torque = -torque;
        self.set_clutch(clutch_capacity);
        self.clutch_torque_sum = 0.0;
        self.clutch_steps = 0;
    }

    /// The clutch torque on the driven side (forwards positive), averaged over the steps
    /// since `set_load`.
    pub fn clutch_torque(&self) -> f64 {
        if self.clutch_steps == 0 {
            0.0
        } else {
            self.clutch_torque_sum / self.clutch_steps as f64
        }
    }

    /// The driven side's speed (forwards positive).
    pub fn load_speed(&self) -> f64 {
        -self.mech.load().v_theta
    }

    pub fn load_body(&self) -> &Body {
        self.mech.load()
    }

    pub fn set_clutch(&mut self, max_torque: f64) {
        self.mech.controls.clutch_capacity = max_torque;
    }

    pub fn set_drag(&mut self, max_torque: f64) {
        self.mech.controls.drag = max_torque;
    }

    pub fn timestep(&self) -> f64 {
        self.timestep
    }

    pub fn cylinder_pressure(&self, i: usize) -> f64 {
        self.cylinders[i].system.pressure()
    }

    pub fn cylinder_volume(&self, i: usize) -> f64 {
        self.cylinders[i].system.volume()
    }

    /// Combustion heat released in all cylinders so far (J).
    pub fn combustion_energy(&self) -> f64 {
        self.cylinders.iter().map(|c| c.released).sum()
    }

    /// The share of cylinder `i`'s charge the flame has reached since its last spark (0..1;
    /// it holds its last value once the burn ends).
    pub fn cylinder_burned(&self, i: usize) -> f64 {
        let f = &self.cylinders[i].flame;
        if f.total_n > 0.0 {
            f.lit_n / f.total_n
        } else {
            0.0
        }
    }

    /// The lumped exhaust collector's pressure (without ducts), and its wastegate's opening.
    pub fn exhaust_pressure(&self, i: usize) -> f64 {
        self.exhausts[i].system.pressure()
    }

    /// Air drawn through the throttles (kg/s, averaged over ~0.03 s; kept with a voice).
    pub fn air_mass_flow(&self) -> f64 {
        self.air_mass_flow
    }

    pub fn wastegate(&self) -> f64 {
        self.turbo.as_ref().map_or(0.0, |t| t.wastegate)
    }

    pub fn exhaust_runner_pressure(&self, i: usize) -> f64 {
        self.cylinders[i].exhaust_runner.pressure()
    }

    pub fn exhaust_runner_temperature(&self, i: usize) -> f64 {
        self.cylinders[i].exhaust_runner.temperature()
    }

    /// Pressure behind the throttle, in the (first) intake plenum.
    pub fn manifold_pressure(&self) -> f64 {
        self.intakes.first().map_or(ATM, |i| i.system.pressure())
    }

    /// The turbocharger's boost over the atmosphere (Pa) and shaft speed (rad/s); zero
    /// without one.
    pub fn turbo_state(&self) -> (f64, f64) {
        self.turbo.as_ref().map_or((0.0, 0.0), |t| (t.charge_pressure - ATM, t.omega))
    }

    /// One simulation step (Simulator::simulateStep).
    pub fn step(&mut self) {
        let dt = self.timestep;

        // The mechanism, with the gas pressures and the wall friction as forces on the pistons.
        let mut forces = [0.0; 16];
        let forces = &mut forces[..self.cylinders.len()];
        for (i, c) in self.cylinders.iter().enumerate() {
            let bank = &self.banks[c.bank];
            let area = (bank.bore * bank.bore / 4.0) * PI;
            let v_s = c.piston_v;
            let wall_force = self.mech.wall_force(i);
            let pressure_differential = c.system.pressure() - CRANKCASE_PRESSURE;
            let force = -area * pressure_differential;
            const LIMIT: f64 = 1e-3;
            let abs_v_s = v_s.abs().min(LIMIT);
            let attenuation = abs_v_s / LIMIT;
            let f = friction_force(v_s, wall_force) * attenuation;
            let f_fric = if v_s > 0.0 { -f } else { f };
            forces[i] = force + f_fric;
        }
        self.mech.step(dt, forces, self.load_torque);

        self.clutch_torque_sum += -self.mech.clutch_torque();
        self.clutch_steps += 1;

        // Engine::update: the throttle linkage.
        for intake in &mut self.intakes {
            intake.throttle = self.throttle_position;
        }

        // Filtered engine speed (engine-sim's, in rpm).
        let alpha = dt / (100.0 + dt);
        self.filtered_rpm = alpha * self.filtered_rpm + (1.0 - alpha) * self.rpm();

        // Keep the crank angle bounded.
        self.mech.wrap_theta();
        self.update_pistons();

        // Dyno torque over the cycle.
        let index = ((512.0 * self.cycle_angle() / (4.0 * PI)).floor() as usize).min(511);
        let torque = self.dyno_torque();
        self.dyno_torque_samples[index] = torque;
        if self.last_dyno_torque_sample != index {
            let step: isize = if self.crank_speed() <= 0.0 { 1 } else { -1 };
            let mut i = self.last_dyno_torque_sample as isize + step;
            while i != index as isize {
                if i >= 512 {
                    i = -1;
                } else if i < 0 {
                    i = 512;
                } else {
                    self.dyno_torque_samples[i as usize] = torque;
                }
                i += step;
            }
            self.last_dyno_torque_sample = index;
        }

        self.simulate_step(dt);
        self.write_audio();
    }

    fn update_ignition(&mut self, dt: f64) {
        let cycle_angle = self.cycle_angle();
        let v_theta = self.crank_speed();
        let manifold = self.intakes.first().map_or(ATM, |i| i.system.pressure());
        let ign = &mut self.ignition;
        if ign.full_load.is_some() {
            ign.map += (manifold - ign.map) * (dt / 0.01).min(1.0);
        }
        if ign.enabled && ign.rev_limit_timer == 0.0 {
            let four_pi = 4.0 * PI;
            let mut advance = ign.timing.sample_triangle(-v_theta);
            if let Some(full) = &ign.full_load {
                let load = ((ign.map - ATM) / (full.pressure - ATM).max(1.0)).clamp(0.0, 1.0);
                advance += load * (full.timing.sample_triangle(-v_theta) - advance);
            }
            advance -= ign.retard;
            for c in &mut self.cylinders {
                let mut adjusted_angle = positive_mod(c.firing_angle - advance, four_pi);
                let r0 = ign.last_cycle_angle;
                let mut r1 = cycle_angle;
                if v_theta < 0.0 {
                    if r1 < r0 {
                        r1 += four_pi;
                        adjusted_angle += four_pi;
                    }
                    if adjusted_angle >= r0 && adjusted_angle < r1 {
                        c.ignition_event = true;
                    }
                } else {
                    if r1 > r0 {
                        r1 -= four_pi;
                        adjusted_angle -= four_pi;
                    }
                    if adjusted_angle >= r1 && adjusted_angle < r0 {
                        c.ignition_event = true;
                    }
                }
                if c.ignition_event && ign.cut > 0.0 {
                    ign.cut_debt += ign.cut;
                    if ign.cut_debt >= 1.0 {
                        ign.cut_debt -= 1.0;
                        c.ignition_event = false;
                    }
                }
            }
        }
        ign.rev_limit_timer -= dt;
        if v_theta.abs() > ign.rev_limit {
            ign.rev_limit_timer = ign.limiter_duration;
        }
        if ign.rev_limit_timer < 0.0 {
            ign.rev_limit_timer = 0.0;
        }
        ign.last_cycle_angle = cycle_angle;
    }

    /// PistonEngineSimulator::simulateStep_: ignition, then the gas dynamics in substeps.
    fn simulate_step(&mut self, dt: f64) {
        self.update_ignition(dt);

        let mut cylinders = std::mem::take(&mut self.cylinders);
        {
            let (crank_angle, cycle_angle) = (self.crank_angle(), self.cycle_angle());
            let ctx = Context {
                wall_temperature: self.wall_temperature,
                crank_speed: self.physical.then(|| self.mech.omega()),
                banks: &self.banks,
                heads: &self.heads,
                fuel: &self.fuel,
                turbulence: &self.turbulence,
                crank_angle,
                cycle_angle,
                ducted: self.ducts.is_some(),
                cycle_variation: self.cycle_variation,
            };
            let rng = &mut self.rng;
            for c in cylinders.iter_mut() {
                if c.ignition_event {
                    c.ignite(&ctx, rng);
                }
                c.update(&ctx);
            }
        }

        let fluid_dt = dt / self.fluid_steps as f64;
        // What passes the throttles and leaves the exhausts over the step, for the turbo.
        let (mut air, mut outflow, mut outflow_energy) = (0.0, 0.0, 0.0);
        let mut turbine_work = 0.0;
        for _ in 0..self.fluid_steps {
            for e in self.exhausts.iter_mut().filter(|_| self.ducts.is_none()) {
                e.process(fluid_dt);
                self.exhaust_moles += e.outflow;
                if self.turbo.is_some() && e.outflow > 0.0 {
                    outflow += e.outflow;
                    outflow_energy += e.outflow * e.system.temperature();
                }
            }
            for i in &mut self.intakes {
                i.process(fluid_dt);
                air += i.flow;
            }
            let ctx = Context {
                wall_temperature: self.wall_temperature,
                crank_speed: self.physical.then(|| self.mech.omega()),
                banks: &self.banks,
                heads: &self.heads,
                fuel: &self.fuel,
                turbulence: &self.turbulence,
                crank_angle: 0.0,
                cycle_angle: 0.0,
                ducted: self.ducts.is_some(),
                cycle_variation: self.cycle_variation,
            };
            for c in cylinders.iter_mut() {
                let (intake, exhaust) = (&mut self.intakes[c.intake], &mut self.exhausts[c.exhaust]);
                c.flow(fluid_dt, &ctx, intake, exhaust);
            }
            if let Some(d) = &mut self.ducts {
                d.set_flaps(self.mech.omega().abs(), self.speed_control, fluid_dt);
                turbine_work += Self::step_ducts(d, &mut cylinders, fluid_dt);
                self.exhaust_moles += d.outlets.iter().map(|&l| d.net.links[l].flow.m).sum::<f64>() * fluid_dt / AIR_MOLECULAR_MASS;
            }
        }
        for c in cylinders.iter_mut() {
            c.ignition_event = false;
        }
        self.cylinders = cylinders;
        self.air_moles += air;

        if let Some(turbo) = &mut self.turbo {
            let n = self.exhausts.len() as f64;
            let exhaust_pressure = self.exhausts.iter().map(|e| e.system.pressure()).sum::<f64>() / n;
            let exhaust_temperature = if outflow > 0.0 { outflow_energy / outflow } else { 0.0 };
            let manifold = self.intakes.first().map_or(ATM, |i| i.system.pressure());
            // The flow splits between turbine and wastegate by their openings.
            let wastegate_k = self.wastegate_flow_k * turbo.wastegate;
            let turbine_share = self.exhausts.first().map_or(1.0, |e| {
                e.turbine_flow_rate / (e.turbine_flow_rate + wastegate_k).max(1e-30)
            });
            turbo.update(
                dt,
                &TurboInputs {
                    air: air / dt,
                    exhaust: outflow * turbine_share / dt,
                    exhaust_temperature,
                    exhaust_pressure,
                    manifold,
                    crank: self.mech.omega(),
                    turbine_power: self.ducts.as_ref().map(|_| turbine_work / dt),
                },
            );
            if let Some(d) = &mut self.ducts {
                for &(o, area) in &d.wastegates {
                    d.net.orifices[o].cda = area * turbo.wastegate;
                }
                for &o in &d.turbines {
                    d.net.orifices[o].efficiency = turbo.turbine_efficiency();
                }
            }
            for i in &mut self.intakes {
                i.charge_pressure = turbo.charge_pressure;
                i.charge_temperature = turbo.charge_temperature;
            }
            let wastegate_k = self.wastegate_flow_k * turbo.wastegate;
            for e in &mut self.exhausts {
                e.outlet_flow_rate = e.turbine_flow_rate + wastegate_k;
            }
        }
    }

    /// Exchanges the exhaust valves' flow with the ducts over a fluid substep: each
    /// cylinder's gas becomes its node's for the step (its volume held, as over a substep
    /// here), and what the step leaves there goes back. Returns the turbines' work (J).
    fn step_ducts(d: &mut crate::duct::Ducts, cylinders: &mut [Cylinder], dt: f64) -> f64 {
        const M: f64 = AIR_MOLECULAR_MASS;
        for (c, &(node, link)) in cylinders.iter().zip(&d.cylinders) {
            let (n, e) = c.system.totals();
            let mix = c.system.mix();
            let nd = &mut d.net.nodes[node];
            nd.volume = c.system.volume();
            nd.m = n * M;
            nd.e = e;
            nd.mf = n * M * mix.p_fuel;
            nd.mo = n * M * mix.p_o2;
            // The port's flow constant as an effective area (kArea's inverse).
            d.net.links[link].cda = c.exhaust_flow_rate * M.sqrt();
        }
        d.net.step(dt);
        for (c, &(node, _)) in cylinders.iter_mut().zip(&d.cylinders) {
            let nd = &d.net.nodes[node];
            let (n0, _) = c.system.totals();
            let n = nd.m / M;
            let (f, o) = (nd.mf / nd.m, nd.mo / nd.m);
            c.system.set_totals(n, nd.e, Mix { p_fuel: f, p_inert: 1.0 - f - o, p_o2: o });
            c.exhaust_flow = n0 - n;
            c.exhaust_moles += n0 - n;
        }
        d.listen();
        d.turbines.iter().map(|&o| d.net.work[o]).sum()
    }

    /// The exhaust pulses as the synthesizer's input (writeToSynthesizer).
    fn write_audio(&mut self) {
        if self.voice {
            return self.write_voice();
        }
        for s in &mut self.staging {
            *s = 0.0;
        }
        let attenuation = self.filtered_rpm.abs().min(40.0) / 40.0;
        let attenuation_3 = attenuation * attenuation * attenuation;
        let count = self.cylinders.len() as f64;
        let dt = self.timestep;
        for c in self.cylinders.iter_mut() {
            c.exhaust_moles = 0.0;
            let exhaust = &self.exhausts[c.exhaust];
            let exhaust_length = c.primary_length + exhaust.length;
            let runner = &c.exhaust_runner;
            let exhaust_flow = if self.audio_flow {
                attenuation_3 * 1600.0 * c.exhaust_wave / dt
            } else {
                attenuation_3
                    * 1600.0
                    * (1.0 * (runner.pressure() - ATM)
                        + 0.1 * runner.dynamic_pressure(1.0, 0.0)
                        + 0.1 * runner.dynamic_pressure(-1.0, 0.0))
            };
            c.exhaust_wave = 0.0;
            let delayed = c.delay.f(exhaust_flow);
            self.staging[c.exhaust] += c.sound_attenuation
                * (exhaust.audio_volume * delayed / count)
                * (1.0 / (exhaust_length * exhaust_length));
        }
        self.exhaust_moles = 0.0;
        if self.collect_audio {
            for (out, s) in self.audio.iter_mut().zip(&self.staging) {
                out.push(*s as f32);
            }
        }
    }

    /// voice.rs's control stream for the step: each cylinder's exhaust valve flow, then the
    /// state the voice's other sources follow (crate::voice::Global).
    fn write_voice(&mut self) {
        let dt = self.timestep;
        let mean = (dt / MEAN_FLOW_TIME).min(1.0);
        for (i, e) in self.exhausts.iter().enumerate() {
            self.back_temperature[i] += (e.system.temperature() - self.back_temperature[i]) * mean;
        }
        // Each valve's flow over the step, as the volume velocity the exhaust's acoustics
        // take: they are linear, at the atmosphere's density, so the flow at that density
        // (at the gas's temperature), which launches the wave of the pressure the real flow
        // does, c m' / A, whatever the runner's mean pressure. (As a volume velocity at the
        // runner's pressure, a turbo engine's back-pressure would shrink its full-throttle
        // pulses by itself.) Its low frequencies are the simulation's prediction, waves in
        // the runners and all; the voice makes its high ones (voice.rs).
        for (i, c) in self.cylinders.iter_mut().enumerate() {
            let n = c.exhaust_moles / dt;
            c.exhaust_moles = 0.0;
            let t = if n > 0.0 { c.system.temperature() } else { c.exhaust_runner.temperature() };
            self.staging[i] = n * crate::units::R * t / ATM;
        }
        let flow = self.exhaust_moles * AIR_MOLECULAR_MASS / dt;
        self.exhaust_mass_flow += (flow - self.exhaust_mass_flow) * mean;
        self.exhaust_moles = 0.0;
        let air = self.air_moles * AIR_MOLECULAR_MASS / dt;
        self.air_mass_flow += (air - self.air_mass_flow) * mean;
        self.air_moles = 0.0;
        use crate::voice::Global as G;
        let manifold = self.manifold_pressure();
        let g = &mut self.staging[self.cylinders.len()..];
        let n = self.exhausts.len() as f64;
        g[G::CrankSpeed as usize] = self.mech.omega().abs();
        g[G::ExhaustFlow as usize] = self.exhaust_mass_flow;
        g[G::ExhaustTemperature as usize] = self.back_temperature.iter().sum::<f64>() / n;
        g[G::AirFlow as usize] = self.air_mass_flow;
        g[G::Manifold as usize] = manifold;
        g[G::Throttle as usize] = self.speed_control;
        let (charge, shaft, vented, wastegate) =
            self.turbo.as_ref().map_or((ATM, 0.0, 0.0, 0.0), |t| (t.charge_pressure, t.omega, t.vented(), t.wastegate));
        g[G::Charge as usize] = charge;
        g[G::TurboSpeed as usize] = shaft;
        g[G::Vented as usize] = vented;
        g[G::Wastegate as usize] = wastegate;
        if self.collect_audio {
            for (out, s) in self.audio.iter_mut().zip(&self.staging) {
                out.push(*s as f32);
            }
        }
    }

    /// The last step's exhaust signal per exhaust system, at full precision.
    pub fn staging(&self) -> &[f64] {
        &self.staging
    }
}

/// Averaging time of the voice's mean flows and temperature: a few engine cycles.
const MEAN_FLOW_TIME: f64 = 0.03;

/// The engine's audio channels: with a voice, its control stream (each cylinder's exhaust
/// valve flow, then crate::voice::GLOBALS of state); else engine-sim's, one per exhaust
/// system.
pub fn audio_channels(spec: &EngineSpec) -> usize {
    if spec.audio.voice.is_some() {
        spec.banks.iter().map(|b| b.cylinders.len()).sum::<usize>() + crate::voice::GLOBALS
    } else {
        spec.exhausts.len()
    }
}
