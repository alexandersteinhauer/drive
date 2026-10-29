//! The piston engine and its simulation step (engine-sim's Engine, PistonEngineSimulator and
//! the parts they own: crankshaft, pistons, rods, heads and camshafts, combustion chambers,
//! intakes, exhausts, ignition, fuel).

use crate::function::Function;
use crate::gas::{FlowParameters, GasSystem, Mix};
use crate::rng::Rng;
use crate::solver::{Body, Constraint, Kind, State, System};
use crate::spec::{CamSpec, EngineSpec, FuelSpec};
use crate::units::{celcius, to_rpm, ATM, E, PI, ROOT_2};
use std::collections::VecDeque;

const STATE_SAMPLES: usize = 256;
const CRANKCASE_PRESSURE: f64 = ATM;

struct Bank {
    angle: f64,
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
        self.atmosphere.reset(ATM, celcius(25.0), fuel_air_mix);
        self.flow = GasSystem::flow(&params, &mut self.atmosphere, &mut self.system);

        self.atmosphere.reset(ATM, celcius(25.0), fuel_mix);
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
    audio_volume: f64,
    velocity_decay: f64,
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
        GasSystem::flow(&params, &mut self.atmosphere, &mut self.system);
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
    piston_body: usize,
    wall_constraint: usize,
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
    pressure_history: [f64; STATE_SAMPLES],
    piston_speed_history: [f64; STATE_SAMPLES],
}

/// Inputs read by the cylinders during a step that live elsewhere in the engine.
struct Context<'a> {
    bodies: &'a [Body],
    banks: &'a [Bank],
    heads: &'a [Head],
    fuel: &'a Fuel,
    turbulence: &'a Function,
    crank_angle: f64,
    cycle_angle: f64,
}

impl Cylinder {
    fn volume(&self, ctx: &Context) -> f64 {
        let head = &ctx.heads[self.bank];
        let bank = &ctx.banks[self.bank];
        let body = &ctx.bodies[self.piston_body];
        let area = bank.bore_surface_area();
        let s = (body.p_x - bank.x) * bank.dx + (body.p_y - bank.y) * bank.dy;
        let sweep = area * (bank.deck_height - s - self.compression_height);
        sweep + head.chamber_volume - self.displacement
    }

    fn piston_speed(&self, ctx: &Context) -> f64 {
        let bank = &ctx.banks[self.bank];
        let body = &ctx.bodies[self.piston_body];
        body.v_x * bank.dx + body.v_y * bank.dy
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
        let turbulence = ctx.turbulence.sample_triangle(self.mean_piston_speed());
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
        self.piston_speed_history[i] = self.piston_speed(ctx).abs();
        self.pressure_history[i] = self.system.pressure();

        let head = &ctx.heads[self.bank];
        self.intake_flow_rate = head
            .intake_port_flow
            .sample_triangle(head.intake_cam.valve_lift(self.index, ctx.crank_angle));
        self.exhaust_flow_rate = head
            .exhaust_port_flow
            .sample_triangle(head.exhaust_cam.valve_lift(self.index, ctx.crank_angle));
    }

    fn flow(&mut self, dt: f64, ctx: &Context, intake: &mut Intake, exhaust: &mut Exhaust) {
        let head = &ctx.heads[self.bank];
        let bank = &ctx.banks[self.bank];

        let volume = self.volume(ctx);
        let cylinder_height = volume / self.cross_section_area;
        let cylinder_surface_area =
            cylinder_height * PI * bank.bore + self.cross_section_area * 2.0;

        let dt_temp = celcius(90.0) - self.system.temperature();
        self.system.change_energy(dt_temp * cylinder_surface_area * 100.0 * dt);

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

        self.intake_runner.update_velocity(dt, intake.velocity_decay);
        self.system.update_velocity(dt, 0.5);
        self.exhaust_runner.update_velocity(dt, exhaust.velocity_decay);

        if intake_flow.abs() > 1e-9 && self.lit {
            self.lit = false;
        }

        self.exhaust_flow = exhaust_flow;

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
    rev_limit: f64,
    limiter_duration: f64,
    rev_limit_timer: f64,
    last_cycle_angle: f64,
    enabled: bool,
}

/// The engine and everything it drives, stepped at the simulation frequency.
pub struct Engine {
    pub system: System,
    banks: Vec<Bank>,
    heads: Vec<Head>,
    cylinders: Vec<Cylinder>,
    intakes: Vec<Intake>,
    exhausts: Vec<Exhaust>,
    fuel: Fuel,
    turbulence: Function,
    ignition: Ignition,
    rng: Rng,
    crank_body: usize,
    crank_tdc: f64,
    load_body: usize,
    clutch: usize,
    drag: usize,
    dyno: usize,
    starter: usize,
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
    /// Torque on the driven side (the load body), e.g. from the tyres.
    load_torque: f64,
    /// Sum of the clutch torque (forwards, on the load) over the steps since `set_load`.
    clutch_torque_sum: f64,
    clutch_steps: u32,
    staging: Vec<f64>,
    dyno_torque_samples: Vec<f64>,
    last_dyno_torque_sample: usize,
    pub impulse_responses: Vec<(String, f64)>,
    pub audio_params: (f64, f64, f64),
}

impl Engine {
    pub fn new(spec: &EngineSpec) -> Engine {
        let mut system = System::default();
        let cs = &spec.crankshaft;

        // Crankshaft: fixed in place, with bearing friction.
        let crank_body = system.add_body(Body {
            p_x: cs.x,
            p_y: cs.y,
            m: cs.mass + cs.flywheel_mass,
            i: cs.moment_of_inertia,
            ..Body::default()
        });
        system.add_constraint(Constraint::new(
            Kind::FixedPosition { local_x: 0.0, local_y: 0.0, world_x: cs.x, world_y: cs.y },
            &[crank_body],
        ));
        system.add_constraint(Constraint::new(
            Kind::RotationFriction { min_torque: -cs.friction_torque, max_torque: cs.friction_torque },
            &[crank_body],
        ));

        // What the engine drives, through the clutch: one rotating body.
        let load_body = 1;
        let clutch = system.add_constraint(Constraint::new(
            Kind::Clutch { min_torque: 0.0, max_torque: 0.0 },
            &[crank_body, load_body],
        ));
        let drag = system.add_constraint(Constraint::new(Kind::Drag { max_torque: 0.0 }, &[load_body]));
        let lb = system.add_body(Body { m: 1.0, i: 1.0, ..Body::default() });
        debug_assert_eq!(lb, load_body);

        let banks: Vec<Bank> = spec
            .banks
            .iter()
            .map(|b| Bank {
                angle: b.angle,
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
                    audio_volume: s.audio_volume,
                    velocity_decay: s.velocity_decay,
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
                    piston_body: 0,
                    wall_constraint: 0,
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
                    pressure_history: [0.0; STATE_SAMPLES],
                    piston_speed_history: [0.0; STATE_SAMPLES],
                });
            }
        }

        // Pistons and rods.
        let mut i = 0;
        for (bank_index, b) in spec.banks.iter().enumerate() {
            let bank = &banks[bank_index];
            for c in &b.cylinders {
                let wrist_pin = c.piston.wrist_pin_position;
                let little_end = (c.rod.length / 2.0) - c.rod.center_of_mass;
                let big_end = -(c.rod.length / 2.0) + c.rod.center_of_mass;
                let journal_angle = cs.journals[c.journal];
                let (journal_x, journal_y) = (journal_angle.cos() * cs.throw, journal_angle.sin() * cs.throw);

                let wall = Kind::Line {
                    local_x: 0.0,
                    local_y: wrist_pin,
                    p0_x: bank.x,
                    p0_y: bank.y,
                    dx: (bank.angle + PI / 2.0).cos(),
                    dy: (bank.angle + PI / 2.0).sin(),
                };
                let piston_body = system.add_body(Body { m: c.piston.mass, i: 1.0, ..Body::default() });
                let rod_body = system.add_body(Body {
                    m: c.rod.mass,
                    i: c.rod.moment_of_inertia,
                    ..Body::default()
                });
                system.add_constraint(Constraint::new(
                    Kind::Link { local_x_1: 0.0, local_y_1: little_end, local_x_2: 0.0, local_y_2: wrist_pin },
                    &[rod_body, piston_body],
                ));
                system.add_constraint(Constraint::new(
                    Kind::Link { local_x_1: 0.0, local_y_1: big_end, local_x_2: journal_x, local_y_2: journal_y },
                    &[rod_body, crank_body],
                ));
                let wall_constraint = system.add_constraint(Constraint::new(wall, &[piston_body]));
                cylinders[i].piston_body = piston_body;
                cylinders[i].wall_constraint = wall_constraint;

                // placeCylinder
                let crank = system.bodies[crank_body];
                let (p_x, p_y) = (journal_x + crank.p_x, journal_y + crank.p_y);
                let a = bank.dx * bank.dx + bank.dy * bank.dy;
                let bb = -2.0 * bank.dx * (p_x - bank.x) - 2.0 * bank.dy * (p_y - bank.y);
                let cc = (p_x - bank.x) * (p_x - bank.x) + (p_y - bank.y) * (p_y - bank.y)
                    - c.rod.length * c.rod.length;
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
                        piston.theta = bank.angle + PI;
                    }
                }
                i += 1;
            }
        }

        let dyno = system.add_constraint(Constraint::new(
            Kind::Dynamometer {
                speed: 0.0,
                max_torque: 10000.0 * crate::units::FT_LB,
                hold: false,
                enabled: false,
            },
            &[crank_body],
        ));
        let starter = system.add_constraint(Constraint::new(
            Kind::Starter { speed: -spec.starter_speed, max_torque: spec.starter_torque, enabled: false },
            &[crank_body],
        ));

        let mut turbulence = Function::new(1.0);
        for i in 0..30 {
            let s = i as f64;
            turbulence.add_sample(s, s * 0.5);
        }

        let mut engine = Engine {
            system,
            banks,
            heads,
            cylinders,
            intakes,
            exhausts,
            fuel: Fuel::new(&spec.fuel),
            turbulence,
            ignition: Ignition {
                timing: spec.ignition.timing.clone(),
                rev_limit: spec.ignition.rev_limit,
                limiter_duration: spec.ignition.limiter_duration,
                rev_limit_timer: 0.0,
                last_cycle_angle: 0.0,
                enabled: false,
            },
            rng: Rng::new(1),
            crank_body,
            crank_tdc: cs.tdc,
            load_body,
            clutch,
            drag,
            dyno,
            starter,
            throttle_gamma: spec.throttle_gamma,
            throttle_position: 1.0,
            speed_control: 0.0,
            timestep: 1.0 / spec.simulation_frequency,
            fluid_steps: spec.fluid_steps,
            filtered_rpm: 0.0,
            audio: vec![Vec::new(); spec.exhausts.len()],
            collect_audio: true,
            load_torque: 0.0,
            clutch_torque_sum: 0.0,
            clutch_steps: 0,
            staging: vec![0.0; spec.exhausts.len()],
            dyno_torque_samples: vec![0.0; 512],
            last_dyno_torque_sample: 0,
            impulse_responses: spec
                .exhausts
                .iter()
                .map(|e| (e.impulse_response.clone(), e.impulse_volume))
                .collect(),
            audio_params: (spec.audio.hf_gain, spec.audio.noise, spec.audio.jitter),
        };

        // Chambers filled with air at the placed pistons.
        for i in 0..engine.cylinders.len() {
            let v = {
                let ctx = engine.context();
                engine.cylinders[i].volume(&ctx)
            };
            engine.cylinders[i].system.initialize(ATM, v, celcius(25.0));
        }
        engine.ignition.last_cycle_angle = engine.cycle_angle();
        engine
    }

    fn context(&self) -> Context<'_> {
        Context {
            bodies: &self.system.bodies,
            banks: &self.banks,
            heads: &self.heads,
            fuel: &self.fuel,
            turbulence: &self.turbulence,
            crank_angle: self.crank_angle(),
            cycle_angle: self.cycle_angle(),
        }
    }

    fn crank_angle(&self) -> f64 {
        self.system.bodies[self.crank_body].theta - self.crank_tdc
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
        self.system.bodies[self.crank_body].v_theta
    }

    pub fn set_crank_speed(&mut self, v: f64) {
        self.system.bodies[self.crank_body].v_theta = v;
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

    pub fn set_dyno(&mut self, enabled: bool, hold: bool, speed: f64) {
        if let Kind::Dynamometer { speed: s, hold: h, enabled: e, .. } = &mut self.system.constraints[self.dyno].kind {
            *s = speed;
            *h = hold;
            *e = enabled;
        }
    }

    pub fn set_starter(&mut self, enabled: bool) {
        if let Kind::Starter { enabled: e, .. } = &mut self.system.constraints[self.starter].kind {
            *e = enabled;
        }
    }

    /// Torque the dyno applies to the crank (from the last step).
    pub fn dyno_torque(&self) -> f64 {
        let c = &self.system.constraints[self.dyno];
        if self.crank_speed() > 0.0 {
            -c.f_t[0][0]
        } else {
            c.f_t[0][0]
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
        let b = &mut self.system.bodies[self.load_body];
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
        -self.system.bodies[self.load_body].v_theta
    }

    pub fn load_body(&self) -> &Body {
        &self.system.bodies[self.load_body]
    }

    pub fn set_clutch(&mut self, max_torque: f64) {
        if let Kind::Clutch { min_torque, max_torque: m } = &mut self.system.constraints[self.clutch].kind {
            *min_torque = -max_torque;
            *m = max_torque;
        }
    }

    pub fn set_drag(&mut self, max_torque: f64) {
        if let Kind::Drag { max_torque: m } = &mut self.system.constraints[self.drag].kind {
            *m = max_torque;
        }
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

    pub fn exhaust_runner_pressure(&self, i: usize) -> f64 {
        self.cylinders[i].exhaust_runner.pressure()
    }

    /// One simulation step (Simulator::simulateStep).
    pub fn step(&mut self) {
        let dt = self.timestep;

        // Rigid bodies, with the gas pressures as forces on the pistons.
        {
            // Wall reaction forces from the last solve, for the piston friction.
            let wall_forces: Vec<f64> = self
                .cylinders
                .iter()
                .map(|c| {
                    let wall = &self.system.constraints[c.wall_constraint];
                    (wall.f_x[0][0] * wall.f_x[0][0] + wall.f_y[0][0] * wall.f_y[0][0]).sqrt()
                })
                .collect();
            let cylinders = &self.cylinders;
            let banks = &self.banks;
            let (load_body, load_torque) = (self.load_body, self.load_torque);
            self.system.process(dt, |state: &mut State| {
                state.t[load_body] += load_torque;
                for (c, &wall_force) in cylinders.iter().zip(&wall_forces) {
                    let bank = &banks[c.bank];
                    let area = (bank.bore * bank.bore / 4.0) * PI;
                    let b = c.piston_body;
                    let v_s = state.v_x[b] * bank.dx + state.v_y[b] * bank.dy;
                    let pressure_differential = c.system.pressure() - CRANKCASE_PRESSURE;
                    let force = -area * pressure_differential;
                    const LIMIT: f64 = 1e-3;
                    let abs_v_s = v_s.abs().min(LIMIT);
                    let attenuation = abs_v_s / LIMIT;
                    let f = friction_force(v_s, wall_force) * attenuation;
                    let f_fric = if v_s > 0.0 { -f } else { f };
                    state.apply_force((force + f_fric) * bank.dx, (force + f_fric) * bank.dy, b);
                }
            });
        }

        self.clutch_torque_sum += -self.system.constraints[self.clutch].f_t[0][1];
        self.clutch_steps += 1;

        // Engine::update: the throttle linkage.
        for intake in &mut self.intakes {
            intake.throttle = self.throttle_position;
        }

        // Filtered engine speed (engine-sim's, in rpm).
        let alpha = dt / (100.0 + dt);
        self.filtered_rpm = alpha * self.filtered_rpm + (1.0 - alpha) * self.rpm();

        // Keep the crank angle bounded.
        let crank = &mut self.system.bodies[self.crank_body];
        crank.theta %= 4.0 * PI;

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
        let ign = &mut self.ignition;
        if ign.enabled && ign.rev_limit_timer == 0.0 {
            let four_pi = 4.0 * PI;
            let advance = ign.timing.sample_triangle(-v_theta);
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
                bodies: &self.system.bodies,
                banks: &self.banks,
                heads: &self.heads,
                fuel: &self.fuel,
                turbulence: &self.turbulence,
                crank_angle,
                cycle_angle,
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
        for _ in 0..self.fluid_steps {
            for e in &mut self.exhausts {
                e.process(fluid_dt);
            }
            for i in &mut self.intakes {
                i.process(fluid_dt);
            }
            let ctx = Context {
                bodies: &self.system.bodies,
                banks: &self.banks,
                heads: &self.heads,
                fuel: &self.fuel,
                turbulence: &self.turbulence,
                crank_angle: 0.0,
                cycle_angle: 0.0,
            };
            for c in cylinders.iter_mut() {
                let (intake, exhaust) = (&mut self.intakes[c.intake], &mut self.exhausts[c.exhaust]);
                c.flow(fluid_dt, &ctx, intake, exhaust);
            }
        }
        for c in cylinders.iter_mut() {
            c.ignition_event = false;
        }
        self.cylinders = cylinders;
    }

    /// The exhaust pulses as the synthesizer's input (writeToSynthesizer).
    fn write_audio(&mut self) {
        for s in &mut self.staging {
            *s = 0.0;
        }
        let attenuation = self.filtered_rpm.abs().min(40.0) / 40.0;
        let attenuation_3 = attenuation * attenuation * attenuation;
        let count = self.cylinders.len() as f64;
        for c in &mut self.cylinders {
            let exhaust = &self.exhausts[c.exhaust];
            let exhaust_length = c.primary_length + exhaust.length;
            let runner = &c.exhaust_runner;
            let exhaust_flow = attenuation_3
                * 1600.0
                * (1.0 * (runner.pressure() - ATM)
                    + 0.1 * runner.dynamic_pressure(1.0, 0.0)
                    + 0.1 * runner.dynamic_pressure(-1.0, 0.0));
            let delayed = c.delay.f(exhaust_flow);
            self.staging[c.exhaust] += c.sound_attenuation
                * (exhaust.audio_volume * delayed / count)
                * (1.0 / (exhaust_length * exhaust_length));
        }
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
