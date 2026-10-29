//! A turbocharger, which engine-sim doesn't have: a turbine in the exhaust driving a
//! compressor on one shaft, a wastegate holding the boost, a blow-off valve and an
//! intercooler. The game's own model, kept to the shaft's power balance:
//!
//! - The exhaust leaves each collector through the turbine and, beside it, the wastegate:
//!   two openings in parallel to the atmosphere, sharing the flow by their sizes. The turbine
//!   takes the isentropic work of its share (the gas's temperature, the pressure drop) at
//!   its efficiency. A boost controller opens the wastegate as the shaft passes the speed of
//!   the target boost, which also lowers the back-pressure the engine pushes against.
//! - The compressor's work per kg of air is Euler's, `SLIP * U^2` at the wheel's tip speed U;
//!   its pressure ratio follows from that work at its efficiency, and the power it takes is
//!   that work times the air the engine draws through the throttle.
//! - The shaft's inertia and bearing losses set how fast it spools and runs down: that is
//!   the turbo lag.
//! - The blow-off valve, held shut by a spring and the manifold's pressure, opens once the
//!   charge exceeds the manifold by more than the spring: with the throttle closing, it caps
//!   the charge there and vents the rest of the compressor's flow to the atmosphere.

use crate::function::Function;
use crate::spec::TurboSpec;
use crate::units::{celcius, ATM, AIR_MOLECULAR_MASS, R};

/// Work per kg as a fraction of the tip speed squared (radial wheel with backswept blades).
const SLIP: f64 = 0.9;
/// The gas's cp (J/kg K) and (gamma - 1) / gamma at `t`: engine-sim's diatomic gas (gamma
/// 1.4), or with real gases the engine's gamma(T) (gas.rs), which for hot exhaust lets the
/// turbine take ~20 % more work per kg than air's would.
fn properties(real: bool, t: f64) -> (f64, f64) {
    let gamma = if real { crate::gas::real_gamma(t) } else { 1.4 };
    (gamma / (gamma - 1.0) * R / AIR_MOLECULAR_MASS, (gamma - 1.0) / gamma)
}
/// The boost controller: wastegate opening per unit of relative shaft-speed error, and per
/// second of it (a PI controller; the actuator follows within ~0.1 s).
const WASTEGATE_P: f64 = 4.0;
const WASTEGATE_I: f64 = 30.0;
/// The blow-off valve's spring: how far the charge may exceed the manifold (Pa).
const BLOW_OFF: f64 = 0.35e5;

/// What the turbo sees of the engine over a step.
#[derive(Clone, Copy)]
pub struct TurboInputs {
    /// Air drawn through the throttle (mol/s).
    pub air: f64,
    /// Gas leaving the exhaust collectors through the turbine (mol/s), its temperature and
    /// their pressure.
    pub exhaust: f64,
    pub exhaust_temperature: f64,
    pub exhaust_pressure: f64,
    /// Pressure behind the throttle.
    pub manifold: f64,
    /// Crank speed (rad/s).
    pub crank: f64,
}

pub struct Turbo {
    radius: f64,
    inertia: f64,
    compressor_efficiency: f64,
    turbine_efficiency: f64,
    boost: Function,
    friction: f64,
    intercooler: f64,
    vent_flow: f64,
    /// Shaft speed at the highest target boost (rad/s), for the vented flow.
    full_speed: f64,
    /// Real-gas properties (with the engine's).
    real: bool,
    /// Shaft speed (rad/s).
    pub omega: f64,
    /// Wastegate opening, 0..1, and the controller's integral of the speed error.
    pub wastegate: f64,
    integral: f64,
    /// Cap on the boost target (gauge Pa).
    pub limit: f64,
    pub blow_off: bool,
    /// What the throttle sees: pressure (Pa) and temperature (K) after the intercooler.
    pub charge_pressure: f64,
    pub charge_temperature: f64,
}

impl Turbo {
    pub fn new(s: &TurboSpec, real: bool) -> Turbo {
        let mut t = Turbo {
            real,
            radius: s.compressor_radius,
            inertia: s.inertia,
            compressor_efficiency: s.compressor_efficiency,
            turbine_efficiency: s.turbine_efficiency,
            boost: s.boost.clone(),
            friction: s.friction,
            intercooler: s.intercooler,
            vent_flow: s.vent_flow,
            full_speed: 1.0,
            omega: 0.0,
            wastegate: 0.0,
            integral: 0.0,
            limit: f64::INFINITY,
            blow_off: false,
            charge_pressure: ATM,
            charge_temperature: celcius(25.0),
        };
        let max_boost = s.boost.samples.iter().fold(0.0f64, |m, &(_, b)| m.max(b));
        t.full_speed = t.speed_for(max_boost).max(1.0);
        t
    }

    /// Compressor work per kg of air at shaft speed `omega` (J/kg).
    fn work(&self, omega: f64) -> f64 {
        let u = omega * self.radius;
        SLIP * u * u
    }

    /// Pressure ratio the compressor makes with `work` per kg.
    fn pressure_ratio(&self, work: f64) -> f64 {
        let (cp, kappa) = properties(self.real, ambient_temperature());
        (1.0 + self.compressor_efficiency * work / (cp * ambient_temperature())).powf(1.0 / kappa)
    }

    /// Shaft speed at which the compressor makes `boost` (gauge Pa).
    fn speed_for(&self, boost: f64) -> f64 {
        let ratio = 1.0 + boost.max(0.0) / ATM;
        let (cp, kappa) = properties(self.real, ambient_temperature());
        let work = cp * ambient_temperature() * (ratio.powf(kappa) - 1.0) / self.compressor_efficiency;
        (work / SLIP).sqrt() / self.radius
    }

    /// Air the blow-off valve vents (kg/s), while it is open.
    pub fn vented(&self) -> f64 {
        if self.blow_off {
            self.vent_flow * self.omega / self.full_speed
        } else {
            0.0
        }
    }

    /// Advances the shaft by `dt`.
    pub fn update(&mut self, dt: f64, inputs: &TurboInputs) {
        let TurboInputs { air, exhaust, exhaust_temperature, exhaust_pressure, manifold, crank } =
            *inputs;
        let omega = self.omega;
        let work = self.work(omega);
        let ratio = self.pressure_ratio(work);

        // The blow-off valve: open while the throttle holds the manifold well below the charge.
        let relief = (manifold + BLOW_OFF).max(ATM);
        self.blow_off = ATM * ratio > relief;
        let mut air_mass = air.max(0.0) * AIR_MOLECULAR_MASS;
        if self.blow_off {
            air_mass += self.vent_flow * omega / self.full_speed;
        }
        let compressor = air_mass * work;
        let friction = self.friction * omega * omega;

        // The turbine's share, cut by the wastegate once the shaft reaches the target speed.
        let (cp_exhaust, kappa_exhaust) = properties(self.real, exhaust_temperature);
        let expansion = if exhaust_pressure > ATM {
            1.0 - (ATM / exhaust_pressure).powf(kappa_exhaust)
        } else {
            0.0
        };
        let available = self.turbine_efficiency
            * exhaust.max(0.0)
            * AIR_MOLECULAR_MASS
            * cp_exhaust
            * exhaust_temperature
            * expansion;
        let turbine = available;
        // The boost controller opens the wastegate as the shaft passes its target speed.
        let target = self.speed_for(self.boost.sample_triangle(crank.abs()).min(self.limit));
        let error = if target > 0.0 { omega / target - 1.0 } else { 1.0 };
        self.integral = (self.integral + WASTEGATE_I * error * dt).clamp(0.0, 1.0);
        self.wastegate = (WASTEGATE_P * error + self.integral).clamp(0.0, 1.0);

        let energy = 0.5 * self.inertia * omega * omega + (turbine - compressor - friction) * dt;
        self.omega = (2.0 * energy.max(0.0) / self.inertia).sqrt();

        let work = self.work(self.omega);
        let ratio = self.pressure_ratio(work);
        let t1 = ambient_temperature();
        let (_, kappa) = properties(self.real, t1);
        let t2 = t1 * (1.0 + (ratio.powf(kappa) - 1.0) / self.compressor_efficiency);
        self.charge_pressure = (ATM * ratio).min(relief);
        self.charge_temperature = t2 - self.intercooler * (t2 - t1);
    }
}

fn ambient_temperature() -> f64 {
    celcius(25.0)
}
