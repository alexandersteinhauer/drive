//! The description of an engine, as plain data (what engine-sim's .mr scripts evaluate to).
//! All values are SI: m, kg, s, rad, Pa, K, mol; flow constants as engine-sim's k_carb and
//! k_28inH2O produce them.

use crate::function::Function;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineSpec {
    /// Simulation steps per second.
    pub simulation_frequency: f64,
    /// Gas-dynamics substeps per simulation step.
    #[serde(default = "default_fluid_steps")]
    pub fluid_steps: usize,
    pub starter_torque: f64,
    pub starter_speed: f64,
    /// Pedal to throttle plate: plate = 1 - pedal^gamma.
    pub throttle_gamma: f64,
    pub fuel: FuelSpec,
    pub crankshaft: CrankshaftSpec,
    pub banks: Vec<BankSpec>,
    pub intakes: Vec<IntakeSpec>,
    pub exhausts: Vec<ExhaustSpec>,
    pub ignition: IgnitionSpec,
    pub audio: AudioSpec,
}

fn default_fluid_steps() -> usize {
    8
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FuelSpec {
    pub molecular_mass: f64,
    pub energy_density: f64,
    pub molecular_afr: f64,
    pub max_burning_efficiency: f64,
    pub burning_efficiency_randomness: f64,
    pub low_efficiency_attenuation: f64,
    pub max_turbulence_effect: f64,
    pub max_dilution_effect: f64,
    pub turbulence_to_flame_speed_ratio: Function,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrankshaftSpec {
    pub throw: f64,
    pub mass: f64,
    pub flywheel_mass: f64,
    pub moment_of_inertia: f64,
    pub friction_torque: f64,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    pub tdc: f64,
    /// Rod journal angles.
    pub journals: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BankSpec {
    pub angle: f64,
    pub bore: f64,
    pub deck_height: f64,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    pub head: HeadSpec,
    pub cylinders: Vec<CylinderSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadSpec {
    pub chamber_volume: f64,
    pub intake_runner_volume: f64,
    pub intake_runner_area: f64,
    pub exhaust_runner_volume: f64,
    pub exhaust_runner_area: f64,
    /// Valve lift (m) to flow constant.
    pub intake_port_flow: Function,
    pub exhaust_port_flow: Function,
    pub intake_cam: CamSpec,
    pub exhaust_cam: CamSpec,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CamSpec {
    #[serde(default)]
    pub advance: f64,
    /// Cam angle from the lobe centre to valve lift.
    pub profile: Function,
    /// Lobe centrelines (crank angle), one per cylinder of the bank.
    pub lobes: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CylinderSpec {
    /// Index of the rod journal on the crankshaft.
    pub journal: usize,
    pub piston: PistonSpec,
    pub rod: RodSpec,
    pub intake: usize,
    pub exhaust: usize,
    pub primary_length: f64,
    pub sound_attenuation: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PistonSpec {
    pub mass: f64,
    /// Blow-by flow constant.
    pub blowby: f64,
    pub compression_height: f64,
    #[serde(default)]
    pub wrist_pin_position: f64,
    #[serde(default)]
    pub displacement: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RodSpec {
    pub mass: f64,
    pub moment_of_inertia: f64,
    #[serde(default)]
    pub center_of_mass: f64,
    pub length: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntakeSpec {
    pub plenum_volume: f64,
    pub plenum_area: f64,
    pub flow_k: f64,
    pub idle_flow_k: f64,
    pub runner_flow_k: f64,
    pub molecular_afr: f64,
    pub idle_throttle_plate: f64,
    pub runner_length: f64,
    pub velocity_decay: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExhaustSpec {
    pub length: f64,
    pub collector_area: f64,
    pub outlet_flow_k: f64,
    pub primary_tube_length: f64,
    pub primary_flow_k: f64,
    pub velocity_decay: f64,
    pub audio_volume: f64,
    /// Name of the exhaust impulse response (see synth::impulse_response).
    pub impulse_response: String,
    pub impulse_volume: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnitionSpec {
    /// Crank speed (rad/s) to advance (rad).
    pub timing: Function,
    pub rev_limit: f64,
    pub limiter_duration: f64,
    /// Firing angle per cylinder (in bank order).
    pub firing: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSpec {
    pub hf_gain: f64,
    pub noise: f64,
    pub jitter: f64,
}
