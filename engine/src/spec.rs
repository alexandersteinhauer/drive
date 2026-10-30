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
    /// A turbocharger feeding every intake from every exhaust (turbo.rs); none: aspirated.
    #[serde(default)]
    pub turbo: Option<TurboSpec>,
    /// Physics beyond engine-sim's (all off: engine-sim's own, step for step).
    #[serde(default)]
    pub physics: PhysicsSpec,
    /// The exhaust as pipes and volumes for duct.rs's gas dynamics, in place of engine-sim's
    /// lumped runners and collectors (the reference simulation; needs `physics.real_gas`).
    #[serde(default)]
    pub ducts: Option<DuctSpec>,
}

/// An exhaust (duct.rs) of pipes between named nodes: "cylinder N" (1-based: its exhaust
/// valve, opening as the port's flow-bench curve has it), "atmosphere" (an open end,
/// radiating), "closed" (a wall), or one of `volumes`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctSpec {
    pub pipes: Vec<DuctPipeSpec>,
    #[serde(default)]
    pub volumes: Vec<DuctVolumeSpec>,
    #[serde(default)]
    pub orifices: Vec<DuctOrificeSpec>,
    #[serde(default)]
    pub turbines: Vec<DuctTurbineSpec>,
    #[serde(default)]
    pub perforations: Vec<DuctPerforationSpec>,
    #[serde(default)]
    pub flaps: Vec<DuctFlapSpec>,
    #[serde(default)]
    pub nozzles: Vec<DuctNozzlesSpec>,
    /// The listener's distance from the open ends (m).
    pub listener_distance: f64,
    /// The highest frequency the cells resolve (Hz): 30 cells to its wavelength at each
    /// pipe's temperature.
    #[serde(default = "default_resolution")]
    pub resolution: f64,
}

fn default_resolution() -> f64 {
    3000.0
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctPipeSpec {
    /// A name for perforations to refer to it by.
    #[serde(default)]
    pub name: Option<String>,
    pub a: String,
    pub b: String,
    /// One section's (none with `sections`).
    #[serde(default)]
    pub length: f64,
    /// Hydraulic diameter: a round pipe's own (at end `a`).
    #[serde(default)]
    pub diameter: f64,
    /// A cone's diameter at end `b`, the diameter running linearly from `diameter`.
    #[serde(default)]
    pub diameter_b: Option<f64>,
    /// Flow area; none: a round pipe's. A catalyst's monolith: its channels' open area,
    /// `diameter` theirs.
    #[serde(default)]
    pub area: Option<f64>,
    /// Square channels (a monolith's): their laminar friction and heat transfer.
    #[serde(default)]
    pub square: bool,
    /// A packing's flow resistivity (Pa s/m^2).
    #[serde(default)]
    pub flow_resistivity: f64,
    /// The gas's temperature at the start (K), and what the cells are sized for.
    pub temperature: f64,
    /// The wall's temperature (K); none: adiabatic.
    #[serde(default)]
    pub wall_temperature: Option<f64>,
    /// Wall roughness (m); none: drawn steel tube's.
    #[serde(default)]
    pub roughness: Option<f64>,
    /// Stretches end to end from `a` to `b`, each its own shape (straight, cone, monolith,
    /// packed), in place of the fields above but `temperature`, which sizes the cells and
    /// gives the walls' temperature and roughness where a section doesn't.
    #[serde(default)]
    pub sections: Vec<DuctSectionSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctSectionSpec {
    pub length: f64,
    pub diameter: f64,
    #[serde(default)]
    pub diameter_b: Option<f64>,
    #[serde(default)]
    pub area: Option<f64>,
    #[serde(default)]
    pub square: bool,
    #[serde(default)]
    pub wall_temperature: Option<f64>,
    #[serde(default)]
    pub flow_resistivity: f64,
    #[serde(default)]
    pub roughness: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctVolumeSpec {
    pub node: String,
    pub volume: f64,
    pub temperature: f64,
}

/// An opening between two nodes, of `area` and discharge coefficient `discharge`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctOrificeSpec {
    pub a: String,
    pub b: String,
    pub area: f64,
    pub discharge: f64,
}

/// A perforated stretch of pipe `pipe`'s wall, `from` to `to` (m along it), opening onto a
/// node or a pipe (named), alongside from `outer_from` along it.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctPerforationSpec {
    pub pipe: String,
    pub from: f64,
    pub to: f64,
    pub outer: String,
    #[serde(default)]
    pub outer_from: f64,
    /// Open fraction of the wall, hole diameter, wall thickness.
    pub porosity: f64,
    pub hole_diameter: f64,
    pub thickness: f64,
}

/// A flap (a silencer's bypass valve) at end `end` ("a" or "b") of pipe `pipe`, where it
/// meets its node: shut, a wall; opening with the crank speed from `speed[0]` to `speed[1]`
/// (rad/s) or the pedal from `throttle[0]` to `throttle[1]`, whichever opens it more (an
/// ECU's flap logic), its actuator taking ~0.15 s.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctFlapSpec {
    pub pipe: String,
    pub end: String,
    pub speed: [f64; 2],
    pub throttle: [f64; 2],
}

/// A turbine from node `a` to node `b`: a nozzle of effective area `area` taking the turbo's
/// turbine efficiency of the isentropic work of what passes (with `nozzles` from volutes into
/// `a`, the rotor of a two-nozzle turbine), and beside it a wastegate of effective area
/// `wastegate` when fully open, opened by the boost controller (turbo.rs).
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctTurbineSpec {
    pub a: String,
    pub b: String,
    pub area: f64,
    pub wastegate: f64,
    /// A two-nozzle turbine's rotor: the nodes whose mean pressure is the stage's inlet (its
    /// work the stage's isentropic work from there); none: its own drop.
    #[serde(default)]
    pub stage: Vec<String>,
    /// The wastegate's ports (in the volutes), its area shared between them; none: from `a`.
    #[serde(default)]
    pub wastegate_from: Vec<String>,
    /// The rotor's gas slug length (its blade passages and end corrections, m); none: a thin
    /// plate's end corrections.
    #[serde(default)]
    pub length: Option<f64>,
}

/// Nozzles from a stretch of pipe `pipe` (`from` to `to`, m) into `node`, `area` the total
/// effective area spread evenly over its cells: a turbine volute's stator channels.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuctNozzlesSpec {
    pub pipe: String,
    pub from: f64,
    pub to: f64,
    pub node: String,
    pub area: f64,
    /// The channels' gas slug length (passage and end corrections, m).
    pub length: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsSpec {
    /// Real gases (gas.rs): heat capacity ratio falling with temperature, and flows carrying
    /// enthalpy. Without: engine-sim's constant 1.4 and internal energy only.
    #[serde(default)]
    pub real_gas: bool,
    /// Woschni's heat transfer from the cylinder gas to walls at this mean temperature (K).
    /// Without: engine-sim's fixed coefficient towards 90 C walls.
    #[serde(default)]
    pub wall_temperature: Option<f64>,
    /// Cycle-to-cycle variation of combustion: each burn's flame speed scaled by a normal
    /// random factor of this standard deviation (the early flame's growth, set by the
    /// turbulence and mixture at the spark, varies from cycle to cycle: Heywood 9.4).
    #[serde(default)]
    pub cycle_variation: Option<f64>,
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

/// A turbocharger (turbo.rs); twin turbos as one, with their flows and inertias summed.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurboSpec {
    /// Compressor wheel tip radius (m).
    pub compressor_radius: f64,
    /// Rotating assemblies' moment of inertia (kg m^2).
    pub inertia: f64,
    pub compressor_efficiency: f64,
    pub turbine_efficiency: f64,
    /// Gauge boost the wastegate holds (Pa), by crank speed (rad/s).
    pub boost: Function,
    /// Bearing losses, W per (rad/s)^2 of shaft speed.
    pub friction: f64,
    /// Intercooler effectiveness, 0..1.
    pub intercooler: f64,
    /// Air the compressor pumps through the open blow-off valve at full boost speed (kg/s).
    pub vent_flow: f64,
    /// Flow constant of each exhaust's wastegate, fully open; the turbine is the exhaust's
    /// outlet (`outlet_flow_k`).
    pub wastegate_flow_k: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IgnitionSpec {
    /// Crank speed (rad/s) to advance (rad).
    pub timing: Function,
    /// The advance at full load, by crank speed, and the manifold pressure (Pa) it holds
    /// from: the advance blends from `timing` at atmospheric pressure to this, as an ECU's
    /// spark table by speed and load does (best-torque timing at part load, knock-limited
    /// under boost). None: `timing` alone, as in engine-sim.
    #[serde(default)]
    pub full_load: Option<FullLoadTiming>,
    pub rev_limit: f64,
    pub limiter_duration: f64,
    /// Firing angle per cylinder (in bank order).
    pub firing: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FullLoadTiming {
    pub timing: Function,
    pub pressure: f64,
}

/// The exhaust sound's source.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum AudioSource {
    /// engine-sim's: each exhaust runner's pressure above the atmosphere's. Without a turbo
    /// the runners sit near atmospheric, so this is mostly the pulses; with one they sit at
    /// the turbine's back-pressure, many times the pulses, and its swings (whenever
    /// combustion stops or resumes) come out as thumps.
    #[default]
    Pressure,
    /// The flow through each exhaust valve, as the pressure of the wave it launches into the
    /// runner, c m' / A (the plane-wave relation; Pa, on engine-sim's scale). Flow is what
    /// drives an exhaust's acoustics. Its steady part, the mean flow, makes no sound: a
    /// tailpipe radiates the rate of change of its volume velocity (a monopole). The computed
    /// acoustics include that (hf gain 0); through a recorded impulse response the
    /// synthesizer's derivative path takes it (engine-sim's hf gain, 0.01, makes it nearly
    /// all of the signal above ~16 Hz).
    Flow,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSpec {
    pub hf_gain: f64,
    pub noise: f64,
    pub jitter: f64,
    /// What the exhaust's sound is driven by.
    #[serde(default)]
    pub source: AudioSource,
    /// The exhaust's acoustics (acoustics.rs), for the voice (which they need). Without
    /// them, each exhaust system's pulses through its recorded impulse response.
    #[serde(default)]
    pub acoustics: Option<AcousticsSpec>,
    /// The sound as voice.rs makes it, in place of engine-sim's synthesizer: the engine then
    /// writes voice.rs's control stream (each exhaust valve's flow and the state the other
    /// sources follow) rather than an exhaust signal.
    #[serde(default)]
    pub voice: Option<VoiceSpec>,
}

/// voice.rs's sources beyond the exhaust valves' flow, and its output level. Each level is
/// a multiple of the source's physical estimate (voice.rs), 0 leaving it out.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSpec {
    /// Sound pressure (Pa) at the listener that is full scale.
    pub full_scale: f64,
    /// The exhaust's mean mass flow (kg/s) its acoustics' flow losses are computed for:
    /// one fixed damping (voice.rs explains why not the engine's own).
    #[serde(default)]
    pub acoustic_flow: f64,
    /// Cycle-to-cycle variation of the blowdowns beyond the simulation's (its combustion
    /// is the same every cycle but for the flame's): the standard deviation of each one's
    /// strength, relative.
    #[serde(default)]
    pub variation: f64,
    /// Turbulence of the exhaust valves' jets, carried through the exhaust's acoustics.
    #[serde(default)]
    pub port_noise: f64,
    /// Noise of the jets leaving the tailpipes (Lighthill's), with their pulsing velocity.
    #[serde(default)]
    pub tail_noise: f64,
    /// The engine bay, heard through its cover: the exhaust manifolds' and turbine
    /// housings' thin shells ringing with the pulses inside.
    #[serde(default)]
    pub bay: f64,
    /// Valves seating, and direct injectors ticking.
    #[serde(default)]
    pub valvetrain: f64,
    #[serde(default)]
    pub injectors: f64,
    /// Turbocharger: the compressor's broadband flow noise, its blade-pass whine, and the
    /// blow-off valve venting boost when the throttle shuts.
    #[serde(default)]
    pub whoosh: f64,
    #[serde(default)]
    pub whistle: f64,
    #[serde(default)]
    pub compressor_blades: f64,
    #[serde(default)]
    pub blow_off: f64,
    /// Scales recorded impulse responses (used without acoustics) to the computed level.
    #[serde(default = "one")]
    pub impulse_gain: f64,
}

/// The exhaust as an acoustic network of named junctions (acoustics.rs). Temperatures in K.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticsSpec {
    pub pipes: Vec<AcousticPipe>,
    #[serde(default)]
    pub orifices: Vec<AcousticOrifice>,
    #[serde(default)]
    pub volumes: Vec<AcousticVolume>,
    /// The junction behind each cylinder's exhaust valve, in cylinder order.
    pub ports: Vec<String>,
    pub outlets: Vec<AcousticOutlet>,
    pub listener_distance: f64,
    /// Scales the impulse responses (sets the level against engine-sim's synthesizer).
    pub gain: f64,
    /// Where the pulses turn nonlinear, for the voice (voice.rs) to steepen them there.
    #[serde(default)]
    pub nonlinear: Option<NonlinearSpec>,
    /// Valves (a silencer's bypass) an ECU opens. The voice blends the responses past
    /// `nonlinear.nodes` by them, so they need those.
    #[serde(default)]
    pub valves: Option<ValvesSpec>,
    /// A turbocharger's wastegates, fully open: orifices around the turbines (to the
    /// junctions of `nonlinear.nodes`, which they need), opened as the engine's wastegate is.
    #[serde(default)]
    pub wastegate: Vec<AcousticOrifice>,
}

/// What opening an exhaust's valves adds to its network (a silencer's bypass: a straight
/// pipe past its chambers), and when an ECU opens them: between the crank speeds (rad/s)
/// in `speed`, or the pedal positions in `throttle`, whichever is further. The voice blends
/// the responses closed and open.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValvesSpec {
    #[serde(default)]
    pub pipes: Vec<AcousticPipe>,
    #[serde(default)]
    pub orifices: Vec<AcousticOrifice>,
    pub speed: [f64; 2],
    pub throttle: [f64; 2],
}

impl AcousticsSpec {
    /// The network with its valves open (unchanged without any).
    pub fn opened(&self) -> AcousticsSpec {
        let mut s = self.clone();
        if let Some(v) = &self.valves {
            s.pipes.extend(v.pipes.iter().cloned());
            s.orifices.extend(v.orifices.iter().cloned());
        }
        s
    }

    /// The network with its wastegates `opening` (0..1) of the way open: their area so much
    /// of its full.
    pub fn with_wastegate(&self, opening: f64) -> AcousticsSpec {
        let mut s = self.clone();
        if opening > 0.0 {
            s.orifices.extend(self.wastegate.iter().map(|o| AcousticOrifice { area: o.area * opening, ..o.clone() }));
        }
        s
    }
}

/// Pipes in which the exhaust's pulses steepen as they travel, their crests outrunning
/// their troughs: from each exhaust system's junction in `nodes` (exhaust order; the only
/// way on from its cylinders), or from each port, along `length` of pipe at `temperature`
/// (0: not past the junctions, which then only split the responses, as for wastegates).
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NonlinearSpec {
    /// None: in each cylinder's own pipe, from its port (an exhaust without one junction
    /// all its pulses pass, as a turbine is).
    #[serde(default)]
    pub nodes: Vec<String>,
    pub length: f64,
    pub temperature: f64,
    /// With `nodes`, each cylinder's own pipe too, ahead of them (a turbo engine's
    /// primaries, where the pulses are largest).
    #[serde(default)]
    pub primaries: Option<PrimariesSpec>,
}

/// Each cylinder's pipe from its port, for its pulses to steepen in.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrimariesSpec {
    pub length: f64,
    pub diameter: f64,
    pub temperature: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticPipe {
    pub a: String,
    pub b: String,
    pub length: f64,
    pub diameter: f64,
    pub temperature: f64,
    /// Wall losses as a multiple of a smooth pipe's.
    #[serde(default = "one")]
    pub damping: f64,
    /// Absorption of a lining (a perforated pipe in packing), Np/m per kHz.
    #[serde(default)]
    pub absorption: f64,
    /// Losses alike at every frequency (Np/m): large pulses' to their own flow at bends and
    /// junctions, which linear acoustics leave out.
    #[serde(default)]
    pub loss: f64,
}

fn one() -> f64 {
    1.0
}

/// A perforated cone, baffle or cap: a mass of gas through `area` over `length` (with end
/// corrections), and a resistance in units of the gas's characteristic impedance there,
/// without flow; the flow through it adds its jet's (acoustics.rs).
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticOrifice {
    pub a: String,
    pub b: String,
    pub area: f64,
    pub length: f64,
    pub resistance: f64,
    pub temperature: f64,
    /// Discharge coefficient: the jet's area over `area` (0.6-0.7 for sharp holes).
    #[serde(default = "default_discharge")]
    pub discharge: f64,
}

fn default_discharge() -> f64 {
    0.65
}

/// A volume small against the wavelength, at a junction.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticVolume {
    pub node: String,
    pub volume: f64,
    pub temperature: f64,
}

/// An open pipe end radiating into the air.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticOutlet {
    pub node: String,
    pub diameter: f64,
}
