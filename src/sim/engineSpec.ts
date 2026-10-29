/**
 * Combustion engines as data, for the simulated engine (engine/, a Rust port of engine-sim):
 * what engine-sim's .mr scripts evaluate to, in SI units (m, kg, s, rad, Pa, mol), with the
 * helpers those scripts use. Keep in sync with engine/src/spec.rs.
 */

/** A sampled function, read with a triangle filter of `filterRadius`. */
export interface SampledFunction {
  filterRadius: number;
  samples: [x: number, y: number][];
}

export interface EngineSpec {
  /** Simulation steps per second. */
  simulationFrequency: number;
  /** Gas-dynamics substeps per step (default 8). */
  fluidSteps?: number;
  starterTorque: number;
  starterSpeed: number;
  /** Pedal to throttle plate opening: 1 - (1 - pedal^gamma). */
  throttleGamma: number;
  fuel: {
    molecularMass: number;
    energyDensity: number;
    molecularAfr: number;
    maxBurningEfficiency: number;
    burningEfficiencyRandomness: number;
    lowEfficiencyAttenuation: number;
    maxTurbulenceEffect: number;
    maxDilutionEffect: number;
    turbulenceToFlameSpeedRatio: SampledFunction;
  };
  crankshaft: {
    throw: number;
    mass: number;
    flywheelMass: number;
    momentOfInertia: number;
    frictionTorque: number;
    x?: number;
    y?: number;
    tdc: number;
    /** Rod journal angles. */
    journals: number[];
  };
  banks: {
    angle: number;
    bore: number;
    deckHeight: number;
    x?: number;
    y?: number;
    head: {
      chamberVolume: number;
      intakeRunnerVolume: number;
      intakeRunnerArea: number;
      exhaustRunnerVolume: number;
      exhaustRunnerArea: number;
      /** Valve lift to flow constant. */
      intakePortFlow: SampledFunction;
      exhaustPortFlow: SampledFunction;
      intakeCam: CamSpec;
      exhaustCam: CamSpec;
    };
    cylinders: {
      /** Rod journal on the crankshaft. */
      journal: number;
      piston: {
        mass: number;
        /** Blow-by flow constant. */
        blowby: number;
        compressionHeight: number;
        wristPinPosition?: number;
        displacement?: number;
      };
      rod: {
        mass: number;
        momentOfInertia: number;
        centerOfMass?: number;
        length: number;
      };
      intake: number;
      exhaust: number;
      primaryLength: number;
      soundAttenuation: number;
    }[];
  }[];
  intakes: {
    plenumVolume: number;
    plenumArea: number;
    flowK: number;
    idleFlowK: number;
    runnerFlowK: number;
    molecularAfr: number;
    idleThrottlePlate: number;
    runnerLength: number;
    velocityDecay: number;
  }[];
  exhausts: {
    length: number;
    collectorArea: number;
    outletFlowK: number;
    primaryTubeLength: number;
    primaryFlowK: number;
    velocityDecay: number;
    audioVolume: number;
    /** Impulse response file under public/audio/impulse/, and its volume. */
    impulseResponse: string;
    impulseVolume: number;
  }[];
  ignition: {
    /** Crank speed (rad/s) to spark advance (rad). */
    timing: SampledFunction;
    /**
     * The advance at full load, by crank speed, and the manifold pressure (Pa) it holds from:
     * the advance blends from `timing` at atmospheric pressure to this, as an ECU's spark
     * table by speed and load (best-torque timing at part load, knock-limited under boost).
     */
    fullLoad?: { timing: SampledFunction; pressure: number };
    revLimit: number;
    limiterDuration: number;
    /** Firing angle per cylinder, banks in order. */
    firing: number[];
  };
  audio: {
    /**
     * The synthesizer's derivative path, mixed against its direct one (engine-sim's 0.01:
     * nearly all of the signal above ~16 Hz). With the flow source and a recorded impulse
     * response, it is the tailpipe's radiation; computed acoustics include that (0).
     */
    hfGain: number;
    /** Noise multiplying the direct path (engine-sim's "air noise"). */
    noise: number;
    jitter: number;
    /**
     * What drives the sound (engine/src/spec.rs AudioSource): "pressure", engine-sim's,
     * each exhaust runner's pressure above the atmosphere's; "flow", each exhaust valve's
     * flow as the pressure of the wave it launches into the runner (c m' / A), whose steady
     * part makes no sound (nor does a turbine's back-pressure).
     */
    source?: "pressure" | "flow";
    /**
     * The exhaust's acoustics, computed from its geometry (engine/src/acoustics.rs), in
     * place of the exhausts' recorded responses. The voice's: needs `voice`.
     */
    acoustics?: Acoustics;
    /**
     * The sound as engine/src/voice.rs makes it, in place of engine-sim's synthesizer
     * (and the fields above but `acoustics`): from each exhaust valve's flow, event by
     * event, through the acoustics (or, without them, the exhausts' recorded responses),
     * with the noise and the other sources a car makes.
     */
    voice?: VoiceSpec;
  };
  /** A turbocharger feeding every intake from every exhaust (engine/src/turbo.rs). */
  turbo?: TurboSpec;
  /** Physics beyond engine-sim's (engine/README.md); absent: engine-sim's own. */
  physics?: {
    /** Heat capacity ratio falling with temperature, and flows carrying enthalpy. */
    realGas?: boolean;
    /** Woschni's heat transfer to cylinder walls at this mean temperature (K). */
    wallTemperature?: number;
  };
}

/**
 * engine/src/voice.rs's sources beyond the exhaust valves' flow, and its output level
 * (engine/src/spec.rs VoiceSpec). Each level multiplies the source's physical estimate; 0
 * or absent leaves it out.
 */
export interface VoiceSpec {
  /** Sound pressure (Pa) at the listener that is full scale. */
  fullScale: number;
  /** Mean exhaust mass flow (kg/s) the acoustics' flow losses are fixed at. */
  acousticFlow?: number;
  /** Each blowdown's random strength beyond the simulation's (standard deviation). */
  variation?: number;
  /** Turbulence of the valve jets, through the acoustics. */
  portNoise?: number;
  /** The tailpipe jets' noise (Lighthill's), pulsing with their flow. */
  tailNoise?: number;
  /** The engine bay: manifold and turbine shells ringing with the pulses. */
  bay?: number;
  /** Valves seating; direct injectors ticking. */
  valvetrain?: number;
  injectors?: number;
  /** Turbocharger: compressor noise, its blade-pass whine (with the blade count). */
  whoosh?: number;
  whistle?: number;
  compressorBlades?: number;
  /** The blow-off valve venting boost as the throttle shuts. */
  blowOff?: number;
  /** Scales recorded impulse responses (without acoustics) to the computed level. */
  impulseGain?: number;
}

/** Twin turbos count as one, with their flows and inertias summed. */
export interface TurboSpec {
  /** Compressor wheel tip radius (m). */
  compressorRadius: number;
  /** Rotating assemblies' moment of inertia (kg m^2). */
  inertia: number;
  compressorEfficiency: number;
  turbineEfficiency: number;
  /** Gauge boost the wastegate holds (Pa), by crank speed (rad/s). */
  boost: SampledFunction;
  /** Bearing losses, W per (rad/s)^2 of shaft speed. */
  friction: number;
  /** Intercooler effectiveness, 0..1. */
  intercooler: number;
  /** Air the compressor pumps through the open blow-off valve at full boost speed (kg/s). */
  ventFlow: number;
  /**
   * Flow constant of each exhaust's wastegate, fully open: it bypasses the turbine (the
   * exhaust's outlet), opened by a boost controller as the shaft reaches its target.
   */
  wastegateFlowK: number;
}

/** A pipe between two junctions of an exhaust's acoustics. */
export interface AcousticPipe {
  a: string;
  b: string;
  length: number;
  diameter: number;
  temperature: number;
  /** Wall losses as a multiple of a smooth pipe's. */
  damping?: number;
  /** Absorption of a lining (a perforated pipe in packing), Np/m per kHz. */
  absorption?: number;
  /**
   * Losses alike at every frequency (Np/m): large pulses' to their own flow at bends and
   * junctions, which linear acoustics leave out.
   */
  loss?: number;
}

/**
 * A perforated cone, baffle or cap: gas mass through `area` over `length`, resistance in
 * units of the characteristic impedance there without flow; the flow through adds its
 * jet's loss, by the discharge coefficient (default 0.65).
 */
export interface AcousticOrifice {
  a: string;
  b: string;
  area: number;
  length: number;
  resistance: number;
  temperature: number;
  discharge?: number;
}

/** The exhaust as an acoustic network of named junctions. Temperatures in K. */
export interface Acoustics {
  pipes: AcousticPipe[];
  orifices?: AcousticOrifice[];
  /** Volumes small against the wavelength (a turbine's volute, a small chamber). */
  volumes?: { node: string; volume: number; temperature: number }[];
  /** The junction behind each cylinder's exhaust valve, in cylinder order. */
  ports: string[];
  /** Open pipe ends, radiating. */
  outlets: { node: string; diameter: number }[];
  listenerDistance: number;
  /** Scales the impulse responses, against the synthesizer's fixed gain. */
  gain: number;
  /**
   * Pipes the pulses steepen in, crests outrunning troughs (engine/src/voice.rs): from
   * each exhaust system's junction in `nodes` (exhaust order; the only way on from its
   * cylinders), along `length` of pipe at `temperature`.
   */
  nonlinear?: {
    nodes?: string[];
    length: number;
    temperature: number;
    /** With `nodes`, each cylinder's own pipe ahead of them too (a turbo's primaries). */
    primaries?: { length: number; diameter: number; temperature: number };
  };
  /**
   * Valves an ECU opens (a silencer's bypass): the elements opening them adds, and the
   * crank speeds (rad/s) or pedal positions they open between, whichever is further. The
   * voice blends the responses past `nonlinear.nodes` closed and open (so needs those).
   */
  valves?: {
    pipes?: AcousticPipe[];
    orifices?: AcousticOrifice[];
    speed: [number, number];
    throttle: [number, number];
  };
  /**
   * A turbocharger's wastegates, fully open: orifices around the turbines (to the
   * junctions of `nonlinear.nodes`, which they need). The voice blends the manifolds'
   * responses by the engine's wastegate opening.
   */
  wastegate?: AcousticOrifice[];
}

export interface CamSpec {
  advance?: number;
  /** Cam angle from the lobe centre to valve lift. */
  profile: SampledFunction;
  /** Lobe centrelines (crank angle), one per cylinder of the bank. */
  lobes: number[];
}

/** engine-sim's units (es/constants/units.mr), so values come out the same. */
export const u = (() => {
  const pi = 3.14159265359;
  const N = 1.0;
  const lbf = N * 4.44822;
  const kg = 1.0;
  const g = kg / 1000.0;
  const m = 1.0;
  const cm = m / 100.0;
  const mm = m / 1000.0;
  const inch = cm * 2.54;
  const foot = inch * 12.0;
  const thou = inch / 1000.0;
  const cc = cm * cm * cm;
  const kPa = 1000.0;
  const inHg = 3386.3886666666713;
  return {
    bar: 100000.0,
    pi,
    N,
    lbf,
    kg,
    g,
    m,
    cm,
    mm,
    inch,
    foot,
    thou,
    lb_ft: lbf * foot,
    cc,
    L: cc * 1000.0,
    cm2: cm * cm,
    deg: 1.0 * (pi / 180.0),
    rpm: 0.104719755,
    kJ: 1000.0,
    atm: 101.325 * kPa,
    inHg,
    inH2O: inHg * 0.0734824,
    scfm: (0.002641 * 453.59237) / 60,
    R: 8.31446261815324,
    celcius: (t: number) => t + 273.15,
  };
})();

/** engine-sim's GasSystem::flowConstant: k such that `flow` passes at `pressureDrop`. */
function flowConstant(
  flow: number,
  p: number,
  pressureDrop: number,
  t: number,
): number {
  const hcr = 1.0 + 2.0 / 5;
  const pT = p - pressureDrop;
  const chokedFlowLimit = Math.pow(2.0 / (hcr + 1), hcr / (hcr - 1));
  const ratio = pT / p;
  let rate: number;
  if (ratio <= chokedFlowLimit) {
    rate =
      Math.sqrt(hcr) * Math.pow(2 / (hcr + 1), (hcr + 1) / (2 * (hcr - 1)));
  } else {
    rate = (2 * hcr) / (hcr - 1);
    rate *= 1 - Math.pow(ratio, (hcr - 1) / hcr);
    rate = Math.sqrt(rate);
    rate *= Math.pow(ratio, 1 / hcr);
  }
  rate *= p / Math.sqrt(u.R * t);
  return flow / rate;
}

/** Flow constant passing `cfm` at a carburettor's 1.5 inHg. */
/** The engine state engine/src/voice.rs's control stream carries after the cylinders. */
export const VOICE_GLOBALS = 10;

/**
 * The engine's audio channels (engine/src/engine.rs audio_channels): with a voice, its
 * control stream (each cylinder's exhaust valve flow, then VOICE_GLOBALS of state); else
 * engine-sim's, one per exhaust system.
 */
export function audioChannels(spec: EngineSpec): number {
  if (!spec.audio.voice) return spec.exhausts.length;
  return spec.banks.reduce((n, b) => n + b.cylinders.length, 0) + VOICE_GLOBALS;
}

export function kCarb(cfm: number): number {
  return flowConstant(cfm * u.scfm, 1.0 * u.atm, 1.5 * u.inHg, u.celcius(25));
}

/** Flow constant passing `cfm` at a flow bench's 28 inH2O. */
export function k28inH2O(cfm: number): number {
  return flowConstant(cfm * u.scfm, 1.0 * u.atm, 28.0 * u.inH2O, u.celcius(25));
}

/** Flow constant of an opening of `area` with discharge coefficient `cd`. */
export function kArea(area: number, cd = 1): number {
  // engine-sim's flow is k * p * psi / sqrt(R T) mol/s; an orifice passes
  // cd * A * p * psi / sqrt(M R T).
  return (cd * area) / Math.sqrt(28.97 * u.g);
}

export function diskInertia(mass: number, radius: number): number {
  return (1 / 2.0) * mass * radius * radius;
}

export function rodInertia(mass: number, length: number): number {
  return (1 / 12.0) * mass * length * length;
}

export function sampled(
  filterRadius: number,
  samples: [number, number][],
): SampledFunction {
  return { filterRadius, samples };
}

/** Port flow from flow-bench numbers: [lift (thou), flow (cfm at 28 inH2O)]. */
export function portFlow(samples: [number, number][]): SampledFunction {
  return sampled(
    50 * u.thou,
    samples.map(([lift, cfm]) => [lift * u.thou, k28inH2O(cfm)]),
  );
}

/** engine-sim's harmonic_cam_lobe. */
export function harmonicCamLobe(p: {
  durationAt50Thou: number;
  gamma: number;
  lift: number;
  steps: number;
}): SampledFunction {
  const angle = p.durationAt50Thou / 4;
  const s = Math.pow((2 * (50 * u.thou)) / p.lift, 1 / p.gamma) - 1;
  const k = Math.acos(s) / angle;
  const extents = u.pi / k;
  const step = extents / (p.steps - 5.0);
  const samples: [number, number][] = [];
  for (let i = 0; i < p.steps; ++i) {
    if (i === 0) {
      samples.push([0.0, p.lift]);
    } else {
      const x = i * step;
      const lift =
        x >= extents
          ? 0.0
          : p.lift * Math.pow(0.5 + 0.5 * Math.cos(k * x), p.gamma);
      samples.push([x, lift], [-x, lift]);
    }
  }
  return sampled(step, samples);
}

/** Gasoline's lower heating value (Heywood, Internal Combustion Engine Fundamentals, App. D). */
export const GASOLINE_LHV = (44.0 * u.kJ) / u.g;
/**
 * Mean cylinder wall temperature for Woschni's heat transfer, fired (Klein, "Single-zone
 * cylinder pressure modeling", Linköping 2004).
 */
export const WALL_TEMPERATURE = 440;

/** engine-sim's default fuel (gasoline). */
export function gasoline(
  overrides: Partial<EngineSpec["fuel"]> = {},
): EngineSpec["fuel"] {
  return {
    molecularMass: 100 * u.g,
    energyDensity: (48.1 * u.kJ) / u.g,
    molecularAfr: 25 / 2.0,
    maxBurningEfficiency: 0.8,
    burningEfficiencyRandomness: 0.5,
    lowEfficiencyAttenuation: 0.6,
    maxTurbulenceEffect: 2.0,
    maxDilutionEffect: 10.0,
    turbulenceToFlameSpeedRatio: sampled(
      5.0,
      [0, 5, 10, 15, 20, 25, 30, 35, 40, 45].map((x) => [
        x,
        x === 0 ? 3.0 : 1.5 * x,
      ]),
    ),
    ...overrides,
  };
}
