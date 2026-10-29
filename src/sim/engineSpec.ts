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
    revLimit: number;
    limiterDuration: number;
    /** Firing angle per cylinder, banks in order. */
    firing: number[];
  };
  audio: {
    hfGain: number;
    noise: number;
    jitter: number;
    /**
     * The exhaust's acoustics, computed from its geometry (engine/src/acoustics.rs): one
     * impulse response per cylinder in place of the exhausts' recorded ones.
     */
    acoustics?: Acoustics;
  };
}

/** The exhaust as an acoustic network of named junctions. Temperatures in K. */
export interface Acoustics {
  pipes: {
    a: string;
    b: string;
    length: number;
    diameter: number;
    temperature: number;
    /** Wall losses as a multiple of a smooth pipe's. */
    damping?: number;
    /** Absorption of a lining (a perforated pipe in packing), Np/m per kHz. */
    absorption?: number;
  }[];
  /** Perforated cones, baffles, caps: gas mass through `area` over `length`, resistance in
   * units of the characteristic impedance there. */
  orifices?: {
    a: string;
    b: string;
    area: number;
    length: number;
    resistance: number;
    temperature: number;
  }[];
  /** Volumes small against the wavelength (a cylinder behind its open valve). */
  volumes?: { node: string; volume: number; temperature: number }[];
  /** The junction behind each cylinder's exhaust valve, in cylinder order. */
  ports: string[];
  /** Open pipe ends, radiating. */
  outlets: { node: string; diameter: number }[];
  listenerDistance: number;
  /** Scales the impulse responses, against the synthesizer's fixed gain. */
  gain: number;
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
