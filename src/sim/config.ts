import type { EngineSpec } from "./engineSpec";
import { f154cd } from "./engines/f154cd";
import { vw1200 } from "./engines/vw1200";
import type { TireConfig } from "./tire";

export interface AxleConfig {
  /** Longitudinal offset from the centre of mass; forward is -Z. */
  z: number;
  halfTrack: number;
  restLength: number;
  /** Compression at which the bump stop engages. */
  bumpStop: number;
  springRate: number;
  bumpDamping: number;
  reboundDamping: number;
  /** Force per metre of left/right compression difference. */
  antiRoll: number;
  wheelRadius: number;
  wheelInertia: number;
  brakeTorque: number;
  steered: boolean;
  driven: boolean;
}

/** The clutch and idle speed, whatever the engine. */
interface EngineBase {
  idleRpm: number;
  /** Auto-clutch starts engaging here and is fully engaged at launchRpm. */
  engageRpm: number;
  launchRpm: number;
  /** Torque the fully engaged clutch can transmit before slipping. */
  clutchTorque: number;
}

/** An engine described by its full-throttle torque curve. */
export interface MappedEngineConfig extends EngineBase {
  type: "mapped";
  /** Fuel cut above this. */
  limiterRpm: number;
  /** Throttle the idle governor adds at idle; fades out by 1.3x idle. */
  idleThrottle: number;
  /** [rpm, Nm] at full throttle. */
  torqueCurve: [number, number][];
  inertia: number;
  frictionTorque: number;
  frictionPerRpm: number;
}

/**
 * An engine simulated piston by piston (engine/, a port of engine-sim): its torque, idle,
 * rev limit and sound all come from the simulation.
 */
export interface SimulatedEngineConfig extends EngineBase {
  type: "simulated";
  spec: EngineSpec;
  /**
   * Variable torque management (turbo engines): the ECU's cap on the boost in each gear,
   * [rpm, gauge bar], from first gear up; gears beyond the list get the engine's full map.
   */
  boostByGear?: [number, number][][];
}

export type EngineConfig = MappedEngineConfig | SimulatedEngineConfig;

export interface GearboxConfig {
  ratios: number[];
  reverse: number;
  final: number;
  efficiency: number;
  /** Shift points at full throttle and at zero throttle; interpolated in between. */
  shiftUpRpm: number;
  shiftDownRpm: number;
  shiftUpRpmLight: number;
  shiftDownRpmLight: number;
  shiftTime: number;
  /** Time over which the clutch comes back after a shift (s); 0 or absent: at once. */
  clutchRelease?: number;
  /** Shifts up without lifting (a simulated engine's sparks are cut instead). */
  dualClutch?: boolean;
  /**
   * Under braking, steps down once the revs fall below this, by the brake pedal (an
   * automated gearbox keeping the engine in its band for engine braking and the exit);
   * absent: only the light-load point.
   */
  shiftDownRpmBraking?: number;
  /**
   * A downshift's longest (s): the oncoming clutch waits until the blip has brought the
   * engine up to the lower gear's speed, at least `shiftTime`; absent: `shiftTime`.
   */
  downshiftTime?: number;
}

export interface CombustionConfig {
  type: "combustion";
  engine: EngineConfig;
  gearbox: GearboxConfig;
}

/** Single motor on a fixed reduction; torque-limited at low speed, power-limited above. */
export interface ElectricConfig {
  type: "electric";
  /** Constant torque up to torqueRpm, then rising power to maxPower at powerRpm, then constant power. */
  maxTorque: number;
  torqueRpm: number;
  maxPower: number;
  powerRpm: number;
  maxRpm: number;
  /** Electronic top speed limit (m/s). */
  speedLimit: number;
  inertia: number;
  ratio: number;
  efficiency: number;
  /** Lift-off regenerative braking limits. */
  regenTorque: number;
  regenPower: number;
}

/**
 * Electronic stability control. Compares the measured yaw rate with the one the driver asks
 * for (steering and speed through a linear bicycle model, capped by available grip) and
 * corrects the difference by braking single wheels and cutting drive torque.
 */
export interface EscConfig {
  enabled: boolean;
  /** Understeer gradient of the reference model, rad per m/s^2 of lateral acceleration. */
  understeerGradient: number;
  /** Yaw rate error ignored before intervening: fixed part (rad/s) plus a fraction of the target. */
  deadband: number;
  deadbandRatio: number;
  /** Reference yaw rate lag (s), like the car's own yaw response, so quick steering is not an error. */
  yawLag: number;
  /**
   * Rear axle slip angle (rad) allowed before it counts as oversteer, and its weight in rad/s per
   * rad. Measured at the rear axle, as the body slip at the centre of mass is large in tight
   * low-speed turns without any sliding.
   */
  slipAngleLimit: number;
  slipAngleGain: number;
  /** Brake torque per rad/s of yaw rate error beyond the deadband, and its cap per wheel. */
  gain: number;
  maxTorque: number;
}

/** Clutch-type limited slip differential. */
export interface DiffConfig {
  preload: number;
  powerRamp: number;
  coastRamp: number;
}

export interface CarConfig {
  mass: number;
  /** Principal moments: pitch (x), yaw (y), roll (z). */
  inertia: [number, number, number];
  cgHeight: number;
  front: AxleConfig;
  rear: AxleConfig;
  tire: TireConfig;
  powertrain: CombustionConfig | ElectricConfig;
  diff: DiffConfig;
  dragArea: number;
  airDensity: number;
  steering: {
    lock: number;
    /** 0 = parallel, 1 = full Ackermann. */
    ackermann: number;
    /** Limits steering to roughly the front tires' useful slip range, while allowing countersteer. */
    limitAssist: boolean;
  };
  abs: boolean;
  /** Traction control: cuts throttle when the driven wheels spin past peak slip. */
  tcs: boolean;
  esc: EscConfig;
  handbrakeTorque: number;
  /** Collision box relative to the centre of mass. */
  body: {
    center: [number, number, number];
    halfExtents: [number, number, number];
  };
}

/**
 * The car with its engine heard through engine-sim's exhaust impulse responses (each
 * exhaust's `impulseResponse` file) instead of the acoustics computed from its geometry.
 */
export function withImpulseExhaust(cfg: CarConfig): CarConfig {
  const out = structuredClone(cfg);
  if (
    out.powertrain.type === "combustion" &&
    out.powertrain.engine.type === "simulated"
  ) {
    const audio = out.powertrain.engine.spec.audio;
    delete audio.acoustics;
    // engine-sim's synthesizer: its derivative path (the radiation, for the flow source),
    // with air noise on the rest.
    audio.hfGain = 0.01;
    audio.noise = 1.0;
  }
  return out;
}

/**
 * BMW i4 eDrive40 (G26, 2021-). Published by BMW (press kit technical data): 2125 kg (EU),
 * 45.1/54.9 % front/rear, 4783 x 1852 x 1448 mm, 2856 mm wheelbase, 1601/1630 mm track,
 * 125 mm ground clearance, 12.5 m turning circle, Cd 0.24 x 2.31 m^2, 250 kW / 430 Nm
 * (430 Nm from 0-5000 rpm, peak power 8000-17000 rpm), single-speed 8.774:1, 225/55 R17,
 * 0-100 km/h 5.7 s, 190 km/h limited.
 *
 * Estimated (not published): centre of mass height, inertias, spring/damper/anti-roll rates
 * (from ~1.4/1.6 Hz ride frequencies), tyre grip, brake torques, regen strength.
 */
const I4_WHEELBASE = 2.856;
const I4_REAR_SHARE = 0.549;
const i4Axle = {
  restLength: 0.32,
  bumpStop: 0.24,
  wheelRadius: 0.34,
  wheelInertia: 1.2,
};

export const bmwI4: CarConfig = {
  mass: 2125,
  inertia: [4000, 4300, 850],
  cgHeight: 0.49,
  front: {
    ...i4Axle,
    z: -I4_WHEELBASE * I4_REAR_SHARE,
    halfTrack: 1.601 / 2,
    springRate: 37000,
    bumpDamping: 2600,
    reboundDamping: 3800,
    antiRoll: 36000,
    brakeTorque: 3200,
    steered: true,
    driven: false,
  },
  rear: {
    ...i4Axle,
    z: I4_WHEELBASE * (1 - I4_REAR_SHARE),
    halfTrack: 1.63 / 2,
    springRate: 59000,
    bumpDamping: 3600,
    reboundDamping: 5300,
    antiRoll: 18000,
    brakeTorque: 1600,
    steered: false,
    driven: true,
  },
  tire: {
    muX: 1.05,
    muY: 1.0,
    peakSlipX: 0.1,
    peakSlipY: Math.tan((8 * Math.PI) / 180),
    shapeC: 1.45,
    shapeE: 0.2,
    relaxX: 0.12,
    relaxY: 0.35,
    loadSensitivity: 0.12,
    nominalLoad: 5000,
    lowSpeedDampingX: 0.04,
    lowSpeedDampingY: 0.1,
    lowSpeedThreshold: 3,
    rollingResistance: 0.009,
  },
  powertrain: {
    type: "electric",
    maxTorque: 430,
    torqueRpm: 5000,
    maxPower: 250000,
    powerRpm: 8000,
    maxRpm: 17000,
    speedLimit: 190 / 3.6,
    inertia: 0.05,
    ratio: 8.774,
    efficiency: 0.97,
    // ~0.13 g of lift-off regen, like the default (medium) recuperation rather than one-pedal B mode.
    regenTorque: 100,
    regenPower: 70000,
  },
  diff: { preload: 60, powerRamp: 0.35, coastRamp: 0.2 },
  dragArea: 0.24 * 2.31,
  airDensity: 1.2,
  // Centre-of-mass to front axle 1.568 m; the 12.5 m turning circle gives ~31 deg at the centre.
  steering: { lock: (31 * Math.PI) / 180, ackermann: 0.8, limitAssist: true },
  abs: true,
  tcs: true,
  esc: {
    enabled: true,
    // Matches the measured understeer gradient of this setup, so ESC stays quiet in normal driving.
    understeerGradient: 0.0007,
    deadband: 0.04,
    deadbandRatio: 0.15,
    yawLag: 0.15,
    slipAngleLimit: (6 * Math.PI) / 180,
    slipAngleGain: 1.5,
    gain: 6000,
    maxTorque: 2200,
  },
  handbrakeTorque: 2500,
  // 125 mm clearance up to the 1448 mm roof; ~0.85 m front and ~1.08 m rear overhang.
  body: { center: [0, 0.297, -0.027], halfExtents: [0.926, 0.662, 2.392] },
};

/**
 * VW Beetle 1200 (Type 1, 1963). Published by VW (1963 brochure / owner's manual): 34 PS DIN
 * (25 kW) at 3600 rpm, 82 Nm (8.4 mkp) at 2000 rpm, 1192 cc air-cooled flat four in the rear,
 * all-synchromesh 4-speed 3.80/2.06/1.32/0.89, reverse 3.88, final drive 4.375, 5.60-15
 * cross-ply tyres (~0.315 m rolling radius), 740 kg unladen, 4070 x 1540 x 1500 mm,
 * 2400 mm wheelbase, 11 m turning circle, 115 km/h top speed. Autocar (1961, 34 bhp):
 * 0-50 mph (80 km/h) 18.0 s, 0-60 mph 32.1 s, 72 mph (116 km/h) top speed.
 *
 * The model is ~5 % larger than the real car (scaled to its length, its wheels measure
 * 0.376 m and the track 1.372 m); the axles and wheels follow the model so the wheels sit in
 * its arches, and the final drive is raised by the same factor to keep the real road speed
 * per engine rpm.
 *
 * Estimated (not published): ~58 % on the rear axle with a driver, centre of mass height,
 * inertias, torsion bar/damper rates (soft, ~1.25/1.35 Hz), tyre grip, drum brake torques.
 */
const BEETLE_WHEELBASE = 2.423;
const BEETLE_REAR_SHARE = 0.58;
const BEETLE_RADIUS = 0.375;
const beetleAxle = {
  halfTrack: 1.372 / 2,
  restLength: 0.34,
  bumpStop: 0.26,
  wheelRadius: BEETLE_RADIUS,
  wheelInertia: 0.9,
};

export const vwBeetle: CarConfig = {
  // 740 kg unladen plus fuel and a driver.
  mass: 840,
  inertia: [1150, 1300, 320],
  cgHeight: 0.53,
  front: {
    ...beetleAxle,
    z: -BEETLE_WHEELBASE * BEETLE_REAR_SHARE,
    springRate: 11000,
    bumpDamping: 850,
    reboundDamping: 1200,
    antiRoll: 9000,
    brakeTorque: 900,
    steered: true,
    driven: false,
  },
  rear: {
    ...beetleAxle,
    z: BEETLE_WHEELBASE * (1 - BEETLE_REAR_SHARE),
    springRate: 17500,
    bumpDamping: 1250,
    reboundDamping: 1750,
    antiRoll: 2000,
    brakeTorque: 600,
    steered: false,
    driven: true,
  },
  // Narrow cross-ply tyres: less grip, a wider and softer peak.
  tire: {
    muX: 0.82,
    muY: 0.8,
    peakSlipX: 0.12,
    peakSlipY: Math.tan((10 * Math.PI) / 180),
    shapeC: 1.4,
    shapeE: 0.1,
    relaxX: 0.15,
    relaxY: 0.45,
    loadSensitivity: 0.1,
    nominalLoad: 2200,
    lowSpeedDampingX: 0.04,
    lowSpeedDampingY: 0.1,
    lowSpeedThreshold: 3,
    rollingResistance: 0.013,
  },
  powertrain: {
    type: "combustion",
    // The 1200, simulated (src/sim/engines/vw1200.ts).
    engine: {
      type: "simulated",
      spec: vw1200,
      idleRpm: 850,
      engageRpm: 1000,
      launchRpm: 1900,
      clutchTorque: 160,
    },
    gearbox: {
      ratios: [3.8, 2.06, 1.32, 0.89],
      reverse: 3.88,
      final: (4.375 * BEETLE_RADIUS) / 0.315,
      efficiency: 0.88,
      shiftUpRpm: 4200,
      shiftDownRpm: 2000,
      shiftUpRpmLight: 2600,
      shiftDownRpmLight: 1200,
      // No hurry: a long lever and a cable clutch.
      shiftTime: 0.5,
      clutchRelease: 0.35,
    },
  },
  // Swing axles with an open differential.
  diff: { preload: 0, powerRamp: 0, coastRamp: 0 },
  // Cd 0.48 x 1.8 m^2.
  dragArea: 0.48 * 1.8,
  airDensity: 1.2,
  steering: { lock: (32 * Math.PI) / 180, ackermann: 0.8, limitAssist: true },
  abs: true,
  tcs: true,
  esc: {
    enabled: true,
    // Matches the measured understeer gradient (close to neutral), as for the i4.
    understeerGradient: 0.0005,
    deadband: 0.04,
    deadbandRatio: 0.15,
    yawLag: 0.15,
    slipAngleLimit: (6 * Math.PI) / 180,
    slipAngleGain: 1.5,
    gain: 3000,
    maxTorque: 900,
  },
  handbrakeTorque: 1200,
  // From the model: 0.27 m under the floor pan up to the 1.61 m roof, bumper to bumper.
  body: { center: [0, 0.409, -0.068], halfExtents: [0.79, 0.669, 2.035] },
};

/**
 * Ferrari 488 Pista (2018-20). Published by Ferrari (press kit, ferrari.com): 720 cv (530 kW)
 * at 8000 rpm, 770 Nm at 3000 rpm in 7th gear (Variable Torque Management holds lower gears
 * below that), 3902 cc twin-turbo V8, 8000 rpm, 1385 kg kerb with the lightweight options
 * (1280 kg dry), 41.5/58.5 % front/rear, 4605 x 1975 x 1206 mm, 2650 mm wheelbase,
 * 1679/1649 mm track, 245/35 ZR20 and 305/30 ZR20 (Michelin Pilot Sport Cup 2), carbon-ceramic
 * discs 398/360 mm, 7-speed dual-clutch, E-Diff3, 0-100 km/h 2.85 s, 0-200 km/h 7.6 s,
 * > 340 km/h. Gear ratios from the 488 GTB owner's manual (Ferrari lists none for the Pista,
 * whose gearbox only shifts faster): 3.334/2.285/1.728/1.369/1.115/0.875/0.642, reverse
 * 2.979, final drive 4.375.
 * Car and Driver (2019 test): 0-60 mph 2.7 s, 0-100 mph 5.2 s, 0-150 mph 10.9 s, quarter
 * mile 10.1 s at 144 mph, 1.10 g on the skidpad.
 *
 * The model measures as the real car (2.651 m wheelbase, 1.676/1.631 m track, tyres
 * 0.342/0.348 m), so the axles and wheels follow it.
 *
 * Estimated (not published): centre of mass height, inertias, spring/damper/anti-roll rates
 * (from ~1.9/2.1 Hz ride frequencies), tyre grip (for the skidpad figure), brake torques
 * (for Ferrari's 29.5 m from 100 km/h). No downforce (the physics has none); the drag area is
 * set for the published top speed. In the game it runs 0-100 km/h in ~3.9 s and the quarter
 * mile in ~11 s at 144 mph: without launch control the engine first revs up from idle.
 */
const PISTA_WHEELBASE = 2.65;
const PISTA_REAR_SHARE = 0.585;
/**
 * Variable Torque Management: Ferrari limits the boost below 7th gear, most in the lowest
 * ones, so the torque rises with the revs like an aspirated engine's: 770 Nm only in 7th,
 * up to 700 Nm in 1st-6th, less at low revs in the low gears (a 488 GTB logged ~11 psi of
 * manifold boost in the low gears against 16 in 7th), and full power at 8000 in every gear
 * (Ferrari's briefing to PistonHeads, 2015). The torque each gear is held to (estimated
 * from that), and the boost cap that gives it on the simulated engine (calibrate.rs `boost`):
 *
 * - 1st-3rd: 540 Nm at 3000 rpm, 600 at 4000, 650 at 5000, 700 at 6000;
 * - 4th: 610 at 3000, 670 at 4000, 700 from 5000; 5th: 660 at 3000, 700 from 4000;
 *   6th: 700 from 3000;
 * - all: 700 Nm to 6500 rpm, then the 7th-gear curve to 650 at 8000.
 */
const VTM_TOP: [number, number][] = [
  [6500, 1.3],
  [7000, 1.29],
  [7500, 1.3],
  [8000, 1.26],
];
const vtm = (low: [number, number][]): [number, number][] => [
  ...low,
  ...VTM_TOP,
];
const PISTA_VTM = [
  ...Array.from({ length: 3 }, () =>
    vtm([
      [3000, 0.82],
      [4000, 0.99],
      [5000, 1.12],
      [6000, 1.26],
    ]),
  ),
  vtm([
    [3000, 1.07],
    [4000, 1.21],
    [5000, 1.27],
    [6000, 1.26],
  ]),
  vtm([
    [3000, 1.23],
    [4000, 1.3],
    [5000, 1.27],
    [6000, 1.26],
  ]),
  vtm([
    [3000, 1.35],
    [4000, 1.3],
    [5000, 1.27],
    [6000, 1.26],
  ]),
];

const pistaAxle = {
  restLength: 0.22,
  bumpStop: 0.15,
};

export const ferrari488Pista: CarConfig = {
  // 1385 kg kerb plus a driver.
  mass: 1460,
  inertia: [1950, 2100, 480],
  cgHeight: 0.44,
  front: {
    ...pistaAxle,
    z: -PISTA_WHEELBASE * PISTA_REAR_SHARE,
    halfTrack: 1.676 / 2,
    wheelRadius: 0.342,
    wheelInertia: 0.9,
    springRate: 43000,
    bumpDamping: 2000,
    reboundDamping: 3000,
    antiRoll: 34000,
    brakeTorque: 3000,
    steered: true,
    driven: false,
  },
  rear: {
    ...pistaAxle,
    z: PISTA_WHEELBASE * (1 - PISTA_REAR_SHARE),
    halfTrack: 1.631 / 2,
    wheelRadius: 0.348,
    wheelInertia: 1.1,
    springRate: 74000,
    bumpDamping: 3100,
    reboundDamping: 4700,
    antiRoll: 26000,
    brakeTorque: 1900,
    steered: false,
    driven: true,
  },
  // Semi-slick Cup 2s: high, sharp peak grip.
  tire: {
    muX: 1.4,
    muY: 1.25,
    peakSlipX: 0.09,
    peakSlipY: Math.tan((7 * Math.PI) / 180),
    shapeC: 1.5,
    shapeE: 0.25,
    relaxX: 0.1,
    relaxY: 0.3,
    loadSensitivity: 0.12,
    nominalLoad: 4500,
    lowSpeedDampingX: 0.04,
    lowSpeedDampingY: 0.1,
    lowSpeedThreshold: 3,
    rollingResistance: 0.011,
  },
  powertrain: {
    type: "combustion",
    // The F154 CD, simulated (src/sim/engines/f154cd.ts).
    engine: {
      type: "simulated",
      spec: f154cd,
      boostByGear: PISTA_VTM,
      idleRpm: 950,
      // Launch control holds the revs up while the clutch slips, so the turbos spool.
      engageRpm: 2200,
      launchRpm: 4000,
      clutchTorque: 1100,
    },
    gearbox: {
      ratios: [3.334, 2.285, 1.728, 1.369, 1.115, 0.875, 0.642],
      reverse: 2.979,
      final: 4.375,
      efficiency: 0.92,
      shiftUpRpm: 7900,
      shiftDownRpm: 3500,
      shiftUpRpmLight: 3200,
      shiftDownRpmLight: 1500,
      // Dual clutch: one clutch hands over to the other.
      shiftTime: 0.06,
      clutchRelease: 0.05,
      dualClutch: true,
      // Under braking it steps down from 4500 rpm (into the next gear at ~5500-6500), the
      // ECU blipping the throttle to match the revs before the clutch closes.
      shiftDownRpmBraking: 4500,
      downshiftTime: 0.3,
    },
  },
  // E-Diff3: an electronically locked differential, modelled as a strong clutch type.
  diff: { preload: 80, powerRamp: 0.45, coastRamp: 0.3 },
  dragArea: 0.9,
  airDensity: 1.2,
  steering: { lock: (30 * Math.PI) / 180, ackermann: 0.6, limitAssist: true },
  abs: true,
  tcs: true,
  esc: {
    enabled: true,
    understeerGradient: 0.0007,
    deadband: 0.04,
    deadbandRatio: 0.15,
    yawLag: 0.12,
    slipAngleLimit: (6 * Math.PI) / 180,
    slipAngleGain: 1.5,
    gain: 4500,
    maxTorque: 1800,
  },
  handbrakeTorque: 2000,
  // From the model: ~0.1 m ground clearance up to the 1.19 m roof; 1.154 m front and 0.80 m
  // rear overhang; 1975 mm wide without the mirrors.
  body: { center: [0, 0.205, -0.403], halfExtents: [0.987, 0.545, 2.3] },
};

/**
 * The Pista's opt-in learned exhaust pilot. Its teacher ran at 20 kHz; the game keeps
 * its existing engine rate and physics, and the audio adapter resamples to the model.
 * This is pressure/exhaust only, not the voice's bay, turbo or mechanical sources.
 */
export function withPistaSurrogateExhaust(): CarConfig {
  const out = structuredClone(ferrari488Pista);
  if (
    out.powertrain.type === "combustion" &&
    out.powertrain.engine.type === "simulated"
  ) {
    out.powertrain.engine.spec.audio.surrogate = {
      model: "audio/surrogate/pista-pilot.json",
      fullScale: 16,
      flapSpeed: [261.7993875, 366.5191425],
      flapThrottle: [0.3, 0.6],
    };
  }
  return out;
}
