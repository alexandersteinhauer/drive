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

export interface EngineConfig {
  idleRpm: number;
  /** Fuel cut above this. */
  limiterRpm: number;
  /** Auto-clutch starts engaging here and is fully engaged at launchRpm. */
  engageRpm: number;
  launchRpm: number;
  /** Throttle the idle governor adds at idle; fades out by 1.3x idle. */
  idleThrottle: number;
  /** Torque the fully engaged clutch can transmit before slipping. */
  clutchTorque: number;
  /** [rpm, Nm] at full throttle. */
  torqueCurve: [number, number][];
  inertia: number;
  frictionTorque: number;
  frictionPerRpm: number;
}

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
    engine: {
      idleRpm: 850,
      limiterRpm: 4600,
      engageRpm: 1000,
      launchRpm: 1900,
      idleThrottle: 0.1,
      clutchTorque: 160,
      torqueCurve: [
        [0, 50],
        [1000, 68],
        [2000, 82],
        [3000, 77],
        [3600, 67],
        [4000, 58],
        [4600, 44],
        [5000, 30],
      ],
      // Heavy flywheel.
      inertia: 0.15,
      frictionTorque: 6,
      frictionPerRpm: 0.0025,
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
