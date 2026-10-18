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
  /** Body slip angle (rad) allowed before it counts as oversteer, and its weight in rad/s per rad. */
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

const axle = {
  halfTrack: 0.78,
  restLength: 0.3,
  bumpStop: 0.2,
  wheelRadius: 0.32,
  wheelInertia: 1.2,
};

/** A ~1250 kg, ~300 hp rear-wheel-drive coupe. */
export const sportsCoupe: CarConfig = {
  mass: 1250,
  inertia: [2100, 2300, 520],
  cgHeight: 0.5,
  front: {
    ...axle,
    z: -1.25,
    springRate: 34000,
    bumpDamping: 2400,
    reboundDamping: 3400,
    antiRoll: 26000,
    brakeTorque: 2200,
    steered: true,
    driven: false,
  },
  rear: {
    ...axle,
    z: 1.35,
    springRate: 38000,
    bumpDamping: 2400,
    reboundDamping: 3400,
    antiRoll: 12000,
    brakeTorque: 1100,
    steered: false,
    driven: true,
  },
  tire: {
    muX: 1.15,
    muY: 1.1,
    peakSlipX: 0.1,
    peakSlipY: Math.tan((8 * Math.PI) / 180),
    shapeC: 1.45,
    shapeE: 0.2,
    relaxX: 0.12,
    relaxY: 0.35,
    loadSensitivity: 0.1,
    nominalLoad: 3000,
    lowSpeedDampingX: 0.04,
    lowSpeedDampingY: 0.1,
    lowSpeedThreshold: 3,
    rollingResistance: 0.012,
  },
  powertrain: {
    type: "combustion",
    engine: {
      idleRpm: 900,
      limiterRpm: 7400,
      engageRpm: 1100,
      launchRpm: 3000,
      idleThrottle: 0.12,
      clutchTorque: 600,
      torqueCurve: [
        [0, 150],
        [1000, 200],
        [2000, 280],
        [3000, 330],
        [4000, 360],
        [5000, 370],
        [6000, 350],
        [7000, 310],
        [8000, 260],
      ],
      inertia: 0.2,
      frictionTorque: 15,
      frictionPerRpm: 0.006,
    },
    gearbox: {
      ratios: [3.6, 2.19, 1.54, 1.21, 1.0, 0.83],
      reverse: 3.4,
      final: 3.42,
      efficiency: 0.92,
      shiftUpRpm: 6900,
      shiftDownRpm: 3000,
      shiftUpRpmLight: 3200,
      shiftDownRpmLight: 1400,
      shiftTime: 0.18,
    },
  },
  diff: { preload: 60, powerRamp: 0.35, coastRamp: 0.2 },
  dragArea: 0.68,
  airDensity: 1.2,
  steering: { lock: (34 * Math.PI) / 180, ackermann: 0.8, limitAssist: true },
  abs: true,
  tcs: true,
  esc: {
    enabled: true,
    understeerGradient: 0.0015,
    deadband: 0.04,
    deadbandRatio: 0.15,
    slipAngleLimit: (6 * Math.PI) / 180,
    slipAngleGain: 1.5,
    gain: 6000,
    maxTorque: 2200,
  },
  handbrakeTorque: 2500,
  body: { center: [0, 0.28, 0.05], halfExtents: [0.9, 0.62, 2.15] },
};

/** The same chassis as an electric car: heavier, lower centre of mass, one fixed gear. */
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
  ...sportsCoupe,
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
    ...sportsCoupe.tire,
    muX: 1.05,
    muY: 1.0,
    nominalLoad: 5000,
    loadSensitivity: 0.12,
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
    regenTorque: 200,
    regenPower: 70000,
  },
  dragArea: 0.24 * 2.31,
  // Centre-of-mass to front axle 1.568 m; the 12.5 m turning circle gives ~31 deg at the centre.
  steering: { ...sportsCoupe.steering, lock: (31 * Math.PI) / 180 },
  // Matches the measured understeer gradient of this setup, so ESC stays quiet in normal driving.
  esc: { ...sportsCoupe.esc, understeerGradient: 0.0007 },
  // 125 mm clearance up to the 1448 mm roof; ~0.85 m front and ~1.08 m rear overhang.
  body: { center: [0, 0.297, -0.027], halfExtents: [0.926, 0.662, 2.392] },
};
