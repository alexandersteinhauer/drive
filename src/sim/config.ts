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
  maxTorque: number;
  maxPower: number;
  maxRpm: number;
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
export const electricCoupe: CarConfig = {
  ...sportsCoupe,
  mass: 1650,
  inertia: [2700, 2950, 640],
  cgHeight: 0.45,
  front: {
    ...sportsCoupe.front,
    springRate: 44000,
    bumpDamping: 3000,
    reboundDamping: 4200,
    antiRoll: 32000,
  },
  rear: {
    ...sportsCoupe.rear,
    springRate: 48000,
    bumpDamping: 3000,
    reboundDamping: 4200,
    antiRoll: 15000,
  },
  tire: { ...sportsCoupe.tire, nominalLoad: 4000 },
  powertrain: {
    type: "electric",
    maxTorque: 440,
    maxPower: 250000,
    maxRpm: 16000,
    inertia: 0.08,
    ratio: 9.0,
    efficiency: 0.95,
    regenTorque: 150,
    regenPower: 60000,
  },
  dragArea: 0.5,
};
