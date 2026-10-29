import {
  diskInertia,
  gasoline,
  harmonicCamLobe,
  k28inH2O,
  kCarb,
  portFlow,
  rodInertia,
  sampled,
  u,
  type CamSpec,
  type EngineSpec,
  type SampledFunction,
} from "../engineSpec";

/**
 * VW 1200 (Type 1, 1961-65): 1192 cc air-cooled flat four, 34 PS DIN at 3600 rpm, 82 Nm at
 * 2000 rpm.
 *
 * Published: 77 mm bore, 64 mm stroke, 7.0:1 compression, single central camshaft with
 * pushrods and two valves per cylinder, firing order 1-4-3-2, one Solex 28 PICT carburettor
 * on a long central manifold, heat exchangers into a single transverse muffler.
 * Estimated: masses and inertias, port flow, cam lobe shapes, timing curve, pipe lengths;
 * tuned so the simulated engine makes about the published torque and power.
 *
 * Laid out like engine-sim's Subaru EJ25 example (also a flat four): two banks at +-90
 * degrees, each with its own camshaft half.
 */

const stroke = 64 * u.mm;
const bore = 77 * u.mm;
const rodLength = 137 * u.mm;
const rodMass = 600 * u.g;
const compressionHeight = 36 * u.mm;
const crankMass = 7.5 * u.kg;
const flywheelMass = 7.7 * u.kg;
const flywheelRadius = 130 * u.mm;
const cycle = 2 * 360 * u.deg;

const lobeCenter = 108 * u.deg;
const rot360 = 360 * u.deg;
/** One physical camshaft, split per bank the way engine-sim lays out a flat four. */
function cams(
  bank: 0 | 1,
  lobe: SampledFunction,
): { intakeCam: CamSpec; exhaustCam: CamSpec } {
  const quarters = bank === 0 ? [0.0, 1.0] : [2.0, 3.0];
  return {
    intakeCam: {
      profile: lobe,
      lobes: quarters.map((q) => rot360 + lobeCenter + (q / 4) * cycle),
    },
    exhaustCam: {
      profile: lobe,
      lobes: quarters.map((q) => rot360 - lobeCenter + (q / 4) * cycle),
    },
  };
}

function head(
  bank: 0 | 1,
  lobe: SampledFunction,
): EngineSpec["banks"][number]["head"] {
  return {
    // 7.0:1 on 298 cc per cylinder.
    chamberVolume: 50 * u.cc,
    intakeRunnerVolume: 45.0 * u.cc,
    intakeRunnerArea: 5.3 * u.cm2,
    exhaustRunnerVolume: 35.0 * u.cc,
    exhaustRunnerArea: 5.5 * u.cm2,
    // Small valves (about 30 mm).
    intakePortFlow: portFlow([
      [0, 0],
      [50, 26],
      [100, 46],
      [150, 62],
      [200, 72],
      [250, 78],
      [300, 80],
      [350, 81],
    ]),
    exhaustPortFlow: portFlow([
      [0, 0],
      [50, 22],
      [100, 38],
      [150, 51],
      [200, 60],
      [250, 63],
      [300, 66],
      [350, 67],
    ]),
    ...cams(bank, lobe),
  };
}

const piston = (blowby: number) => ({
  mass: (390 + 110) * u.g, // piston + pin
  blowby: k28inH2O(blowby),
  compressionHeight,
  wristPinPosition: 0.0,
  displacement: 0.0,
});
const rod = {
  mass: rodMass,
  momentOfInertia: rodInertia(rodMass, rodLength),
  centerOfMass: 0.0,
  length: rodLength,
};
const cylinder = (
  journal: number,
  blowby: number,
  primaryInches: number,
  soundAttenuation: number,
) => ({
  journal,
  piston: piston(blowby),
  rod,
  intake: 0,
  exhaust: 0,
  primaryLength: primaryInches * u.inch,
  soundAttenuation,
});

const flameSpeedFactors = [
  1.0, 1.0, 1.0, 1.1, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
];

/**
 * What the game tunes. engine/reference/vw1200.mr is this engine in engine-sim's script
 * language with REFERENCE_TUNING, which the port is checked against.
 */
export interface Vw1200Tuning {
  /** Throttle plate position at idle (1 = closed): sets the idle speed. */
  idleThrottlePlate: number;
  /** Cam duration at 0.050" lift (rad): longer breathes better at high rpm, worse low. */
  camDuration: number;
  /** Carburettor flow (cfm at 1.5 inHg): caps the top end. */
  carburettorCfm: number;
}

export const REFERENCE_TUNING: Vw1200Tuning = {
  idleThrottlePlate: 0.9965,
  camDuration: 200 * u.deg,
  carburettorCfm: 70.0,
};

/** The engine with the given tuning. */
export function vw1200Engine(tuning: Vw1200Tuning): EngineSpec {
  // Mild stock cam: about 2.5 deg BTDC to 37.5 deg ABDC at 200 deg (and mirrored).
  const lobe = harmonicCamLobe({
    durationAt50Thou: tuning.camDuration,
    gamma: 1.5,
    lift: 7.9 * u.mm,
    steps: 100,
  });
  return {
    simulationFrequency: 10000,
    starterTorque: 40 * u.lb_ft,
    starterSpeed: 300 * u.rpm,
    throttleGamma: 2.0,
    fuel: gasoline({
      maxBurningEfficiency: 0.9,
      turbulenceToFlameSpeedRatio: sampled(
        5.0,
        flameSpeedFactors.map((f, i) => {
          const x = i * 5.0;
          return [x, i === 0 ? f * 3.0 : f * 1.5 * x];
        }),
      ),
    }),
    crankshaft: {
      throw: stroke / 2,
      mass: crankMass,
      flywheelMass,
      momentOfInertia:
        diskInertia(crankMass, stroke / 2) +
        diskInertia(flywheelMass, flywheelRadius) +
        // Cooling fan and generator (belt driven ~1.8x), pulley, cam gear.
        diskInertia(6 * u.kg, 8.0 * u.cm),
      // Bearings plus the cooling fan, which takes a kilowatt or two at speed.
      frictionTorque: 4.0 * u.lb_ft,
      x: 0.0,
      y: 0.0,
      tdc: 180 * u.deg,
      journals: [0, 180 * u.deg, 0, 180 * u.deg],
    },
    banks: [
      {
        angle: 90.0 * u.deg,
        bore,
        deckHeight: stroke / 2 + rodLength + compressionHeight,
        head: head(0, lobe),
        cylinders: [
          cylinder(0, 0.002, 10.0, 0.95),
          cylinder(3, 0.003, 13.0, 1.0),
        ],
      },
      {
        angle: -90.0 * u.deg,
        bore,
        deckHeight: stroke / 2 + rodLength + compressionHeight,
        head: head(1, lobe),
        cylinders: [
          cylinder(1, 0.002, 11.0, 1.05),
          cylinder(2, 0.003, 14.0, 0.95),
        ],
      },
    ],
    // Solex 28 PICT on a long central manifold.
    intakes: [
      {
        plenumVolume: 0.35 * u.L,
        plenumArea: 6.0 * u.cm2,
        flowK: kCarb(tuning.carburettorCfm),
        idleFlowK: kCarb(0.0),
        runnerFlowK: kCarb(50.0),
        molecularAfr: 25.0 / 2.0,
        idleThrottlePlate: tuning.idleThrottlePlate,
        runnerLength: 14.0 * u.inch,
        velocityDecay: 0.5,
      },
    ],
    // Heat exchangers on both sides into one transverse muffler: near-equal lengths.
    exhausts: [
      {
        length: 400 * u.mm,
        collectorArea: u.pi * (2.0 * u.inch) * (2.0 * u.inch),
        outletFlowK: kCarb(250.0),
        primaryTubeLength: 30.0 * u.inch,
        primaryFlowK: kCarb(150.0),
        velocityDecay: 1.0,
        audioVolume: 0.6 * 0.02,
        impulseResponse: "new/mild_exhaust.wav",
        impulseVolume: 0.01,
      },
    ],
    ignition: {
      // Distributor: 10 deg static, centrifugal advance to ~30 deg.
      timing: sampled(1000 * u.rpm, [
        [0, 10 * u.deg],
        [1000 * u.rpm, 12 * u.deg],
        [2000 * u.rpm, 22 * u.deg],
        [3000 * u.rpm, 28 * u.deg],
        [4000 * u.rpm, 30 * u.deg],
        [5000 * u.rpm, 30 * u.deg],
      ]),
      revLimit: 4800 * u.rpm,
      limiterDuration: 0.1,
      // Firing order 1-4-3-2.
      firing: [0.0, 1.0, 2.0, 3.0].map((q) => (q / 4.0) * cycle),
    },
    audio: { hfGain: 0.01, noise: 1.0, jitter: 0.6 },
  };
}

/**
 * Tuned for the game against the published figures: 82 Nm at 2000 rpm, 34 PS at 3600 rpm
 * (torque falling to ~58 Nm by 4000), which give a 115 km/h top speed.
 */
export const vw1200: EngineSpec = vw1200Engine({
  // Idles at ~850 rpm against its friction (and the fan's), unloaded.
  idleThrottlePlate: 0.984,
  camDuration: 190 * u.deg,
  carburettorCfm: 60.0,
});
