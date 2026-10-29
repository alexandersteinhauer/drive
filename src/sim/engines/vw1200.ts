import {
  GASOLINE_LHV,
  WALL_TEMPERATURE,
  diskInertia,
  gasoline,
  harmonicCamLobe,
  k28inH2O,
  kCarb,
  portFlow,
  rodInertia,
  sampled,
  u,
  kArea,
  type Acoustics,
  type CamSpec,
  type EngineSpec,
  type SampledFunction,
  type VoiceSpec,
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
  exhaustRunnerArea: number,
): EngineSpec["banks"][number]["head"] {
  return {
    // 7.0:1 on 298 cc per cylinder.
    chamberVolume: 50 * u.cc,
    intakeRunnerVolume: 45.0 * u.cc,
    intakeRunnerArea: 5.3 * u.cm2,
    exhaustRunnerVolume: 35.0 * u.cc,
    exhaustRunnerArea,
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
  primaryLength: number,
  soundAttenuation: number,
) => ({
  journal,
  piston: piston(blowby),
  rod,
  intake: 0,
  exhaust: 0,
  primaryLength,
  soundAttenuation,
});

/**
 * The flame speed with physical gases: at VW's published distributor advance, the burn
 * is half done ~9 degrees after top dead centre from 1000 to 4500 rpm, where best torque
 * has it (Heywood), in 19-33 degrees from 10 to 90 %, as a 1960s two-valve engine burns.
 */
const PHYSICAL_FLAME_SPEED = 2.7;

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
  /** The exhaust as gas volumes and openings (engine-sim's model). */
  exhaust: {
    /** Cross-section of each cylinder's port and pipe to the muffler. */
    runnerArea: number;
    /** Pipe from each port to the muffler, in cylinder order (shared part below). */
    primaryLengths: [number, number, number, number];
    primaryTubeLength: number;
    /** Opening from the pipes into the muffler. */
    primaryFlowK: number;
    /** The muffler as a volume: cross-section times length. */
    collectorArea: number;
    length: number;
    /** The tailpipes' opening. */
    outletFlowK: number;
  };
  /** Its acoustics (none: engine-sim's recorded impulse response). */
  acoustics?: Acoustics;
  /** Heard through the voice (engine/src/voice.rs) rather than engine-sim's synthesizer. */
  voice?: VoiceSpec;
  /**
   * Physics beyond engine-sim's (engine/README.md): real gases, Woschni's heat transfer,
   * gasoline's lower heating value, and the flame speed that goes with them. Without it,
   * engine-sim's own, which its slow burn makes up for.
   */
  physical?: boolean;
}

export const REFERENCE_TUNING: Vw1200Tuning = {
  idleThrottlePlate: 0.9965,
  camDuration: 200 * u.deg,
  carburettorCfm: 70.0,
  // engine-sim's Subaru example, scaled down.
  exhaust: {
    runnerArea: 5.5 * u.cm2,
    primaryLengths: [
      10.0 * u.inch,
      13.0 * u.inch,
      11.0 * u.inch,
      14.0 * u.inch,
    ],
    primaryTubeLength: 30.0 * u.inch,
    primaryFlowK: kCarb(150.0),
    collectorArea: u.pi * (2.0 * u.inch) * (2.0 * u.inch),
    length: 400 * u.mm,
    outletFlowK: kCarb(250.0),
  },
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
      ...(tuning.physical && { energyDensity: GASOLINE_LHV }),
      turbulenceToFlameSpeedRatio: sampled(
        5.0,
        flameSpeedFactors.map((f, i) => {
          const x = i * 5.0;
          const k = tuning.physical ? PHYSICAL_FLAME_SPEED : 1;
          return [x, k * (i === 0 ? f * 3.0 : f * 1.5 * x)];
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
        head: head(0, lobe, tuning.exhaust.runnerArea),
        cylinders: [
          cylinder(0, 0.002, tuning.exhaust.primaryLengths[0], 0.95),
          cylinder(3, 0.003, tuning.exhaust.primaryLengths[1], 1.0),
        ],
      },
      {
        angle: -90.0 * u.deg,
        bore,
        deckHeight: stroke / 2 + rodLength + compressionHeight,
        head: head(1, lobe, tuning.exhaust.runnerArea),
        cylinders: [
          cylinder(1, 0.002, tuning.exhaust.primaryLengths[2], 1.05),
          cylinder(2, 0.003, tuning.exhaust.primaryLengths[3], 0.95),
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
        length: tuning.exhaust.length,
        collectorArea: tuning.exhaust.collectorArea,
        outletFlowK: tuning.exhaust.outletFlowK,
        primaryTubeLength: tuning.exhaust.primaryTubeLength,
        primaryFlowK: tuning.exhaust.primaryFlowK,
        velocityDecay: 1.0,
        audioVolume: 0.6 * 0.02,
        impulseResponse: "new/mild_exhaust.wav",
        // engine-sim's; with the flow source, at the same level.
        impulseVolume: tuning.physical ? 0.0057 : 0.01,
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
    audio: {
      // Computed acoustics radiate the valves' flow themselves: no derivative path, and no
      // air noise on what is then the whole signal.
      hfGain: tuning.acoustics ? 0 : 0.01,
      noise: tuning.acoustics ? 0 : 1.0,
      jitter: 0.6,
      ...(tuning.physical && { source: "flow" as const }),
      acoustics: tuning.acoustics,
      ...(tuning.voice && { voice: tuning.voice }),
    },
    ...(tuning.physical && {
      physics: { realGas: true, wallTemperature: WALL_TEMPERATURE },
    }),
  };
}

/**
 * The 1963-65 exhaust (fresh-air heat exchangers, twin-tip muffler), as measured on stock
 * parts by owners: the front cylinders (1, 3) blow through the heat exchangers' J-tubes
 * (35 mm outside), entering the muffler through perforated cones at its ends; the rear ones
 * (2, 4) enter it directly and cross over inside it in pipes capped at the far end, so all
 * four have about the same length of pipe. One transverse box (~66 cm x 11 cm, ~6 L), two
 * packed tailpipes 26.5 cm long (~26 mm inside), and the intake's heat riser joining the
 * box's two ends. Temperatures are rough exhaust-gas figures. Cylinders in engine-sim's
 * order: 1, 4, 3, 2 (the firing order).
 */
const BOX = { length: 0.66, diameter: 0.107, temperature: 550 };
/** Junctions along the box, from the left end (m). */
const box: [string, number][] = [
  ["left end", 0],
  ["left cone", 0.06],
  ["left tail", 0.18],
  ["middle", 0.33],
  ["right tail", 0.48],
  ["right cone", 0.6],
  ["right end", 0.66],
];
const jTube = (cyl: string) => [
  { a: cyl, b: `${cyl} cone`, length: 0.55, diameter: 0.032, temperature: 650 },
];
const crossPipe = (cyl: string) => [
  { a: cyl, b: `${cyl} cap`, length: 0.47, diameter: 0.029, temperature: 650 },
];
const tailpipe = (side: string) => ({
  a: `${side} tail`,
  b: `${side} tip`,
  length: 0.265,
  diameter: 0.026,
  temperature: 500,
  // Perforated inner tube in fibre packing: an absorptive silencer, ~13 dB at 1.5 kHz
  // over its length.
  absorption: 4,
});
const STOCK_ACOUSTICS: Acoustics = {
  pipes: [
    ...jTube("cylinder 1"),
    ...jTube("cylinder 3"),
    ...crossPipe("cylinder 2"),
    ...crossPipe("cylinder 4"),
    ...box.slice(1).map(([b, x], i) => ({
      a: box[i]![0],
      b,
      length: x - box[i]![1],
      diameter: BOX.diameter,
      temperature: BOX.temperature,
    })),
    tailpipe("left"),
    tailpipe("right"),
    // The heat riser: from both ends of the box up to the intake manifold and back.
    {
      a: "left end",
      b: "right end",
      length: 0.9,
      diameter: 0.02,
      temperature: 450,
    },
  ],
  orifices: [
    // Perforated cones from the heat exchangers.
    ...[
      ["cylinder 1 cone", "right cone"],
      ["cylinder 3 cone", "left cone"],
    ].map(([a, b]) => ({
      a: a!,
      b: b!,
      area: 6e-4,
      length: 0.03,
      resistance: 0.5,
      temperature: 600,
    })),
    // Capped, perforated ends of the cross-over pipes: each vents by the far tailpipe.
    ...[
      ["cylinder 2 cap", "left tail"],
      ["cylinder 4 cap", "right tail"],
    ].map(([a, b]) => ({
      a: a!,
      b: b!,
      area: 5e-4,
      length: 0.02,
      resistance: 0.5,
      temperature: 600,
    })),
  ],
  // No volume behind the ports: a blowdown is choked through its valve, a flow the pipe's
  // waves can't push back on (a rigid end to them), and the other valves are shut.
  ports: ["cylinder 1", "cylinder 4", "cylinder 3", "cylinder 2"],
  outlets: [
    { node: "left tip", diameter: 0.026 },
    { node: "right tip", diameter: 0.026 },
  ],
  // The chase camera.
  listenerDistance: 5,
  gain: 1,
  // Nothing on the way to the muffler damps the pulses: they steepen down each port's own
  // pipe (the heat exchangers' J-tubes, the cross-over pipes).
  nonlinear: { length: 0.5, temperature: 750 },
};

/**
 * The voice (engine/src/voice.rs): a carburetted engine's combustion varies more from
 * cycle to cycle than a modern one's; pushrods and solid lifters tick.
 */
const VOICE: VoiceSpec = {
  // Physically some 10 dB quieter than the Ferrari at full throttle: heard a little
  // quieter, as a game plays its cars.
  fullScale: 1.2,
  // About full throttle's exhaust flow (~35 g/s at 34 PS).
  acousticFlow: 0.03,
  variation: 0.04,
  portNoise: 0.03,
  tailNoise: 3,
  valvetrain: 2,
  // engine-sim's recorded response (?exhaust=impulse) at about the computed level.
  impulseGain: 0.18,
};

/**
 * Tuned for the game against the published figures: 82 Nm at 2000 rpm, 34 PS at 3600 rpm
 * (torque falling to ~58 Nm by 4000), which give a 115 km/h top speed. The cam and the
 * small carburettor set the curve's shape: simulated, 81 Nm at 2000, 34.4 PS at 3600 and
 * 61 Nm at 4000, the power flat beyond (with a longer cam and freer carburettor the torque
 * held up past 3600, and the car outran Autocar's figures).
 */
export const vw1200: EngineSpec = vw1200Engine({
  physical: true,
  // Idles at ~850 rpm against its friction (and the fan's), unloaded.
  idleThrottlePlate: 0.9857,
  camDuration: 180 * u.deg,
  carburettorCfm: 66.0,
  exhaust: {
    // The ports and J-tubes, ~30 mm.
    runnerArea: u.pi * 0.015 * 0.015,
    // Heat exchangers for 1 and 3, cross-over pipes for 4 and 2.
    primaryLengths: [0.55, 0.47, 0.55, 0.47],
    primaryTubeLength: 0,
    // Through the perforated cones and caps.
    primaryFlowK: kArea(5.5 * u.cm2),
    collectorArea: u.pi * (BOX.diameter / 2) ** 2,
    length: BOX.length,
    // Two ~26 mm tailpipes.
    outletFlowK: kArea(2 * u.pi * 0.013 * 0.013, 0.8),
  },
  acoustics: STOCK_ACOUSTICS,
  voice: VOICE,
});
