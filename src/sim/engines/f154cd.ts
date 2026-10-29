import {
  GASOLINE_LHV,
  WALL_TEMPERATURE,
  diskInertia,
  gasoline,
  harmonicCamLobe,
  k28inH2O,
  kArea,
  kCarb,
  portFlow,
  rodInertia,
  sampled,
  u,
  type Acoustics,
  type CamSpec,
  type EngineSpec,
  type SampledFunction,
  type VoiceSpec,
} from "../engineSpec";

/**
 * Ferrari F154 CD (488 Pista, 2018-20): 3902 cc 90-degree twin-turbo V8 with a flat-plane
 * crank, 720 cv (530 kW) at 8000 rpm, 770 Nm at 3000 rpm.
 *
 * Published (Ferrari; 488 GTB owner's manual for the F154 family): 86.5 mm bore, 83 mm
 * stroke, 9.6:1 compression, 8000 rpm with a hard cut ("wall effect"), firing order
 * 1-5-3-7-4-8-2-6, four valves per cylinder with variable timing on both cams, direct
 * injection, two IHI twin-scroll turbochargers with TiAl turbines, two air-to-air
 * intercoolers, peak boost ~1.5 bar (21.6 psi, Car and Driver; 1.5 psi over the 488 GTB),
 * titanium rods, a lighter crank and flywheel (rotating inertia -17 %), cams with 1 mm more
 * lift, a carbon-fibre plenum with short runners, Inconel exhaust manifolds.
 * Estimated: masses and inertias, rod length, cam lobes, pipe sizes, the turbochargers'
 * efficiencies. Simulated with physics beyond engine-sim's (engine/README.md), and then
 * calibrated on the dynamometer (engine/examples/calibrate.rs) against physical criteria
 * and the published figures: the flame speed and part-load timing to the best-torque burn,
 * the full-load timing to an estimated knock limit, the port, turbine and wastegate sizes
 * and the boost map to the published curve and boost.
 *
 * Laid out like engine-sim's Ferrari F136 example (the 458's flat-plane V8): banks at
 * +-45 degrees, cylinders 1-4 on the first, 5-8 on the second, rod journals 0/180/180/0.
 */

const stroke = 83 * u.mm;
const bore = 86.5 * u.mm;
const rodLength = 146 * u.mm;
const rodMass = 420 * u.g; // titanium
const compressionHeight = 28 * u.mm;
const crankMass = 16 * u.kg;
const flywheelMass = 5 * u.kg;
const flywheelRadius = 120 * u.mm;
const cycle = 2 * 360 * u.deg;
const rot = 90 * u.deg;
const rot360 = 360 * u.deg;

/** Firing order 1-5-3-7-4-8-2-6: each cylinder's place in it, in cylinder order. */
const FIRING = [0, 6, 2, 4, 1, 7, 3, 5];

function cams(
  bank: 0 | 1,
  intake: SampledFunction,
  exhaust: SampledFunction,
  intakeCenter: number,
  exhaustCenter: number,
): { intakeCam: CamSpec; exhaustCam: CamSpec } {
  const order = FIRING.slice(bank * 4, bank * 4 + 4);
  return {
    intakeCam: {
      profile: intake,
      lobes: order.map((k) => rot360 + intakeCenter + k * rot),
    },
    exhaustCam: {
      profile: exhaust,
      lobes: order.map((k) => rot360 - exhaustCenter + k * rot),
    },
  };
}

/** What the game tunes. */
export interface F154Tuning {
  /** Throttle plate position at idle (1 = closed): sets the idle speed. */
  idleThrottlePlate: number;
  /** Cam durations at 0.050" lift (rad). */
  intakeDuration: number;
  exhaustDuration: number;
  /** Throttle body flow (cfm at 1.5 inHg). */
  throttleCfm: number;
  /** Peak port flow (cfm at 28 inH2O), both valves. */
  intakePortCfm: number;
  exhaustPortCfm: number;
  /** Gauge boost the wastegate holds (bar), by rpm. */
  boost: [rpm: number, bar: number][];
  /** Effective opening of each bank's turbine (m^2), and of its wastegate, fully open. */
  turbineArea: number;
  wastegateArea: number;
  /** Spark advance (deg) by rpm at part load (atmospheric manifold pressure and below). */
  timing: [rpm: number, deg: number][];
  /** Spark advance (deg) by rpm at full load, from `fullLoadBoost` (bar) on. */
  fullLoadTiming: [rpm: number, deg: number][];
  fullLoadBoost: number;
}

/**
 * The flame speed with physical gases (engine/README.md): the burn takes 20-28 degrees from
 * 10 to 90 %, as in a high-tumble direct-injection engine (Klein's reference burn is
 * 30 degrees for an older port-injected one), and at best-torque timing is half done about
 * 8 degrees after top dead centre, where Heywood puts it.
 */
const FLAME_SPEED = 3;

/** Flow-bench samples every 50 thou of lift: `shape` scaled to `peak` cfm. */
function scale(peak: number, shape: number[]): [number, number][] {
  return shape.map((f, i) => [i * 50, f * peak]);
}

export function f154Engine(t: F154Tuning): EngineSpec {
  const intakeLobe = harmonicCamLobe({
    durationAt50Thou: t.intakeDuration,
    gamma: 1.1,
    lift: 11.0 * u.mm,
    steps: 100,
  });
  const exhaustLobe = harmonicCamLobe({
    durationAt50Thou: t.exhaustDuration,
    gamma: 1.1,
    lift: 10.0 * u.mm,
    steps: 100,
  });
  // Pent-roof four-valve heads: flow keeps rising to about 0.4" of lift.
  const intakePortFlow = portFlow(
    scale(
      t.intakePortCfm,
      [0, 0.2, 0.37, 0.55, 0.71, 0.84, 0.93, 0.98, 1.0, 1.0],
    ),
  );
  const exhaustPortFlow = portFlow(
    scale(
      t.exhaustPortCfm,
      [0, 0.18, 0.34, 0.5, 0.65, 0.78, 0.88, 0.95, 0.99, 1.0],
    ),
  );
  const head = (bank: 0 | 1): EngineSpec["banks"][number]["head"] => ({
    // 9.6:1 on 487.7 cc per cylinder.
    chamberVolume: 56.7 * u.cc,
    intakeRunnerVolume: 90 * u.cc,
    intakeRunnerArea: 13 * u.cm2,
    exhaustRunnerVolume: 45 * u.cc,
    exhaustRunnerArea: 9 * u.cm2,
    intakePortFlow,
    exhaustPortFlow,
    ...cams(bank, intakeLobe, exhaustLobe, 112 * u.deg, 116 * u.deg),
  });
  const piston = {
    mass: (330 + 90) * u.g, // forged piston + DLC-coated pin
    blowby: k28inH2O(0.1),
    compressionHeight,
    wristPinPosition: 0.0,
    displacement: 0.0,
  };
  const rod = {
    mass: rodMass,
    momentOfInertia: rodInertia(rodMass, rodLength),
    centerOfMass: 0.0,
    length: rodLength,
  };
  // Inconel manifolds: four equal-length primaries per bank into its turbine.
  const cylinder = (journal: number, exhaust: number, attenuation: number) => ({
    journal,
    piston,
    rod,
    intake: 0,
    exhaust,
    primaryLength: 0.42,
    soundAttenuation: attenuation,
  });
  const exhaust = {
    // The collector and turbine housing; the turbine is the opening to the rest.
    length: 0.3,
    collectorArea: u.pi * 0.025 * 0.025,
    outletFlowK: kArea(t.turbineArea, 0.9),
    primaryTubeLength: 0,
    primaryFlowK: kArea(4 * 9 * u.cm2, 0.9),
    velocityDecay: 1.0,
    audioVolume: 0.02,
    // engine-sim's exhaust sound, heard with ?exhaust=impulse in place of the acoustics below:
    // about the Beetle's level (the short pipes and two exhausts make this one loud).
    impulseResponse: "new/mild_exhaust.wav",
    impulseVolume: 0.00044,
  };
  return {
    simulationFrequency: 10000,
    // Half engine-sim's: the torque within 3 % of 8 substeps' (tuned for 4), the turbo's
    // transients the same, and the sound needs no more (engine/src/voice.rs).
    fluidSteps: 4,
    starterTorque: 150 * u.lb_ft,
    starterSpeed: 250 * u.rpm,
    throttleGamma: 2.0,
    fuel: gasoline({
      maxBurningEfficiency: 0.95,
      energyDensity: GASOLINE_LHV,
      // Direct injection and strong tumble: a fast burn.
      turbulenceToFlameSpeedRatio: sampled(
        5.0,
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45].map((x, i) => [
          x,
          FLAME_SPEED *
            (i === 0
              ? 3.0
              : [1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0][i - 1]! * x),
        ]),
      ),
    }),
    crankshaft: {
      throw: stroke / 2,
      mass: crankMass,
      flywheelMass,
      momentOfInertia:
        1.5 * diskInertia(crankMass, stroke) +
        diskInertia(flywheelMass, flywheelRadius) +
        // The dual-clutch's drum on the engine side, pumps, alternator, cam drive.
        diskInertia(8 * u.kg, 9 * u.cm),
      // Bearings, dry-sump scavenge and pressure pumps, the oil and water pumps.
      frictionTorque: 18 * u.lb_ft,
      x: 0.0,
      y: 0.0,
      tdc: 90 * u.deg + 45 * u.deg,
      journals: [0, 180 * u.deg, 180 * u.deg, 0],
    },
    banks: [0, 1].map((b) => ({
      angle: (b === 0 ? 45 : -45) * u.deg,
      bore,
      deckHeight: stroke / 2 + rodLength + compressionHeight,
      head: head(b as 0 | 1),
      cylinders: [0, 1, 2, 3].map((j) =>
        cylinder(
          j,
          b,
          [1.0, 0.95, 1.05, 0.98, 0.97, 1.03, 0.96, 1.02][b * 4 + j]!,
        ),
      ),
    })),
    // One plenum behind one throttle, fed by both turbos through their intercoolers.
    intakes: [
      {
        plenumVolume: 3.0 * u.L,
        plenumArea: 40.0 * u.cm2,
        flowK: kCarb(t.throttleCfm),
        idleFlowK: kCarb(0.0),
        runnerFlowK: kCarb(160.0),
        molecularAfr: 25.0 / 2.0,
        idleThrottlePlate: t.idleThrottlePlate,
        runnerLength: 0.15,
        velocityDecay: 0.5,
      },
    ],
    exhausts: [exhaust, { ...exhaust }],
    ignition: {
      timing: timingCurve(t.timing),
      fullLoad: {
        timing: timingCurve(t.fullLoadTiming),
        pressure: u.atm + t.fullLoadBoost * u.bar,
      },
      revLimit: 8000 * u.rpm,
      // The "wall effect": a hard cut right at 8000.
      limiterDuration: 0.04,
      firing: FIRING.map((k) => (k / 8) * cycle),
    },
    audio: {
      // engine-sim's synthesizer's, unused with the voice.
      hfGain: 0,
      noise: 0,
      jitter: 0.3,
      source: "flow",
      acoustics: ACOUSTICS,
      voice: VOICE,
    },
    physics: { realGas: true, wallTemperature: WALL_TEMPERATURE },
    turbo: {
      // Two IHI turbos as one: ~48 mm compressor wheels, TiAl turbine wheels on ball
      // bearings.
      compressorRadius: 24 * u.mm,
      inertia: 2 * 2.2e-5,
      compressorEfficiency: 0.74,
      turbineEfficiency: 0.7,
      // Every 250 rpm: the triangle filter only sees samples within its radius.
      boost: sampled(
        250 * u.rpm,
        Array.from({ length: 41 }, (_, i): [number, number] => [
          i * 250 * u.rpm,
          interpolate(t.boost, i * 250) * u.bar,
        ]),
      ),
      friction: 2 * 2.5e-6,
      intercooler: 0.8,
      ventFlow: 0.35,
      wastegateFlowK: kArea(t.wastegateArea, 0.8),
    },
  };
}

/** A spark advance table by rpm, sampled every 250 rpm (the filter sees only its radius). */
function timingCurve(table: [number, number][]): ReturnType<typeof sampled> {
  return sampled(
    250 * u.rpm,
    Array.from({ length: 41 }, (_, i): [number, number] => [
      i * 250 * u.rpm,
      interpolate(table, i * 250) * u.deg,
    ]),
  );
}

function interpolate(table: [number, number][], x: number): number {
  if (x <= table[0]![0]) return table[0]![1];
  for (let i = 1; i < table.length; i++) {
    const [x1, y1] = table[i]!;
    if (x <= x1) {
      const [x0, y0] = table[i - 1]!;
      return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
  }
  return table[table.length - 1]![1];
}

/**
 * The exhaust, per bank: twin-scroll Inconel manifolds (the outer cylinders, which fire
 * 360 degrees apart, into one scroll, the inner ones into the other) into the turbine,
 * then a downpipe to the catalyst and on to the rear silencer: a transverse box, packed,
 * with a perforated core from each bank's inlet to the tailpipe on its side (the Pista's
 * two large round tips, high in the diffuser) and bypass valves in them. The two banks'
 * pipes reach the box by routes round the gearbox of different lengths. The turbine,
 * between the manifold and the downpipe, blocks much of the pulses: its scrolls are
 * orifices with a high resistance. Cylinders in order 1-4 (bank 0), 5-8 (bank 1).
 *
 * The lengths, areas and losses the published figures leave open were fitted against
 * recordings of a 488 Pista on a dynamometer (engine/README.md, "Sound"): its 1/3-octave
 * spectra and engine orders at full throttle, 3000-7500 rpm.
 */
const ACOUSTICS: Acoustics = (() => {
  const banks = ["right", "left"];
  const cylinders = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => `cylinder ${n}`);
  // Outer (A) and inner (B) cylinders of each bank: firing 360 degrees apart.
  const scroll = (i: number) => (i % 4 === 0 || i % 4 === 3 ? "A" : "B");
  const sides = ["left", "right"];
  const PERFORATION = 80 * u.cm2;
  // From the catalyst to the silencer, per bank.
  const route = [0.8, 0.9];
  return {
    pipes: [
      ...cylinders.map((c, i) => ({
        a: c,
        b: `${banks[Math.floor(i / 4)]} scroll ${scroll(i)}`,
        length: scroll(i) === "A" ? 0.41 : 0.4,
        diameter: 0.036,
        temperature: 1100,
        damping: 4,
        // The pulses here are large (near a bar) and lose much of their energy to their
        // own flow, at the bends and where the primaries meet, alike at every frequency:
        // losses linear acoustics leave out, which would ring the primaries between the
        // valve and the turbine (~700 Hz) as the real car's recordings don't. Fitted to them.
        loss: 4,
      })),
      ...banks.flatMap((b, k) => [
        {
          a: `${b} turbine`,
          b: `${b} cat`,
          length: 0.45,
          diameter: 0.07,
          temperature: 950,
        },
        // The catalyst's monolith: many fine channels, a lossy stretch of pipe.
        {
          a: `${b} cat`,
          b: `${b} cat out`,
          length: 0.15,
          diameter: 0.11,
          temperature: 900,
          damping: 6,
        },
        {
          a: `${b} cat out`,
          b: `${b} inlet`,
          length: route[k]!,
          diameter: 0.065,
          temperature: 800,
        },
      ]),
      // The silencer's chamber, packed with mineral wool, round both banks' perforated
      // cores.
      {
        a: "left box",
        b: "right box",
        length: 0.7,
        diameter: 0.19,
        temperature: 700,
        absorption: 2,
      },
      ...sides.map((side) => ({
        a: `${side} tail`,
        b: `${side} tip`,
        length: 0.22,
        diameter: 0.09,
        temperature: 650,
      })),
    ],
    orifices: [
      // The cores' perforations, into the chamber and out of it to the tailpipes: with
      // the valves shut, the gas has to go round through it.
      ...sides.flatMap((side) =>
        [
          [`${side} inlet`, `${side} box`],
          [`${side} box`, `${side} tail`],
        ].map(([a, b]) => ({
          a: a!,
          b: b!,
          area: PERFORATION,
          length: 0.01,
          resistance: 0.3,
          temperature: 700,
        })),
      ),
      ...banks.flatMap((b) =>
        ["A", "B"].map((sc) => ({
          a: `${b} scroll ${sc}`,
          b: `${b} turbine`,
          // Each scroll's throat, half the turbine's flow area as the simulation has it
          // (turbineArea, Cd 0.9). Its flow resists the pulses by its jet's loss
          // (acoustics.rs), and the vanes and wheel more: Kabral et al. (ETC 2017) measured
          // over 10 dB of transmission loss across a twin-scroll turbine.
          area: 3 * u.cm2,
          discharge: 0.9,
          length: 0.06,
          resistance: 1.5,
          temperature: 1000,
        })),
      ),
    ],
    // No volume behind the ports: a blowdown is choked through its valve, a flow the
    // pipe's waves can't push back on (a rigid end to them), and the other valves are shut.
    volumes: banks.flatMap((b) => [
      { node: `${b} scroll A`, volume: 0.15 * u.L, temperature: 1050 },
      { node: `${b} scroll B`, volume: 0.15 * u.L, temperature: 1050 },
      { node: `${b} turbine`, volume: 0.6 * u.L, temperature: 950 },
    ]),
    ports: cylinders,
    outlets: ["left tip", "right tip"].map((node) => ({
      node,
      diameter: 0.09,
    })),
    // The chase camera.
    listenerDistance: 5,
    gain: 1,
    // The silencer's bypass valves: open, a straight core from each inlet to its
    // tailpipe, past the chamber. Ferrari's "optimised bypass logic" opens them with the
    // load and the revs (as Race mode has it, from ~3500 rpm or half throttle).
    valves: {
      pipes: sides.map((side) => ({
        a: `${side} inlet`,
        b: `${side} tail`,
        length: 0.25,
        diameter: 0.065,
        temperature: 700,
      })),
      speed: [3000 * u.rpm, 3800 * u.rpm],
      throttle: [0.3, 0.6],
    },
    // The wastegates, in the turbine housings: open, a way round the wheel into the
    // downpipe, for each scroll. They open with the boost controller, half way at full
    // throttle and 6000 rpm, and then let the pulses past, far less blocked than through
    // the turbine (the opening wastegate changes the spectrum: Kabral et al., ETC 2017).
    wastegate: banks.flatMap((b) =>
      ["A", "B"].map((sc) => ({
        a: `${b} scroll ${sc}`,
        b: `${b} turbine`,
        // Half the engine's wastegate area (wastegateArea) each, as the simulation has it.
        area: 6 * u.cm2,
        discharge: 0.8,
        length: 0.04,
        resistance: 1,
        temperature: 1000,
      })),
    ),
    // The pulses steepen into shocks in the primaries, where they are near a bar (past
    // the turbines, far smaller, hardly).
    nonlinear: {
      nodes: banks.map((b) => `${b} turbine`),
      length: 0,
      temperature: 900,
      primaries: { length: 0.4, diameter: 0.036, temperature: 1100 },
    },
  };
})();

/**
 * The voice (engine/src/voice.rs). The flow noise near its physical estimates, fitted with
 * the acoustics above (the exhaust's highs are its shocks' harmonics, not noise); the
 * turbochargers' whoosh kept low, as the Pista's compressors breathe through ducts in its
 * flanks.
 */
const VOICE: VoiceSpec = {
  // Full throttle's peaks near the limiter a little below full scale.
  fullScale: 2,
  // About half full power's: the losses with the engine working hard.
  acousticFlow: 0.25,
  variation: 0.02,
  portNoise: 0.03,
  tailNoise: 1,
  bay: 3e-5,
  valvetrain: 1,
  injectors: 3,
  whoosh: 0.01,
  whistle: 0.3,
  compressorBlades: 12,
  // The F154's bypass valves recirculate to the compressor inlet rather than vent: what is
  // heard of a lift is the compressor's flutter through the intake ducts, faintly.
  blowOff: 0.03,
  // engine-sim's recorded response (?exhaust=impulse) at about the computed level.
  impulseGain: 0.34,
};

/**
 * Calibrated on the dynamometer (engine/examples/calibrate.rs): 770 Nm from 3000 rpm and
 * 720 cv at 8000 in 7th gear at up to 1.55 bar, 33-36 % brake efficiency at full load.
 */
export const F154_TUNING: F154Tuning = {
  // Idles at ~950-1000 rpm, warm and unloaded.
  idleThrottlePlate: 0.9968,
  intakeDuration: 220 * u.deg,
  exhaustDuration: 216 * u.deg,
  throttleCfm: 1100,
  // Four-valve ports for an 86.5 mm bore: below engine-sim's F136 example (280/246 cfm for
  // a 94 mm bore), as the valve area scales.
  intakePortCfm: 240,
  exhaustPortCfm: 200,
  // The boost for Ferrari's curve in 7th, fitted on the dynamometer (calibrate.rs
  // `boost`): 770 Nm from 3000 rpm, 720 cv at 8000 (refitted for 4 fluid substeps, which
  // make 1-3 % more torque than 8). It peaks at 1.55 bar, against the published ~1.5
  // (21.6 psi). Below 3000 the turbos can't make it yet (~0.6 bar at 2500).
  boost: [
    [0, 1.545],
    [3000, 1.545],
    [3500, 1.49],
    [4000, 1.46],
    [5000, 1.445],
    [6000, 1.36],
    [7000, 1.286],
    [7500, 1.245],
    [8000, 1.2],
  ],
  // Sized against each other on the dynamometer: the turbine small enough to make the
  // published 1.5 bar by 3000 rpm, the wastegate large enough to hold the boost target at
  // 8000 (a smaller turbine spools no sooner here and costs power; a larger one can't reach
  // the boost).
  turbineArea: 6 * u.cm2,
  wastegateArea: 12 * u.cm2,
  // Best-torque timing at part load: half burned ~8 degrees after top dead centre
  // (engine/examples/calibrate.rs, `ca50 ... 8 base`), smoothed.
  timing: [
    [0, 11],
    [1000, 11],
    [2000, 18.5],
    [3000, 18],
    [4000, 16],
    [5000, 12],
    [6000, 12.5],
    [7000, 10],
    [8000, 9.5],
  ],
  // Knock-limited at full boost: half burned ~22 degrees after top dead centre up to
  // 3000 rpm, easing to ~10 at 8000 (an estimate from the CRC AVFL-20 dynamometer data for a
  // turbocharged direct-injection engine, which retards to 22 degrees by 15 bar BMEP on 91
  // RON at 10.5:1; the F154 runs ~25 bar on 98 RON at 9.6:1).
  fullLoadTiming: [
    [2000, 4],
    [3000, 3.5],
    [4000, 4],
    [5000, 3],
    [6000, 5.5],
    [7000, 7],
    [8000, 7],
  ],
  fullLoadBoost: 1.0,
};
export const f154cd: EngineSpec = f154Engine(F154_TUNING);
