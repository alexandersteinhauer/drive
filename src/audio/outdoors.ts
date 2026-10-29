/**
 * Puts the engine outdoors. The simulated sound is the exhaust heard directly, as if in an
 * anechoic room: no surroundings at all, and the same signal in both ears, which the ear
 * places close, inside a small dead space. Outdoors you hear the direct sound plus, a little
 * later and much quieter, what comes back from the land around: trees, banks, the far side
 * of a valley. Those reflections arrive late (tens to hundreds of metres away), sparse at
 * first, from every direction (different in each ear), and dull with distance (air and
 * foliage take the highs). There are no dense early reflections, which are what make a room.
 *
 * `outdoorImpulse` builds that response; `Outdoors` mixes it, by convolution, with the
 * untouched direct sound.
 */

export interface OutdoorOptions {
  /** Length of the response (s). */
  length: number;
  /** First reflection (s): the nearest surroundings, ~12 m away and back. */
  firstReflection: number;
  /** Time for the reflections to die away by 60 dB (s). */
  decay: number;
  /** Discrete echoes per second per ear, before they merge into the diffuse tail. */
  echoRate: number;
  /** Low-pass cutoff (Hz) at the first reflection, and how fast it falls (s per e-fold). */
  brightness: number;
  dulling: number;
  /** Seed: the same surroundings every time. */
  seed: number;
}

export const OUTDOORS: OutdoorOptions = {
  length: 1.6,
  firstReflection: 0.035,
  decay: 1.3,
  echoRate: 30,
  brightness: 5000,
  dulling: 0.35,
  seed: 1,
};

/** Small seeded generator (mulberry32). */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The surroundings' response for each ear (left, right), without the direct sound. Its
 * gain is ~1 (0 dB) where the filters pass, so a sound's reflections carry about its own
 * energy, before the mix level.
 */
export function outdoorImpulse(
  sampleRate: number,
  o: OutdoorOptions = OUTDOORS,
): [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>] {
  const n = Math.round(o.length * sampleRate);
  const first = Math.round(o.firstReflection * sampleRate);
  // Amplitude decay: -60 dB over `decay`.
  const k = Math.log(1000) / o.decay;
  return [0, 1].map((ear) => {
    const rand = random(o.seed * 2 + ear);
    const x = new Float32Array(n);
    // Discrete echoes off nearer things, scattered, each its own strength and sign.
    for (let t = o.firstReflection; t < o.length;) {
      const i = Math.round(t * sampleRate);
      const strength = 0.4 + 0.6 * rand();
      x[i]! += (rand() < 0.5 ? -1 : 1) * strength * Math.exp(-k * t) * 2.5;
      t += -Math.log(1 - rand()) / o.echoRate;
    }
    // The diffuse rest, rising as more of the land answers (fully by ~4x the first).
    for (let i = first; i < n; i++) {
      const t = i / sampleRate;
      const rise = Math.min(
        1,
        (t - o.firstReflection) / (3 * o.firstReflection),
      );
      x[i]! += (rand() * 2 - 1) * 0.25 * rise * Math.exp(-k * t);
    }
    // Unit energy before filtering: the filters only take away what they stop.
    let energy = 0;
    for (let i = 0; i < n; i++) energy += x[i]! * x[i]!;
    const scale = 1 / Math.sqrt(energy);
    for (let i = 0; i < n; i++) x[i]! *= scale;
    // Duller with distance: a one-pole low-pass, its cutoff falling along the response.
    let y = 0;
    for (let i = first; i < n; i++) {
      const t = (i - first) / sampleRate;
      const fc = Math.max(500, o.brightness * Math.exp(-t / o.dulling));
      const a = 1 - Math.exp((-2 * Math.PI * fc) / sampleRate);
      y += a * (x[i]! - y);
      x[i] = y;
    }
    // A second pole, so the highs fall off steeply rather than leaving a hiss.
    y = 0;
    for (let i = first; i < n; i++) {
      const t = (i - first) / sampleRate;
      const fc = Math.max(500, o.brightness * Math.exp(-t / o.dulling));
      const a = 1 - Math.exp((-2 * Math.PI * fc) / sampleRate);
      y += a * (x[i]! - y);
      x[i] = y;
    }
    // No rumble below what trees and banks reflect.
    let lp = 0;
    const hp = 1 - Math.exp((-2 * Math.PI * 60) / sampleRate);
    for (let i = 0; i < n; i++) {
      lp += hp * (x[i]! - lp);
      x[i] = x[i]! - lp;
    }
    return x;
  }) as [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>];
}

/**
 * Direct sound plus the surroundings: `input` to `output`, stereo. `level` is the
 * surroundings' energy against the direct sound's (dB).
 */
export class Outdoors {
  readonly input: GainNode;
  readonly output: GainNode;
  private readonly wet: GainNode;
  private enabled = true;

  constructor(
    private readonly context: BaseAudioContext,
    private readonly level = -12,
    options: OutdoorOptions = OUTDOORS,
  ) {
    this.input = context.createGain();
    this.output = context.createGain();
    this.wet = context.createGain();
    this.wet.gain.value = 10 ** (level / 20);
    const convolver = context.createConvolver();
    convolver.normalize = false;
    const [left, right] = outdoorImpulse(context.sampleRate, options);
    const buffer = context.createBuffer(2, left.length, context.sampleRate);
    buffer.copyToChannel(left, 0);
    buffer.copyToChannel(right, 1);
    convolver.buffer = buffer;
    // The engine is mono; the response gives each ear its own surroundings.
    this.input.channelCount = 1;
    this.input.channelCountMode = "explicit";
    this.input.connect(this.output);
    this.input.connect(convolver).connect(this.wet).connect(this.output);
  }

  /** On, or the direct sound alone. */
  get on(): boolean {
    return this.enabled;
  }

  set on(on: boolean) {
    this.enabled = on;
    this.wet.gain.setTargetAtTime(
      on ? 10 ** (this.level / 20) : 0,
      this.context.currentTime,
      0.05,
    );
  }
}
