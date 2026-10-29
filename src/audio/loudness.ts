/**
 * Narrows the simulated engine's dynamic range. With engine-sim's own leveler off (it would
 * hold every sound at one loudness), the exhaust of an engine held on a closed throttle is
 * ~20 dB quieter than at full throttle, and idle ~30 dB below full throttle at high rpm:
 * true to the exhaust, but a real car also has its fan, valvetrain and intake, and the gap
 * is too wide to listen to.
 *
 * Two stages: a boost that rises as the throttle closes on overrun (`es_overrun`), in step
 * with it (an envelope follower alone would only notice the drop afterwards, leaving a hole
 * at every lift), then gentle compression around `refDb` by `ratio` for the rest (rpm).
 * The output is delayed by `LOOKAHEAD` samples so the gain moves with engine-sim's own
 * transients (the manifold emptying or filling when the throttle snaps), not after them.
 */
const LOOKAHEAD = 1024; // 23 ms at 44.1 kHz
export class Loudness {
  private env = 0;
  private gain = 1;
  private overrun = 0;
  private readonly delay = new Float32Array(LOOKAHEAD);
  private delayPos = 0;
  /** Boost on a fully closed throttle (dB), and how fast it follows (s). */
  closedBoostDb = 12;
  boostTime = 0.03;

  constructor(
    private readonly sampleRate: number,
    private readonly refDb = -12,
    private readonly ratio = 2,
    private readonly maxGainDb = 24,
    private readonly attack = 0.03,
    private readonly release = 0.03,
  ) {}

  /** Processes `samples` in place; `overrun` (0..1) is how closed the throttle is. */
  process(samples: Float32Array, overrun: number): void {
    const dt = samples.length / this.sampleRate;
    this.overrun +=
      (overrun - this.overrun) * (1 - Math.exp(-dt / this.boostTime));
    const boost = 10 ** ((this.closedBoostDb * this.overrun) / 20);

    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i]! * samples[i]!;
    const power = (sum / samples.length) * boost * boost;
    const tau = power > this.env ? this.attack : this.release;
    this.env += (power - this.env) * (1 - Math.exp(-dt / tau));

    const inDb = 10 * Math.log10(this.env + 1e-12);
    const outDb = this.refDb + (inDb - this.refDb) / this.ratio;
    const target = boost * 10 ** (Math.min(this.maxGainDb, outDb - inDb) / 20);
    // Ramp across the block so gain changes don't click.
    const g0 = this.gain;
    const step = (target - g0) / samples.length;
    for (let i = 0; i < samples.length; i++) {
      const delayed = this.delay[this.delayPos]!;
      this.delay[this.delayPos] = samples[i]!;
      this.delayPos = (this.delayPos + 1) % LOOKAHEAD;
      const v = delayed * (g0 + step * (i + 1));
      // Soft limit near full scale.
      samples[i] = v > 0.9 || v < -0.9 ? Math.tanh(v) : v;
    }
    this.gain = target;
  }
}
