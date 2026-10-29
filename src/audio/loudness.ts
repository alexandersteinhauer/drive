/**
 * The engine sound's loudness on its way out. The voice's levels are physical (its
 * pascals at the listener over a fixed full scale): a car at full throttle is tens of dB
 * louder than at idle, more than is comfortable through speakers. Gentle compression
 * around `refDb` by `ratio` narrows that, leaving most of the step between throttle open
 * and shut, which is what makes an engine sound sporty (Zeitler & Zeller, SAE
 * 2006-01-0098: perceived sportiness tracks the rise in level on a load change), and a
 * soft limit near full scale catches the rest. The output is delayed by `LOOKAHEAD`
 * samples so the gain moves with the sound's own transients, not after them.
 */
const LOOKAHEAD = 1024; // 23 ms at 44.1 kHz
export class Loudness {
  private env = 0;
  private gain = 1;
  private readonly delay = new Float32Array(LOOKAHEAD);
  private delayPos = 0;

  constructor(
    private readonly sampleRate: number,
    private readonly refDb = -14,
    private readonly ratio = 1.4,
    private readonly maxGainDb = 18,
    private readonly attack = 0.02,
    private readonly release = 0.25,
  ) {}

  /** Processes `samples` in place. */
  process(samples: Float32Array): void {
    const dt = samples.length / this.sampleRate;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i]! * samples[i]!;
    const power = sum / samples.length;
    const tau = power > this.env ? this.attack : this.release;
    this.env += (power - this.env) * (1 - Math.exp(-dt / tau));

    const inDb = 10 * Math.log10(this.env + 1e-12);
    const outDb = this.refDb + (inDb - this.refDb) / this.ratio;
    const target = 10 ** (Math.min(this.maxGainDb, outDb - inDb) / 20);
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
