// AudioWorklet turning the simulated engine's exhaust signal (simulation steps, sent from the
// game each frame) into sound with engine-sim's synthesizer (engine/, WebAssembly).
import { EngineModule } from "../sim/engineSim";
import { Loudness } from "./loudness";

export interface EngineProcessorOptions {
  wasm: ArrayBuffer;
  /** The engine: EngineSpec as UTF-8 JSON (worklets may lack TextEncoder). */
  spec: ArrayBuffer;
  /** Its simulation steps per second, and audio channels (cylinders with acoustics). */
  frequency: number;
  channels: number;
  /** Without acoustics in the spec: the recorded impulse response per exhaust system. */
  impulses: { wav: ArrayBuffer; volume: number }[];
}

/** From the game: the steps simulated since the last message. */
export interface EngineInput {
  /** Exhaust signal per exhaust system, one value per step. */
  samples: Float32Array[];
  /** How closed the throttle was on overrun, 0..1 (see Loudness). */
  overrun: number;
}

/** To the game. */
export type EngineMessage =
  | { type: "ready" }
  | { type: "error"; message: string }
  /** Buffered sound (s) and share of real time the synthesizer took, over a second. */
  | { type: "stats"; buffered: number; load: number };

/** Buffered audio to aim for (samples): enough to ride out a late frame. */
const TARGET = 2048;

class EngineProcessor extends AudioWorkletProcessor {
  private module: EngineModule | null = null;
  private synth = 0;
  private readonly loudness = new Loudness(sampleRate);
  private readonly frequency: number;
  private readonly channels: number;
  /** Overrun per buffered stretch of input: [audio samples, overrun]. */
  private readonly overruns: [number, number][] = [];
  private playing = false;
  private busyMs = 0;
  private blocks = 0;

  constructor(options: { processorOptions: EngineProcessorOptions }) {
    super();
    const o = options.processorOptions;
    this.frequency = o.frequency;
    this.channels = o.channels;
    this.port.onmessage = (e: MessageEvent<EngineInput>) => this.input(e.data);
    EngineModule.load(o.wasm)
      .then((m) => {
        const x = m.exports;
        // With acoustics, this computes the exhaust's impulse responses (~0.1 s).
        const json = m.write(new Uint8Array(o.spec));
        this.synth = x.synth_for_engine(json.ptr, json.len);
        m.free(json);
        if (!this.synth) throw new Error("engine spec rejected");
        o.impulses.forEach((imp, i) => {
          const wav = m.write(new Uint8Array(imp.wav));
          const ok = x.synth_set_impulse_wav(
            this.synth,
            i,
            wav.ptr,
            wav.len,
            imp.volume,
          );
          m.free(wav);
          if (!ok) throw new Error("unreadable impulse response");
        });
        // A fixed gain in place of engine-sim's leveller, which would hold every sound at
        // one loudness; Loudness narrows the range instead.
        x.synth_set_gain(this.synth, 0.003);
        this.module = m;
        this.post({ type: "ready" });
      })
      .catch((e: unknown) => this.post({ type: "error", message: String(e) }));
  }

  private post(message: EngineMessage): void {
    this.port.postMessage(message);
  }

  private input({ samples, overrun }: EngineInput): void {
    const m = this.module;
    const steps = samples[0]?.length ?? 0;
    if (!m || steps === 0) return;
    const x = m.exports;
    const before = x.synth_input_available(this.synth);
    // Trim the resampling by up to 2 % to hold the buffer near TARGET: the game's clock
    // and the audio clock drift apart.
    const error = Math.max(-1, Math.min(1, (before - TARGET) / TARGET));
    x.synth_set_input_rate(this.synth, this.frequency * (1 + 0.02 * error));
    const block = m.write(new Uint8Array(steps * this.channels * 4));
    const data = m.f32(block.ptr, steps * this.channels);
    samples.forEach((s, c) => data.set(s, c * steps));
    x.synth_write(this.synth, block.ptr, steps, this.channels);
    m.free(block);
    const added = x.synth_input_available(this.synth) - before;
    this.overruns.push([added, overrun]);
    // Far ahead (the game caught up after a stall): drop the excess.
    const available = x.synth_input_available(this.synth);
    if (available > 4 * TARGET) {
      x.synth_drop_input(this.synth, TARGET);
      this.skip(available - TARGET);
    }
  }

  /** Forgets the overrun of `n` samples of input. */
  private skip(n: number): void {
    while (n > 0 && this.overruns.length) {
      const head = this.overruns[0]!;
      const take = Math.min(n, head[0]);
      head[0] -= take;
      n -= take;
      if (head[0] <= 0) this.overruns.shift();
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const m = this.module;
    const out = outputs[0];
    if (!m || !out || out.length === 0) return true;
    const frames = out[0]!.length;
    const x = m.exports;

    const t0 = Date.now();
    const available = x.synth_input_available(this.synth);
    // After running dry, wait for a full buffer before playing again.
    if (!this.playing && available >= TARGET) this.playing = true;
    if (this.playing && available < frames) this.playing = false;
    if (this.playing) {
      const overrun = this.overruns[0]?.[1] ?? 0;
      const ptr = x.synth_render(this.synth, frames) >> 2;
      const samples = new Float32Array(
        x.memory.buffer,
        ptr << 2,
        frames,
      ).slice();
      this.skip(frames);
      this.loudness.process(samples, overrun);
      for (const channel of out) channel.set(samples);
    }
    this.busyMs += Date.now() - t0;

    if (++this.blocks * frames >= sampleRate) {
      this.post({
        type: "stats",
        buffered: x.synth_input_available(this.synth) / sampleRate,
        load: this.busyMs / ((this.blocks * frames * 1000) / sampleRate),
      });
      this.busyMs = this.blocks = 0;
    }
    return true;
  }
}

registerProcessor("engine-sim", EngineProcessor);
