// Simulated engine channels become sound on the audio thread: the original voice,
// engine-sim's synthesizer, or the opt-in learned exhaust pilot. Never mix their sources.
import { EngineModule } from "../sim/engineSim";
import { Loudness } from "./loudness";

export interface EngineProcessorOptions {
  wasm: ArrayBuffer;
  /** The engine: EngineSpec as UTF-8 JSON (worklets may lack TextEncoder). */
  spec: ArrayBuffer;
  /** Actual game simulation rate, unchanged by the audio backend. */
  frequency: number;
  channels: number;
  voice: boolean;
  /** Optional learned exhaust asset, already checked against the engine's interface. */
  model?: ArrayBuffer;
  fullScale?: number;
  /** Only the original backend uses recorded impulse responses. */
  impulses: { wav: ArrayBuffer; volume: number }[];
}

/** From the game: planar simulation-step channels, or a lifecycle command. */
export type EngineInput =
  { samples: Float32Array[] } | { type: "clear" | "dispose" };

/** To the game. Errors stop this node; EngineSound replaces it with the original voice. */
export type EngineMessage =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "stats"; buffered: number; load: number };

/** Buffered audio to aim for (device frames): enough to ride out a late game frame. */
const TARGET = 2048;

interface Synth {
  setInputRate(rate: number): void;
  write(ptr: number, steps: number, channels: number): void;
  available(): number;
  drop(keep: number): void;
  render(frames: number): number;
  failed(): boolean;
  clear(): void;
  dispose(): void;
}

function synthesizer(m: EngineModule, o: EngineProcessorOptions): Synth {
  const x = m.exports;
  if (o.model) {
    if (!Number.isFinite(o.fullScale) || !(o.fullScale! > 0))
      throw new Error("invalid exhaust full scale");
    const model = m.write(new Uint8Array(o.model));
    let v: number;
    try {
      // The adapter resamples raw engine-rate flow to the model's trained rate, then
      // pressures to the actual device rate. It adds no voice, jet or mechanical noise.
      v = x.surrogate_voice_new(
        model.ptr,
        model.len,
        o.frequency,
        sampleRate,
        o.fullScale!,
      );
    } finally {
      m.free(model);
    }
    if (!v) throw new Error("exhaust model rejected");
    return {
      setInputRate: (r) => x.surrogate_voice_set_input_rate(v, r),
      write: (p, n, c) => x.surrogate_voice_write(v, p, n, c),
      available: () => x.surrogate_voice_input_available(v),
      drop: (k) => x.surrogate_voice_drop_input(v, k),
      render: (f) => x.surrogate_voice_render(v, f),
      failed: () => x.surrogate_voice_failed(v) !== 0,
      clear: () => {
        if (typeof x.surrogate_voice_reset === "function")
          x.surrogate_voice_reset(v);
        else x.surrogate_voice_drop_input(v, 0);
      },
      dispose: () => x.surrogate_voice_free(v),
    };
  }

  const json = m.write(new Uint8Array(o.spec));
  let handle: number;
  try {
    handle = o.voice
      ? x.voice_for_engine(json.ptr, json.len)
      : x.synth_for_engine(json.ptr, json.len);
  } finally {
    m.free(json);
  }
  if (!handle) throw new Error("engine spec rejected");
  const synth: Synth = o.voice
    ? {
        setInputRate: (r) => x.voice_set_input_rate(handle, r),
        write: (p, n, c) => x.voice_write(handle, p, n, c),
        available: () => x.voice_input_available(handle),
        drop: (k) => x.voice_drop_input(handle, k),
        render: (f) => x.voice_render(handle, f),
        failed: () => false,
        clear: () => x.voice_drop_input(handle, 0),
        dispose: () => x.voice_free(handle),
      }
    : {
        setInputRate: (r) => x.synth_set_input_rate(handle, r),
        write: (p, n, c) => x.synth_write(handle, p, n, c),
        available: () => x.synth_input_available(handle),
        drop: (k) => x.synth_drop_input(handle, k),
        render: (f) => x.synth_render(handle, f),
        failed: () => false,
        clear: () => x.synth_drop_input(handle, 0),
        dispose: () => x.synth_free(handle),
      };
  try {
    // Fixed gain instead of engine-sim's leveller. Loudness gently narrows the range.
    if (!o.voice) x.synth_set_gain(handle, 0.003);
    for (const [i, impulse] of o.impulses.entries()) {
      const wav = m.write(new Uint8Array(impulse.wav));
      let ok: number;
      try {
        const set = o.voice ? x.voice_set_impulse_wav : x.synth_set_impulse_wav;
        ok = set(handle, i, wav.ptr, wav.len, impulse.volume);
      } finally {
        m.free(wav);
      }
      if (!ok) throw new Error("unreadable impulse response");
    }
  } catch (error) {
    synth.dispose();
    throw error;
  }
  return synth;
}

export class EngineProcessor extends AudioWorkletProcessor {
  private module: EngineModule | null = null;
  private synth: Synth | null = null;
  private loudness = new Loudness(sampleRate);
  private readonly frequency: number;
  private readonly channels: number;
  private playing = false;
  private busyMs = 0;
  private blocks = 0;
  private failed = false;
  private disposed = false;

  constructor(options: { processorOptions: EngineProcessorOptions }) {
    super();
    const o = options.processorOptions;
    this.frequency = o.frequency;
    this.channels = o.channels;
    this.port.onmessage = (e: MessageEvent<EngineInput>) => this.input(e.data);
    EngineModule.load(o.wasm)
      .then((m) => {
        if (this.disposed) return;
        if (
          !Number.isFinite(o.frequency) ||
          o.frequency <= 0 ||
          !Number.isInteger(o.channels) ||
          o.channels <= 0
        )
          throw new Error("invalid engine audio interface");
        this.synth = synthesizer(m, o);
        this.module = m;
        this.post({ type: "ready" });
      })
      .catch((e: unknown) => this.fail(e));
  }

  private post(message: EngineMessage): void {
    this.port.postMessage(message);
  }

  private fail(error: unknown): void {
    if (this.failed || this.disposed) return;
    this.failed = true;
    this.playing = false;
    const synth = this.synth;
    this.synth = null;
    try {
      synth?.dispose();
    } catch {
      // A trapped module cannot safely be reused; the main thread replaces this node.
    }
    this.post({ type: "error", message: String(error) });
  }

  private input(input: EngineInput): void {
    if ("type" in input) {
      try {
        if (input.type === "dispose") {
          this.disposed = true;
          const synth = this.synth;
          this.synth = null;
          synth?.dispose();
        } else {
          this.synth?.clear();
          this.loudness = new Loudness(sampleRate);
        }
        this.playing = false;
      } catch (error) {
        this.fail(error);
      }
      return;
    }
    const m = this.module;
    const synth = this.synth;
    if (!m || !synth || this.failed || this.disposed) return;
    try {
      const { samples } = input;
      if (samples.length !== this.channels)
        throw new Error(
          `engine audio needs ${this.channels} channels, got ${samples.length}`,
        );
      const steps = samples[0]?.length ?? 0;
      for (const channel of samples) {
        if (!(channel instanceof Float32Array) || channel.length !== steps)
          throw new Error("engine audio channel lengths do not match");
        for (const value of channel)
          if (!Number.isFinite(value))
            throw new Error("non-finite engine audio input");
      }
      if (steps === 0) return;
      const before = synth.available();
      // Trim by up to 2% to hold the buffer near TARGET as game and audio clocks drift.
      const error = Math.max(-1, Math.min(1, (before - TARGET) / TARGET));
      synth.setInputRate(this.frequency * (1 + 0.02 * error));
      const block = m.write(new Uint8Array(steps * this.channels * 4));
      try {
        const data = m.f32(block.ptr, steps * this.channels);
        samples.forEach((s, c) => data.set(s, c * steps));
        synth.write(block.ptr, steps, this.channels);
      } finally {
        m.free(block);
      }
      if (synth.failed()) throw new Error("learned exhaust processing failed");
      // Far ahead (the game caught up after a stall): discard the excess.
      if (synth.available() > 4 * TARGET) synth.drop(TARGET);
    } catch (error) {
      this.fail(error);
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    if (this.disposed) return false;
    const m = this.module;
    const synth = this.synth;
    const out = outputs[0];
    if (!m || !synth || !out?.length) return true;
    const frames = out[0]!.length;
    const t0 = Date.now();
    try {
      const available = synth.available();
      if (synth.failed()) throw new Error("learned exhaust processing failed");
      // After running dry, wait for a full buffer before playing again.
      if (!this.playing && available >= TARGET) this.playing = true;
      if (this.playing && available < frames) this.playing = false;
      if (this.playing) {
        const ptr = synth.render(frames);
        const count = m.exports.synth_rendered();
        if (
          !Number.isInteger(count) ||
          count < 0 ||
          count > frames ||
          synth.failed()
        )
          throw new Error("invalid engine audio output");
        const samples = m.f32(ptr, count).slice();
        for (const value of samples)
          if (!Number.isFinite(value))
            throw new Error("non-finite engine audio output");
        if (count) this.loudness.process(samples);
        for (const channel of out) channel.set(samples);
        if (count < frames) this.playing = false;
      }
      this.busyMs += Date.now() - t0;
      if (++this.blocks * frames >= sampleRate) {
        this.post({
          type: "stats",
          buffered: synth.available() / sampleRate,
          load: this.busyMs / ((this.blocks * frames * 1000) / sampleRate),
        });
        this.busyMs = this.blocks = 0;
      }
    } catch (error) {
      for (const channel of out) channel.fill(0);
      this.fail(error);
    }
    return true;
  }
}

registerProcessor("engine-sim", EngineProcessor);
