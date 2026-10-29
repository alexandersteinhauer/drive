// The worklet bundled on its own (Vite), as a URL for audioWorklet.addModule.
// oxlint-disable-next-line import/default
import workletUrl from "./engineWorklet.ts?worker&url";
import wasmUrl from "../sim/engine.wasm?url";
import type { EngineSpec } from "../sim/engineSpec";
import { Outdoors } from "./outdoors";
import type {
  EngineInput,
  EngineMessage,
  EngineProcessorOptions,
} from "./engineWorklet";

/**
 * The combustion engine's sound: the game simulates the engine and hands over its exhaust
 * signal each frame (`push`); engine-sim's synthesizer turns it into audio on the audio
 * thread.
 */
export class EngineSound {
  /** Buffered sound (s) and the synthesizer's share of real time. */
  stats = { buffered: 0, load: 0 };
  private readonly gain: GainNode;
  /** The surroundings, added to the engine's direct sound. */
  readonly outdoors: Outdoors;
  /** Whether the user has interacted, so the audio may play. */
  private started = false;

  private constructor(
    readonly context: AudioContext,
    readonly node: AudioWorkletNode,
  ) {
    this.gain = context.createGain();
    this.gain.gain.value = 0;
    this.outdoors = new Outdoors(context);
    node.connect(this.outdoors.input);
    this.outdoors.output.connect(this.gain).connect(context.destination);
    // Hidden, the game stops but audio would drone on.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) void context.suspend();
      else if (this.started) void context.resume();
    });
  }

  static async create(spec: EngineSpec): Promise<EngineSound> {
    const context = new AudioContext({
      sampleRate: 44100,
      latencyHint: "interactive",
    });
    const base = import.meta.env.BASE_URL + "audio/impulse/";
    const [wasm, impulses] = await Promise.all([
      fetch(wasmUrl).then((r) => r.arrayBuffer()),
      Promise.all(
        (spec.audio.acoustics ? [] : spec.exhausts).map(async (e) => ({
          wav: await (await fetch(base + e.impulseResponse)).arrayBuffer(),
          volume: e.impulseVolume,
        })),
      ),
      context.audioWorklet.addModule(workletUrl),
    ]);
    const processorOptions: EngineProcessorOptions = {
      wasm,
      spec: new TextEncoder().encode(JSON.stringify(spec))
        .buffer as ArrayBuffer,
      frequency: spec.simulationFrequency,
      channels: spec.audio.acoustics
        ? spec.banks.reduce((n, b) => n + b.cylinders.length, 0)
        : spec.exhausts.length,
      impulses,
    };
    const node = new AudioWorkletNode(context, "engine-sim", {
      numberOfInputs: 0,
      outputChannelCount: [2],
      processorOptions,
    });
    const sound = new EngineSound(context, node);
    await new Promise<void>((resolve, reject) => {
      node.port.onmessage = (e: MessageEvent<EngineMessage>) => {
        const m = e.data;
        if (m.type === "ready") resolve();
        else if (m.type === "error") reject(new Error(m.message));
        else sound.stats = { buffered: m.buffered, load: m.load };
      };
    });
    return sound;
  }

  /** Starts the audio once the page may play it (after a key press or click). */
  resume(): void {
    this.started = true;
    if (this.context.state === "suspended" && !document.hidden)
      void this.context.resume();
  }

  /** The steps simulated since the last call, and how closed the throttle is on overrun. */
  push(samples: Float32Array[], overrun: number): void {
    if (!samples[0]?.length) return;
    const input: EngineInput = { samples, overrun };
    this.node.port.postMessage(
      input,
      samples.map((s) => s.buffer as ArrayBuffer),
    );
  }

  /** Loudness 0..1 (0 mutes). */
  setVolume(volume: number): void {
    this.gain.gain.setTargetAtTime(volume, this.context.currentTime, 0.08);
  }
}
