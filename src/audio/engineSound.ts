// The worklet bundled on its own (Vite), as a URL for audioWorklet.addModule.
// oxlint-disable-next-line import/default
import workletUrl from "./engineWorklet.ts?worker&url";
import type { EngineMessage, EngineProcessorOptions } from "./engineWorklet";

/**
 * The combustion engine's sound, simulated by engine-sim on the audio thread. The car's
 * simulation stays in charge: `update` hands it the engine speed and throttle each frame,
 * and the simulated engine is held at that speed while it works at that throttle.
 */
export class EngineSound {
  /** Share of real time the simulation takes (1 = it can't keep up), engine rpm. */
  stats = { load: 0, rpm: 0 };
  private readonly gain: GainNode;
  private readonly rpm: AudioParam;
  private readonly throttle: AudioParam;
  private readonly engaged: AudioParam;

  private constructor(
    readonly context: AudioContext,
    readonly node: AudioWorkletNode,
  ) {
    this.gain = context.createGain();
    this.gain.gain.value = 0;
    node.connect(this.gain).connect(context.destination);
    this.rpm = node.parameters.get("rpm")!;
    this.throttle = node.parameters.get("throttle")!;
    this.engaged = node.parameters.get("engaged")!;
    // Hidden, the game stops but audio would drone on at the last speed.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) void context.suspend();
      else if (this.started) void context.resume();
    });
  }

  /** Whether the user has interacted, so the audio may play. */
  private started = false;

  /** Loads the simulator and compiles the engine script (`/engines/<name>.mr`). */
  static async create(engine: string, frequency = 0): Promise<EngineSound> {
    // engine-sim synthesizes at 44.1 kHz.
    const context = new AudioContext({
      sampleRate: 44100,
      latencyHint: "interactive",
    });
    const [wasm] = await Promise.all([
      fetch(import.meta.env.BASE_URL + "audio/engine-sim.wasm").then((r) =>
        r.arrayBuffer(),
      ),
      context.audioWorklet.addModule(workletUrl),
    ]);
    const processorOptions: EngineProcessorOptions = {
      wasm,
      script: `/engines/${engine}.mr`,
      frequency,
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
        if (m.type === "ready") {
          console.info(`engine-sim: ${engine} ready in ${m.initMs} ms`);
          resolve();
        } else if (m.type === "error") reject(new Error(m.message));
        else sound.stats = { load: m.load, rpm: m.rpm };
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

  /**
   * Per frame: engine speed (rpm), throttle 0..1, loudness 0..1 (0 mutes), and whether the
   * clutch is engaged (out for a gear change, the simulated engine revs down by itself).
   */
  update(rpm: number, throttle: number, volume: number, engaged = true): void {
    const t = this.context.currentTime;
    this.rpm.setTargetAtTime(rpm, t, 0.01);
    this.throttle.setValueAtTime(throttle, t);
    this.engaged.setValueAtTime(engaged ? 1 : 0, t);
    this.gain.gain.setTargetAtTime(volume, t, 0.08);
  }
}
