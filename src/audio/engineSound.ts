// The worklet bundled on its own (Vite), as a URL for audioWorklet.addModule.
// oxlint-disable-next-line import/default
import workletUrl from "./engineWorklet.ts?worker&url";
import wasmUrl from "../sim/engine.wasm?url";
import { audioChannels, type EngineSpec } from "../sim/engineSpec";
import { fetchSurrogateModel, originalExhaustSpec } from "./surrogateModel";
import type {
  EngineInput,
  EngineMessage,
  EngineProcessorOptions,
} from "./engineWorklet";

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!response.ok)
    throw new Error(`audio request failed (${response.status}): ${url}`);
  return response.arrayBuffer();
}

/** Engine audio, with a fresh original-voice worklet if the optional learned pilot fails. */
export class EngineSound {
  stats = { buffered: 0, load: 0 };
  /** Called before any differently formatted input can reach the replacement node. */
  onBackendChange: ((surrogate: boolean) => void) | null = null;
  onWarning: ((message: string) => void) | null = null;
  warning: string | null = null;
  private readonly gain: GainNode;
  private activeNode: AudioWorkletNode | null = null;
  private learned = false;
  private recovering = false;
  private disposed = false;
  private started = false;
  private volume = 0;

  private constructor(
    readonly context: AudioContext,
    private readonly wasm: ArrayBuffer,
    private readonly fallback: EngineSpec,
  ) {
    this.gain = context.createGain();
    this.gain.gain.value = 0;
    this.gain.connect(context.destination);
    document.addEventListener("visibilitychange", this.visibility);
  }

  get node(): AudioWorkletNode | null {
    return this.activeNode;
  }

  /** Actual backend, not just the requested setting. The simulation must match this. */
  get surrogate(): boolean {
    return this.learned;
  }

  private readonly visibility = (): void => {
    if (document.hidden) void this.context.suspend();
    else if (this.started) void this.context.resume();
  };

  static async create(spec: EngineSpec): Promise<EngineSound> {
    const context = new AudioContext({
      sampleRate: 44100,
      latencyHint: "interactive",
    });
    let sound: EngineSound | null = null;
    try {
      const [wasm] = await Promise.all([
        fetchBytes(wasmUrl),
        context.audioWorklet.addModule(workletUrl),
      ]);
      sound = new EngineSound(context, wasm, originalExhaustSpec(spec));
      if (spec.audio.surrogate) {
        try {
          const model = await fetchSurrogateModel(
            spec,
            import.meta.env.BASE_URL,
          );
          await sound.attach(spec, model);
        } catch (error) {
          sound.warnFallback(error);
          await sound.attach(sound.fallback);
        }
      } else await sound.attach(spec);
      return sound;
    } catch (error) {
      if (sound) sound.dispose();
      else void context.close();
      throw error;
    }
  }

  private async attach(spec: EngineSpec, model?: ArrayBuffer): Promise<void> {
    const base = import.meta.env.BASE_URL + "audio/impulse/";
    const impulses = await Promise.all(
      (model || spec.audio.acoustics ? [] : spec.exhausts).map(async (e) => ({
        wav: await fetchBytes(base + e.impulseResponse),
        volume: e.impulseVolume,
      })),
    );
    if (this.disposed) throw new Error("engine sound disposed");
    const processorOptions: EngineProcessorOptions = {
      wasm: this.wasm,
      spec: new TextEncoder().encode(JSON.stringify(spec))
        .buffer as ArrayBuffer,
      frequency: spec.simulationFrequency,
      channels: audioChannels(spec),
      voice: !!spec.audio.voice,
      model,
      fullScale: spec.audio.surrogate?.fullScale,
      impulses,
    };
    const node = new AudioWorkletNode(this.context, "engine-sim", {
      numberOfInputs: 0,
      outputChannelCount: [2],
      processorOptions,
    });
    try {
      await new Promise<void>((resolve, reject) => {
        let ready = false;
        const timeout = setTimeout(
          () => reject(new Error("engine audio initialization timed out")),
          15000,
        );
        const error = (reason: unknown): void => {
          if (ready) this.fail(node, reason);
          else {
            clearTimeout(timeout);
            reject(reason);
          }
        };
        node.onprocessorerror = () =>
          error(new Error("engine audio worklet stopped"));
        node.port.onmessage = (e: MessageEvent<EngineMessage>) => {
          const message = e.data;
          if (message.type === "ready") {
            ready = true;
            clearTimeout(timeout);
            resolve();
          } else if (message.type === "error")
            error(new Error(message.message));
          else if (node === this.activeNode)
            this.stats = { buffered: message.buffered, load: message.load };
        };
      });
      if (this.disposed) throw new Error("engine sound disposed");
      this.learned = !!model;
      // Clear the old format on the simulation before accepting any new input. A
      // fallback uses an entirely new port, so old queued raw-flow messages cannot leak.
      this.onBackendChange?.(this.learned);
      this.activeNode = node;
      node.connect(this.gain);
      this.setVolume(this.volume);
    } catch (error) {
      this.release(node);
      throw error;
    }
  }

  private warnFallback(error: unknown): void {
    this.warning = "Learned Pista exhaust unavailable. Using original exhaust.";
    console.warn(this.warning, error);
    this.onWarning?.(this.warning);
  }

  private fail(node: AudioWorkletNode, error: unknown): void {
    if (node !== this.activeNode || this.disposed) return;
    this.activeNode = null;
    this.release(node);
    this.mute();
    this.stats = { buffered: 0, load: 0 };
    if (this.learned && !this.recovering) {
      this.recovering = true;
      this.warnFallback(error);
      void this.attach(this.fallback)
        .catch((fallbackError: unknown) => {
          this.warning = "Original engine sound is also unavailable.";
          console.warn(this.warning, fallbackError);
          this.onWarning?.(this.warning);
        })
        .finally(() => {
          this.recovering = false;
        });
    } else {
      this.warning = "Engine sound unavailable.";
      console.warn(this.warning, error);
      this.onWarning?.(this.warning);
    }
  }

  private release(node: AudioWorkletNode): void {
    node.port.onmessage = null;
    node.onprocessorerror = null;
    node.port.postMessage({ type: "dispose" } satisfies EngineInput);
    node.disconnect();
    node.port.close();
  }

  private mute(): void {
    this.gain.gain.cancelScheduledValues(this.context.currentTime);
    this.gain.gain.setValueAtTime(0, this.context.currentTime);
  }

  /** Starts after a key press or click, including an interaction before loading finished. */
  resume(): void {
    this.started = true;
    if (this.context.state === "suspended" && !document.hidden)
      void this.context.resume();
  }

  /** Deactivate a cached car's sound, discarding input so switching cannot leave a drone. */
  stop(): void {
    this.started = false;
    this.volume = 0;
    this.mute();
    this.activeNode?.port.postMessage({ type: "clear" } satisfies EngineInput);
    void this.context.suspend();
  }

  push(samples: Float32Array[]): void {
    if (!this.activeNode || !samples[0]?.length) return;
    const input: EngineInput = { samples };
    this.activeNode.port.postMessage(
      input,
      samples.map((s) => s.buffer as ArrayBuffer),
    );
  }

  /** Loudness 0..1 (0 mutes). */
  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume));
    if (this.activeNode)
      this.gain.gain.setTargetAtTime(
        this.volume,
        this.context.currentTime,
        0.08,
      );
  }

  dispose(): void {
    this.disposed = true;
    document.removeEventListener("visibilitychange", this.visibility);
    this.mute();
    if (this.activeNode) this.release(this.activeNode);
    this.activeNode = null;
    this.gain.disconnect();
    void this.context.close();
  }
}
