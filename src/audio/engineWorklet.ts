// AudioWorklet running engine-sim (compiled to wasm) on the audio thread: each 128-frame
// block it takes the car's engine speed and throttle and simulates the engine that long.
import "./workletGlobals";
import createModule, { type EngineSimModule } from "./engine-sim.mjs";
import { Loudness } from "./loudness";

export interface EngineProcessorOptions {
  wasm: ArrayBuffer;
  script: string;
  /** Simulation steps per second; 0 for the script's own. */
  frequency: number;
}

/** Messages to the main thread. */
export type EngineMessage =
  | { type: "ready"; initMs: number }
  | { type: "error"; message: string }
  /** Share of real time the simulation took over the last second, engine rpm. */
  | { type: "stats"; load: number; rpm: number };

class EngineProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors(): AudioParamDescriptor[] {
    return [
      {
        name: "rpm",
        defaultValue: 850,
        minValue: 0,
        maxValue: 10000,
        automationRate: "k-rate",
      },
      {
        name: "engaged",
        defaultValue: 1,
        minValue: 0,
        maxValue: 1,
        automationRate: "k-rate",
      },
      {
        name: "throttle",
        defaultValue: 0,
        minValue: 0,
        maxValue: 1,
        automationRate: "k-rate",
      },
    ];
  }

  private sim: EngineSimModule | null = null;
  private readonly loudness = new Loudness(sampleRate);
  private readonly block = new Float32Array(128);
  private busyMs = 0;
  private blocks = 0;

  constructor(options: { processorOptions: EngineProcessorOptions }) {
    super();
    const { wasm, script, frequency } = options.processorOptions;
    createModule({
      instantiateWasm: (imports, done) => {
        void WebAssembly.instantiate(wasm, imports).then((r) =>
          done(r.instance),
        );
        return {};
      },
      print: console.log,
      printErr: console.warn,
    })
      .then((m) => {
        const t0 = Date.now();
        const path = m.stringToNewUTF8(script);
        const status = m._es_init(path, frequency);
        m._free(path);
        if (status !== 0) throw new Error(`engine-sim init failed (${status})`);
        this.sim = m;
        this.post({ type: "ready", initMs: Date.now() - t0 });
      })
      .catch((e: unknown) => this.post({ type: "error", message: String(e) }));
  }

  private post(message: EngineMessage): void {
    this.port.postMessage(message);
  }

  process(
    _inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean {
    const sim = this.sim;
    const out = outputs[0];
    if (!sim || !out || out.length === 0) return true;
    const frames = out[0]!.length;

    const t0 = Date.now();
    sim._es_set_controls(
      parameters.rpm![0]!,
      parameters.throttle![0]!,
      parameters.engaged![0]! >= 0.5 ? 1 : 0,
    );
    const ptr = sim._es_render(frames) >> 2;
    const samples =
      frames === this.block.length ? this.block : new Float32Array(frames);
    samples.set(sim.HEAPF32.subarray(ptr, ptr + frames));
    this.loudness.process(samples, sim._es_overrun());
    for (const channel of out) channel.set(samples);
    this.busyMs += Date.now() - t0;

    if (++this.blocks * frames >= sampleRate) {
      this.post({
        type: "stats",
        load: this.busyMs / ((this.blocks * frames * 1000) / sampleRate),
        rpm: sim._es_engine_rpm(),
      });
      this.busyMs = this.blocks = 0;
    }
    return true;
  }
}

registerProcessor("engine-sim", EngineProcessor);
