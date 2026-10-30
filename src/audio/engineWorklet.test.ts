import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  spyOn,
  test,
} from "bun:test";
import { EngineModule } from "../sim/engineSim";
import type {
  EngineInput,
  EngineMessage,
  EngineProcessorOptions,
} from "./engineWorklet";

class Port {
  onmessage: ((event: MessageEvent<EngineInput>) => void) | null = null;
  sent: EngineMessage[] = [];
  postMessage(message: EngineMessage): void {
    this.sent.push(message);
  }
  send(input: EngineInput): void {
    this.onmessage?.({ data: input } as MessageEvent<EngineInput>);
  }
}

const oldGlobals = {
  AudioWorkletProcessor: Object.getOwnPropertyDescriptor(
    globalThis,
    "AudioWorkletProcessor",
  ),
  registerProcessor: Object.getOwnPropertyDescriptor(
    globalThis,
    "registerProcessor",
  ),
  sampleRate: Object.getOwnPropertyDescriptor(globalThis, "sampleRate"),
};
let Processor: typeof import("./engineWorklet").EngineProcessor;
beforeAll(async () => {
  Object.assign(globalThis, {
    AudioWorkletProcessor: class {
      port = new Port();
    },
    registerProcessor: () => {},
    sampleRate: 48000,
  });
  Processor = (await import("./engineWorklet")).EngineProcessor;
});
afterAll(() => {
  for (const [name, descriptor] of Object.entries(oldGlobals)) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
});
let loadSpy: ReturnType<typeof spyOn<typeof EngineModule, "load">> | undefined;
afterEach(() => loadSpy?.mockRestore());

function harness(options: Partial<EngineProcessorOptions> = {}) {
  const calls: { name: string; args: number[] }[] = [];
  let available = 4096;
  let failed = false;
  let output = 0.2;
  let ptr = 64;
  const memory = new WebAssembly.Memory({ initial: 1 });
  const record =
    (name: string, result = 0) =>
    (...args: number[]) => {
      calls.push({ name, args });
      return result;
    };
  const exports = {
    memory,
    alloc: (len: number) => {
      const out = ptr;
      ptr += len;
      return out;
    },
    dealloc: record("dealloc"),
    surrogate_voice_new: record("surrogate_voice_new", 7),
    surrogate_voice_free: record("surrogate_voice_free"),
    surrogate_voice_reset: record("surrogate_voice_reset"),
    surrogate_voice_set_input_rate: record("surrogate_voice_set_input_rate"),
    surrogate_voice_write: record("surrogate_voice_write"),
    surrogate_voice_input_available: () => available,
    surrogate_voice_drop_input: record("surrogate_voice_drop_input"),
    surrogate_voice_render: (_v: number, frames: number) => {
      new Float32Array(memory.buffer, 32000, frames).fill(output);
      return 32000;
    },
    surrogate_voice_failed: () => Number(failed),
    voice_for_engine: record("voice_for_engine", 9),
    voice_free: record("voice_free"),
    voice_set_input_rate: record("voice_set_input_rate"),
    voice_write: record("voice_write"),
    voice_input_available: () => available,
    voice_drop_input: record("voice_drop_input"),
    voice_render: record("voice_render", 32000),
    synth_for_engine: record("synth_for_engine", 11),
    synth_set_gain: record("synth_set_gain"),
    synth_free: record("synth_free"),
    synth_rendered: () => 128,
  };
  const module = new EngineModule({
    exports,
  } as unknown as WebAssembly.Instance);
  loadSpy = spyOn(EngineModule, "load").mockResolvedValue(module);
  const processor = new Processor({
    processorOptions: {
      wasm: new ArrayBuffer(0),
      spec: new ArrayBuffer(0),
      model: new ArrayBuffer(4),
      fullScale: 16,
      frequency: 10000,
      channels: 13,
      voice: true,
      impulses: [],
      ...options,
    },
  });
  const port = processor.port as unknown as Port;
  return {
    processor,
    port,
    calls,
    setFailed: () => {
      failed = true;
    },
    setOutput: (value: number) => {
      output = value;
    },
    setAvailable: (value: number) => {
      available = value;
    },
  };
}

/** Finish the explicitly mocked module initialization, not a wall-clock wait. */
async function ready(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}
const block = (channels = 13, steps = 32) =>
  Array.from({ length: channels }, () => new Float32Array(steps));

describe("engine worklet learned exhaust", () => {
  test("selects only the surrogate and passes actual device rate", async () => {
    const h = harness();
    await ready();
    expect(h.port.sent).toEqual([{ type: "ready" }]);
    const create = h.calls.find((c) => c.name === "surrogate_voice_new");
    expect(create?.args.slice(2)).toEqual([10000, 48000, 16]);
    expect(h.calls.some((c) => c.name === "voice_for_engine")).toBe(false);
    h.port.send({ samples: block() });
    expect(
      h.calls.find((c) => c.name === "surrogate_voice_write")?.args.slice(2),
    ).toEqual([32, 13]);
  });

  test("original backend does not instantiate the model", async () => {
    const h = harness({ model: undefined, channels: 18 });
    await ready();
    expect(h.calls.some((c) => c.name === "surrogate_voice_new")).toBe(false);
    expect(h.calls.some((c) => c.name === "voice_for_engine")).toBe(true);
    h.port.send({ samples: block(18) });
    expect(h.calls.some((c) => c.name === "voice_write")).toBe(true);
  });

  test("bad channel counts stop once with an error, rather than corrupting memory", async () => {
    const h = harness();
    await ready();
    h.port.send({ samples: block(18) });
    h.port.send({ samples: block(18) });
    expect(h.port.sent.filter((m) => m.type === "error")).toHaveLength(1);
    expect(h.calls.some((c) => c.name === "surrogate_voice_write")).toBe(false);
    expect(
      h.calls.filter((c) => c.name === "surrogate_voice_free"),
    ).toHaveLength(1);
  });

  test.each(["unequal lengths", "nonfinite input"] as const)(
    "%s cannot reach the model",
    async (kind) => {
      const h = harness();
      await ready();
      const samples = block();
      if (kind === "unequal lengths") samples[2] = new Float32Array(1);
      else samples[2]![0] = NaN;
      h.port.send({ samples });
      expect(h.port.sent.at(-1)?.type).toBe("error");
      expect(h.calls.some((c) => c.name === "surrogate_voice_write")).toBe(
        false,
      );
    },
  );

  test("adapter processing failure signals fallback and leaves output silent", async () => {
    const h = harness();
    await ready();
    h.setFailed();
    h.port.send({ samples: block() });
    const out = [[new Float32Array(128), new Float32Array(128)]];
    expect(h.processor.process([], out)).toBe(true);
    expect(h.port.sent.at(-1)?.type).toBe("error");
    expect(out[0]![0]!.every((v) => v === 0)).toBe(true);
  });

  test("nonfinite render is silenced before loudness or speakers", async () => {
    const h = harness();
    await ready();
    h.setOutput(Infinity);
    const out = [[new Float32Array(128), new Float32Array(128)]];
    h.processor.process([], out);
    expect(h.port.sent.at(-1)?.type).toBe("error");
    expect(out[0]![0]!.every((v) => v === 0)).toBe(true);
  });

  test("clear discards inactive car audio; dispose frees backend once", async () => {
    const h = harness();
    await ready();
    h.port.send({ type: "clear" });
    expect(
      h.calls.find((c) => c.name === "surrogate_voice_reset")?.args,
    ).toEqual([7]);
    h.port.send({ type: "dispose" });
    expect(h.processor.process([], [[new Float32Array(128)]])).toBe(false);
    expect(
      h.calls.filter((c) => c.name === "surrogate_voice_free"),
    ).toHaveLength(1);
  });
});
