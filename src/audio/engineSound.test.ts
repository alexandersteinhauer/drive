import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";
import { ferrari488Pista } from "../sim/config";
import type { EngineSpec } from "../sim/engineSpec";
import type {
  EngineInput,
  EngineMessage,
  EngineProcessorOptions,
} from "./engineWorklet";
import { originalExhaustSpec } from "./surrogateModel";

mock.module("./engineWorklet.ts?worker&url", () => ({
  default: "/worklet.js",
}));
mock.module("../sim/engine.wasm?url", () => ({ default: "/engine.wasm" }));
let EngineSoundClass: typeof import("./engineSound").EngineSound;

class FakePort {
  onmessage: ((e: MessageEvent<EngineMessage>) => void) | null = null;
  sent: EngineInput[] = [];
  closed = false;
  postMessage(message: EngineInput): void {
    this.sent.push(message);
  }
  close(): void {
    this.closed = true;
  }
  emit(message: EngineMessage): void {
    this.onmessage?.({ data: message } as MessageEvent<EngineMessage>);
  }
}
class FakeParam {
  value = 0;
  cancelScheduledValues(): void {}
  setValueAtTime(v: number): void {
    this.value = v;
  }
  setTargetAtTime(v: number): void {
    this.value = v;
  }
}
class FakeGain {
  gain = new FakeParam();
  connect(): this {
    return this;
  }
  disconnect(): void {}
}
class FakeContext {
  static all: FakeContext[] = [];
  state = "suspended";
  currentTime = 0;
  destination = {};
  gain = new FakeGain();
  audioWorklet = { addModule: async () => {} };
  constructor() {
    FakeContext.all.push(this);
  }
  createGain(): FakeGain {
    return this.gain;
  }
  async resume(): Promise<void> {
    this.state = "running";
  }
  async suspend(): Promise<void> {
    this.state = "suspended";
  }
  async close(): Promise<void> {
    this.state = "closed";
  }
}
class FakeNode {
  static all: FakeNode[] = [];
  static rejectLearned = false;
  static rejectOriginal = false;
  port = new FakePort();
  onprocessorerror: (() => void) | null = null;
  disconnected = false;
  constructor(
    _context: FakeContext,
    _name: string,
    readonly options: { processorOptions: EngineProcessorOptions },
  ) {
    FakeNode.all.push(this);
    queueMicrotask(() =>
      this.port.emit(
        (
          options.processorOptions.model
            ? FakeNode.rejectLearned
            : FakeNode.rejectOriginal
        )
          ? { type: "error", message: "test initialization error" }
          : { type: "ready" },
      ),
    );
  }
  connect(): void {}
  disconnect(): void {
    this.disconnected = true;
  }
}

const globals = [
  "AudioContext",
  "AudioWorkletNode",
  "document",
  "fetch",
] as const;
const saved = new Map(
  globals.map((name) => [
    name,
    Object.getOwnPropertyDescriptor(globalThis, name),
  ]),
);
let missing = false;
let requests: string[] = [];
let warnings: ReturnType<typeof spyOn<typeof console, "warn">>;
const sounds: import("./engineSound").EngineSound[] = [];

beforeAll(async () => {
  EngineSoundClass = (await import("./engineSound")).EngineSound;
});
beforeEach(() => {
  missing = false;
  requests = [];
  FakeContext.all = [];
  FakeNode.all = [];
  FakeNode.rejectLearned = false;
  FakeNode.rejectOriginal = false;
  warnings = spyOn(console, "warn").mockImplementation(() => {});
  Object.assign(globalThis, {
    AudioContext: FakeContext,
    AudioWorkletNode: FakeNode,
    document: {
      hidden: false,
      addEventListener: () => {},
      removeEventListener: () => {},
    },
    fetch: async (url: string) => {
      requests.push(url);
      if (url.includes("pista-coverage-candidate")) {
        if (missing) return new Response("missing", { status: 404 });
        return Response.json({
          cylinders: 8,
          rate: 20000,
          widths: [1, 1, 1, 1, 1],
          kernels: [[[], []]],
        });
      }
      return new Response(new ArrayBuffer(8));
    },
  });
});
afterEach(() => {
  for (const sound of sounds.splice(0)) sound.dispose();
  warnings.mockRestore();
  for (const [name, descriptor] of saved) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
});
afterAll(() => mock.restore());

function spec(): EngineSpec {
  const cfg = structuredClone(ferrari488Pista);
  if (
    cfg.powertrain.type !== "combustion" ||
    cfg.powertrain.engine.type !== "simulated"
  )
    throw new Error("expected simulated Pista");
  return cfg.powertrain.engine.spec;
}
async function create(engine = spec()) {
  const sound = await EngineSoundClass.create(engine);
  sounds.push(sound);
  return sound;
}
const samples = (channels: number) =>
  Array.from({ length: channels }, () => new Float32Array(32));

describe("default learned exhaust fallback lifecycle", () => {
  test("default Pista requests the coverage candidate and initializes learned audio", async () => {
    const sound = await create();
    expect(sound.surrogate).toBe(true);
    expect(requests.filter((url) => url.includes("/surrogate/"))).toEqual([
      `${import.meta.env.BASE_URL}audio/surrogate/pista-coverage-candidate.json`,
    ]);
    expect(FakeNode.all[0]?.options.processorOptions.channels).toBe(13);
    expect(sound.warning).toBeNull();
  });
  test("original mode creates no model request or surrogate worklet", async () => {
    const sound = await create(originalExhaustSpec(spec()));
    expect(sound.surrogate).toBe(false);
    expect(
      requests.some((url) => url.includes("pista-coverage-candidate")),
    ).toBe(false);
    expect(FakeNode.all[0]?.options.processorOptions.model).toBeUndefined();
    expect(sound.warning).toBeNull();
  });

  test("missing asset initializes a matching original-voice stream", async () => {
    missing = true;
    const sound = await create();
    expect(sound.surrogate).toBe(false);
    expect(sound.warning).toContain("Using original");
    expect(FakeNode.all).toHaveLength(1);
    const options = FakeNode.all[0]!.options.processorOptions;
    expect(options.channels).toBe(18);
    expect(
      JSON.parse(new TextDecoder().decode(options.spec)).audio.surrogate,
    ).toBeUndefined();
    expect(warnings).toHaveBeenCalledTimes(1);
  });

  test("worklet model rejection releases its node and initializes original voice", async () => {
    FakeNode.rejectLearned = true;
    const sound = await create();
    expect(sound.surrogate).toBe(false);
    expect(FakeNode.all).toHaveLength(2);
    expect(FakeNode.all[0]!.disconnected).toBe(true);
    expect(FakeNode.all[0]!.port.closed).toBe(true);
    expect(FakeNode.all[1]!.options.processorOptions.channels).toBe(18);
  });

  test("runtime error uses a fresh port and changes simulation format before accepting audio", async () => {
    const sound = await create();
    expect(sound.surrogate).toBe(true);
    const old = FakeNode.all[0]!;
    let changeBeforeAccept = false;
    const changed = new Promise<void>((resolve) => {
      sound.onBackendChange = (learned) => {
        expect(learned).toBe(false);
        changeBeforeAccept = sound.node === null;
        resolve();
      };
    });
    old.port.emit({ type: "error", message: "runtime test failure" });
    sound.push(samples(13));
    expect(old.port.sent.filter((m) => "samples" in m)).toHaveLength(0);
    await changed;
    expect(changeBeforeAccept).toBe(true);
    expect(sound.surrogate).toBe(false);
    expect(old.disconnected).toBe(true);
    sound.push(samples(18));
    expect(
      FakeNode.all[1]!.port.sent.filter((m) => "samples" in m),
    ).toHaveLength(1);
    expect(warnings).toHaveBeenCalledTimes(1);
  });

  test("processorerror also triggers original fallback", async () => {
    const sound = await create();
    const changed = new Promise<void>((resolve) => {
      sound.onBackendChange = () => resolve();
    });
    FakeNode.all[0]!.onprocessorerror?.();
    await changed;
    expect(sound.surrogate).toBe(false);
  });

  test("switching away immediately mutes, suspends and clears cached car audio", async () => {
    const sound = await create();
    sound.resume();
    sound.setVolume(1);
    expect(FakeContext.all[0]!.state).toBe("running");
    sound.stop();
    expect(FakeContext.all[0]!.gain.gain.value).toBe(0);
    expect(FakeContext.all[0]!.state).toBe("suspended");
    expect(FakeNode.all[0]!.port.sent.at(-1)).toEqual({ type: "clear" });
  });

  test("failure of both backends closes the unusable context", async () => {
    FakeNode.rejectLearned = true;
    FakeNode.rejectOriginal = true;
    await expect(create()).rejects.toThrow("initialization error");
    expect(FakeContext.all[0]!.state).toBe("closed");
    expect(FakeNode.all.every((node) => node.disconnected)).toBe(true);
  });
});
