import { beforeAll, describe, expect, test } from "bun:test";
import { withPistaSurrogateExhaust } from "../sim/config";
import { EngineModule, EngineSim } from "../sim/engineSim";
import type { EngineSpec } from "../sim/engineSpec";
import { originalExhaustSpec } from "./surrogateModel";

let bytes: ArrayBuffer;
let spec: EngineSpec;
beforeAll(async () => {
  bytes = await Bun.file(
    new URL("../sim/engine.wasm", import.meta.url),
  ).arrayBuffer();
  const config = withPistaSurrogateExhaust();
  if (
    config.powertrain.type !== "combustion" ||
    config.powertrain.engine.type !== "simulated"
  )
    throw new Error("Pista must use the simulated engine");
  spec = config.powertrain.engine.spec;
});

describe("shipped learned Pista through real WASM", () => {
  test("game-rate raw flow reaches the model and the actual 44.1 kHz stream", async () => {
    // Separate memories, just like the main thread and AudioWorklet. No mocks or teacher inputs.
    const simulation = await EngineModule.load(bytes);
    const audio = await EngineModule.load(bytes);
    const engine = new EngineSim(spec, simulation);
    const x = audio.exports;
    const data = await Bun.file(
      new URL(
        "../../public/audio/surrogate/pista-pilot.json",
        import.meta.url,
      ),
    ).arrayBuffer();
    const model = audio.write(new Uint8Array(data));
    const s = x.surrogate_voice_new(
      model.ptr,
      model.len,
      engine.frequency,
      44100,
      16,
    );
    audio.free(model);
    expect(s).not.toBe(0);
    engine.ignition = true;
    engine.speed = (3000 * Math.PI) / 30;
    engine.dyno(true, engine.speed);
    engine.throttle = 0.8;
    engine.collectAudio = true;
    engine.surrogateAudio = true;
    let frames = 0,
      power = 0,
      peak = 0;
    try {
      for (let step = 0; step < 120; step++) {
        engine.step(167);
        const samples = engine.takeAudio();
        expect(samples.length).toBe(13);
        const n = samples[0]!.length;
        const block = audio.write(new Uint8Array(n * 13 * 4));
        const planar = audio.f32(block.ptr, n * 13);
        samples.forEach((channel, i) => planar.set(channel, i * n));
        x.surrogate_voice_write(s, block.ptr, n, 13);
        audio.free(block);
        while (x.surrogate_voice_input_available(s) > 0) {
          const count = Math.min(128, x.surrogate_voice_input_available(s));
          const ptr = x.surrogate_voice_render(s, count);
          expect(x.synth_rendered()).toBe(count);
          for (const sample of audio.f32(ptr, count)) {
            if (!Number.isFinite(sample))
              throw new Error("nonfinite learned PCM");
            power += sample * sample;
            peak = Math.max(peak, Math.abs(sample));
          }
          frames += count;
        }
      }
      expect(x.surrogate_voice_failed(s)).toBe(0);
      expect(frames).toBeGreaterThan(85000);
      expect(frames).toBeLessThan(90000);
      expect(Math.sqrt(power / frames)).toBeGreaterThan(1e-5);
      expect(peak).toBeLessThan(10);
      engine.surrogateAudio = false;
      engine.step(2);
      expect(engine.takeAudio().length).toBe(18);
      x.surrogate_voice_reset(s);
      expect(x.surrogate_voice_input_available(s)).toBe(0);
    } finally {
      x.surrogate_voice_free(s);
      engine.dispose();
    }
  });

  test("original fallback does not require the optional selector in an older WASM", async () => {
    const actual = await EngineModule.load(bytes);
    const legacy = new EngineModule({
      exports: { ...actual.exports, engine_set_surrogate_audio: undefined },
    } as unknown as WebAssembly.Instance);
    const engine = new EngineSim(originalExhaustSpec(spec), legacy);
    try {
      expect(() => {
        engine.surrogateAudio = false;
      }).not.toThrow();
      expect(() => {
        engine.surrogateAudio = true;
      }).toThrow("updated engine WASM");
    } finally {
      engine.dispose();
    }
  });
});
