import { describe, expect, test } from "bun:test";
import { ferrari488Pista, vwBeetle, withImpulseExhaust } from "../sim/config";
import { f154cd } from "../sim/engines/f154cd";
import { audioChannels, type EngineSpec } from "../sim/engineSpec";
import {
  fetchSurrogateModel,
  originalExhaustSpec,
  validateSurrogateModel,
  type ModelRequest,
} from "./surrogateModel";

function pistaSpec(): EngineSpec {
  const cfg = structuredClone(ferrari488Pista);
  if (
    cfg.powertrain.type !== "combustion" ||
    cfg.powertrain.engine.type !== "simulated"
  )
    throw new Error("expected simulated Pista");
  return cfg.powertrain.engine.spec;
}

function modelBytes(overrides: Record<string, unknown> = {}): ArrayBuffer {
  return new TextEncoder().encode(
    JSON.stringify({
      cylinders: 8,
      rate: 20000,
      widths: [1, 1, 1, 1, 1],
      kernels: [[[], []]],
      ...overrides,
    }),
  ).buffer as ArrayBuffer;
}

describe("default Pista exhaust", () => {
  test("uses the coverage candidate with unchanged calibration and engine physics", () => {
    const spec = pistaSpec();
    expect(spec.audio.surrogate).toEqual({
      model: "audio/surrogate/pista-coverage-candidate.json",
      fullScale: 16,
      flapSpeed: [261.7993875, 366.5191425],
      flapThrottle: [0.3, 0.6],
    });
    expect(originalExhaustSpec(spec)).toEqual(f154cd);
    expect(f154cd.audio.surrogate).toBeUndefined();
    expect(spec.simulationFrequency).toBe(10000);
    expect(spec.fluidSteps).toBe(4);
  });

  test("surrogate stream takes precedence over fallback voice", () => {
    const spec = pistaSpec();
    expect(audioChannels(spec)).toBe(8 + 5);
    expect(audioChannels(originalExhaustSpec(spec))).toBe(8 + 10);
    delete spec.audio.voice;
    expect(audioChannels(spec)).toBe(8 + 5);
    delete spec.audio.surrogate;
    expect(audioChannels(spec)).toBe(spec.exhausts.length);
  });

  test("fallback and diagnostic impulse configs do not mutate the default", () => {
    const before = structuredClone(ferrari488Pista);
    const spec = pistaSpec();
    expect(originalExhaustSpec(spec).audio.surrogate).toBeUndefined();
    expect(spec.audio.surrogate).toBeDefined();
    const impulse = withImpulseExhaust(ferrari488Pista);
    if (
      impulse.powertrain.type !== "combustion" ||
      impulse.powertrain.engine.type !== "simulated"
    )
      throw new Error("expected simulated Pista");
    expect(impulse.powertrain.engine.spec.audio.surrogate).toBeUndefined();
    expect(impulse.powertrain.engine.spec.audio.acoustics).toBeUndefined();
    expect(ferrari488Pista).toEqual(before);
  });
});

describe("learned exhaust model asset", () => {
  test("fallback and other engines never fetch the learned asset", async () => {
    let requests = 0;
    const request = (async () => {
      requests++;
      throw new Error("unexpected fetch");
    }) as ModelRequest;
    if (
      vwBeetle.powertrain.type !== "combustion" ||
      vwBeetle.powertrain.engine.type !== "simulated"
    )
      throw new Error("expected simulated Beetle");
    const results = await Promise.all(
      [
        originalExhaustSpec(pistaSpec()),
        f154cd,
        vwBeetle.powertrain.engine.spec,
      ].map((spec) => fetchSurrogateModel(spec, "/game/", request)),
    );
    expect(results).toEqual([undefined, undefined, undefined]);
    expect(requests).toBe(0);
  });

  test("default loads only the coverage candidate relative to the deployment base", async () => {
    const requested: string[] = [];
    const bytes = modelBytes();
    const request = (async (url: string | URL | Request) => {
      requested.push(String(url));
      return new Response(bytes);
    }) as ModelRequest;
    expect(await fetchSurrogateModel(pistaSpec(), "/game/", request)).toEqual(
      bytes,
    );
    expect(requested).toEqual([
      "/game/audio/surrogate/pista-coverage-candidate.json",
    ]);
  });

  test("default loads the real coverage candidate with unchanged calibration", async () => {
    const spec = pistaSpec();
    const path = spec.audio.surrogate!.model;
    const bytes = await Bun.file(`public/${path}`).arrayBuffer();
    expect(() => validateSurrogateModel(bytes, spec)).not.toThrow();
    const model = JSON.parse(new TextDecoder().decode(bytes));
    expect(model.rate).toBe(20000);
    expect(model.nativeRate).toBe(10000);
    expect(model.kernels.length).toBe(6);
    expect(model.kernels[0][0][0].length).toBe(512);
    expect(spec.audio.surrogate!.fullScale).toBe(16);
  });

  test("failed HTTP, network errors, and malformed model reject for voice fallback", async () => {
    const notFound = (async () =>
      new Response("missing", { status: 404 })) as ModelRequest;
    await expect(
      fetchSurrogateModel(pistaSpec(), "/", notFound),
    ).rejects.toThrow("404");
    const offline = (async () => {
      throw new Error("offline");
    }) as ModelRequest;
    await expect(
      fetchSurrogateModel(pistaSpec(), "/", offline),
    ).rejects.toThrow("offline");
    const bad = (async () => new Response("not JSON")) as ModelRequest;
    await expect(fetchSurrogateModel(pistaSpec(), "/", bad)).rejects.toThrow();
  });

  test("declared native training rate must match the engine; legacy metadata stays compatible", () => {
    const spec = pistaSpec();
    expect(() =>
      validateSurrogateModel(
        modelBytes({ nativeRate: spec.simulationFrequency }),
        spec,
      ),
    ).not.toThrow();
    expect(() =>
      validateSurrogateModel(modelBytes({ nativeRate: 20000 }), spec),
    ).toThrow("native engine rate");
    expect(() => validateSurrogateModel(modelBytes(), spec)).not.toThrow();
  });

  test("wrong model cylinders, controls, output type or rate are rejected", () => {
    const spec = pistaSpec();
    expect(() => validateSurrogateModel(modelBytes(), spec)).not.toThrow();
    for (const overrides of [
      { cylinders: 4 },
      { widths: [1, 1] },
      { kernels: [[[]]] },
      { rate: 44100 },
    ])
      expect(() =>
        validateSurrogateModel(modelBytes(overrides), spec),
      ).toThrow();
  });
});
