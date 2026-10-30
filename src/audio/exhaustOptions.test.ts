import { describe, expect, test } from "bun:test";
import {
  bmwI4,
  ferrari488Pista,
  vwBeetle,
  withPistaSurrogateExhaust,
} from "../sim/config";
import { audioChannels, type EngineSpec } from "../sim/engineSpec";
import { pistaExhaustMode, pistaExhaustUrl } from "./exhaustOptions";
import {
  fetchSurrogateModel,
  originalExhaustSpec,
  validateSurrogateModel,
  type ModelRequest,
} from "./surrogateModel";

function pistaSpec(learned = true): EngineSpec {
  const cfg = learned ? withPistaSurrogateExhaust() : ferrari488Pista;
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

describe("Pista exhaust selection", () => {
  test("original by default, saved opt-in, explicit URL overrides", () => {
    const url = new URL("https://example.test/game/?seed=17");
    expect(pistaExhaustMode(url, null)).toBe("original");
    expect(pistaExhaustMode(url, "surrogate")).toBe("surrogate");
    url.searchParams.set("exhaust", "original");
    expect(pistaExhaustMode(url, "surrogate")).toBe("original");
    url.searchParams.set("exhaust", "impulse");
    expect(pistaExhaustMode(url, "surrogate")).toBe("original");
    url.searchParams.set("exhaust", "surrogate");
    expect(pistaExhaustMode(url, "original")).toBe("surrogate");
  });

  test("settings link preserves world and starts the Pista", () => {
    const url = new URL("https://example.test/game/?seed=17&foo=bar#x");
    const next = pistaExhaustUrl(url, "surrogate");
    expect(next.searchParams.get("car")).toBe("pista");
    expect(next.searchParams.get("exhaust")).toBe("surrogate");
    expect(next.searchParams.get("seed")).toBe("17");
    expect(next.searchParams.get("foo")).toBe("bar");
    expect(next.hash).toBe("#x");
    expect(url.searchParams.has("exhaust")).toBe(false);
  });

  test("opt-in changes only the Pista audio selector, never engine physics", () => {
    const before = structuredClone([bmwI4, vwBeetle, ferrari488Pista]);
    const spec = pistaSpec();
    expect(spec.audio.surrogate?.fullScale).toBe(16);
    expect(spec.audio.voice).toEqual(pistaSpec(false).audio.voice);
    expect(originalExhaustSpec(spec)).toEqual(pistaSpec(false));
    expect([bmwI4, vwBeetle, ferrari488Pista]).toEqual(before);
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
});

describe("optional exhaust model asset", () => {
  test("original backend never fetches the learned asset", async () => {
    let requests = 0;
    const request = (async () => {
      requests++;
      throw new Error("unexpected fetch");
    }) as ModelRequest;
    expect(
      await fetchSurrogateModel(pistaSpec(false), "/game/", request),
    ).toBeUndefined();
    expect(requests).toBe(0);
  });

  test("selected pilot loads relative to the deployment base", async () => {
    let requested = "";
    const bytes = modelBytes();
    const request = (async (url: string | URL | Request) => {
      requested = String(url);
      return new Response(bytes);
    }) as ModelRequest;
    expect(await fetchSurrogateModel(pistaSpec(), "/game/", request)).toEqual(
      bytes,
    );
    expect(requested).toBe("/game/audio/surrogate/pista-pilot.json");
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
