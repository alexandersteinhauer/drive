import { describe, expect, test } from "bun:test";
import {
  bmwI4,
  ferrari488Pista,
  vwBeetle,
  withPistaSurrogateExhaust,
} from "../sim/config";
import { audioChannels, type EngineSpec } from "../sim/engineSpec";
import {
  pistaExhaustMode,
  pistaExhaustStatus,
  pistaExhaustUrl,
} from "./exhaustOptions";
import {
  fetchSurrogateModel,
  originalExhaustSpec,
  validateSurrogateModel,
  type ModelRequest,
} from "./surrogateModel";

function pistaSpec(
  variant: "pilot" | "retrained" | false = "pilot",
): EngineSpec {
  const cfg =
    variant === false ? ferrari488Pista : withPistaSurrogateExhaust(variant);
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

  test.each(["original", "surrogate", "surrogate-retrained"] as const)(
    "saved %s survives reload and an explicit URL wins every saved mode",
    (mode) => {
      const url = new URL("https://example.test/game/?seed=17");
      expect(pistaExhaustMode(new URL(url.href), mode)).toBe(mode);
      url.searchParams.set("exhaust", mode);
      for (const saved of [
        null,
        "original",
        "surrogate",
        "surrogate-retrained",
      ])
        expect(pistaExhaustMode(url, saved)).toBe(mode);
    },
  );

  test.each(["", "unknown", "impulse"])(
    "unrecognized or legacy %s never opts into a saved learned mode",
    (mode) => {
      const url = new URL("https://example.test/game/");
      expect(pistaExhaustMode(url, mode)).toBe("original");
      url.searchParams.set("exhaust", mode);
      expect(pistaExhaustMode(url, "surrogate")).toBe("original");
      expect(pistaExhaustMode(url, "surrogate-retrained")).toBe("original");
    },
  );

  test.each(["original", "surrogate", "surrogate-retrained"] as const)(
    "settings link for %s preserves world and starts the Pista",
    (mode) => {
      const url = new URL(
        "https://example.test/game/?car=beetle&seed=17&foo=bar#x",
      );
      const next = pistaExhaustUrl(url, mode);
      expect(next.searchParams.get("car")).toBe("pista");
      expect(next.searchParams.get("exhaust")).toBe(mode);
      expect(pistaExhaustMode(next, "original")).toBe(mode);
      expect(next.searchParams.get("seed")).toBe("17");
      expect(next.searchParams.get("foo")).toBe("bar");
      expect(next.hash).toBe("#x");
      expect(url.searchParams.get("car")).toBe("beetle");
      expect(url.searchParams.has("exhaust")).toBe(false);
    },
  );

  test("retrained status discloses the experimental pressure-only preview and residual errors", () => {
    for (const active of [undefined, true]) {
      const status = pistaExhaustStatus("surrogate-retrained", active);
      expect(status).toContain("pressure-only");
      expect(status).toContain("preview");
      expect(status).toContain("experimental");
      expect(status).toContain("residual bass and spectral errors");
    }
    expect(pistaExhaustStatus("surrogate-retrained", false)).toBe(
      "Retrained preview unavailable; original exhaust active.",
    );
    expect(pistaExhaustStatus("original")).toBe("Original exhaust active.");
    expect(pistaExhaustStatus("surrogate", false)).toBe(
      "Pilot unavailable; original exhaust active.",
    );
  });

  test.each(["pilot", "retrained"] as const)(
    "%s opt-in changes only the Pista audio selector, never car or engine physics",
    (variant) => {
      const before = structuredClone([bmwI4, vwBeetle, ferrari488Pista]);
      const cfg = withPistaSurrogateExhaust(variant);
      if (
        cfg.powertrain.type !== "combustion" ||
        cfg.powertrain.engine.type !== "simulated"
      )
        throw new Error("expected simulated Pista");
      const spec = cfg.powertrain.engine.spec;
      expect(spec.audio.surrogate?.fullScale).toBe(16);
      expect(spec.audio.voice).toEqual(pistaSpec(false).audio.voice);
      expect(originalExhaustSpec(spec)).toEqual(pistaSpec(false));
      delete spec.audio.surrogate;
      expect(cfg).toEqual(ferrari488Pista);
      expect([bmwI4, vwBeetle, ferrari488Pista]).toEqual(before);
    },
  );

  test("factory defaults to the old pilot and retrained selects the new coverage candidate", () => {
    const implicit = withPistaSurrogateExhaust();
    expect(implicit).toEqual(withPistaSurrogateExhaust("pilot"));
    const pilot = pistaSpec();
    const retrained = pistaSpec("retrained");
    expect(pilot.audio.surrogate?.model).toBe(
      "audio/surrogate/pista-pilot.json",
    );
    expect(retrained.audio.surrogate?.model).toBe(
      "audio/surrogate/pista-coverage-candidate.json",
    );
    retrained.audio.surrogate!.model = pilot.audio.surrogate!.model;
    expect(retrained).toEqual(pilot);
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

  test.each(["pilot", "retrained"] as const)(
    "selected %s loads only its own model relative to the deployment base",
    async (variant) => {
      const requested: string[] = [];
      const bytes = modelBytes();
      const request = (async (url: string | URL | Request) => {
        requested.push(String(url));
        return new Response(bytes);
      }) as ModelRequest;
      expect(
        await fetchSurrogateModel(pistaSpec(variant), "/game/", request),
      ).toEqual(bytes);
      expect(requested).toEqual([
        variant === "retrained"
          ? "/game/audio/surrogate/pista-coverage-candidate.json"
          : "/game/audio/surrogate/pista-pilot.json",
      ]);
    },
  );

  test("retrained selection loads the real coverage candidate with unchanged calibration", async () => {
    const spec = pistaSpec("retrained");
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
    const spec = pistaSpec("retrained");
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
