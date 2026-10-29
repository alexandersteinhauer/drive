import { expect, test } from "bun:test";
import { REFERENCE_TUNING, vw1200Engine } from "./vw1200";

/**
 * The Beetle's engine as engine-sim built it from engine/reference/vw1200.mr (written by
 * engine/reference/build.sh): the TypeScript version, with the script's tuning, must describe
 * the same engine, which checks the helpers in engineSpec.ts against engine-sim's.
 */
test("matches the engine engine-sim built from the script", async () => {
  const reference = await Bun.file(
    new URL("../../../engine/tests/vw1200.json", import.meta.url),
  ).json();
  const sortFunctions = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(sortFunctions);
    if (v && typeof v === "object") {
      const o = Object.fromEntries(
        Object.entries(v).map(([k, x]) => [k, sortFunctions(x)]),
      );
      if ("samples" in o)
        (o.samples as [number, number][]).sort((a, b) => a[0] - b[0]);
      return o;
    }
    return v;
  };
  const ours = sortFunctions(vw1200Engine(REFERENCE_TUNING)) as Record<
    string,
    unknown
  >;
  const theirs = sortFunctions(reference) as Record<string, unknown>;
  const mismatches: string[] = [];
  const compare = (a: unknown, b: unknown, path: string) => {
    if (typeof b === "number") {
      // Optional fields default to 0.
      const x = a ?? 0;
      const ok =
        typeof x === "number" &&
        Math.abs(x - b) <= 1e-12 * Math.max(Math.abs(x), Math.abs(b), 1e-300);
      if (!ok) mismatches.push(`${path}: ${x} vs ${b}`);
    } else if (Array.isArray(b)) {
      if (!Array.isArray(a) || a.length !== b.length)
        mismatches.push(
          `${path}: length ${(a as unknown[])?.length} vs ${b.length}`,
        );
      else b.forEach((x, i) => compare(a[i], x, `${path}[${i}]`));
    } else if (b && typeof b === "object") {
      for (const k of Object.keys(b))
        compare(
          (a as Record<string, unknown>)?.[k],
          (b as Record<string, unknown>)[k],
          `${path}.${k}`,
        );
    } else if (a !== b) mismatches.push(`${path}: ${a} vs ${b}`);
  };
  compare({ fluidSteps: 8, ...ours }, theirs, "");
  expect(mismatches).toEqual([]);
});
