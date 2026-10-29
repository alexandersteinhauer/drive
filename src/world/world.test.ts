import { describe, expect, test } from "bun:test";
import type { Obstacle } from "../sim/ground";
import { Landscape, Surface } from "./landscape";
import { BUSHES, FIRST_ROCK, STRIDE, scatter } from "./plants";

describe("road", () => {
  const land = new Landscape(1);
  const road = land.road;
  const t0 = performance.now();
  road.ensure(20_000);
  const genMs = performance.now() - t0;

  test("progresses steadily and stays within the heading limit", () => {
    const n = road.finalCount;
    let maxHeading = 0;
    let minRadius = Infinity;
    let maxGrade = 0;
    let maxGradeChange = 0;
    let turn = 0;
    for (let i = 1; i < n; i++) {
      maxHeading = Math.max(maxHeading, Math.abs(road.headings[i]!));
      const k = Math.abs(road.curvatures[i]!);
      if (k > 0) minRadius = Math.min(minRadius, 1 / k);
      turn += Math.abs(road.headings[i]! - road.headings[i - 1]!);
      const grade =
        (road.heights[i]! - road.heights[i - 1]!) / road.cfg.spacing;
      maxGrade = Math.max(maxGrade, Math.abs(grade));
      if (i > 1) {
        const prev =
          (road.heights[i - 1]! - road.heights[i - 2]!) / road.cfg.spacing;
        maxGradeChange = Math.max(maxGradeChange, Math.abs(grade - prev));
      }
      expect(road.zs[i]!).toBeLessThan(road.zs[i - 1]!);
    }
    const length = (n - 1) * road.cfg.spacing;
    const heights = road.heights.slice(0, n);
    console.log(
      `road: ${(length / 1000).toFixed(1)} km in ${genMs.toFixed(0)} ms, max heading ${((maxHeading * 180) / Math.PI).toFixed(0)} deg, ` +
        `min radius ${minRadius.toFixed(0)} m, turning ${(((turn / length) * 1000 * 180) / Math.PI).toFixed(0)} deg/km, ` +
        `max grade ${(maxGrade * 100).toFixed(1)}%, max grade change ${(maxGradeChange * 100).toFixed(2)}%/sample, ` +
        `height range ${Math.min(...heights).toFixed(0)}..${Math.max(...heights).toFixed(0)} m`,
    );
    expect(maxHeading).toBeLessThanOrEqual(road.cfg.maxHeading + 0.01);
    expect(minRadius).toBeGreaterThanOrEqual(road.cfg.minRadius - 1);
    expect(maxGrade).toBeLessThanOrEqual(road.cfg.maxGrade + 1e-9);
  });

  test("is deterministic", () => {
    const other = new Landscape(1).road;
    other.ensure(3000);
    for (let i = 0; i < other.finalCount; i += 97) {
      expect(other.xs[i]).toBe(road.xs[i]!);
      expect(other.heights[i]).toBe(road.heights[i]!);
    }
  });

  test("the landscape matches the road surface and is continuous", () => {
    const i = 3000;
    const x = road.xs[i]!;
    const z = road.zs[i]!;
    const s = land.sample(x, z);
    expect(s.surface).toBe(Surface.Asphalt);
    expect(Math.abs(s.height - road.heights[i]!)).toBeLessThan(1e-6);
    // Walk across the road and out into the hills; heights must not jump.
    const h = road.headings[i]!;
    let prev = land.height(x, z);
    let maxStep = 0;
    for (let d = 0; d <= 80; d += 0.1) {
      const y = land.height(x + Math.cos(h) * d, z + Math.sin(h) * d);
      maxStep = Math.max(maxStep, Math.abs(y - prev));
      prev = y;
    }
    console.log(`cross-section max step per 10 cm: ${maxStep.toFixed(3)} m`);
    expect(maxStep).toBeLessThan(0.1);
  });

  test("sampling is fast enough for physics and meshing", () => {
    const t = performance.now();
    let acc = 0;
    const n = 200_000;
    for (let k = 0; k < n; k++) {
      const i = 1000 + (k % 5000);
      acc += land.height(road.xs[i]! + ((k * 7) % 60) - 30, road.zs[i]!);
    }
    const us = ((performance.now() - t) * 1000) / n;
    console.log(`sample: ${us.toFixed(2)} us`);
    expect(Number.isFinite(acc)).toBe(true);
    expect(us).toBeLessThan(5);
  });
});

describe("obstacles", () => {
  test("every tree and rock the scatter places has a hitbox, and none stands on the road", () => {
    const land = new Landscape(3);
    land.road.ensure(3000);
    const size = 128;
    const found: Obstacle[] = [];
    let trees = 0;
    let rocks = 0;
    for (let tz = -6; tz < 2; tz++)
      for (let tx = -2; tx < 2; tx++) {
        const items = scatter(land, tx, tz, size, false);
        for (let i = 0; i < items.length; i += STRIDE) {
          const [x, , z, , , kind] = items.subarray(i, i + STRIDE);
          const isRock = kind! >= FIRST_ROCK;
          const isTree = kind! < BUSHES[0]!;
          const n = land.obstaclesNear(x!, z!, 0, found);
          // A rock's capsule is centred on its footprint, a little off its origin.
          if (isRock) {
            rocks++;
            expect(n).toBeGreaterThan(0);
          } else if (isTree) {
            trees++;
            const own = found
              .slice(0, n)
              .some((o) => Math.hypot(o.x - x!, o.z - z!) < 1e-3);
            expect(own).toBe(true);
          }
        }
      }
    expect(trees).toBeGreaterThan(50);
    expect(rocks).toBeGreaterThan(5);
    // Nothing stands on the road.
    const road = land.road;
    for (let i = 0; i < road.finalCount; i += 5)
      expect(
        land.obstaclesNear(road.xs[i]!, road.zs[i]!, road.cfg.halfWidth, found),
      ).toBe(0);
  });
});
