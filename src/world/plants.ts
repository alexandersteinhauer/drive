import type { Vector3 } from "three";
import type { Landscape, LandSample } from "./landscape";
import { hash2 } from "./noise";

/**
 * What grows where: trees in clumpy forests, bushes at their edges and in the meadows,
 * rocks (more on slopes). Pure functions of the landscape, so the world workers and the
 * main thread place exactly the same plants.
 *
 * Kinds index vegetation.ts's KINDS: oaks and ashes 0-4, pines 5-7, bushes 8-10, rocks
 * 11-13.
 */
export const BROADLEAF = [0, 1, 2, 3, 4];
export const PINE = [5, 6, 7];
export const BUSHES = [8, 9, 10];
export const FIRST_ROCK = 11;

/** Floats per placed item: x, y, z, yaw, scale, kind. */
export const STRIDE = 6;
/** Spacing of the grid trees and bushes are planted on (m). */
export const PLANT_STEP = 5;

/**
 * Crowns per tree kind, at scale 1, from the models' leaves: bottom and top above the
 * ground and radius (m), and whether a cone (pines) or rounded. The sun field stands
 * them on the ground as the trees' shadow casters.
 */
export const CROWN: [number, number, number, boolean][] = [
  [2.0, 9.6, 3.6, false],
  [2.4, 9.6, 3.1, false],
  [2.4, 11.5, 4.7, false],
  [2.3, 10.7, 4.1, false],
  [1.7, 10.4, 3.7, false],
  [1.3, 10.9, 2.2, true],
  [1.0, 10.8, 2.2, true],
  [1.0, 13.0, 2.9, true],
];
/** Widest crown at the largest scale (m): trees this far outside an area still shade it. */
export const CROWN_REACH = 4.7 * 1.8;

/** A tree or bush planted at a grid cell (see `plantAt`). */
export interface Plant {
  x: number;
  y: number;
  z: number;
  yaw: number;
  scale: number;
  kind: number;
  tree: boolean;
}

const sample: LandSample = { height: 0, surface: 2, roadDistance: Infinity };
/** Scratch normal (Landscape.normal only calls set() and normalize()). */
class Normal {
  x = 0;
  y = 0;
  z = 0;
  set(x: number, y: number, z: number): this {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  normalize(): this {
    const l = Math.hypot(this.x, this.y, this.z) || 1;
    this.x /= l;
    this.y /= l;
    this.z /= l;
    return this;
  }
}
const normal = new Normal() as unknown as Vector3;
const plant: Plant = {
  x: 0,
  y: 0,
  z: 0,
  yaw: 0,
  scale: 1,
  kind: 0,
  tree: true,
};

export function smoothstep(x: number, a: number, b: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/**
 * What grows in grid cell (gx, gz), if anything: the one place trees are decided, so
 * everything that depends on them (the meshes, their shadows) agrees. `treesOnly` skips
 * the bushes (as far tiles do).
 */
export function plantAt(
  land: Landscape,
  gx: number,
  gz: number,
  treesOnly: boolean,
  out: Plant,
): boolean {
  const step = PLANT_STEP;
  const x = (gx + hash2(gx, gz, 11)) * step;
  const z = (gz + hash2(gx, gz, 12)) * step;
  const d = forest(x, z);
  const roll = hash2(gx, gz, 13);
  // Dense woods, plus lone trees in the meadows.
  const isTree = roll < d * 0.92 + 0.02;
  const isBush =
    !treesOnly && !isTree && roll < d * 0.96 + 0.03 + 0.08 * edgeOf(d);
  if (!isTree && !isBush) return false;
  land.sample(x, z, sample);
  // Woods stop a varying distance from the road, not in a straight wall.
  const clear = isTree ? 5 + 10 * hash2(gx, gz, 18) ** 2 : 3;
  if (sample.roadDistance < land.road.edge + clear) return false;
  const y = sample.height;
  if (land.normal(x, z, normal).y < 0.8) return false;
  out.x = x;
  out.y = y;
  out.z = z;
  out.yaw = hash2(gx, gz, 15) * Math.PI * 2;
  out.tree = isTree;
  if (isTree) {
    // Pines on the higher ground, broadleaf below, mixed in between.
    const pine = hash2(gx, gz, 16) < 0.2 + 0.7 * smoothstep(y, 5, 35);
    const kinds = pine ? PINE : BROADLEAF;
    out.kind = kinds[Math.floor(hash2(gx, gz, 17) * kinds.length)]!;
    out.scale = 0.7 + hash2(gx, gz, 14) ** 1.3 * 1.1;
  } else {
    out.kind = BUSHES[Math.floor(hash2(gx, gz, 17) * BUSHES.length)]!;
    out.scale = 0.7 + hash2(gx, gz, 14) * 0.7;
  }
  return true;
}

/**
 * A tile's vegetation as packed records (STRIDE floats each). `far` tiles get only the
 * trees: they are only ever seen as billboards.
 */
export function scatter(
  land: Landscape,
  tx: number,
  tz: number,
  size: number,
  far: boolean,
): Float32Array {
  const out: number[] = [];
  const x0 = tx * size;
  const z0 = tz * size;
  const put = (
    x: number,
    y: number,
    z: number,
    yaw: number,
    s: number,
    kind: number,
  ) => out.push(x, y, z, yaw, s, kind);
  // Trees and bushes on a 5 m grid; far tiles the same trees, so nothing changes when a
  // tile turns from near to far.
  const step = PLANT_STEP;
  for (let j = 0; j < size / step; j++)
    for (let i = 0; i < size / step; i++) {
      const gx = Math.floor(x0 / step) + i;
      const gz = Math.floor(z0 / step) + j;
      if (!plantAt(land, gx, gz, far, plant)) continue;
      // Far, sunk a little: the coarse terrain there is up to a metre off.
      const sink = plant.tree ? (far ? 0.8 : 0.15) : 0.1;
      put(plant.x, plant.y - sink, plant.z, plant.yaw, plant.scale, plant.kind);
    }
  if (far) return new Float32Array(out);
  // Rocks on a 24 m grid, more on slopes.
  for (let j = 0; j < size / ROCK_STEP; j++)
    for (let i = 0; i < size / ROCK_STEP; i++)
      rocksAt(
        land,
        Math.floor(x0 / ROCK_STEP) + i,
        Math.floor(z0 / ROCK_STEP) + j,
        put,
      );
  return new Float32Array(out);
}

/** Spacing of the grid rock groups are placed on (m). */
export const ROCK_STEP = 24;

/**
 * The rock group in grid cell (gx, gz), if any: one big, partly buried rock and a few
 * smaller ones around it, each passed to `put` (x, y, z, yaw, scale, kind). The one place
 * rocks are decided, so the meshes and the obstacles the car hits agree.
 */
export function rocksAt(
  land: Landscape,
  gx: number,
  gz: number,
  put: (
    x: number,
    y: number,
    z: number,
    yaw: number,
    scale: number,
    kind: number,
  ) => void,
): void {
  const edge = land.road.edge;
  const x = (gx + hash2(gx, gz, 21)) * ROCK_STEP;
  const z = (gz + hash2(gx, gz, 22)) * ROCK_STEP;
  const slope = 1 - land.normal(x, z, normal).y;
  if (hash2(gx, gz, 23) > 0.22 + slope * 4) return;
  land.sample(x, z, sample);
  if (sample.roadDistance < edge + 2) return;
  const s = 0.6 + hash2(gx, gz, 24) ** 2 * 1.6;
  const kind = FIRST_ROCK + Math.floor(hash2(gx, gz, 25) * 3);
  put(x, sample.height - 0.35 * s, z, hash2(gx, gz, 26) * 6.283, s, kind);
  const extra = Math.floor(hash2(gx, gz, 27) * 4);
  for (let e = 0; e < extra; e++) {
    const a = hash2(gx, gz, 30 + e) * 6.283;
    const r = s * (1.6 + hash2(gx, gz, 40 + e) * 1.6);
    const ex = x + Math.cos(a) * r;
    const ez = z + Math.sin(a) * r;
    const es = s * (0.25 + hash2(gx, gz, 50 + e) * 0.35);
    land.sample(ex, ez, sample);
    if (sample.roadDistance < edge + 2) continue;
    put(
      ex,
      sample.height - 0.3 * es,
      ez,
      a * 3.1,
      es,
      FIRST_ROCK + ((kind + e) % 3),
    );
  }
}

/** Forest density (0..1) at a point: the woods' extent. */
export function forest(x: number, z: number): number {
  const large =
    0.5 +
    0.5 *
      Math.sin(x * 0.006 + 1.3) *
      Math.sin(z * 0.0047 + Math.sin(x * 0.002) * 2);
  const small =
    0.5 +
    0.5 *
      Math.sin(x * 0.021 + Math.sin(z * 0.017) * 1.7) *
      Math.sin(z * 0.019 + 0.7);
  return smoothstep(large * 0.7 + small * 0.45, 0.35, 0.65);
}

/** Bushes gather where the woods thin out. */
function edgeOf(d: number): number {
  return Math.max(0, 1 - Math.abs(d - 0.25) / 0.2);
}
