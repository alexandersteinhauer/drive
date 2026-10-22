import { Surface, type Landscape, type LandSample } from "./landscape";
import { hash2 } from "./noise";
import {
  CROWN,
  CROWN_REACH,
  forest,
  PLANT_STEP,
  plantAt,
  scatter,
  smoothstep,
  type Plant,
} from "./plants";

/**
 * World generation: terrain tiles, grass chunks and sun-field tiles, as plain typed arrays.
 * Pure functions of the landscape, run in the world workers (worldWorkers.ts) so none of it
 * costs the main thread a frame.
 */

/** Terrain under the road is pushed down so it never pokes through the road mesh. */
export const UNDER_ROAD = 0.3;
/** Depth of the skirts that hide cracks between terrain tiles of different detail. */
const SKIRT = 6;

const sample: LandSample = {
  height: 0,
  surface: Surface.Grass,
  roadDistance: Infinity,
};

/** Large dry/green patches across the meadows (0 = lush, 1 = dry), plus a little noise. */
export function variationAt(x: number, z: number): number {
  const dry = 0.5 + 0.5 * Math.sin(x * 0.011 + Math.sin(z * 0.007) * 3);
  return Math.min(1, dry * 0.9 + hash2(x, z, 3) * 0.1);
}

/** How far the terrain is sunk under the road at a distance from its centreline. */
function underRoad(road: number, edge: number): number {
  return UNDER_ROAD * (1 - Math.min(1, Math.max(0, (road - edge + 1.2) / 1.2)));
}

export interface TerrainRequest {
  x0: number;
  z0: number;
  size: number;
  segments: number;
  /** Size of the vegetation tiles, and which to scatter (tx, tz, far). */
  tile: number;
  treeTiles: [number, number, boolean][];
}

export interface TerrainResult {
  position: Float32Array;
  normal: Float32Array;
  variation: Float32Array;
  road: Float32Array;
  forest: Float32Array;
  index: Uint32Array;
  /** Bounding sphere: centre x, y, z and radius. */
  sphere: [number, number, number, number];
  /** Vegetation per tile: key "tx,tz" and packed records. */
  trees: [string, Float32Array][];
}

/** A terrain tile (or far block): grid, skirts, per-vertex attributes, and its vegetation. */
export function terrainTile(land: Landscape, w: TerrainRequest): TerrainResult {
  const { x0, z0, segments: n } = w;
  const step = w.size / n;
  // Heights on an (n + 3)^2 grid with a one-sample border, for normals that match across tiles.
  const g = n + 3;
  const heights = new Float32Array(g * g);
  const nearRoad = new Float32Array(g * g);
  for (let j = 0; j < g; j++)
    for (let i = 0; i < g; i++) {
      land.sample(x0 + (i - 1) * step, z0 + (j - 1) * step, sample);
      heights[j * g + i] = sample.height;
      nearRoad[j * g + i] = sample.roadDistance;
    }

  const v = n + 1;
  const ring = 4 * n;
  const count = v * v + ring * 2;
  const pos = new Float32Array(count * 3);
  const nor = new Float32Array(count * 3);
  const variation = new Float32Array(count);
  const roadDist = new Float32Array(count);
  const forestDensity = new Float32Array(count);
  const edge = land.road.edge;
  const put = (k: number, i: number, j: number, drop: number) => {
    const h = heights[(j + 1) * g + i + 1]!;
    const hl = heights[(j + 1) * g + i]!;
    const hr = heights[(j + 1) * g + i + 2]!;
    const hd = heights[j * g + i + 1]!;
    const hu = heights[(j + 2) * g + i + 1]!;
    const road = nearRoad[(j + 1) * g + i + 1]!;
    pos[k * 3] = x0 + i * step;
    // Sunk under the road surface, rising to exactly road height at the shoulder edge,
    // where the road mesh ends and the terrain (with its own gravel verge) takes over.
    pos[k * 3 + 1] = h - drop - underRoad(road, edge);
    pos[k * 3 + 2] = z0 + j * step;
    const nx = -(hr - hl);
    const ny = 2 * step;
    const nz = -(hu - hd);
    const len = Math.hypot(nx, ny, nz);
    nor[k * 3] = nx / len;
    nor[k * 3 + 1] = ny / len;
    nor[k * 3 + 2] = nz / len;
    variation[k] = variationAt(x0 + i * step, z0 + j * step);
    forestDensity[k] = forest(x0 + i * step, z0 + j * step);
    roadDist[k] = Math.min(road, 100);
  };
  for (let j = 0; j < v; j++)
    for (let i = 0; i < v; i++) put(j * v + i, i, j, 0);

  const index = new Uint32Array((n * n + ring) * 6);
  let ix = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const a = j * v + i;
      index[ix++] = a;
      index[ix++] = a + v;
      index[ix++] = a + 1;
      index[ix++] = a + 1;
      index[ix++] = a + v;
      index[ix++] = a + v + 1;
    }
  // Skirts: the boundary loop duplicated and dropped, stitched to the edge.
  const loop: [number, number][] = [];
  for (let i = 0; i < n; i++) loop.push([i, 0]);
  for (let j = 0; j < n; j++) loop.push([n, j]);
  for (let i = n; i > 0; i--) loop.push([i, n]);
  for (let j = n; j > 0; j--) loop.push([0, j]);
  const base = v * v;
  loop.forEach(([i, j], k) => {
    put(base + k, i, j, 0);
    put(base + ring + k, i, j, SKIRT);
  });
  for (let k = 0; k < ring; k++) {
    const a = base + k;
    const b = base + ((k + 1) % ring);
    const c = base + ring + k;
    const d = base + ring + ((k + 1) % ring);
    index[ix++] = a;
    index[ix++] = b;
    index[ix++] = c;
    index[ix++] = b;
    index[ix++] = d;
    index[ix++] = c;
  }

  let lo = Infinity;
  let hi = -Infinity;
  for (let k = 1; k < pos.length; k += 3) {
    lo = Math.min(lo, pos[k]!);
    hi = Math.max(hi, pos[k]!);
  }
  const sphere: [number, number, number, number] = [
    x0 + w.size / 2,
    (lo + hi) / 2,
    z0 + w.size / 2,
    Math.hypot(w.size / 2, w.size / 2, (hi - lo) / 2),
  ];
  const trees: [string, Float32Array][] = [];
  for (const [tx, tz, far] of w.treeTiles) {
    const items = scatter(land, tx, tz, w.tile, far);
    if (items.length) trees.push([`${tx},${tz}`, items]);
  }
  return {
    position: pos,
    normal: nor,
    variation,
    road: roadDist,
    forest: forestDensity,
    index,
    sphere,
    trees,
  };
}

/** Grass chunks: square (m), on the terrain's full-detail vertex spacing, clumps spacing. */
export const GRASS_CHUNK = 32;
const GRASS_CELL = 2;
const GRASS_SPACING = 0.3;

export interface GrassResult {
  /** Per clump, sorted by rank: x, y, z, yaw | height, rank, variation, forest | normal x, z. */
  root: Float32Array;
  info: Float32Array;
  ground: Float32Array;
  count: number;
  minY: number;
  maxY: number;
}

/**
 * A chunk of grass clumps on a jittered grid: dense in the open, mown short along the
 * verge, thin in the woods and gone on rock and steep cuttings. Clumps sit exactly on the
 * terrain's triangles. Only clumps of rank up to `maxRank` (a distant chunk draws just its
 * lowest ranks). Null if none grow there.
 */
export function grassChunk(
  land: Landscape,
  cx: number,
  cz: number,
  maxRank = 1,
): GrassResult | null {
  const CHUNK = GRASS_CHUNK;
  const CELL = GRASS_CELL;
  const SPACING = GRASS_SPACING;
  const edge = land.road.edge;
  const x0 = cx * CHUNK;
  const z0 = cz * CHUNK;
  // The terrain's own vertices (with a one-sample border for normals).
  const n = CHUNK / CELL;
  const g = n + 3;
  const hs = new Float32Array(g * g);
  const rs = new Float32Array(g * g);
  for (let j = 0; j < g; j++)
    for (let i = 0; i < g; i++) {
      land.sample(x0 + (i - 1) * CELL, z0 + (j - 1) * CELL, sample);
      hs[j * g + i] = sample.height;
      rs[j * g + i] = Math.min(sample.roadDistance, 100);
    }
  const v = n + 1;
  // Per vertex: height, normal x/y/z, road distance, variation, forest.
  const vert = new Float32Array(v * v * 7);
  for (let j = 0; j < v; j++)
    for (let i = 0; i < v; i++) {
      const at = (di: number, dj: number) => hs[(j + 1 + dj) * g + i + 1 + di]!;
      const road = rs[(j + 1) * g + i + 1]!;
      const nx = -(at(1, 0) - at(-1, 0));
      const ny = 2 * CELL;
      const nz = -(at(0, 1) - at(0, -1));
      const len = Math.hypot(nx, ny, nz);
      const x = x0 + i * CELL;
      const z = z0 + j * CELL;
      const o = (j * v + i) * 7;
      vert[o] = at(0, 0) - underRoad(road, edge);
      vert[o + 1] = nx / len;
      vert[o + 2] = ny / len;
      vert[o + 3] = nz / len;
      vert[o + 4] = road;
      vert[o + 5] = variationAt(x, z);
      vert[o + 6] = forest(x, z);
    }

  // Interpolates the vertex values over the terrain's triangles (split along the
  // (i+1, j)-(i, j+1) diagonal), into `p`.
  const p = new Float32Array(7);
  const interpolate = (x: number, z: number) => {
    const fx = (x - x0) / CELL;
    const fz = (z - z0) / CELL;
    const i = Math.min(n - 1, Math.floor(fx));
    const j = Math.min(n - 1, Math.floor(fz));
    const u = fx - i;
    const w = fz - j;
    const a = (j * v + i) * 7;
    const b = a + 7; // (i+1, j)
    const c = a + v * 7; // (i, j+1)
    const d = c + 7; // (i+1, j+1)
    if (u + w <= 1)
      for (let k = 0; k < 7; k++)
        p[k] =
          vert[a + k]! +
          u * (vert[b + k]! - vert[a + k]!) +
          w * (vert[c + k]! - vert[a + k]!);
    else
      for (let k = 0; k < 7; k++)
        p[k] =
          vert[d + k]! +
          (1 - u) * (vert[c + k]! - vert[d + k]!) +
          (1 - w) * (vert[b + k]! - vert[d + k]!);
  };

  const cells = Math.round(CHUNK / SPACING);
  const out: number[] = [];
  for (let j = 0; j < cells; j++)
    for (let i = 0; i < cells; i++) {
      const gx = Math.round(x0 / SPACING) + i;
      const gz = Math.round(z0 / SPACING) + j;
      const x = (gx + hash2(gx, gz, 41)) * SPACING;
      const z = (gz + hash2(gx, gz, 42)) * SPACING;
      if (x < x0 || x >= x0 + CHUNK || z < z0 || z >= z0 + CHUNK) continue;
      const rank = hash2(gx, gz, 46);
      if (rank > maxRank) continue;
      interpolate(x, z);
      const r = p[4]! - edge;
      if (r < 0.7) continue;
      const slope = 1 - p[2]!;
      const woods = p[6]!;
      // Patches of taller and thinner grass across the meadows.
      const patch =
        0.5 +
        0.5 *
          Math.sin(x * 0.13 + Math.sin(z * 0.09) * 2.1) *
          Math.sin(z * 0.11 + x * 0.04);
      const density =
        smoothstep(r, 0.7, 2.2) *
        (1 - smoothstep(slope, 0.08, 0.17)) *
        (1 - 0.85 * woods) *
        (0.75 + 0.25 * patch);
      if (hash2(gx, gz, 43) > density) continue;
      // Mown short along the verge, longer out in the meadow, longest in its patches.
      const field = smoothstep(r, 2, 7);
      const height =
        (0.45 + field * (0.4 + 0.6 * patch * patch)) *
        (0.7 + 0.6 * hash2(gx, gz, 44)) *
        (1 - 0.35 * woods);
      out.push(
        x,
        p[0]!,
        z,
        hash2(gx, gz, 45) * Math.PI * 2,
        height,
        rank,
        p[5]!,
        woods,
        p[1]!,
        p[3]!,
      );
    }
  const count = out.length / 10;
  if (count === 0) return null;
  // Sorted by rank, so drawing a prefix thins the chunk out evenly.
  const order = [...Array(count).keys()].toSorted(
    (a, b) => out[a * 10 + 5]! - out[b * 10 + 5]!,
  );
  const root = new Float32Array(count * 4);
  const info = new Float32Array(count * 4);
  const ground = new Float32Array(count * 2);
  let minY = Infinity;
  let maxY = -Infinity;
  order.forEach((src, k) => {
    const s = src * 10;
    for (let q = 0; q < 4; q++) root[k * 4 + q] = out[s + q]!;
    for (let q = 0; q < 4; q++) info[k * 4 + q] = out[s + 4 + q]!;
    ground[k * 2] = out[s + 8]!;
    ground[k * 2 + 1] = out[s + 9]!;
    minY = Math.min(minY, out[s + 1]!);
    maxY = Math.max(maxY, out[s + 1]!);
  });
  return { root, info, ground, count, minY, maxY };
}

const plant: Plant = {
  x: 0,
  y: 0,
  z: 0,
  yaw: 0,
  scale: 1,
  kind: 0,
  tree: true,
};

/**
 * Sun-field tiles (`size` texels of `texel` m across, by tile coordinates): ground
 * heights, and the occluders over it (the ground, and every tree crown stood on it).
 */
export function sunTiles(
  land: Landscape,
  tiles: number[],
  size: number,
  texel: number,
): { heights: Float32Array; tops: Float32Array } {
  const n = size * size;
  const heights = new Float32Array((tiles.length / 2) * n);
  const tops = new Float32Array((tiles.length / 2) * n);
  for (let t = 0; t < tiles.length / 2; t++) {
    const x = tiles[t * 2]!;
    const z = tiles[t * 2 + 1]!;
    const base = t * n;
    for (let lz = 0; lz < size; lz++)
      for (let lx = 0; lx < size; lx++) {
        land.sample((x * size + lx) * texel, (z * size + lz) * texel, sample);
        heights[base + lz * size + lx] = sample.height;
      }
    tops.set(heights.subarray(base, base + n), base);
    const x0 = x * size * texel;
    const z0 = z * size * texel;
    const span = size * texel;
    const g0x = Math.floor((x0 - CROWN_REACH) / PLANT_STEP);
    const g1x = Math.floor((x0 + span + CROWN_REACH) / PLANT_STEP);
    const g0z = Math.floor((z0 - CROWN_REACH) / PLANT_STEP);
    const g1z = Math.floor((z0 + span + CROWN_REACH) / PLANT_STEP);
    for (let gz = g0z; gz <= g1z; gz++)
      for (let gx = g0x; gx <= g1x; gx++) {
        if (!plantAt(land, gx, gz, true, plant)) continue;
        const [bottom, top, radius, cone] = CROWN[plant.kind]!;
        const s = plant.scale;
        const r = radius * s;
        const lo = plant.y + bottom * s;
        const hi = plant.y + top * s;
        const ix0 = Math.max(0, Math.ceil((plant.x - r - x0) / texel));
        const ix1 = Math.min(size - 1, Math.floor((plant.x + r - x0) / texel));
        const iz0 = Math.max(0, Math.ceil((plant.z - r - z0) / texel));
        const iz1 = Math.min(size - 1, Math.floor((plant.z + r - z0) / texel));
        for (let iz = iz0; iz <= iz1; iz++)
          for (let ix = ix0; ix <= ix1; ix++) {
            const d =
              Math.hypot(x0 + ix * texel - plant.x, z0 + iz * texel - plant.z) /
              r;
            if (d >= 1) continue;
            const h = cone
              ? hi - (hi - lo) * d
              : (lo + hi) / 2 + ((hi - lo) / 2) * Math.sqrt(1 - d * d);
            const i = base + iz * size + ix;
            if (h > tops[i]!) tops[i] = h;
          }
      }
  }
  return { heights, tops };
}
