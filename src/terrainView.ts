import * as THREE from "three/webgpu";
import { Surface, type Landscape, type LandSample } from "./world/landscape";
import { hash2 } from "./world/noise";
import { terrainMaterial } from "./terrainMaterial";
import { forest, type Vegetation } from "./vegetation";

const TILE = 128;
const RADIUS = 2200;
/** Vertex spacing per level of detail, and the distance (tile centre to camera) each is used up to. */
const LODS = [
  { segments: 64, within: 300 },
  { segments: 32, within: 700 },
  { segments: 16, within: 1400 },
  { segments: 8, within: Infinity },
];
/** Trees are planted this far out (billboards beyond the near tiles). */
const TREES_WITHIN = 2050;
/**
 * Beyond this distance, whole blocks of BLOCK x BLOCK tiles are built as one mesh (same
 * vertex spacing), cutting draw calls: each costs CPU time and garbage in the renderer.
 */
const COARSE_FROM = 700;
const SKIRT = 6;
/** Terrain under the road is pushed down so it never pokes through the road mesh. */
export const UNDER_ROAD = 0.3;

/** Terrain blocks (and vegetation pools) span BLOCK x BLOCK tiles. */
const BLOCK = 4;

interface Tile {
  key: string;
  /** What it was built as; rebuilt when the wanted one differs. */
  sig: string;
  mesh: THREE.Mesh;
  /** Vegetation it scattered, per 128 m tile key. */
  trees: [string, Float32Array][];
}

/** A square of terrain to build: one tile, or a whole block far away. */
interface Wanted {
  key: string;
  sig: string;
  x0: number;
  z0: number;
  size: number;
  segments: number;
  /** Tiles (tx, tz) to scatter vegetation on, and whether as far (trees only, sparser). */
  treeTiles: [number, number, boolean][];
  dist: number;
}

const sample: LandSample = {
  height: 0,
  surface: Surface.Grass,
  roadDistance: Infinity,
};

/** Streams terrain tiles (with skirts to hide LOD cracks) and trees around a focus point. */
export class TerrainView {
  readonly group = new THREE.Group();
  private readonly tiles = new Map<string, Tile>();
  private readonly material: THREE.Material;

  constructor(
    private readonly land: Landscape,
    private readonly vegetation: Vegetation,
  ) {
    this.material = terrainMaterial(land.road.edge);
  }

  /** Squares wanted around the last focus tile, nearest first, and how many are done. */
  private wanted: Wanted[] = [];
  private done = 0;
  private scanKey = "";
  private keep = new Set<string>();
  private job?: Generator<void, void>;
  /** Terrain no longer wanted, kept until its replacement is built so no gaps show. */
  private retiring = new Map<string, Tile>();

  /**
   * Builds missing terrain nearest-first until `budgetMs` is spent; drops far terrain. The
   * wanted set is only recomputed when the focus enters another tile or the road grows (it
   * allocated ~200 kB a frame when rescanned every frame).
   */
  update(focus: THREE.Vector3, budgetMs: number): void {
    const start = performance.now();
    const fx = Math.floor(focus.x / TILE);
    const fz = Math.floor(focus.z / TILE);
    const gate = Math.floor(this.land.road.finalProgress / TILE);
    const scanKey = `${fx},${fz},${gate}`;
    if (scanKey !== this.scanKey) {
      this.scanKey = scanKey;
      this.rescan(fx, fz);
    }
    // Builds run as resumable jobs, a few rows per step, so no frame overruns the budget.
    for (;;) {
      if (!this.job) {
        if (this.done >= this.wanted.length) break;
        const w = this.wanted[this.done++]!;
        const existing = this.tiles.get(w.key);
        if (existing && existing.sig === w.sig) continue;
        this.job = this.build(w);
      }
      if (performance.now() - start > budgetMs) break;
      if (this.job.next().done) this.job = undefined;
    }
    if (!this.job && this.done === this.wanted.length && this.retiring.size) {
      for (const [key, tile] of this.retiring) this.drop(key, tile);
      this.retiring.clear();
    }
  }

  private rescan(fx: number, fz: number): void {
    // Distances from the focus tile's centre, so the set only depends on the tile.
    const cx = (fx + 0.5) * TILE;
    const cz = (fz + 0.5) * TILE;
    const final = this.land.road.finalProgress;
    // Only build where the road can no longer change: its future lies further along -Z.
    const settled = (tz: number) => -tz * TILE + 64 <= final;
    const span = BLOCK * TILE;
    const reach = Math.ceil(RADIUS / span) + 1;
    const bx0 = Math.floor(fx / BLOCK);
    const bz0 = Math.floor(fz / BLOCK);
    const wanted: Wanted[] = [];
    for (let bx = bx0 - reach; bx <= bx0 + reach; bx++)
      for (let bz = bz0 - reach; bz <= bz0 + reach; bz++) {
        const nearest = Math.hypot(
          Math.max(bx * span - cx, 0, cx - (bx + 1) * span),
          Math.max(bz * span - cz, 0, cz - (bz + 1) * span),
        );
        if (nearest > RADIUS) continue;
        if (nearest > COARSE_FROM && settled(bz * BLOCK)) {
          const per = nearest <= LODS[2]!.within ? 16 : 8;
          const treeTiles: [number, number, boolean][] = [];
          for (let i = 0; i < BLOCK; i++)
            for (let j = 0; j < BLOCK; j++) {
              const tx = bx * BLOCK + i;
              const tz = bz * BLOCK + j;
              const d = Math.hypot(
                (tx + 0.5) * TILE - cx,
                (tz + 0.5) * TILE - cz,
              );
              if (d <= TREES_WITHIN) treeTiles.push([tx, tz, true]);
            }
          wanted.push({
            key: `b${bx},${bz}`,
            sig: `${per}`,
            x0: bx * span,
            z0: bz * span,
            size: span,
            segments: per * BLOCK,
            treeTiles,
            dist: nearest,
          });
          continue;
        }
        for (let i = 0; i < BLOCK; i++)
          for (let j = 0; j < BLOCK; j++) {
            const tx = bx * BLOCK + i;
            const tz = bz * BLOCK + j;
            const dist = Math.hypot(
              (tx + 0.5) * TILE - cx,
              (tz + 0.5) * TILE - cz,
            );
            if (dist > RADIUS || !settled(tz)) continue;
            const lod = LODS.findIndex((l) => dist <= l.within);
            wanted.push({
              key: `${tx},${tz}`,
              sig: `${lod}`,
              x0: tx * TILE,
              z0: tz * TILE,
              size: TILE,
              segments: LODS[lod]!.segments,
              treeTiles: dist <= TREES_WITHIN ? [[tx, tz, false]] : [],
              dist,
            });
          }
      }
    const keep = new Set(wanted.map((w) => w.key));
    this.keep = keep;
    for (const [key, tile] of this.tiles)
      if (!keep.has(key)) {
        this.tiles.delete(key);
        this.retiring.set(key, tile);
      }
    // Wanted again before it was dropped: take it back.
    for (const key of keep) {
      const tile = this.retiring.get(key);
      if (tile && !this.tiles.has(key)) {
        this.retiring.delete(key);
        this.tiles.set(key, tile);
      }
    }
    wanted.sort((a, b) => a.dist - b.dist);
    this.wanted = wanted;
    this.done = 0;
  }

  private drop(key: string, tile: Tile): void {
    this.group.remove(tile.mesh);
    tile.mesh.geometry.dispose();
    // Only its own trees: a replacement may have planted the same tile already.
    for (const [t, trees] of tile.trees) this.vegetation.remove(t, trees);
    if (this.tiles.get(key) === tile) this.tiles.delete(key);
  }

  /** Builds a tile or block, yielding between rows; the result replaces any old version. */
  private *build(w: Wanted): Generator<void, void> {
    const { key, x0, z0, segments: n } = w;
    const step = w.size / n;
    // Heights on an (n + 3)^2 grid with a one-sample border, for normals that match across tiles.
    const g = n + 3;
    const heights = new Float32Array(g * g);
    const nearRoad = new Float32Array(g * g);
    for (let j = 0; j < g; j++) {
      for (let i = 0; i < g; i++) {
        this.land.sample(x0 + (i - 1) * step, z0 + (j - 1) * step, sample);
        heights[j * g + i] = sample.height;
        nearRoad[j * g + i] = sample.roadDistance;
      }
      if (j % 4 === 3) yield;
    }

    const v = n + 1;
    const ring = 4 * n;
    const count = v * v + ring * 2;
    const pos = new Float32Array(count * 3);
    const nor = new Float32Array(count * 3);
    const variation = new Float32Array(count);
    const roadDist = new Float32Array(count);
    const forestDensity = new Float32Array(count);
    const edge = this.land.road.edge;
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
      const under =
        UNDER_ROAD * (1 - Math.min(1, Math.max(0, (road - edge + 1.2) / 1.2)));
      pos[k * 3 + 1] = h - drop - under;
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
    for (let j = 0; j < v; j++) {
      for (let i = 0; i < v; i++) put(j * v + i, i, j, 0);
      if (j % 8 === 7) yield;
    }

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
    yield;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    geo.setAttribute("variation", new THREE.BufferAttribute(variation, 1));
    geo.setAttribute("road", new THREE.BufferAttribute(roadDist, 1));
    geo.setAttribute("forest", new THREE.BufferAttribute(forestDensity, 1));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    const planted: [string, Float32Array][] = [];
    for (const [tx, tz, far] of w.treeTiles) {
      const items = yield* this.vegetation.scatter(tx, tz, TILE, far);
      if (items.length) planted.push([`${tx},${tz}`, items]);
    }
    // No longer wanted after all (the focus moved on while it was built).
    if (!this.keep.has(key)) {
      geo.dispose();
      return;
    }
    const old = this.tiles.get(key);
    if (old) this.drop(key, old);
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    this.group.add(mesh);
    const tile: Tile = { key, sig: w.sig, mesh, trees: planted };
    for (const [treeKey, items] of planted) {
      const [tx, tz] = treeKey.split(",").map(Number) as [number, number];
      this.vegetation.add(
        treeKey,
        (tx + 0.5) * TILE,
        (tz + 0.5) * TILE,
        blockKey(treeKey),
        items,
      );
    }
    this.tiles.set(key, tile);
  }
}

/** Key of the tree block holding tile "tx,tz". */
function blockKey(tileKey: string): string {
  const [tx, tz] = tileKey.split(",").map(Number) as [number, number];
  return `${Math.floor(tx / BLOCK)},${Math.floor(tz / BLOCK)}`;
}

/** Large dry/green patches across the meadows (0 = lush, 1 = dry), plus a little noise. */
export function variationAt(x: number, z: number): number {
  const dry = 0.5 + 0.5 * Math.sin(x * 0.011 + Math.sin(z * 0.007) * 3);
  return Math.min(1, dry * 0.9 + hash2(x, z, 3) * 0.1);
}
