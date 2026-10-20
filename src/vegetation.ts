import * as THREE from "three/webgpu";
import {
  attribute,
  cameraPosition,
  dFdx,
  dFdy,
  float,
  interleavedGradientNoise,
  log2,
  max,
  mix,
  normalViewGeometry,
  positionWorld,
  screenCoordinate,
  smoothstep,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
} from "three/tsl";
import type { Node } from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { Surface, type Landscape, type LandSample } from "./world/landscape";
import { hash2 } from "./world/noise";

/**
 * Trees and bushes generated with EZ-Tree (MIT), grass, ferns and rocks from Quaternius'
 * Stylized Nature MegaKit (CC0); see public/models/CREDITS.md.
 *
 * Terrain tiles scatter items into packed records (x, y, z, yaw, scale, kind). Near the car
 * every kind is drawn from a shared instanced mesh per kind and level of detail, refilled as
 * the car moves; things that end (ground cover, bushes, rocks) grow in and out at the edge
 * of their range instead of popping. The nearest trees cast their shadow from the medium
 * level, so it keeps its shape when they switch. Distant trees are billboards from an atlas
 * rendered at load, pooled per 512 m block so off-screen blocks are culled.
 */

const STRIDE = 6;
/**
 * Trees and bushes are full 3D up close and billboards beyond, cross-faded with a dither
 * between these distances (m); billboards fade out again towards the planting limit.
 */
const NEAR_START = 35;
const NEAR_END = 55;
const HORIZON_START = 650;
const HORIZON_END = 790;
const TILE_HALF = 64;
/** Pools hold at least this many instances: smaller ones would bake their matrices into the
 *  shader (a compile per size). */
const MIN_CAPACITY = 1100;
/** Refill near pools after the focus moves this far (m). */
const REFILL = 10;

interface Kind {
  model: string;
  /** Far end of each detail level (m); the last level ends the kind (or hands over to far). */
  ranges: number[];
  /** Far family (index into FAR), for trees. */
  family?: number;
  shadows: boolean;
}

const tree = (model: string, family: number): Kind => ({
  model,
  ranges: [NEAR_END],
  family,
  shadows: true,
});
const KINDS: Kind[] = [
  ...["Oak_1", "Oak_2", "Oak_3"].map((m) => tree(m, 0)),
  ...["Ash_1", "Ash_2"].map((m) => tree(m, 1)),
  ...["Pine_1", "Pine_2", "Pine_3"].map((m) => tree(m, 2)),
  ...["Bush_1", "Bush_2", "Bush_3"].map((m) => tree(m, 3)),
  { model: "Rock_Medium_1", ranges: [80, 320], shadows: true },
  { model: "Rock_Medium_2", ranges: [80, 320], shadows: true },
  { model: "Rock_Medium_3", ranges: [80, 320], shadows: true },
  ...[
    "Grass_Common_Short",
    "Grass_Common_Tall",
    "Grass_Wispy_Short",
    "Grass_Wispy_Tall",
    "Clover_1",
    "Fern_1",
  ].map((model) => ({ model, ranges: [110], shadows: false })),
];
/** Farthest any kind reaches, and ground cover (kinds from FIRST_COVER on). */
const MAX_RANGE = Math.max(...KINDS.map((k) => k.ranges.at(-1)!));
const FIRST_COVER = KINDS.findIndex((k) => !k.shadows);
const COVER_RANGE = KINDS[FIRST_COVER]!.ranges.at(-1)!;
const K = Object.fromEntries(KINDS.map((k, i) => [k.model, i]));
const BROADLEAF = [0, 1, 2, 3, 4];
const PINE = [5, 6, 7];
const BUSHES = [8, 9, 10];
const GRASS = [
  K.Grass_Common_Short!,
  K.Grass_Common_Tall!,
  K.Grass_Wispy_Short!,
  K.Grass_Wispy_Tall!,
];
const FLOWERS = [K.Clover_1!, K.Fern_1!];
/** Far stand-in model per family, and how its height compares to each kind's (by index). */
/** Billboard atlas: one cell (px) per tree kind, in a grid of columns x rows. */
const BILLBOARD = 512;
const ATLAS_COLS = 4;
const ATLAS_ROWS = 3;

/** Direction towards the sun and its light (colour times intensity), for translucency. */
const sunDirection = uniform(new THREE.Vector3(0, 1, 0));
const sunLight = uniform(new THREE.Color(1, 1, 1));

/** Leaf tints (white textures in the kit). */
const TINTS: Record<string, THREE.ColorRepresentation> = {
  // EZ-Tree's leaf textures are coloured; only evened out a little.
  leaves_oak: 0xf8f8ec,
  leaves_ash: 0xf8f8ec,
  leaves_pine: 0xf4f4e8,
  plants: 0xe8ecd0,
  // Towards the terrain's yellow-olive meadow grass.
  grass: 0xc9c47e,
  // Greyer, less orange than the kit's stylized bark.
  bark: 0xd2cac2,
};

const sample: LandSample = {
  height: 0,
  surface: Surface.Grass,
  roadDistance: Infinity,
};
const normal = new THREE.Vector3();

interface Pool {
  mesh: THREE.InstancedMesh;
  count: number;
}

interface TileItems {
  cx: number;
  cz: number;
  block: string;
  /** Packed records, ground cover first. */
  records: Float32Array;
  /** Offset where ground cover ends. */
  coverEnd: number;
}

export class Vegetation {
  readonly group = new THREE.Group();
  private readonly tiles = new Map<string, TileItems>();
  /** Geometry per kind and level, and the far stand-ins; null until loaded. */
  private geometries: THREE.BufferGeometry[][] | null = null;
  /** Distant tree billboards: shared quads, and each kind's atlas cell and size. */
  private farGeometry: THREE.BufferGeometry | null = null;
  private farMaterial: THREE.Material | null = null;
  private farBoxes: { width: number; bottom: number; top: number }[] = [];
  private cellOf: number[] = [];
  /** The billboard atlas (dev: inspect with readRenderTargetPixelsAsync). */
  atlas: THREE.RenderTarget | null = null;
  /** Tiles per far block. */
  private readonly byBlock = new Map<string, Set<TileItems>>();
  private readonly near = new Map<string, Pool>();
  private readonly far = new Map<string, Pool>();
  private readonly dirtyBlocks = new Set<string>();
  private nearDirty = true;
  private readonly lastFill = new THREE.Vector3(Infinity, 0, Infinity);

  constructor(
    private readonly land: Landscape,
    url: string,
    private readonly renderer: THREE.WebGPURenderer,
  ) {
    void this.load(url);
  }

  private async load(url: string): Promise<void> {
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(url);
    gltf.scene.updateMatrixWorld(true);
    const byName = new Map<string, THREE.Object3D>();
    gltf.scene.traverse((o) => byName.set(o.name, o));
    const materials = new Map<string, THREE.Material>();
    const geometry = (name: string) => {
      const node = byName.get(name);
      if (!node) throw new Error(`nature.glb: no node ${name}`);
      const parts: THREE.Mesh[] = [];
      node.traverse((o) => {
        if (o instanceof THREE.Mesh) parts.push(o);
      });
      const geos = parts.map((m) => {
        const src = m.material as THREE.MeshStandardMaterial;
        if (!materials.has(src.name))
          materials.set(src.name, makeMaterial(src));
        return bake(m);
      });
      const merged = mergeGeometries(geos, true)!;
      merged.userData.materials = parts.map((m) =>
        materials.get((m.material as THREE.Material).name)!,
      );
      merged.computeBoundingSphere();
      merged.computeBoundingBox();
      return merged;
    };
    this.geometries = KINDS.map((k) =>
      k.ranges.map((_, lod) => geometry(`${k.model}_lod${lod}`)),
    );
    this.billboards();
    this.nearDirty = true;
    for (const block of this.byBlock.keys()) this.dirtyBlocks.add(block);
  }

  /** Registers a tile's items (from `scatter`). */
  add(
    key: string,
    cx: number,
    cz: number,
    block: string,
    records: Float32Array,
  ): void {
    const old = this.tiles.get(key);
    if (old) this.unlink(old);
    const t: TileItems = {
      cx,
      cz,
      block,
      records,
      coverEnd: coverFirst(records),
    };
    this.tiles.set(key, t);
    let set = this.byBlock.get(block);
    if (!set) this.byBlock.set(block, (set = new Set()));
    set.add(t);
    this.dirtyBlocks.add(block);
    this.nearDirty = true;
  }

  /** Unregisters a tile's items, if `records` are still the registered ones. */
  remove(key: string, records: Float32Array): void {
    const t = this.tiles.get(key);
    if (!t || t.records !== records) return;
    this.tiles.delete(key);
    this.unlink(t);
    this.nearDirty = true;
  }

  private unlink(t: TileItems): void {
    const set = this.byBlock.get(t.block);
    set?.delete(t);
    if (set && set.size === 0) this.byBlock.delete(t.block);
    this.dirtyBlocks.add(t.block);
  }

  /**
   * Scatters a tile's vegetation: trees in clumpy forests, bushes at their edges and in the
   * meadows, rocks (more on slopes) and, if `cover`, grass and flowers, dense along the road.
   */
  *scatter(
    tx: number,
    tz: number,
    size: number,
    cover: boolean,
  ): Generator<void, Float32Array> {
    const out: number[] = [];
    const land = this.land;
    const edge = land.road.edge;
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
    // Trees and bushes on an 8 m grid.
    const step = 8;
    for (let j = 0; j < size / step; j++) {
      for (let i = 0; i < size / step; i++) {
        const gx = Math.floor(x0 / step) + i;
        const gz = Math.floor(z0 / step) + j;
        const x = (gx + hash2(gx, gz, 11)) * step;
        const z = (gz + hash2(gx, gz, 12)) * step;
        const d = forest(x, z);
        const roll = hash2(gx, gz, 13);
        // Woods, plus the odd lone tree in the meadows.
        const isTree = roll < d * 0.85 + 0.012;
        const isBush = !isTree && roll < d * 0.95 + 0.025 + 0.06 * edgeOf(d);
        if (!isTree && !isBush) continue;
        land.sample(x, z, sample);
        // Woods stop a varying distance from the road, not in a straight wall.
        const clear = isTree ? 7 + 14 * hash2(gx, gz, 18) ** 2 : 3;
        if (sample.roadDistance < edge + clear) continue;
        if (land.normal(x, z, normal).y < 0.8) continue;
        const yaw = hash2(gx, gz, 15) * Math.PI * 2;
        if (isTree) {
          // Pines on the higher ground, broadleaf below, mixed in between.
          const pine =
            hash2(gx, gz, 16) <
            0.2 + 0.7 * THREE.MathUtils.smoothstep(sample.height, 5, 35);
          const kinds = pine ? PINE : BROADLEAF;
          const kind = kinds[Math.floor(hash2(gx, gz, 17) * kinds.length)]!;
          put(
            x,
            sample.height - 0.15,
            z,
            yaw,
            0.7 + hash2(gx, gz, 14) ** 1.3 * 1.1,
            kind,
          );
          // Undergrowth at its foot: ferns and long grass.
          if (cover)
            for (let e = 0; e < 6; e++) {
              if (hash2(gx, gz, 100 + e) > 0.75) continue;
              const a = hash2(gx, gz, 110 + e) * 6.283;
              const r = 0.8 + hash2(gx, gz, 120 + e) * 3;
              put(
                x + Math.cos(a) * r,
                sample.height - 0.1,
                z + Math.sin(a) * r,
                a,
                0.45 + hash2(gx, gz, 130 + e) * 0.35,
                e % 2 === 0 ? K.Fern_1! : K.Grass_Wispy_Tall!,
              );
            }
        } else {
          const kind = BUSHES[Math.floor(hash2(gx, gz, 17) * BUSHES.length)]!;
          put(
            x,
            sample.height - 0.1,
            z,
            yaw,
            0.7 + hash2(gx, gz, 14) * 0.7,
            kind,
          );
        }
      }
      yield;
    }
    // Rocks on a 24 m grid, more on slopes.
    for (let j = 0; j < size / 24; j++)
      for (let i = 0; i < size / 24; i++) {
        const gx = Math.floor(x0 / 24) + i;
        const gz = Math.floor(z0 / 24) + j;
        const x = (gx + hash2(gx, gz, 21)) * 24;
        const z = (gz + hash2(gx, gz, 22)) * 24;
        const slope = 1 - land.normal(x, z, normal).y;
        if (hash2(gx, gz, 23) > 0.22 + slope * 4) continue;
        land.sample(x, z, sample);
        if (sample.roadDistance < edge + 2) continue;
        // A group of one big, partly buried rock and a few smaller ones around it.
        const s = 0.6 + hash2(gx, gz, 24) ** 2 * 1.6;
        const kind = K.Rock_Medium_1! + Math.floor(hash2(gx, gz, 25) * 3);
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
            K.Rock_Medium_1! + ((kind + e) % 3),
          );
        }
      }
    if (cover) {
      // Ground cover: a dense band along the road verge, and sparse clumps everywhere.
      const g = 1.05;
      for (let j = 0; j < size / g; j++) {
        for (let i = 0; i < size / g; i++) {
          const gx = Math.floor(x0 / g) + i;
          const gz = Math.floor(z0 / g) + j;
          const x = (gx + hash2(gx, gz, 31)) * g;
          const z = (gz + hash2(gx, gz, 32)) * g;
          const roll = hash2(gx, gz, 33);
          if (roll > 0.75) continue;
          land.sample(x, z, sample);
          const r = sample.roadDistance - edge;
          if (r < 0.25) continue;
          // Dense near the road, thinning out to sparse meadow clumps by 16 m; in patches.
          const clump =
            0.55 +
            0.45 *
              Math.sin(x * 0.31 + Math.sin(z * 0.23) * 2) *
              Math.sin(z * 0.27 + x * 0.05);
          const keep =
            (0.75 * (1 - THREE.MathUtils.smoothstep(r, 6, 16)) + 0.04) * clump;
          if (roll > keep) continue;
          // Flowers in patches; mostly the cheap short grass tufts, tall and wispy between.
          const patch =
            0.5 +
            0.5 *
              Math.sin(x * 0.07 + Math.sin(z * 0.05) * 2) *
              Math.sin(z * 0.061);
          const flower = hash2(gx, gz, 34) < 0.04 + 0.3 * patch ** 3;
          const pick = hash2(gx, gz, 35);
          const kind = flower
            ? FLOWERS[Math.floor(pick * FLOWERS.length)]!
            : GRASS[pick < 0.45 ? 0 : pick < 0.7 ? 1 : pick < 0.88 ? 2 : 3]!;
          // Shorter right at the shoulder, like a mown verge.
          const base = flower
            ? 0.13 + hash2(gx, gz, 36) * 0.09
            : 0.25 + hash2(gx, gz, 36) * 0.25;
          const s = base * Math.min(1, 0.6 + r / 5);
          put(x, sample.height - 0.08, z, hash2(gx, gz, 37) * 6.283, s, kind);
          // Grass grows in clumps: a few more tufts close around this one.
          if (flower) continue;
          const extra = Math.floor(hash2(gx, gz, 38) * 3.5);
          for (let e = 0; e < extra; e++) {
            const a = hash2(gx, gz, 60 + e) * 6.283;
            const d = 0.18 + hash2(gx, gz, 70 + e) * 0.3;
            put(
              x + Math.cos(a) * d,
              sample.height - 0.08,
              z + Math.sin(a) * d,
              a * 2.3,
              s * (0.7 + hash2(gx, gz, 90 + e) * 0.4),
              GRASS[Math.floor(hash2(gx, gz, 80 + e) * 2)]!,
            );
          }
        }
        if (j % 8 === 7) yield;
      }
    }
    return new Float32Array(out);
  }

  /** Camera position, for the near/far cross-fade. */
  setViewer(position: THREE.Vector3): void {
    viewer.value.copy(position);
  }

  /** Sun (or moon) direction and light, for light shining through the leaves. */
  setSun(light: THREE.DirectionalLight): void {
    sunDirection.value.copy(light.position).normalize();
    sunLight.value.copy(light.color).multiplyScalar(light.intensity);
  }

  /** Refills pools for the focus position. */
  update(focus: THREE.Vector3): void {
    if (!this.geometries) return;
    const moved = Math.hypot(
      focus.x - this.lastFill.x,
      focus.z - this.lastFill.z,
    );
    if (moved > REFILL || this.nearDirty) {
      this.lastFill.copy(focus);
      this.nearDirty = false;
      this.fillNear(focus);
      // Blocks near the focus hand trees to and from the near pools.
      // Blocks across the near/far boundary hand trees over; the others only update shadows.
      const handover = KINDS[0]!.ranges.at(-1)! + 2 * REFILL;
      for (const block of this.byBlock.keys())
        if (blockDistance(block, focus) < handover) this.dirtyBlocks.add(block);
    }
    for (const block of this.dirtyBlocks) this.fillFar(block, this.lastFill);
    this.dirtyBlocks.clear();
  }

  /**
   * Refills the near pools: two passes over the records in range (count, then write straight
   * into the pools), skipping tiles too far for any of their kinds.
   */
  private fillNear(focus: THREE.Vector3): void {
    const levels = 3;
    const counts = new Int32Array(KINDS.length * levels);
    const visit = (
      fn: (
        i: number,
        r: Float32Array,
        kind: number,
        lod: number,
        s: number,
      ) => void,
    ) => {
      for (const t of this.tiles.values()) {
        // Nearest point of the tile's square.
        const dx = Math.max(Math.abs(t.cx - focus.x) - TILE_HALF, 0);
        const dz = Math.max(Math.abs(t.cz - focus.z) - TILE_HALF, 0);
        const near = Math.hypot(dx, dz);
        if (near > MAX_RANGE) continue;
        const r = t.records;
        // Ground cover (stored first) only matters close by.
        const from = near > COVER_RANGE ? t.coverEnd : 0;
        for (let i = from; i < r.length; i += STRIDE) {
          const k = r[i + 5]!;
          const kind = KINDS[k]!;
          const ex = r[i]! - focus.x;
          const ez = r[i + 2]! - focus.z;
          const d = Math.sqrt(ex * ex + ez * ez);
          const ranges = kind.ranges;
          let lod = 0;
          while (lod < ranges.length && d > ranges[lod]!) lod++;
          if (lod === ranges.length) continue;
          let s = r[i + 4]!;
          // Grow in and out over the last 20 m of a kind that ends there.
          if (lod === ranges.length - 1 && kind.family === undefined) {
            const t = Math.min(1, Math.max(0, (ranges[lod]! - d) / 20));
            s *= t * t * (3 - 2 * t);
            if (s < 0.02) continue;
          }
          fn(i, r, k, lod, s);
        }
      }
    };
    visit((_i, _r, k, lod) => counts[k * levels + lod]!++);
    const pools: (Pool | undefined)[] = [];
    for (const [key, pool] of this.near) {
      pool.count = 0;
      pool.mesh.count = 0;
      pools[Number(key)] = pool;
    }
    counts.forEach((n, key) => {
      if (!n) return;
      const k = Math.floor(key / levels);
      pools[key] = this.pool(
        this.near,
        String(key),
        this.geometries![k]![key % levels]!,
        n,
        KINDS[k]!.shadows,
      );
    });
    visit((i, r, k, lod, s) => {
      write(
        pools[k * levels + lod]!,
        r[i]!,
        r[i + 1]!,
        r[i + 2]!,
        r[i + 3]!,
        s,
      );
    });
    counts.forEach((n, key) => {
      if (n) finish(pools[key]!, false);
    });
  }

  /** Far stand-ins for the trees of one block that are beyond the near pools. */
  private fillFar(block: string, focus: THREE.Vector3): void {
    const start = NEAR_START;
    const tiles = this.byBlock.get(block);
    const visit = (fn: (r: Float32Array, i: number) => void) => {
      for (const t of tiles ?? []) {
        const r = t.records;
        // Trees come after the ground cover.
        for (let i = t.coverEnd; i < r.length; i += STRIDE) {
          if (KINDS[r[i + 5]!]!.family === undefined) continue;
          const ex = r[i]! - focus.x;
          const ez = r[i + 2]! - focus.z;
          if (ex * ex + ez * ez <= start * start) continue;
          fn(r, i);
        }
      }
    };
    let n = 0;
    visit(() => n++);
    if (n === 0) {
      const p = this.far.get(block);
      if (p) {
        this.group.remove(p.mesh);
        p.mesh.dispose();
        this.far.delete(block);
      }
      return;
    }
    const p = this.pool(this.far, block, this.farGeometry!, n, true, [
      this.farMaterial!,
    ]);
    // Own copy of the quads per pool, for its per-instance atlas cells.
    let cells = p.mesh.geometry.getAttribute("cell") as
      THREE.InstancedBufferAttribute | undefined;
    if (!cells || cells.count < p.mesh.instanceMatrix.count) {
      p.mesh.geometry = this.farGeometry!.clone();
      cells = new THREE.InstancedBufferAttribute(
        new Float32Array(p.mesh.instanceMatrix.count),
        1,
      );
      p.mesh.geometry.setAttribute("cell", cells);
    }
    const cellArray = cells.array as Float32Array;
    visit((r, i) => {
      const kind = r[i + 5]!;
      const box = this.farBoxes[kind]!;
      const s = r[i + 4]!;
      cellArray[p.count] = this.cellOf[kind]!;
      // Unit quads stretched to the kind's crown width and height, standing on its base.
      writeScaled(
        p,
        r[i]!,
        r[i + 1]! + box.bottom * s,
        r[i + 2]!,
        r[i + 3]!,
        box.width * s,
        (box.top - box.bottom) * s,
      );
    });
    cells.needsUpdate = true;
    finish(p, true);
  }

  /**
   * Distant stand-ins for trees: every tree kind's side view rendered once (unlit colour and
   * alpha) into a cell of one atlas, shown on two crossed quads lit with crown normals like
   * the near leaves. Each far instance picks its kind's cell, so a block takes one draw.
   */
  private billboards(): void {
    const kinds = KINDS.flatMap((k, i) => (k.family === undefined ? [] : [i]));
    const target = new THREE.RenderTarget(
      BILLBOARD * ATLAS_COLS,
      BILLBOARD * ATLAS_ROWS,
      {
        generateMipmaps: true,
        minFilter: THREE.LinearMipmapLinearFilter,
      },
    );
    const r = this.renderer;
    const prevTarget = r.getRenderTarget();
    const prevColor = r.getClearColor(new THREE.Color());
    const prevAlpha = r.getClearAlpha();
    const prevAuto = r.autoClear;
    // Leaf-coloured but transparent background, so mipmaps don't bleed dark edges in.
    r.setClearColor(new THREE.Color(0.16, 0.22, 0.08), 0);
    r.setRenderTarget(target);
    r.clear();
    r.autoClear = false;
    this.farBoxes = KINDS.map(() => ({ width: 1, bottom: 0, top: 1 }));
    kinds.forEach((kind, cell) => {
      const tree = this.geometries![kind]![0]!;
      const box = tree.boundingBox!;
      const w = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) * 1.02;
      this.farBoxes[kind] = { width: w, bottom: box.min.y, top: box.max.y };
      const flat = (
        tree.userData.materials as THREE.MeshStandardNodeMaterial[]
      ).map((m) => {
        const b = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
        b.colorNode = m.colorNode;
        if (m.maskShadowNode) b.maskNode = m.maskShadowNode;
        return b;
      });
      const capture = new THREE.Scene();
      capture.add(new THREE.Mesh(tree, flat));
      const camera = new THREE.OrthographicCamera(
        -w / 2,
        w / 2,
        box.max.y,
        box.min.y,
        0.1,
        400,
      );
      camera.position.set(0, 0, 200);
      camera.lookAt(0, 0, 0);
      target.viewport.set(
        (cell % ATLAS_COLS) * BILLBOARD,
        Math.floor(cell / ATLAS_COLS) * BILLBOARD,
        BILLBOARD,
        BILLBOARD,
      );
      r.render(capture, camera);
      for (const m of flat) m.dispose();
    });
    target.viewport.set(0, 0, target.width, target.height);
    r.autoClear = prevAuto;
    r.setRenderTarget(prevTarget);
    r.setClearColor(prevColor, prevAlpha);
    this.cellOf = KINDS.map((_, i) => kinds.indexOf(i));
    this.atlas = target;

    // Two crossed unit quads (x, z in -0.5..0.5, y in 0..1); crown normals out from the
    // centre and a little up.
    const pos: number[] = [];
    const nor: number[] = [];
    const uvs: number[] = [];
    const index: number[] = [];
    for (const axis of [0, 1]) {
      const base = pos.length / 3;
      for (const [u, v] of [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ] as const) {
        const a = u - 0.5;
        const p = axis === 0 ? [a, v, 0] : [0, v, a];
        pos.push(...p);
        const n = new THREE.Vector3(
          p[0]! * 2,
          (v - 0.5) * 2 + 0.3,
          p[2]! * 2,
        ).normalize();
        nor.push(n.x, n.y, n.z);
        uvs.push(u, v);
      }
      index.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(index);
    geo.addGroup(0, index.length, 0);
    this.farGeometry = geo;

    const m = new THREE.MeshStandardNodeMaterial({
      roughness: 0.9,
      metalness: 0,
      side: THREE.DoubleSide,
    });
    const cell = attribute<"float">("cell", "float");
    // Render targets are stored top row first and sampled unflipped in WebGPU: the cell's top
    // (the treetop) is at its smallest v.
    const cellUv = vec2(cell.mod(ATLAS_COLS), cell.div(ATLAS_COLS).floor())
      .add(vec2(uv().x, uv().y.oneMinus()))
      .div(vec2(ATLAS_COLS, ATLAS_ROWS));
    const tex = texture(target.texture, cellUv);
    // A touch brighter: the capture has no sky light bouncing inside the crown.
    m.colorNode = tex.rgb.mul(1.08);
    const t = uv().mul(BILLBOARD);
    const lod = log2(max(max(dFdx(t).length(), dFdy(t).length()), 1));
    m.maskNode = tex.a.mul(lod.mul(0.3).add(1)).greaterThan(0.45).and(showFar);
    m.maskShadowNode = tex.a.greaterThan(0.5).and(showFar);
    m.normalNode = normalViewGeometry;
    m.emissiveNode = translucency(tex.rgb);
    this.farMaterial = m;
  }

  /** The pool for `key`, with room for `n` instances (recreated larger if needed). */
  private pool(
    pools: Map<string, Pool>,
    key: string,
    geometry: THREE.BufferGeometry,
    n: number,
    shadows: boolean,
    materials?: THREE.Material[],
  ): Pool {
    let p = pools.get(key);
    if (p && p.mesh.instanceMatrix.count < n) {
      this.group.remove(p.mesh);
      p.mesh.dispose();
      p = undefined;
    }
    if (!p) {
      let capacity = MIN_CAPACITY;
      while (capacity < n) capacity *= 2;
      const mesh = new THREE.InstancedMesh(
        geometry,
        materials ?? (geometry.userData.materials as THREE.Material[]),
        capacity,
      );
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(capacity * 3),
        3,
      );
      mesh.castShadow = shadows;
      mesh.receiveShadow = true;
      mesh.frustumCulled = pools === this.far;
      this.group.add(mesh);
      p = { mesh, count: 0 };
      pools.set(key, p);
    }
    p.count = 0;
    return p;
  }
}

/** Reorders packed records in place so ground cover comes first; returns where it ends. */
function coverFirst(r: Float32Array): number {
  const n = r.length / STRIDE;
  const isCover = (i: number) => (r[i * STRIDE + 5]! >= FIRST_COVER ? 1 : 0);
  const order = [...Array(n).keys()].sort((a, b) => isCover(b) - isCover(a));
  const copy = r.slice();
  order.forEach((src, dst) =>
    r.set(copy.subarray(src * STRIDE, src * STRIDE + STRIDE), dst * STRIDE),
  );
  let end = 0;
  while (end < n && isCover(end)) end++;
  return end * STRIDE;
}

/**
 * Forest density (0..1): groves and woods with open meadows between them, from a large
 * field broken up by a smaller one.
 */
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
  return THREE.MathUtils.smoothstep(large * 0.7 + small * 0.45, 0.42, 0.8);
}

/** Highest at a forest's edge (density around 0.1-0.4), for bushes. */
function edgeOf(d: number): number {
  return Math.max(0, 1 - Math.abs(d - 0.25) / 0.2);
}

/** Distance from a point to a block's 512 m square (block key "bx,bz"). */
function blockDistance(block: string, p: THREE.Vector3): number {
  const [bx, bz] = block.split(",").map(Number) as [number, number];
  const span = 512;
  const dx = Math.max(bx * span - p.x, 0, p.x - (bx + 1) * span);
  const dz = Math.max(bz * span - p.z, 0, p.z - (bz + 1) * span);
  return Math.hypot(dx, dz);
}

function writeScaled(
  p: Pool,
  x: number,
  y: number,
  z: number,
  yaw: number,
  sxz: number,
  sy: number,
): void {
  const m = p.mesh.instanceMatrix.array as Float32Array;
  const o = p.count * 16;
  const c = Math.cos(yaw);
  const n = Math.sin(yaw);
  m.set(
    [c * sxz, 0, -n * sxz, 0, 0, sy, 0, 0, n * sxz, 0, c * sxz, 0, x, y, z, 1],
    o,
  );
  tint(p, x, z);
  p.count++;
}

function write(
  p: Pool,
  x: number,
  y: number,
  z: number,
  yaw: number,
  s: number,
): void {
  writeScaled(p, x, y, z, yaw, s, s);
}

/** Per-plant tint: brightness and a slight warm/cool shift, from its position. */
function tint(p: Pool, x: number, z: number): void {
  const b = 0.82 + 0.3 * hash2(x, z, 41);
  const h = hash2(x, z, 42) * 2 - 1;
  const col = p.mesh.instanceColor!.array as Float32Array;
  col[p.count * 3] = b * (1 + 0.07 * h);
  col[p.count * 3 + 1] = b;
  col[p.count * 3 + 2] = b * (1 - 0.12 * h);
}

function finish(p: Pool, bounds: boolean): void {
  p.mesh.count = p.count;
  p.mesh.instanceMatrix.needsUpdate = true;
  p.mesh.instanceColor!.needsUpdate = true;
  if (bounds && p.count) p.mesh.computeBoundingSphere();
}

/** A mesh's geometry in plain floats, with its node transform applied (quantized input). */
function bake(mesh: THREE.Mesh): THREE.BufferGeometry {
  const src = mesh.geometry;
  const geo = new THREE.BufferGeometry();
  for (const name of ["position", "normal", "uv", "color"]) {
    const a = src.getAttribute(name);
    const size = name === "uv" ? 2 : 3;
    const out = new Float32Array(src.getAttribute("position").count * size);
    if (a)
      for (let i = 0; i < a.count; i++)
        for (let k = 0; k < size; k++) out[i * size + k] = a.getComponent(i, k);
    else out.fill(1);
    geo.setAttribute(name, new THREE.BufferAttribute(out, size));
  }
  geo.setIndex(src.getIndex());
  geo.applyMatrix4(mesh.matrixWorld);
  return geo;
}

/** Camera position, for the near/far cross-fade (also in shadow passes). */
const viewer = uniform(new THREE.Vector3());

/** Screen-space dither (0..1) and how far a fragment is into the billboard range (0..1). */
function farness(): Node<"float"> {
  return smoothstep(NEAR_START, NEAR_END, positionWorld.distance(viewer));
}
const dither = interleavedGradientNoise(screenCoordinate.xy);
/** Keep for the 3D trees, and for the billboards (complementary pixels in the cross-fade). */
const showNear = dither.greaterThanEqual(farness());
const showFar = dither
  .lessThan(farness())
  .and(
    dither.lessThan(
      smoothstep(
        HORIZON_START,
        HORIZON_END,
        positionWorld.distance(viewer),
      ).oneMinus(),
    ),
  );

/**
 * Light through the leaves: strongest looking towards the sun, plus a little all round, so
 * backlit and shaded crowns don't go black.
 */
function translucency(albedo: Node<"vec3">): Node<"vec3"> {
  const view = positionWorld.sub(cameraPosition).normalize();
  const back = max(view.dot(sunDirection), 0).pow(3);
  return albedo.mul(sunLight).mul(back.mul(0.2).add(0.035));
}

/** Node material for a kit material: textured, vertex-coloured, leaves alpha-masked. */
function makeMaterial(src: THREE.MeshStandardMaterial): THREE.Material {
  const m = new THREE.MeshStandardNodeMaterial({
    name: src.name,
    roughness: src.name === "rock" ? 0.85 : 0.9,
    metalness: 0,
    vertexColors: true,
    // Blades, cards and leaves are single planes; only trunks and rocks are closed.
    side: /bark|rock/.test(src.name) ? THREE.FrontSide : THREE.DoubleSide,
  });
  const map = src.map!;
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 4;
  const tex = texture(map, uv());
  const tint = new THREE.Color(TINTS[src.name] ?? 0xffffff);
  // Bark half desaturated: the kit's is a stylized orange-brown.
  const rgb =
    src.name === "bark"
      ? mix(vec3(tex.rgb.dot(vec3(0.2126, 0.7152, 0.0722))), tex.rgb, 0.3)
      : tex.rgb;
  m.colorNode = rgb.mul(vec3(tint.r, tint.g, tint.b));
  if (src.normalMap) m.normalMap = src.normalMap;
  if (src.alphaTest > 0) {
    // Alpha-tested leaves thin out in the smaller mip levels until distant trees vanish; the
    // alpha is boosted with the mip level to keep their coverage.
    const size = float((map.image as { width?: number } | null)?.width ?? 1024);
    const t = uv().mul(size);
    const lod = log2(max(max(dFdx(t).length(), dFdy(t).length()), 1));
    m.maskNode = tex.a.mul(lod.mul(0.35).add(1)).greaterThan(0.4);
    // Shadow maps sample the texture coarsely, where the boost would make crowns solid.
    m.maskShadowNode = tex.a.greaterThan(0.5);
  }
  // Generated trees and bushes hand over to billboards with distance.
  if (/^(bark|leaves)_/.test(src.name)) {
    m.maskNode = m.maskNode
      ? (m.maskNode as Node<"bool">).and(showNear)
      : showNear;
    m.maskShadowNode = m.maskShadowNode
      ? (m.maskShadowNode as Node<"bool">).and(showNear)
      : showNear;
  }
  // Generated leaves carry crown normals (outwards from the crown); three would flip them on
  // back faces, which turns half the cards towards the trunk and darkens the crown.
  if (src.name.startsWith("leaves_")) {
    m.normalNode = normalViewGeometry;
    // Light through the leaves: strongest looking towards the sun, plus a little all round,
    // so backlit and shaded crowns don't go black.
    m.emissiveNode = translucency(rgb.mul(vec3(tint.r, tint.g, tint.b)));
  }
  return m;
}
