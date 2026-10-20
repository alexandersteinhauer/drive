import * as THREE from "three/webgpu";
import {
  dFdx,
  dFdy,
  float,
  log2,
  max,
  mix,
  texture,
  uv,
  vec3,
} from "three/tsl";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { Surface, type Landscape, type LandSample } from "./world/landscape";
import { hash2 } from "./world/noise";

/**
 * Trees, bushes, rocks and ground cover from Quaternius' Stylized Nature MegaKit (CC0).
 *
 * Terrain tiles scatter items into packed records (x, y, z, yaw, scale, kind). Near the car
 * every kind is drawn from a shared instanced mesh per kind and level of detail, refilled as
 * the car moves; things that end (ground cover, bushes, rocks) grow in and out at the edge
 * of their range instead of popping. Distant trees use one cheap model per family (broadleaf,
 * pine), pooled per 512 m block so off-screen blocks are culled.
 */

const STRIDE = 6;
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
  ranges: [50, 160],
  family,
  shadows: true,
});
const KINDS: Kind[] = [
  ...[1, 2, 3, 4, 5].map((i) => tree(`CommonTree_${i}`, 0)),
  ...[1, 2, 3, 4, 5].map((i) => tree(`Pine_${i}`, 1)),
  { model: "Bush_Common", ranges: [60, 260], shadows: true },
  { model: "Bush_Common_Flowers", ranges: [60, 260], shadows: true },
  { model: "Rock_Medium_1", ranges: [80, 320], shadows: true },
  { model: "Rock_Medium_2", ranges: [80, 320], shadows: true },
  { model: "Rock_Medium_3", ranges: [80, 320], shadows: true },
  ...[
    "Grass_Common_Short",
    "Grass_Common_Tall",
    "Grass_Wispy_Short",
    "Grass_Wispy_Tall",
    "Clover_1",
    "Flower_3_Group",
    "Flower_4_Group",
    "Fern_1",
  ].map((model) => ({ model, ranges: [110], shadows: false })),
];
const K = Object.fromEntries(KINDS.map((k, i) => [k.model, i]));
const BROADLEAF = [0, 1, 2, 3, 4];
const PINE = [5, 6, 7, 8, 9];
const GRASS = [
  K.Grass_Common_Short!,
  K.Grass_Common_Tall!,
  K.Grass_Wispy_Short!,
  K.Grass_Wispy_Tall!,
];
const FLOWERS = [K.Clover_1!, K.Flower_3_Group!, K.Flower_4_Group!, K.Fern_1!];
/** Far stand-in model per family, and how its height compares to each kind's (by index). */
const FAR = ["CommonTree_3_lod2", "Pine_2_lod2"];
/** Far trees in blocks nearer than this (m, to the block's square) cast shadows. */
const FAR_SHADOWS = 250;
/** Distance (m) beyond which trees are left out entirely (the terrain stops planting them). */
const TREE_END = 1000;

/** Leaf tints (white textures in the kit). */
const TINTS: Record<string, THREE.ColorRepresentation> = {
  leaves: 0x96b552,
  needles: 0x5f8a44,
  bush: 0x88ad50,
  plants: 0xe8ecd0,
  // Towards the terrain's yellow-olive meadow grass.
  grass: 0xc9c47e,
  // Greyer, less orange than the kit's stylized bark.
  bark: 0x9a948e,
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
  records: Float32Array;
}

export class Vegetation {
  readonly group = new THREE.Group();
  private readonly tiles = new Map<string, TileItems>();
  /** Geometry per kind and level, and the far stand-ins; null until loaded. */
  private geometries: THREE.BufferGeometry[][] | null = null;
  private farGeometries: THREE.BufferGeometry[] = [];
  private readonly near = new Map<string, Pool>();
  private readonly far = new Map<string, Pool>();
  private readonly dirtyBlocks = new Set<string>();
  private nearDirty = true;
  private readonly lastFill = new THREE.Vector3(Infinity, 0, Infinity);

  constructor(
    private readonly land: Landscape,
    url: string,
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
    this.farGeometries = FAR.map(geometry);
    this.nearDirty = true;
    for (const t of this.tiles.values()) this.dirtyBlocks.add(t.block);
  }

  /** Registers a tile's items (from `scatter`). */
  add(
    key: string,
    cx: number,
    cz: number,
    block: string,
    records: Float32Array,
  ): void {
    this.tiles.set(key, { cx, cz, block, records });
    this.dirtyBlocks.add(block);
    this.nearDirty = true;
  }

  /** Unregisters a tile's items, if `records` are still the registered ones. */
  remove(key: string, records: Float32Array): void {
    const t = this.tiles.get(key);
    if (!t || t.records !== records) return;
    this.tiles.delete(key);
    this.dirtyBlocks.add(t.block);
    this.nearDirty = true;
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
            THREE.MathUtils.smoothstep(sample.height, 5, 35);
          const kinds = pine ? PINE : BROADLEAF;
          const kind = kinds[Math.floor(hash2(gx, gz, 17) * kinds.length)]!;
          put(
            x,
            sample.height - 0.1,
            z,
            yaw,
            1 + hash2(gx, gz, 14) * 0.9,
            kind,
          );
        } else {
          const kind =
            hash2(gx, gz, 17) < 0.7 ? K.Bush_Common! : K.Bush_Common_Flowers!;
          put(
            x,
            sample.height - 0.1,
            z,
            yaw,
            0.7 + hash2(gx, gz, 14) * 0.8,
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
      const g = 1.1;
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
        }
        if (j % 8 === 7) yield;
      }
    }
    return new Float32Array(out);
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
      const reach = KINDS[0]!.ranges.at(-1)! + REFILL + 400;
      for (const t of this.tiles.values())
        if (Math.hypot(t.cx - focus.x, t.cz - focus.z) < reach)
          this.dirtyBlocks.add(t.block);
    }
    for (const block of this.dirtyBlocks) this.fillFar(block, this.lastFill);
    this.dirtyBlocks.clear();
  }

  private fillNear(focus: THREE.Vector3): void {
    const maxRange = Math.max(...KINDS.map((k) => k.ranges.at(-1)!));
    const counts = new Map<string, number>();
    const writes: [string, number, number, number, number, number, number][] =
      [];
    for (const t of this.tiles.values()) {
      if (Math.hypot(t.cx - focus.x, t.cz - focus.z) > maxRange + 100) continue;
      const r = t.records;
      for (let i = 0; i < r.length; i += STRIDE) {
        const kind = KINDS[r[i + 5]!]!;
        const d = Math.hypot(r[i]! - focus.x, r[i + 2]! - focus.z);
        const lod = kind.ranges.findIndex((end) => d <= end);
        if (lod < 0) continue;
        let s = r[i + 4]!;
        // Grow in and out over the last 20 m of a kind that ends there.
        if (lod === kind.ranges.length - 1 && kind.family === undefined)
          s *= THREE.MathUtils.smoothstep(kind.ranges[lod]! - d, 0, 20);
        if (s < 0.02) continue;
        const key = `${r[i + 5]}:${lod}`;
        counts.set(key, (counts.get(key) ?? 0) + 1);
        writes.push([
          key,
          r[i]!,
          r[i + 1]!,
          r[i + 2]!,
          r[i + 3]!,
          s,
          r[i + 5]!,
        ]);
      }
    }
    for (const [key, pool] of this.near) {
      pool.count = 0;
      if (!counts.has(key)) pool.mesh.count = 0;
    }
    for (const [key, n] of counts) {
      const [kind, lod] = key.split(":").map(Number) as [number, number];
      this.pool(
        this.near,
        key,
        this.geometries![kind]![lod]!,
        n,
        KINDS[kind]!.shadows,
      );
    }
    for (const [key, x, y, z, yaw, s] of writes)
      write(this.near.get(key)!, x, y, z, yaw, s);
    for (const [key] of counts) finish(this.near.get(key)!, false);
  }

  /** Far stand-ins for the trees of one block that are beyond the near pools. */
  private fillFar(block: string, focus: THREE.Vector3): void {
    const start = KINDS[0]!.ranges.at(-1)!;
    const heights = this.geometries!.map((g) => g[0]!.boundingBox?.max.y ?? 1);
    const pools = FAR.map((_, f) => this.far.get(`${block}:${f}`));
    for (const p of pools) if (p) p.count = 0;
    const entries: [number, number, number, number, number, number][] = [];
    for (const t of this.tiles.values()) {
      if (t.block !== block) continue;
      const r = t.records;
      for (let i = 0; i < r.length; i += STRIDE) {
        const kind = KINDS[r[i + 5]!]!;
        if (kind.family === undefined) continue;
        const d = Math.hypot(r[i]! - focus.x, r[i + 2]! - focus.z);
        if (d <= start || d > TREE_END) continue;
        entries.push([
          kind.family,
          r[i]!,
          r[i + 1]!,
          r[i + 2]!,
          r[i + 3]!,
          r[i + 4]! * (heights[r[i + 5]!]! / this.farHeight(kind.family)),
        ]);
      }
    }
    FAR.forEach((_, f) => {
      const n = entries.filter((e) => e[0] === f).length;
      const key = `${block}:${f}`;
      if (n === 0) {
        const p = this.far.get(key);
        if (p) {
          this.group.remove(p.mesh);
          p.mesh.dispose();
          this.far.delete(key);
        }
        return;
      }
      this.pool(this.far, key, this.farGeometries[f]!, n, true);
      // Distant blocks cast no shadows: the last cascade ends at 450 m.
      this.far.get(key)!.mesh.castShadow =
        blockDistance(block, focus) < FAR_SHADOWS;
    });
    for (const [f, x, y, z, yaw, s] of entries)
      write(this.far.get(`${block}:${f}`)!, x, y, z, yaw, s);
    FAR.forEach((_, f) => {
      const p = this.far.get(`${block}:${f}`);
      if (p) finish(p, true);
    });
  }

  private farHeight(family: number): number {
    const g = this.farGeometries[family]!;
    if (!g.boundingBox) g.computeBoundingBox();
    return g.boundingBox!.max.y;
  }

  /** The pool for `key`, with room for `n` instances (recreated larger if needed). */
  private pool(
    pools: Map<string, Pool>,
    key: string,
    geometry: THREE.BufferGeometry,
    n: number,
    shadows: boolean,
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
        geometry.userData.materials as THREE.Material[],
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

/**
 * Forest density (0..1): groves and woods with open meadows between them, from a large
 * field broken up by a smaller one.
 */
function forest(x: number, z: number): number {
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

function write(
  p: Pool,
  x: number,
  y: number,
  z: number,
  yaw: number,
  s: number,
): void {
  const m = p.mesh.instanceMatrix.array as Float32Array;
  const o = p.count * 16;
  const c = Math.cos(yaw) * s;
  const n = Math.sin(yaw) * s;
  m[o] = c;
  m[o + 1] = 0;
  m[o + 2] = -n;
  m[o + 3] = 0;
  m[o + 4] = 0;
  m[o + 5] = s;
  m[o + 6] = 0;
  m[o + 7] = 0;
  m[o + 8] = n;
  m[o + 9] = 0;
  m[o + 10] = c;
  m[o + 11] = 0;
  m[o + 12] = x;
  m[o + 13] = y;
  m[o + 14] = z;
  m[o + 15] = 1;
  // Per-plant tint: brightness and a slight warm/cool shift, from its position.
  const b = 0.82 + 0.3 * hash2(x, z, 41);
  const h = hash2(x, z, 42) * 2 - 1;
  const col = p.mesh.instanceColor!.array as Float32Array;
  col[p.count * 3] = b * (1 + 0.07 * h);
  col[p.count * 3 + 1] = b;
  col[p.count * 3 + 2] = b * (1 - 0.12 * h);
  p.count++;
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

/** Node material for a kit material: textured, vertex-coloured, leaves alpha-masked. */
function makeMaterial(src: THREE.MeshStandardMaterial): THREE.Material {
  const m = new THREE.MeshStandardNodeMaterial({
    name: src.name,
    roughness: src.name === "rock" ? 0.85 : 0.9,
    metalness: 0,
    vertexColors: true,
    side:
      src.alphaTest > 0 || src.transparent ? THREE.DoubleSide : THREE.FrontSide,
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
  }
  return m;
}
