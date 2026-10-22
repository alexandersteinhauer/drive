import * as THREE from "three/webgpu";
import {
  abs,
  atan,
  attribute,
  cross,
  cameraPosition,
  cameraWorldMatrix,
  positionGeometry,
  transformNormalToView,
  dFdx,
  dFdy,
  exp,
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
  vec4,
} from "three/tsl";
import type { Node } from "three/webgpu";
import { MatteNodeMaterial } from "./matte";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { hash2 } from "./world/noise";
import { STRIDE } from "./world/plants";

/**
 * Trees and bushes generated with EZ-Tree (MIT), rocks from Quaternius'
 * Stylized Nature MegaKit (CC0); see public/models/CREDITS.md.
 *
 * Terrain tiles scatter items into packed records (x, y, z, yaw, scale, kind). Near the car
 * every kind is drawn from a shared instanced mesh per kind and level of detail, refilled as
 * the car moves; things that end (rocks) grow in and out at the edge
 * of their range instead of popping. None of it casts into the shadow map: the trees'
 * shadows come from the SunField (sunField.ts). Distant trees are camera-facing billboards
 * from a multi-view atlas rendered at load, pooled per 512 m block so off-screen blocks are
 * culled.
 */

/**
 * Trees and bushes are full 3D up close and billboards beyond, cross-faded with a dither
 * between these distances (m); billboards fade out again towards the planting limit.
 */
const NEAR_START = 40;
const NEAR_END = 60;
const HORIZON_START = 1750;
const HORIZON_END = 2000;
const TILE_HALF = 64;
/** Pools hold at least this many instances: smaller ones would bake their matrices into the
 *  shader (a compile per size). */
const MIN_CAPACITY = 1100;
/** Near pools are made this large at load (a dense wood holds ~250 of a kind in range). */
const NEAR_CAPACITY = 2048;
/**
 * Floats per billboard instance: quad base x, y, z and height | atlas row, yaw, width |
 * tint rgb | crown centre x, y, z and horizontal radius | vertical radius.
 */
const FAR_STRIDE = 15;
/** Billboard blocks rebuilt per frame at most. */
const BLOCKS_PER_FRAME = 2;
/** Refill near pools after the focus moves this far (m). */
const REFILL = 10;

interface Kind {
  model: string;
  /** Far end of each detail level (m); the last level ends the kind (or hands over to far). */
  ranges: number[];
  /** Far family (index into FAR), for trees. */
  family?: number;
}

const tree = (model: string, family: number): Kind => ({
  model,
  ranges: [NEAR_END],
  family,
});
const KINDS: Kind[] = [
  ...["Oak_1", "Oak_2", "Oak_3"].map((m) => tree(m, 0)),
  ...["Ash_1", "Ash_2"].map((m) => tree(m, 1)),
  ...["Pine_1", "Pine_2", "Pine_3"].map((m) => tree(m, 2)),
  ...["Bush_1", "Bush_2", "Bush_3"].map((m) => tree(m, 3)),
  { model: "Rock_Medium_1", ranges: [80, 320] },
  { model: "Rock_Medium_2", ranges: [80, 320] },
  { model: "Rock_Medium_3", ranges: [80, 320] },
];
/** Farthest any kind reaches. */
const MAX_RANGE = Math.max(...KINDS.map((k) => k.ranges.at(-1)!));
/** Far stand-in model per family, and how its height compares to each kind's (by index). */
/** Billboard atlas: one cell (px) per tree kind, in a grid of columns x rows. */
/**
 * Billboard atlas: each tree kind (a row) seen from VIEWS directions around it (columns),
 * CELL px square each.
 */
const CELL = 384;
/** Billboard brightness relative to the captured (unshaded) crown colour. */
const BILLBOARD_SHADE = 0.5;
/** How quickly sunlight fades through foliage (per metre), and the skylight left at a
 *  crown's centre. */
const CROWN_DENSITY = 0.3;
const CROWN_AMBIENT = 0.5;
const VIEWS = 8;

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

/** A block's billboards (see `fillBlock`). */
interface FarBlock {
  mesh: THREE.Mesh<THREE.InstancedBufferGeometry>;
  capacity: number;
}

/** One tile's billboard instances, and its extent. */
interface TileBillboards {
  data: Float32Array;
  count: number;
  minY: number;
  maxY: number;
  min: THREE.Vector2;
  max: THREE.Vector2;
}

interface Pool {
  mesh: THREE.InstancedMesh;
  count: number;
}

interface TileItems {
  cx: number;
  cz: number;
  block: string;
  /** Packed records. */
  records: Float32Array;
}

export class Vegetation {
  readonly group = new THREE.Group();
  private readonly tiles = new Map<string, TileItems>();
  /** Geometry per kind and level, and the far stand-ins; null until loaded. */
  private geometries: THREE.BufferGeometry[][] | null = null;
  /** Per kind with leaves: crown centre height, horizontal and vertical radius (scale 1). */
  private crowns: ([number, number, number] | null)[] = [];
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
  private readonly far = new Map<string, FarBlock>();
  /** Billboard instance data per tile (FAR_STRIDE floats each), built once per tile. */
  private readonly billboardsOf = new WeakMap<TileItems, TileBillboards>();
  private readonly dirtyBlocks = new Set<string>();
  private nearDirty = true;
  private readonly lastFill = new THREE.Vector3(Infinity, 0, Infinity);

  /** Resolves once the models are in and every pool exists (see `load`). */
  readonly ready: Promise<void>;

  constructor(
    url: string,
    private readonly renderer: THREE.WebGPURenderer,
  ) {
    this.ready = this.load(url);
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
    this.crowns = this.geometries.map((levels) => crownOf(levels[0]!));
    // Every near pool up front, at a capacity no drive outgrows: an InstancedMesh compiles
    // its own shaders (~10 ms), so none may be created mid-drive.
    this.geometries.forEach((levels, k) =>
      levels.forEach((geometry, lod) => {
        const p = this.pool(
          this.near,
          String(k * 3 + lod),
          geometry,
          NEAR_CAPACITY,
        );
        if (this.crowns[k]) crownAttributes(p);
      }),
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
    const t: TileItems = { cx, cz, block, records };
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

  /** Camera position, for the near/far cross-fade. */
  setViewer(position: THREE.Vector3): void {
    viewer.value.copy(position);
  }

  /** Sun (or moon) direction and light, for light shining through the leaves. */
  setSun(light: THREE.DirectionalLight): void {
    // Towards the light: it follows the car (shadow box), so not its position alone.
    sunDirection.value
      .copy(light.position)
      .sub(light.target.position)
      .normalize();
    sunLight.value.copy(light.color).multiplyScalar(light.intensity);
  }

  /**
   * Refills the near pools for the focus position (every REFILL m), and rebuilds the
   * billboard blocks whose tiles changed (a few per frame: each a copy and an upload).
   */
  update(focus: THREE.Vector3, blocks = BLOCKS_PER_FRAME): void {
    if (!this.geometries) return;
    const moved = Math.hypot(
      focus.x - this.lastFill.x,
      focus.z - this.lastFill.z,
    );
    if (moved > REFILL || this.nearDirty) {
      this.lastFill.copy(focus);
      this.nearDirty = false;
      this.fillNear(focus);
    }
    let n = 0;
    for (const block of this.dirtyBlocks) {
      if (n++ >= blocks) break;
      this.dirtyBlocks.delete(block);
      this.fillBlock(block);
    }
  }

  /** Brings everything up to date at once (start-up). */
  flush(focus: THREE.Vector3): void {
    this.update(focus, Infinity);
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
        for (let i = 0; i < r.length; i += STRIDE) {
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
      );
    });
    // Nearest first: the GPU then rejects most hidden leaves by depth before shading them
    // (alpha-tested leaves can't be culled after the fact), several times fewer in woods.
    const order: [number, number, Float32Array, number, number][] = [];
    visit((i, r, k, lod, s) => {
      const ex = r[i]! - focus.x;
      const ez = r[i + 2]! - focus.z;
      order.push([ex * ex + ez * ez, i, r, k * levels + lod, s]);
    });
    order.sort((a, b) => a[0] - b[0]);
    for (const [key, p] of pools.entries())
      if (p && this.crowns[Math.floor(key / levels)]) crownAttributes(p);
    for (const [, i, r, key, s] of order) {
      const p = pools[key]!;
      const crown = this.crowns[Math.floor(key / levels)];
      if (crown) writeCrown(p, crown, r[i]!, r[i + 1]!, r[i + 2]!, s);
      write(p, r[i]!, r[i + 1]!, r[i + 2]!, r[i + 3]!, s);
    }
    for (const pool of this.near.values()) finish(pool);
  }

  /**
   * A block's billboards: every tree of its tiles, near ones included (the shader drops
   * those the 3D trees stand for), so a block only changes when its tiles do. One plain
   * mesh per block over instanced geometry: unlike an InstancedMesh, it shares the
   * billboard shader with every other block.
   */
  private fillBlock(block: string): void {
    const tiles = [...(this.byBlock.get(block) ?? [])].map((t) =>
      this.tileBillboards(t),
    );
    const count = tiles.reduce((n, t) => n + t.count, 0);
    let b = this.far.get(block);
    if (count === 0) {
      if (b) {
        this.group.remove(b.mesh);
        b.mesh.geometry.dispose();
        this.far.delete(block);
      }
      return;
    }
    if (!b || b.capacity < count) {
      let capacity = 1024;
      while (capacity < count) capacity *= 2;
      const geo = new THREE.InstancedBufferGeometry();
      geo.index = this.farGeometry!.index;
      for (const name of ["position", "uv"])
        geo.setAttribute(name, this.farGeometry!.getAttribute(name));
      const buf = new THREE.InstancedInterleavedBuffer(
        new Float32Array(capacity * FAR_STRIDE),
        FAR_STRIDE,
      );
      geo.setAttribute("inst", new THREE.InterleavedBufferAttribute(buf, 4, 0));
      geo.setAttribute("cell", new THREE.InterleavedBufferAttribute(buf, 3, 4));
      geo.setAttribute("tint", new THREE.InterleavedBufferAttribute(buf, 3, 7));
      geo.setAttribute(
        "crownA",
        new THREE.InterleavedBufferAttribute(buf, 4, 10),
      );
      geo.setAttribute(
        "crownB",
        new THREE.InterleavedBufferAttribute(buf, 1, 14),
      );
      if (b) {
        b.mesh.geometry.dispose();
        b.mesh.geometry = geo;
      } else {
        const mesh = new THREE.Mesh(geo, this.farMaterial!);
        mesh.matrixAutoUpdate = false;
        this.group.add(mesh);
        b = { mesh, capacity };
        this.far.set(block, b);
      }
      b.capacity = capacity;
    }
    const geo = b.mesh.geometry;
    const buf = (geo.getAttribute("inst") as THREE.InterleavedBufferAttribute)
      .data as THREE.InstancedInterleavedBuffer;
    const array = buf.array as Float32Array;
    let at = 0;
    let lo = Infinity;
    let hi = -Infinity;
    const from = new THREE.Vector2(Infinity, Infinity);
    const to = new THREE.Vector2(-Infinity, -Infinity);
    for (const t of tiles) {
      array.set(t.data, at);
      at += t.data.length;
      lo = Math.min(lo, t.minY);
      hi = Math.max(hi, t.maxY);
      from.min(t.min);
      to.max(t.max);
    }
    buf.clearUpdateRanges();
    buf.addUpdateRange(0, at);
    buf.needsUpdate = true;
    geo.instanceCount = count;
    const centre = new THREE.Vector3(
      (from.x + to.x) / 2,
      (lo + hi) / 2,
      (from.y + to.y) / 2,
    );
    geo.boundingSphere = new THREE.Sphere(
      centre,
      Math.hypot(to.x - from.x, hi - lo, to.y - from.y) / 2,
    );
  }

  /** A tile's billboard instances (built once, on first use, from its records). */
  private tileBillboards(t: TileItems): TileBillboards {
    let b = this.billboardsOf.get(t);
    if (b) return b;
    const r = t.records;
    let count = 0;
    for (let i = 0; i < r.length; i += STRIDE)
      if (KINDS[r[i + 5]!]!.family !== undefined) count++;
    const data = new Float32Array(count * FAR_STRIDE);
    let minY = Infinity;
    let maxY = -Infinity;
    let o = 0;
    for (let i = 0; i < r.length; i += STRIDE) {
      const kind = r[i + 5]!;
      if (KINDS[kind]!.family === undefined) continue;
      const box = this.farBoxes[kind]!;
      const crown = this.crowns[kind]!;
      const s = r[i + 4]!;
      const x = r[i]!;
      const y = r[i + 1]!;
      const z = r[i + 2]!;
      // Per-plant tint, as the near trees' (see `tint`).
      const bright = 0.82 + 0.3 * hash2(x, z, 41);
      const hue = hash2(x, z, 42) * 2 - 1;
      data.set(
        [
          x,
          y + box.bottom * s,
          z,
          (box.top - box.bottom) * s,
          this.cellOf[kind]!,
          r[i + 3]!,
          box.width * s,
          bright * (1 + 0.07 * hue),
          bright,
          bright * (1 - 0.12 * hue),
          x,
          y + crown[0] * s,
          z,
          crown[1] * s,
          crown[2] * s,
        ],
        o,
      );
      o += FAR_STRIDE;
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y + box.top * s);
    }
    b = {
      data,
      count,
      minY,
      maxY,
      min: new THREE.Vector2(t.cx - TILE_HALF, t.cz - TILE_HALF),
      max: new THREE.Vector2(t.cx + TILE_HALF, t.cz + TILE_HALF),
    };
    this.billboardsOf.set(t, b);
    return b;
  }

  /**
   * Distant stand-ins for trees: every tree kind rendered once from VIEWS sides (unlit colour
   * and alpha) into one atlas, shown on a quad that faces the camera, with the view matching
   * the direction it's seen from (so the outline doesn't change at the hand-over to 3D), lit
   * with crown normals like the near leaves. A block takes one draw.
   */
  private billboards(): void {
    const kinds = KINDS.flatMap((k, i) => (k.family === undefined ? [] : [i]));
    const target = new THREE.RenderTarget(CELL * VIEWS, CELL * kinds.length, {
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
    });
    const r = this.renderer;
    const prevTarget = r.getRenderTarget();
    const prevColor = r.getClearColor(new THREE.Color());
    const prevAlpha = r.getClearAlpha();
    const prevAuto = r.autoClear;
    // Transparent black (the backend premultiplies the clear colour anyway): filtered and
    // mipmapped texels are then colour times coverage, which the billboard shader divides
    // back out (see `colour`).
    r.setClearColor(new THREE.Color(0, 0, 0), 0);
    r.setRenderTarget(target);
    r.clear();
    r.autoClear = false;
    this.farBoxes = KINDS.map(() => ({ width: 1, bottom: 0, top: 1 }));
    kinds.forEach((kind, cell) => {
      const tree = this.geometries![kind]![0]!;
      const box = tree.boundingBox!;
      // Wide enough for the crown from any side: twice its farthest reach from the trunk.
      const at = tree.getAttribute("position");
      let reach = 0;
      for (let i = 0; i < at.count; i++)
        reach = Math.max(reach, Math.hypot(at.getX(i), at.getZ(i)));
      const w = reach * 2 * 1.02;
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
      // View k looks at the tree from angle 2 pi k / VIEWS (from +Z towards +X).
      for (let k = 0; k < VIEWS; k++) {
        const a = (k / VIEWS) * Math.PI * 2;
        camera.position.set(Math.sin(a) * 200, 0, Math.cos(a) * 200);
        camera.lookAt(0, 0, 0);
        target.viewport.set(k * CELL, cell * CELL, CELL, CELL);
        r.render(capture, camera);
      }
      for (const m of flat) m.dispose();
    });
    target.viewport.set(0, 0, target.width, target.height);
    r.autoClear = prevAuto;
    r.setRenderTarget(prevTarget);
    r.setClearColor(prevColor, prevAlpha);
    this.cellOf = KINDS.map((_, i) => kinds.indexOf(i));
    this.atlas = target;

    // One unit quad (x in -0.5..0.5, y in 0..1), turned to the camera in the shader.
    const geo = new THREE.PlaneGeometry(1, 1).translate(0, 0.5, 0);
    geo.addGroup(0, geo.index!.count, 0);
    this.farGeometry = geo;

    const m = new MatteNodeMaterial({
      roughness: 0.9,
      metalness: 0,
      side: THREE.DoubleSide,
    });
    // Per instance: quad base and height; atlas row (kind), yaw and width; tint.
    const inst = attribute<"vec4">("inst", "vec4");
    const cell = attribute<"vec3">("cell", "vec3");
    const tintOf = attribute<"vec3">("tint", "vec3");
    // The quad faces the camera (turning about the vertical only): its right-hand side is
    // perpendicular to the camera's backward axis.
    const back = cameraWorldMatrix.mul(vec4(0, 0, 1, 0)).xyz;
    const back2 = vec2(back.x, back.z).normalize();
    const right = vec3(back2.y, 0, back2.x.negate());
    // The quad (x -0.5..0.5, y 0..1) stood on its base, its sides along `right`. Quads
    // wholly within NEAR_START of the viewer collapse to a point: the 3D trees stand for
    // them, and they'd only be discarded fragment by fragment.
    const size = vec2(cell.z, inst.w);
    const centre = inst.xyz.add(vec3(0, inst.w.mul(0.5), 0));
    const near = centre
      .distance(viewer)
      .add(size.length().mul(0.5))
      .lessThan(NEAR_START);
    const scale = near.select(float(0), float(1));
    m.positionNode = inst.xyz.add(
      right
        .mul(positionGeometry.x.mul(cell.z))
        .add(vec3(0, positionGeometry.y.mul(inst.w), 0))
        .mul(scale),
    );
    // The captured view nearest the direction the tree is seen from, two neighbours mixed
    // with a dither in between, so the outline matches the 3D tree's from any side.
    const toCamera = cameraPosition.xz.sub(positionWorld.xz);
    const views = atan(toCamera.x, toCamera.y)
      .sub(cell.y)
      .div(Math.PI * 2)
      .fract()
      .mul(VIEWS);
    const first = views.floor();
    const pick = interleavedGradientNoise(
      screenCoordinate.xy.add(vec2(17.3, 41.9)),
    )
      .lessThan(views.fract())
      .select(first.add(1).mod(VIEWS), first);
    // Render targets are stored top row first and sampled unflipped in WebGPU: the cell's top
    // (the treetop) is at its smallest v.
    const inCell = vec2(uv().x, uv().y.oneMinus()).div(
      vec2(VIEWS, kinds.length),
    );
    const cellUv = vec2(pick, cell.x)
      .div(vec2(VIEWS, kinds.length))
      .add(inCell);
    // Gradients from the continuous coordinates: the dithered view jumps between
    // neighbouring pixels would otherwise pick the smallest mip level.
    const tex = texture(target.texture, cellUv).grad(
      dFdx(inCell),
      dFdy(inCell),
    );
    // The crown's own colour up to its outline: texels the filter or the mipmaps mix with
    // the transparent background are darkened by its black, by their coverage (alpha).
    // Kept fragments have alpha above ~0.13 (see maskNode), so this stays well-conditioned.
    const colour = tex.rgb.div(max(tex.a, 1 / 64));
    // Darker than the capture: a 3D crown shades itself (its inner and far-side leaves are
    // in shadow), which a flat card can't; matched to the 3D trees at the hand-over.
    m.colorNode = colour.mul(BILLBOARD_SHADE).mul(tintOf);
    const t = uv().mul(CELL);
    const lod = log2(max(max(dFdx(t).length(), dFdy(t).length()), 1));
    m.maskNode = tex.a.mul(lod.mul(0.3).add(1)).greaterThan(0.45).and(showFar);
    // Crown normals: out from the centre, a little up, and towards the camera, like the
    // near trees' leaves.
    const crown = right
      .mul(uv().x.sub(0.5).mul(2))
      .add(vec3(0, uv().y.sub(0.5).mul(2).add(0.3), 0))
      .add(vec3(back2.x, 0, back2.y).mul(0.9))
      .normalize();
    m.normalNode = transformNormalToView(crown);
    // The same crown shading as the 3D trees, at the visible (camera-facing) surface of
    // the crown's ellipsoid behind this point of the card.
    const radii = vec3(crownA.w, crownB, crownA.w);
    const off = positionWorld.sub(crownA.xyz);
    const lateral = off.dot(right).div(crownA.w);
    const height = off.y.div(crownB);
    const depth = float(1)
      .sub(lateral.mul(lateral))
      .sub(height.mul(height))
      .max(0)
      .sqrt();
    const front = right
      .mul(off.dot(right))
      .add(vec3(0, off.y, 0))
      .add(vec3(back2.x, 0, back2.y).mul(depth.mul(crownA.w)));
    const sun = throughCrown(front, radii);
    (m as unknown as { sunFieldPosition: Node<"vec3"> }).sunFieldPosition =
      crownA.xyz.add(front);
    (
      m as unknown as { receivedShadowNode: (s: Node<"float">) => Node }
    ).receivedShadowNode = (shadow) => shadow.mul(sun);
    m.emissiveNode = translucency(colour.mul(BILLBOARD_SHADE));
    this.farMaterial = m;
  }

  /**
   * The near pool for `key`, with room for `n` instances (recreated larger if needed,
   * which costs a shader compile: see NEAR_CAPACITY).
   */
  private pool(
    pools: Map<string, Pool>,
    key: string,
    geometry: THREE.BufferGeometry,
    n: number,
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
      // No vegetation casts into the shadow map (it holds only the car): trees' shadows
      // come from the SunField, which plants the same crowns (world/plants.ts).
      mesh.receiveShadow = true;
      // Refilled around the focus; always partly in view.
      mesh.frustumCulled = false;
      this.group.add(mesh);
      p = { mesh, count: 0 };
      pools.set(key, p);
    }
    p.count = 0;
    return p;
  }
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

/** A kind's crown, from its leaves: centre height, horizontal and vertical radius. */
function crownOf(geo: THREE.BufferGeometry): [number, number, number] | null {
  const mats = geo.userData.materials as THREE.Material[];
  const pos = geo.getAttribute("position");
  const index = geo.index!;
  let lo = Infinity;
  let hi = -Infinity;
  const radii: number[] = [];
  for (const g of geo.groups) {
    if (!mats[g.materialIndex ?? 0]?.name.startsWith("leaves_")) continue;
    for (let i = g.start; i < g.start + g.count; i++) {
      const v = index.getX(i);
      const y = pos.getY(v);
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
      radii.push(Math.hypot(pos.getX(v), pos.getZ(v)));
    }
  }
  if (!radii.length) return null;
  radii.sort((a, b) => a - b);
  return [(lo + hi) / 2, radii[Math.floor(radii.length * 0.9)]!, (hi - lo) / 2];
}

/** Gives a pool's own geometry the per-instance crown attributes, with room for all. */
function crownAttributes(p: Pool): void {
  const n = p.mesh.instanceMatrix.count;
  const a = p.mesh.geometry.getAttribute("crownA") as
    THREE.InstancedBufferAttribute | undefined;
  if (a && a.count >= n) return;
  // Near pools start out on the kind's shared geometry: an own copy for own attributes.
  if (!a) p.mesh.geometry = p.mesh.geometry.clone();
  p.mesh.geometry.setAttribute(
    "crownA",
    new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4),
  );
  p.mesh.geometry.setAttribute(
    "crownB",
    new THREE.InstancedBufferAttribute(new Float32Array(n), 1),
  );
}

/** The next instance's crown: centre and radii in the world. */
function writeCrown(
  p: Pool,
  crown: [number, number, number],
  x: number,
  y: number,
  z: number,
  s: number,
): void {
  const a = p.mesh.geometry.getAttribute("crownA")!.array as Float32Array;
  const b = p.mesh.geometry.getAttribute("crownB")!.array as Float32Array;
  a.set([x, y + crown[0] * s, z, crown[1] * s], p.count * 4);
  b[p.count] = crown[2] * s;
}

function finish(p: Pool): void {
  const used = p.count;
  p.mesh.count = used;
  // Only the instances in use go to the GPU (pools are sized for the densest woods);
  // nothing at all for an empty pool.
  if (used === 0) return;
  const upload = (a: THREE.BufferAttribute | undefined) => {
    if (!a) return;
    a.clearUpdateRanges();
    a.addUpdateRange(0, used * a.itemSize);
    a.needsUpdate = true;
  };
  upload(p.mesh.instanceMatrix);
  upload(p.mesh.instanceColor!);
  const geo = p.mesh.geometry;
  upload(geo.getAttribute("crownA") as THREE.BufferAttribute | undefined);
  upload(geo.getAttribute("crownB") as THREE.BufferAttribute | undefined);
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

/** Camera position, for the near/far cross-fade. */
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

/** Per near instance: crown centre (world) and horizontal radius, and vertical radius. */
const crownA = attribute<"vec4">("crownA", "vec4");
const crownB = attribute<"float">("crownB", "float");

/**
 * Sunlight left after passing through a crown, from a point at `q` relative to its centre:
 * the leaves are a soft volume (an ellipsoid of radii `r`), and light falls off with the
 * length of foliage it crosses on its way (Beer-Lambert). Sunny sides stay bright, the
 * interior and far side go dark, as a shadow map did, at the cost of a little arithmetic.
 * 3D trees and their billboards use the same crowns, so the hand-over stays seamless.
 */
function throughCrown(q: Node<"vec3">, r: Node<"vec3">): Node<"float"> {
  const u = q.div(r);
  const d = sunDirection.div(r);
  const a = d.dot(d);
  const b = u.dot(d);
  const c = u.dot(u).sub(1);
  const root = b.mul(b).sub(a.mul(c)).max(0).sqrt();
  const exit = root.sub(b).div(a);
  const entry = root.negate().sub(b).div(a);
  const path = exit.sub(entry.max(0)).max(0);
  return exp(path.mul(-CROWN_DENSITY));
}

/** Skylight inside a crown: less towards its centre (`depth` 0 at the centre, 1 at the edge). */
function crownAmbient(depth: Node<"float">): Node<"float"> {
  return mix(float(CROWN_AMBIENT), float(1), smoothstep(0.15, 1, depth));
}

/** Node material for a kit material: textured, vertex-coloured, leaves alpha-masked. */
function makeMaterial(src: THREE.MeshStandardMaterial): THREE.Material {
  const m = new MatteNodeMaterial({
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
  // Generated leaves are named per clump size (leaves_oak_x16): tinted by type.
  const tint = new THREE.Color(
    TINTS[src.name.replace(/_x\d+$/, "")] ?? 0xffffff,
  );
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
    let alpha = tex.a.mul(lod.mul(0.35).add(1));
    // Generated leaves are a few large clump cards: seen edge-on, one reads as a long
    // streak, so cards dissolve as they turn edge-on (face normal from the derivatives).
    if (src.name.startsWith("leaves_")) {
      const face = cross(dFdx(positionWorld), dFdy(positionWorld)).normalize();
      const facing = abs(
        face.dot(positionWorld.sub(cameraPosition).normalize()),
      );
      alpha = alpha.mul(smoothstep(0.08, 0.35, facing));
    }
    m.maskNode = alpha.greaterThan(0.4);
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
    // The crown shades itself: less sun the deeper in (towards the sun's far side), and
    // less skylight towards its centre.
    const r = vec3(crownA.w, crownB, crownA.w);
    const q = positionWorld.sub(crownA.xyz);
    const sun = throughCrown(q, r);
    (
      m as unknown as { receivedShadowNode: (s: Node<"float">) => Node }
    ).receivedShadowNode = (shadow) => shadow.mul(sun);
    m.aoNode = crownAmbient(q.div(r).length());
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
