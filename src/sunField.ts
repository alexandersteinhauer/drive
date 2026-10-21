import * as THREE from "three/webgpu";
import {
  cameraPosition,
  Fn,
  If,
  float,
  instanceIndex,
  int,
  ivec2,
  max,
  mix,
  normalWorld,
  positionWorld,
  renderGroup,
  select,
  smoothstep,
  storage,
  texture,
  textureStore,
  uniform,
  vec2,
  vec4,
} from "three/tsl";
import type { WorldWorkers } from "./worldWorkers";

type F = THREE.Node<"float">;
type I = THREE.Node<"int">;
type V2 = THREE.Node<"vec2">;
type V3 = THREE.Node<"vec3">;

const buffer = (attr: THREE.StorageBufferAttribute) =>
  storage(attr, "float", N * N);
type Buf = ReturnType<typeof buffer>;

/** Texels across the field, its tiles and the texel size (m): 4 km around the focus. */
const N = 2048;
const TS = 32;
const NT = N / TS;
const T = 2;
/** Padding beyond the field's own window edge where its shadows fade out (m). */
const EDGE_FADE = 96;
/**
 * Doubling passes: the k-th looks 2^(k-1) texels further towards the sun, so K passes
 * reach 2^(K-1) texels (~2 km); fewer are needed with the sun high (see `passesFor`).
 */
const MAX_PASSES = 11;
/** Sun flecks under the trees: noise above FLECKS lets the sun through (~a tenth of the
 *  ground), at FLECK_LIGHT strength; FLECK_SCALE is the noise tile's size (m). */
const FLECKS = 0.76;
const FLECK_LIGHT = 0.8;
const FLECK_SCALE = 10;
/** World size of the edge noise's tile (m), and how far it moves the shadow line across
 *  the ground (m). */
const NOISE_SCALE = 6;
const EDGE_NOISE = 0.6;
/** Highest relief that can cast a shadow (m): bounds how far towards the sun to look. */
const RELIEF = 180;

/** Tiles per worker job, and jobs in flight while driving. */
const BATCH = 32;
const IN_FLIGHT = 2;

/**
 * Sun shadows of the land, and the shade of the woods, as world-space data: the same point
 * always gets the same answer, wherever the camera is, so nothing pops or slides as it
 * moves (unlike a shadow map's cascades).
 *
 * Heights on a 2 m grid, 4 km across, follow the focus a 64 m tile at a time (generated in
 * the world workers; storage wraps around, so nothing is moved). Tree crowns stand on them
 * as solid shapes (rounded, or cones for pines), placed exactly as the trees are planted
 * (world/plants.ts), so the trees cast shadows too, near and far alike, and shade each
 * other and themselves (a crown's far side is in its own shadow).
 * The GPU turns them into a "shadow height" per texel, the height below which a point
 * there is in the sun's shadow: max over t of h(p + t·d) - t·tan(elevation), towards the
 * sun along d. Rather than marching every texel, K passes double the reach each time
 * (H_k(p) = max(H_k-1(p), H_k-1(p + s·d) - s·tan e), s = 2^(k-1) texels), spread over a few
 * frames and published together. Anything lit by the sun compares its own height with it:
 * the ground, grass, trees (tops catch the sun while the valley floor is in shade), the
 * road and the car.
 */
export class SunField {
  private readonly heights = new Float32Array(N * N);
  private readonly tops = new Float32Array(N * N);
  private readonly heightAttr = new THREE.StorageBufferAttribute(
    this.heights,
    1,
  );
  private readonly topAttr = new THREE.StorageBufferAttribute(this.tops, 1);
  /** Shadow heights, ping-ponged through the passes: with the trees (a, b) and of the
   *  land alone (c, d), which tells the trees' shade from the hills'. */
  private readonly a = new THREE.StorageBufferAttribute(N * N, 1);
  private readonly b = new THREE.StorageBufferAttribute(N * N, 1);
  private readonly c = new THREE.StorageBufferAttribute(N * N, 1);
  private readonly d = new THREE.StorageBufferAttribute(N * N, 1);
  /** What materials sample: shadow height, ground height, shadow height of the land
   *  alone, and how high a crown stands over the point. */
  private readonly out = new THREE.StorageTexture(N, N);
  /** Tileable value noise for the shadows' edges. */
  private readonly noise = valueNoise(64);

  /** Window origin (world texels, tile-aligned) the passes work on and was published. */
  private readonly origin = uniform(new THREE.Vector2());
  /** Read by materials: in the shared group, see `shadowNode`. */
  private readonly published = uniform(new THREE.Vector2()).setGroup(
    renderGroup,
  );
  /** tan(sun elevation), for materials (shared group too). */
  private readonly slope = uniform(0.4).setGroup(renderGroup);
  private readonly dir = uniform(new THREE.Vector2(1, 0));
  /** Height the line towards the sun climbs per texel along it. */
  private readonly rise = uniform(0.4);
  private readonly shift = uniform(1);

  private readonly pass0: THREE.ComputeNode;
  private readonly passAB: THREE.ComputeNode;
  private readonly passBA: THREE.ComputeNode;
  private readonly finalA: THREE.ComputeNode;
  private readonly finalB: THREE.ComputeNode;

  /** Tile coordinates of the window's first tile (the wanted one, and the filled one). */
  private wantX = NaN;
  private wantZ = NaN;
  private readonly filled = new Map<number, string>();
  private queue: [number, number][] = [];
  /** Tiles being generated (slot -> key), and finished batches to copy in. */
  private readonly building = new Map<number, string>();
  private readonly finished: {
    tiles: number[];
    heights: Float32Array;
    tops: Float32Array;
  }[] = [];
  private batches = 0;
  private dirty: number[] = [];
  /** Recompute in progress: passes left, and which buffer holds the last one. */
  private pass = -1;
  private passes = 0;
  private inA = true;
  private stale = true;
  private lightKey = "";

  constructor(
    private readonly renderer: THREE.WebGPURenderer,
    private readonly workers: WorldWorkers,
  ) {
    this.out.type = THREE.HalfFloatType;
    this.out.wrapS = this.out.wrapT = THREE.RepeatWrapping;
    this.out.minFilter = this.out.magFilter = THREE.LinearFilter;
    this.out.generateMipmaps = false;

    const heights = buffer(this.heightAttr).toReadOnly() as Buf;
    const tops = buffer(this.topAttr).toReadOnly() as Buf;
    const a = buffer(this.a);
    const b = buffer(this.b);
    const c = buffer(this.c);
    const d = buffer(this.d);
    const origin = this.origin;

    /** World texel of this invocation's storage index. */
    const texel = (): [I, I, I, I] => {
      const i = int(instanceIndex);
      const tile = i.shiftRight(10);
      const local = i.bitAnd(TS * TS - 1);
      const wx = tile
        .bitAnd(NT - 1)
        .mul(TS)
        .add(local.bitAnd(TS - 1));
      const wz = tile.shiftRight(6).mul(TS).add(local.shiftRight(5));
      const ox = int(origin.x);
      const oz = int(origin.y);
      return [
        ox.add(wx.sub(ox).bitAnd(N - 1)) as I,
        oz.add(wz.sub(oz).bitAnd(N - 1)) as I,
        wx as I,
        wz as I,
      ];
    };
    /** Bilinear lookup at a world texel position; -infinity outside the window. */
    const lookup = (src: Buf, fx: F, fz: F): F => {
      const x0 = int(fx.floor());
      const z0 = int(fz.floor());
      const tx = fx.fract();
      const tz = fz.fract();
      const at = (dx: number, dz: number): F => {
        const x = x0.add(dx);
        const z = z0.add(dz);
        const inside = x
          .greaterThanEqual(int(origin.x))
          .and(x.lessThan(int(origin.x).add(N)))
          .and(z.greaterThanEqual(int(origin.y)))
          .and(z.lessThan(int(origin.y).add(N)));
        return select(
          inside,
          src.element(index(x as I, z as I)) as unknown as F,
          float(-1e4),
        ) as F;
      };
      return mix(
        mix(at(0, 0), at(1, 0), tx),
        mix(at(0, 1), at(1, 1), tx),
        tz,
      ) as F;
    };

    // Pass 0: the ground one texel towards the sun.
    this.pass0 = Fn(() => {
      const [gx, gz] = texel();
      const fx = float(gx).add(this.dir.x);
      const fz = float(gz).add(this.dir.y);
      a.element(instanceIndex).assign(lookup(tops, fx, fz).sub(this.rise));
      c.element(instanceIndex).assign(lookup(heights, fx, fz).sub(this.rise));
    })().compute(N * N, [256]);

    const double = (src: Buf, dst: Buf, srcLand: Buf, dstLand: Buf) =>
      Fn(() => {
        const [gx, gz] = texel();
        const fx = float(gx).add(this.dir.x.mul(this.shift));
        const fz = float(gz).add(this.dir.y.mul(this.shift));
        const drop = this.rise.mul(this.shift);
        dst
          .element(instanceIndex)
          .assign(
            max(src.element(instanceIndex), lookup(src, fx, fz).sub(drop)),
          );
        dstLand
          .element(instanceIndex)
          .assign(
            max(
              srcLand.element(instanceIndex),
              lookup(srcLand, fx, fz).sub(drop),
            ),
          );
      })().compute(N * N, [256]);
    this.passAB = double(a, b, c, d);
    this.passBA = double(b, a, d, c);

    const publish = (src: Buf, land: Buf) =>
      Fn(() => {
        const [, , wx, wz] = texel();
        textureStore(
          this.out,
          ivec2(wx, wz),
          vec4(
            src.element(instanceIndex),
            heights.element(instanceIndex),
            land.element(instanceIndex),
            // How high a crown stands over this point (0 in the open).
            tops.element(instanceIndex).sub(heights.element(instanceIndex)),
          ),
        ).toWriteOnly();
      })().compute(N * N, [256]);
    this.finalA = publish(a, c);
    this.finalB = publish(b, d);
  }

  /**
   * Lit fraction (0..1) of the sun at a world position, from the land's shadows. 1 beyond
   * the field. `bias` (m) absorbs the mesh being coarser than the field far away.
   */
  shadow(p: V3, distance: F, leafyEdges = true): F {
    const s = this.at(p);
    const bias = float(0.25).add(distance.mul(0.0015));
    // Leafy, ragged edges: the shadow line moves by a fixed world-space noise (so it stays
    // put as the camera moves), fading out with distance where it would only shimmer.
    // Vertical jitter that moves the line by EDGE_NOISE across the ground: scaled by the
    // sun's slope, or long low-sun shadows (whose height barely changes) turn to speckle.
    const leafy = texture(this.noise, p.xz.div(NOISE_SCALE))
      .r.sub(0.5)
      .mul(this.slope.mul(leafyEdges ? EDGE_NOISE * 2 : 0))
      .mul(float(1).sub(smoothstep(40, 160, distance)));
    // Soft edge: ~1 m across where the shadow line runs over level ground, wider with a
    // low sun, as a penumbra.
    const lit = smoothstep(-0.3, 0.3, p.y.add(bias).add(leafy).sub(s.x));
    // Shade from the land alone: solid. The rest is the trees': sun flecks come through
    // the gaps in their crowns (a fixed world-space pattern, so they stay put).
    const land = smoothstep(-0.3, 0.3, p.y.add(bias).sub(s.z));
    // Only under a crown: the long shadows woods cast onto clearings and the road stay
    // whole (their edges are leafy anyway).
    const underCrown = smoothstep(1, 3, s.w);
    const flecks = smoothstep(
      FLECKS - 0.06,
      FLECKS + 0.06,
      texture(this.noise, p.xz.div(FLECK_SCALE)).r,
    )
      .mul(underCrown)
      .mul(float(1).sub(smoothstep(60, 200, distance)));
    const trees = mix(flecks.mul(FLECK_LIGHT), float(1), lit);
    return mix(float(1), land.mul(trees), this.inside(p)) as F;
  }

  /** The sun's shadow node: the car's shadow map, times the land's and the trees' shadows. */
  shadowNode(light: THREE.DirectionalLight): THREE.ShadowNode {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const field = this;
    // The shadow box's matrix and depth bias; the matrix is updated in place by
    // Atmosphere.follow before each frame.
    // Shared (render) group: refreshed for every object each render. Per-object uniforms
    // only refresh for node materials or objects that changed, so the static road (a
    // plain MeshStandardMaterial) would keep the matrix it was first drawn with.
    const matrix = uniform(light.shadow.matrix).setGroup(renderGroup);
    const bias = uniform(light.shadow.bias);
    class SunShadowNode extends THREE.ShadowNode {
      override setup(builder: THREE.NodeBuilder) {
        // The base class renders the map (the car); its own lookup's matrix and bias sit in
        // a uniform group that never refreshed here (the car's shadow never showed), so the
        // lookup is done with our own uniforms.
        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const node = this as unknown as {
          _node: unknown;
          _currentShadowType: number;
          shadowMap: THREE.RenderTarget;
          setupShadowPosition(b: THREE.NodeBuilder): void;
          setupShadow(b: THREE.NodeBuilder): unknown;
        };
        const p = positionWorld as unknown as V3;
        return Fn(() => {
          // The base class's setup, minus its lookup: creates the map it renders.
          if (node._node === null) {
            node.setupShadowPosition(builder);
            node._node = node.setupShadow(builder);
            node._currentShadowType = builder.renderer.shadowMap.type;
          }
          const depth = node.shadowMap.depthTexture!;
          const lit = float(1).toVar();
          // Only the car is in the map: the lookup (and filtering) only inside its box.
          {
            const c = matrix.mul(
              vec4(p.add(normalWorld.mul(light.shadow.normalBias)), 1),
            );
            const ndc = c.xyz.div(c.w);
            const uv = vec2(ndc.x, ndc.y.oneMinus());
            const z = ndc.z.add(bias);
            const inside = uv.x
              .greaterThanEqual(0)
              .and(uv.x.lessThanEqual(1))
              .and(uv.y.greaterThanEqual(0))
              .and(uv.y.lessThanEqual(1))
              .and(z.lessThanEqual(1));
            If(inside, () => {
              // 2 x 2 taps, a texel apart, each a filtered comparison: soft enough edge.
              const t = float(1 / light.shadow.mapSize.x);
              const tap = (x: number, y: number) =>
                texture(depth, uv.add(vec2(t.mul(x), t.mul(y)))).compare(z);
              lit.assign(
                tap(-0.5, -0.5)
                  .add(tap(0.5, -0.5))
                  .add(tap(-0.5, 0.5))
                  .add(tap(0.5, 0.5))
                  .mul(0.25),
              );
            });
          }
          // Materials can look the land's and trees' shadows up elsewhere: billboards at the
          // crown surface they stand for, not on their flat card.
          const at =
            (builder.material as unknown as { sunFieldPosition?: V3 })
              .sunFieldPosition ?? p;
          // No leafy edges on the car (the one thing that casts into the map): a shadow line
          // crossing its smooth paint with them reads as dirt.
          const out = lit.mul(
            field.shadow(
              at,
              at.distance(cameraPosition) as F,
              builder.object.castShadow !== true,
            ),
          );
          // A material's own adjustment (as the base class applies it): the trees' crowns.
          const own = (
            builder.material as unknown as {
              receivedShadowNode?: (s: F) => F;
            }
          ).receivedShadowNode;
          return own ? own(out) : out;
        })();
      }
    }
    return new SunShadowNode(light, light.shadow);
  }

  /** Ground height under a world position. */
  ground(p: V3): F {
    return this.at(p).y as F;
  }

  private at(p: V3) {
    const uv = p.xz.div(T).add(0.5).div(N) as V2;
    return texture(this.out, uv);
  }

  /** 1 well inside the published window, fading to 0 at its edge. */
  private inside(p: V3): F {
    const lo = this.published.mul(T);
    const hi = lo.add(N * T);
    const dx = p.x.sub(lo.x).min(hi.x.sub(p.x));
    const dz = p.z.sub(lo.y).min(hi.y.sub(p.z));
    return smoothstep(T, EDGE_FADE, dx.min(dz)) as F;
  }

  /**
   * Follows `focus` and the light: fills new tiles within `budgetMs` of CPU time, then
   * recomputes the shadows over the next frames, two passes per frame.
   */
  update(
    focus: THREE.Vector3,
    light: THREE.Vector3,
    inFlight = IN_FLIGHT,
  ): void {
    const tx = Math.floor(focus.x / (T * TS)) - NT / 2;
    const tz = Math.floor(focus.z / (T * TS)) - NT / 2;
    if (tx !== this.wantX || tz !== this.wantZ) {
      this.wantX = tx;
      this.wantZ = tz;
      this.queue = [];
      for (let j = 0; j < NT; j++)
        for (let i = 0; i < NT; i++) {
          const x = tx + i;
          const z = tz + j;
          const key = `${x},${z}`;
          if (this.filled.get(slot(x, z)) === key) continue;
          if (this.building.get(slot(x, z)) === key) continue;
          this.queue.push([x, z]);
        }
      // Nearest first.
      const cx = tx + NT / 2;
      const cz = tz + NT / 2;
      this.queue.sort(
        (p, q) =>
          Math.hypot(q[0] - cx, q[1] - cz) - Math.hypot(p[0] - cx, p[1] - cz),
      );
    }
    // Copy finished tiles in (a few microseconds each).
    for (const r of this.finished.splice(0)) this.store(r);
    while (this.batches < inFlight && this.queue.length) {
      const tiles: number[] = [];
      while (tiles.length < BATCH * 2 && this.queue.length) {
        const [x, z] = this.queue.pop()!;
        this.building.set(slot(x, z), `${x},${z}`);
        tiles.push(x, z);
      }
      this.batches++;
      void this.workers
        .run({ kind: "sun", tiles, size: TS, texel: T })
        .then(({ heights, tops }) => {
          this.batches--;
          this.finished.push({ tiles, heights, tops });
        });
    }

    const key = `${light.x.toFixed(4)},${light.y.toFixed(4)},${light.z.toFixed(4)}`;
    const moved = key !== this.lightKey;
    this.lightKey = key;
    const ready =
      this.batches === 0 && this.queue.length === 0 && !this.finished.length;
    if (moved && ready) {
      // The light moved (the time is being changed): all passes at once (~1 ms), so the
      // shadows follow the sun smoothly instead of freezing until it stops.
      this.begin(light);
      this.step(MAX_PASSES + 1);
      this.stale = false;
      return;
    }
    if (moved) this.stale = true;
    // A new window: start over once every tile is in, a couple of passes per frame.
    if (this.stale && ready) {
      this.stale = false;
      this.begin(light);
    }
    this.step(2);
  }

  /** Fills everything and computes the shadows at once (start-up). */
  async prime(focus: THREE.Vector3, light: THREE.Vector3): Promise<void> {
    this.update(focus, light, Infinity);
    while (this.batches)
      await new Promise<void>((resolve) => setTimeout(resolve, 10));
    this.update(focus, light, 0);
    this.step(MAX_PASSES + 1);
    // The GPU buffers were created from the whole arrays just now; three doesn't clear the
    // ranges on creation, and they'd all go out (32 MB) with the first new tiles.
    this.heightAttr.clearUpdateRanges();
    this.topAttr.clearUpdateRanges();
  }

  /** A generated batch into storage, for the next recompute. */
  private store({
    tiles,
    heights,
    tops,
  }: {
    tiles: number[];
    heights: Float32Array;
    tops: Float32Array;
  }): void {
    const n = TS * TS;
    for (let t = 0; t < tiles.length / 2; t++) {
      const x = tiles[t * 2]!;
      const z = tiles[t * 2 + 1]!;
      const s = slot(x, z);
      const key = `${x},${z}`;
      if (this.building.get(s) === key) this.building.delete(s);
      // The window moved on and this slot now belongs to another tile.
      const ox = this.wantX;
      const oz = this.wantZ;
      if (x < ox || z < oz || x >= ox + NT || z >= oz + NT) continue;
      const base = s * n;
      this.heights.set(heights.subarray(t * n, t * n + n), base);
      this.tops.set(tops.subarray(t * n, t * n + n), base);
      this.filled.set(s, key);
      this.dirty.push(base);
      this.stale = true;
    }
  }

  private begin(light: THREE.Vector3): void {
    for (const base of this.dirty) {
      this.heightAttr.addUpdateRange(base, TS * TS);
      this.topAttr.addUpdateRange(base, TS * TS);
    }
    if (this.dirty.length) {
      this.heightAttr.needsUpdate = true;
      this.topAttr.needsUpdate = true;
    }
    this.dirty = [];
    const flat = Math.hypot(light.x, light.z) || 1e-6;
    this.dir.value.set(light.x / flat, light.z / flat);
    // Below a degree or so the light is gone anyway; keep the reach finite.
    const tan = Math.max(light.y / flat, 0.02);
    this.rise.value = tan * T;
    this.slope.value = tan;
    this.origin.value.set(this.wantX * TS, this.wantZ * TS);
    this.passes = passesFor(tan);
    this.pass = 0;
  }

  /** Runs up to `n` passes of the recompute in progress; the last one publishes. */
  private step(n: number): void {
    for (; n > 0 && this.pass >= 0; n--) {
      if (this.pass === 0) {
        this.renderer.compute(this.pass0);
        this.inA = true;
      } else if (this.pass < this.passes) {
        this.shift.value = 2 ** (this.pass - 1);
        this.renderer.compute(this.inA ? this.passAB : this.passBA);
        this.inA = !this.inA;
      } else {
        this.renderer.compute(this.inA ? this.finalA : this.finalB);
        this.published.value.copy(this.origin.value);
        this.pass = -1;
        return;
      }
      this.pass++;
    }
  }
}

/** Storage index of a world texel (tiles of TS x TS, wrapped). */
function index(gx: I, gz: I): I {
  const tx = gx.shiftRight(5).bitAnd(NT - 1);
  const tz = gz.shiftRight(5).bitAnd(NT - 1);
  return tz
    .mul(NT)
    .add(tx)
    .mul(TS * TS)
    .add(gz.bitAnd(TS - 1).mul(TS))
    .add(gx.bitAnd(TS - 1)) as I;
}

/** Smooth tileable value noise: random lattice values, bilinear (and mip-mapped). */
function valueNoise(size: number): THREE.DataTexture {
  let seed = 12345;
  const data = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    data.fill(seed >>> 24, i * 4, i * 4 + 4);
  }
  const t = new THREE.DataTexture(data, size, size);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

/** Storage slot of a tile (wrapping). */
function slot(x: number, z: number): number {
  const m = (v: number) => ((v % NT) + NT) % NT;
  return m(z) * NT + m(x);
}

/** Doubling passes needed to look as far as RELIEF can cast a shadow at this sun height. */
function passesFor(tan: number): number {
  const reach = Math.min(N, RELIEF / tan / T);
  return Math.min(MAX_PASSES, 1 + Math.ceil(Math.log2(Math.max(2, reach))));
}
