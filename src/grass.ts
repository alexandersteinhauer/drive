import * as THREE from "three/webgpu";
import {
  attribute,
  cameraPosition,
  cos,
  float,
  max,
  mix,
  positionGeometry,
  positionWorld,
  sin,
  smoothstep,
  sqrt,
  transformNormalToView,
  uniform,
  varying,
  vec3,
} from "three/tsl";
import type { Node } from "three/webgpu";
import { MatteNodeMaterial } from "./matte";
import { rng } from "./world/noise";
import { meadowColor } from "./terrainMaterial";
import { GRASS_CHUNK, type GrassResult } from "./world/gen";
import type { WorldWorkers } from "./worldWorkers";

/** Grass is streamed in square chunks (m) on a world-aligned grid, generated in the
 *  world workers (world/gen.ts). */
const CHUNK = GRASS_CHUNK;
/** Chunks being generated at once, and finished ones turned into meshes per frame. */
const IN_FLIGHT = 4;
const APPLY_PER_FRAME = 2;
/** Grass reaches this far from the camera, sinking into the ground over the last stretch. */
const RANGE = 180;
const FADE_FROM = 120;
/**
 * Fraction of clumps kept at distance d: all up close, then 1 / (1 + ((d - THIN_FROM) /
 * THIN)^2). Far out that's as many tufts per patch of screen at any distance, each widened
 * to keep the cover (so they keep their size on screen too).
 */
const THIN_FROM = 10;
const THIN = 42;
/**
 * A chunk holds the clumps up to one of these ranks: enough for where it'll be LOOKAHEAD
 * (m) from now. It's regenerated denser as it comes closer.
 */
const RANK_LEVELS = [1 / 8, 1 / 4, 1 / 2, 1];
const LOOKAHEAD = 48;

const BLADES = 14;
/**
 * Blades drawn per tuft: all up close, fewer with distance, down to BLADES_MID at MID_AT
 * and BLADES_FAR at FAR_AT (m), the rest widening to keep the cover.
 */
const MID_FROM = 12;
const MID_AT = 20;
const BLADES_MID = 7;
const FAR_AT = 45;
const BLADES_FAR = 4;
/** Blades straighten into one triangle each (root to tip) from here to FAR_AT; chunks
 *  wholly beyond FAR_AT draw just those triangles. */
const STRAIGHT_FROM = 25;
/** Blade levels (fraction of its length) below the tip. */
const LEVELS = [0, 0.4, 0.75];
/** Index count per full blade, and of the leading straight-triangle (far) blades. */
const BLADE_INDICES = (LEVELS.length - 1) * 6 + 3;
const FAR_INDICES = BLADES_FAR * 3;

const viewer = uniform(new THREE.Vector3());
const sunDirection = uniform(new THREE.Vector3(0, 1, 0));
const sunLight = uniform(new THREE.Color(1, 1, 1));

interface Chunk {
  mesh: THREE.Mesh<THREE.InstancedBufferGeometry>;
  /** Clumps (sorted by rank, see `info`), and the rank they go up to. */
  count: number;
  level: number;
  info: Float32Array;
  cx: number;
  cz: number;
}

/**
 * A meadow of grass blades around the camera: tufts on a jittered grid, dense in the open,
 * mown short along the verge, thin in the woods and gone on rock and steep cuttings. Blades
 * take the ground's colour at their root and are lit with the ground's normal, so the cover
 * merges into the terrain where it thins out and sinks away in the distance.
 */
export class Grass {
  readonly group = new THREE.Group();
  private readonly chunks = new Map<string, Chunk>();
  private readonly material = grassMaterial();
  private readonly clump = clumpGeometry();
  private focusKey = "";
  /** Chunks in range, nearest first (as of the last rescan). */
  private list: { cx: number; cz: number; key: string }[] = [];
  private wanted = new Set<string>();
  /**
   * Chunks being generated (to which rank) until applied, and those finished (null:
   * nothing grows there).
   */
  private readonly building = new Map<string, number>();
  private readonly finished: [number, number, number, GrassResult | null][] =
    [];
  /** Chunks known to be bare up to a rank, so they aren't requested again. */
  private readonly bare = new Map<string, number>();

  constructor(private readonly workers: WorldWorkers) {}

  setSun(light: THREE.DirectionalLight): void {
    // Towards the light: it follows the car (shadow box), so not its position alone.
    sunDirection.value
      .copy(light.position)
      .sub(light.target.position)
      .normalize();
    sunLight.value.copy(light.color).multiplyScalar(light.intensity);
  }

  /** Builds everything around `camera` (start-up), as fast as the workers go. */
  async prime(camera: THREE.Vector3): Promise<void> {
    this.update(camera, Infinity, 0);
    while (this.finished.length < this.building.size)
      await new Promise<void>((resolve) => setTimeout(resolve, 10));
    this.update(camera, 0, Infinity);
  }

  /** Streams chunks around the camera and trims distant ones. */
  update(
    camera: THREE.Vector3,
    inFlight = IN_FLIGHT,
    apply = APPLY_PER_FRAME,
  ): void {
    viewer.value.copy(camera);
    const fx = Math.floor(camera.x / CHUNK);
    const fz = Math.floor(camera.z / CHUNK);
    const focus = `${fx},${fz}`;
    if (focus !== this.focusKey) {
      this.focusKey = focus;
      this.rescan(camera);
    }
    for (let n = 0; n < apply && this.finished.length; n++) {
      const [cx, cz, level, r] = this.finished.shift()!;
      const k = `${cx},${cz}`;
      this.building.delete(k);
      const had = this.chunks.get(k);
      if (!this.wanted.has(k) || (had && had.level >= level)) continue;
      if (!r) {
        this.bare.set(k, Math.max(level, this.bare.get(k) ?? 0));
        continue;
      }
      if (had) this.drop(k, had);
      const chunk = this.chunk(cx, cz, level, r);
      this.chunks.set(k, chunk);
      this.group.add(chunk.mesh);
    }
    // Nearest first: chunks with fewer clumps than they'll need shortly.
    for (const { cx, cz, key } of this.list) {
      if (this.building.size >= inFlight) break;
      if (this.building.has(key)) continue;
      const need = levelFor(squareDistance(cx, cz, camera));
      const have = this.chunks.get(key)?.level ?? this.bare.get(key) ?? 0;
      if (need <= have) continue;
      this.building.set(key, need);
      void this.workers
        .run({ kind: "grass", cx, cz, maxRank: need })
        .then((r) => this.finished.push([cx, cz, need, r]));
    }
    // Draw only as many clumps (sorted by rank) as the nearest point needs, and of each
    // tuft as many blades: just their straight triangles if it's all far enough.
    for (const c of this.chunks.values()) {
      const d = squareDistance(c.cx, c.cz, camera);
      const geo = c.mesh.geometry;
      geo.instanceCount =
        d > RANGE ? 0 : countBelow(c.info, c.count, keep(d) + 0.02);
      if (d >= FAR_AT) geo.setDrawRange(0, FAR_INDICES);
      else
        geo.setDrawRange(
          FAR_INDICES,
          Math.ceil(bladesAt(d) - 1e-3) * BLADE_INDICES,
        );
    }
  }

  private drop(key: string, c: Chunk): void {
    this.group.remove(c.mesh);
    c.mesh.geometry.dispose();
    this.chunks.delete(key);
  }

  private rescan(camera: THREE.Vector3): void {
    const reach = Math.ceil(RANGE / CHUNK) + 1;
    const fx = Math.floor(camera.x / CHUNK);
    const fz = Math.floor(camera.z / CHUNK);
    const wanted = new Set<string>();
    const list: { cx: number; cz: number; key: string; d: number }[] = [];
    for (let j = -reach; j <= reach; j++)
      for (let i = -reach; i <= reach; i++) {
        const cx = fx + i;
        const cz = fz + j;
        const d = squareDistance(cx, cz, camera);
        if (d > RANGE + CHUNK / 2) continue;
        const key = `${cx},${cz}`;
        wanted.add(key);
        list.push({ cx, cz, key, d });
      }
    for (const [k, c] of this.chunks) if (!wanted.has(k)) this.drop(k, c);
    for (const k of this.bare.keys()) if (!wanted.has(k)) this.bare.delete(k);
    this.wanted = wanted;
    this.list = list.toSorted((a, b) => a.d - b.d);
  }

  /** A generated chunk as a mesh: the clump geometry, instanced over its clumps. */
  private chunk(cx: number, cz: number, level: number, r: GrassResult): Chunk {
    const { root, info, ground, count, minY, maxY } = r;
    const x0 = cx * CHUNK;
    const z0 = cz * CHUNK;
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = this.clump.index;
    for (const [name, a] of Object.entries(this.clump.attributes))
      geo.setAttribute(name, a);
    geo.setAttribute("root", new THREE.InstancedBufferAttribute(root, 4));
    geo.setAttribute("info", new THREE.InstancedBufferAttribute(info, 4));
    geo.setAttribute("ground", new THREE.InstancedBufferAttribute(ground, 2));
    geo.instanceCount = count;
    geo.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(x0 + CHUNK / 2, (minY + maxY) / 2, z0 + CHUNK / 2),
      // Distant tufts spread wide, a little past the chunk's edges.
      Math.hypot(CHUNK * 0.71 + 2, (maxY - minY) / 2 + 1.5),
    );
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    return { mesh, count, level, info, cx, cz };
  }
}

/** Fraction of clumps kept at distance `d` (CPU mirror of the shader's thinning). */
function keep(d: number): number {
  const x = Math.max(0, d - THIN_FROM) / THIN;
  return 1 / (1 + x * x);
}

/** The rank a chunk at distance `d` should be generated to (see RANK_LEVELS). */
function levelFor(d: number): number {
  const need = keep(Math.max(0, d - LOOKAHEAD)) + 0.02;
  return RANK_LEVELS.find((l) => l >= need) ?? 1;
}

/** How many of a chunk's clumps (sorted by rank in `info`) rank below `limit`. */
function countBelow(info: Float32Array, count: number, limit: number): number {
  let lo = 0;
  let hi = count;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (info[mid * 4 + 1]! < limit) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Blades per tuft at distance `d` (fractional: the last one is narrowing away). */
function bladesAt(d: number): number {
  const clamp = THREE.MathUtils.clamp;
  return (
    BLADES -
    (BLADES - BLADES_MID) * clamp((d - MID_FROM) / (MID_AT - MID_FROM), 0, 1) -
    (BLADES_MID - BLADES_FAR) * clamp((d - MID_AT) / (FAR_AT - MID_AT), 0, 1)
  );
}

/** Horizontal distance from a point to chunk (cx, cz)'s square. */
function squareDistance(cx: number, cz: number, p: THREE.Vector3): number {
  const dx = Math.max(cx * CHUNK - p.x, 0, p.x - (cx + 1) * CHUNK);
  const dz = Math.max(cz * CHUNK - p.z, 0, p.z - (cz + 1) * CHUNK);
  return Math.hypot(dx, dz);
}

/** The angle between two directions (radians). */
function angleBetween(a: number, b: number): number {
  return Math.abs(
    ((((a - b) % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI)) - Math.PI,
  );
}

/**
 * One tuft of blades, 1 high: each a tapered, curving strip leaning out from the centre,
 * ordered so that any first few lean evenly round the tuft (far away only those are drawn).
 * Attributes: "blade" (fraction along the blade, random tone, blade number), "axis" (the
 * blade's centre line here), and "straight" and "straightAxis", the same on the blade
 * straightened into one triangle from its root to its tip. The index starts with just those
 * triangles for the first BLADES_FAR blades, then has the full blades in order.
 */
function clumpGeometry(): THREE.BufferGeometry {
  const random = rng(7);
  const blades = Array.from({ length: BLADES }, () => {
    const a0 = random() * Math.PI * 2;
    const rr = 0.22 * Math.sqrt(random());
    // Leaning outwards, more for blades further out.
    const lean = a0 + (random() - 0.5) * 1.2;
    return {
      ox: Math.cos(a0) * rr,
      oz: Math.sin(a0) * rr,
      lean,
      amount: 0.15 + 0.5 * random() * (0.4 + rr / 0.22),
      length: 0.55 + 0.45 * random(),
      width: 0.035 + 0.025 * random(),
      twist: (random() - 0.5) * 1.2,
      tone: random(),
    };
  });
  // Each next blade the one leaning furthest from those before it.
  const ordered = [blades.shift()!];
  while (blades.length) {
    const spread = blades.map((b) =>
      Math.min(...ordered.map((o) => angleBetween(o.lean, b.lean))),
    );
    const best = spread.indexOf(Math.max(...spread));
    ordered.push(blades.splice(best, 1)[0]!);
  }

  const pos: number[] = [];
  const blade: number[] = [];
  const axis: number[] = [];
  const straight: number[] = [];
  const straightAxis: number[] = [];
  const index: number[] = [];
  const far: number[] = [];
  ordered.forEach((b, n) => {
    const base = pos.length / 3;
    const { ox, oz, amount, length, width, tone } = b;
    const dx = Math.cos(b.lean);
    const dz = Math.sin(b.lean);
    // Across the blade: perpendicular to its lean, twisted a little.
    const ax = Math.cos(b.lean + Math.PI / 2 + b.twist);
    const az = Math.sin(b.lean + Math.PI / 2 + b.twist);
    const at = (t: number) => {
      const out = amount * length * t * t;
      const up = length * t * (1 - 0.3 * amount * t);
      return [ox + dx * out, up, oz + dz * out] as const;
    };
    const tip = at(1);
    // Straight from the root to the tip, narrowing evenly.
    const line = (t: number) =>
      [ox + (tip[0] - ox) * t, tip[1] * t, oz + (tip[2] - oz) * t] as const;
    for (const t of LEVELS) {
      const [cx, cy, cz] = at(t);
      const [sx, sy, sz] = line(t);
      const w = (width / 2) * (1 - t) ** 0.7;
      const sw = (width / 2) * (1 - t);
      pos.push(cx - ax * w, cy, cz - az * w, cx + ax * w, cy, cz + az * w);
      straight.push(
        sx - ax * sw,
        sy,
        sz - az * sw,
        sx + ax * sw,
        sy,
        sz + az * sw,
      );
      blade.push(t, tone, n, t, tone, n);
      axis.push(cx, cz, cx, cz);
      straightAxis.push(sx, sz, sx, sz);
    }
    pos.push(...tip);
    straight.push(...tip);
    blade.push(1, tone, n);
    axis.push(tip[0], tip[2]);
    straightAxis.push(tip[0], tip[2]);
    for (let l = 0; l < LEVELS.length - 1; l++) {
      const k = base + l * 2;
      index.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
    const k = base + (LEVELS.length - 1) * 2;
    index.push(k, k + 1, k + 2);
    if (n < BLADES_FAR) far.push(base, base + 1, k + 2);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("blade", new THREE.Float32BufferAttribute(blade, 3));
  geo.setAttribute("axis", new THREE.Float32BufferAttribute(axis, 2));
  geo.setAttribute("straight", new THREE.Float32BufferAttribute(straight, 3));
  geo.setAttribute(
    "straightAxis",
    new THREE.Float32BufferAttribute(straightAxis, 2),
  );
  geo.setIndex([...far, ...index]);
  return geo;
}

/**
 * Grass material. With distance, tufts thin out (by rank) and spread wider, lose blades
 * (the rest widening) and straighten them, and finally sink into the ground.
 */
function grassMaterial(): THREE.MeshStandardNodeMaterial {
  const root = attribute("root", "vec4");
  const info = attribute("info", "vec4");
  const ground = attribute("ground", "vec2");
  const blade = attribute("blade", "vec3");
  const t = blade.x;
  const tone = blade.y;
  const height = info.x;
  const rank = info.y;

  const dist = root.xz.distance(viewer.xz);
  const thin = dist.sub(THIN_FROM).max(0).div(THIN);
  const kept = float(1).div(thin.mul(thin).add(1));
  // Clumps beyond their rank shrink away; the rest widen to keep the cover, and all sink
  // into the ground towards the end of the range.
  const show = kept.sub(rank).mul(30).clamp(0, 1);
  const fade = float(1).sub(smoothstep(FADE_FROM, RANGE, dist));
  const h = height.mul(show).mul(fade);
  const widen = sqrt(thin.mul(thin).add(1));
  // Taller tufts spread wider, but less than in proportion.
  const spread = height.mul(0.4).add(0.3).mul(show).mul(fade).mul(widen);
  const c = cos(root.w);
  const s = sin(root.w);
  // Blades straighten with distance, and drop out one by one from the last, narrowing
  // away; the others widen to keep the cover (about each blade's own centre line).
  const straighten = smoothstep(STRAIGHT_FROM, FAR_AT, dist);
  const p = mix(positionGeometry, attribute("straight", "vec3"), straighten);
  const axis = mix(
    attribute("axis", "vec2"),
    attribute("straightAxis", "vec2"),
    straighten,
  );
  const blades = float(BLADES)
    .sub(
      dist
        .sub(MID_FROM)
        .div(MID_AT - MID_FROM)
        .clamp(0, 1)
        .mul(BLADES - BLADES_MID),
    )
    .sub(
      dist
        .sub(MID_AT)
        .div(FAR_AT - MID_AT)
        .clamp(0, 1)
        .mul(BLADES_MID - BLADES_FAR),
    );
  const width = float(BLADES)
    .div(blades)
    .pow(0.85)
    .mul(blades.sub(blade.z).clamp(0, 1));
  const bx = mix(axis.x, p.x, width);
  const bz = mix(axis.y, p.z, width);
  const lx = bx.mul(c).sub(bz.mul(s)).mul(spread);
  const lz = bx.mul(s).add(bz.mul(c)).mul(spread);
  const ny = sqrt(float(1).sub(ground.dot(ground)).max(0.1));
  // Roots follow the slope.
  const ly = p.y.mul(h).sub(ground.x.mul(lx).add(ground.y.mul(lz)).div(ny));
  const m = new MatteNodeMaterial({
    side: THREE.DoubleSide,
    roughness: 0.85,
    metalness: 0,
  });
  m.sheen = 0.01;
  m.positionNode = vec3(root.x.add(lx), root.y.add(ly), root.z.add(lz));

  // The ground's colour at the root, darker down in the tuft, lighter at the tips; some
  // blades dry. Close up the depth shows; far away it evens out into the ground's colour.
  const groundColor = varying(
    meadowColor(root.xz, info.z, info.w, 4) as Node<"vec3">,
  );
  const depth = mix(float(0.4), float(1.2), t.pow(0.8));
  const shade = mix(depth, float(1), smoothstep(FADE_FROM, RANGE, dist));
  const clumpTone = rank.mul(7.31).fract();
  const dry = smoothstep(0.82, 0.95, tone).mul(t);
  const albedo = mix(groundColor, groundColor.mul(vec3(1.3, 1.15, 0.7)), dry)
    .mul(shade.mul(tone.mul(0.16).add(0.92)))
    // Clumps differ a little in shade and hue.
    .mul(mix(vec3(0.9, 0.93, 0.95), vec3(1.12, 1.1, 0.92), clumpTone));
  m.colorNode = albedo;
  m.normalNode = transformNormalToView(vec3(ground.x, ny, ground.y));
  // Light through backlit blades.
  const view = positionWorld.sub(cameraPosition).normalize();
  const back = max(view.dot(sunDirection), 0).pow(3);
  m.emissiveNode = albedo.mul(sunLight).mul(back.mul(t).mul(0.18));
  return m;
}
