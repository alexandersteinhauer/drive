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
const RANGE = 75;
const FADE_FROM = 55;
/** Clumps thin out with distance to this fraction (widening to keep the cover). */
const THIN_FROM = 12;
const THIN_TO = 0.3;

const BLADES = 14;
/** Beyond this distance, chunks draw only the first half of each tuft's blades, wider. */
const FAR_FROM = 20;
const FAR_WIDTH = 1.8;
/** Blade levels (fraction of its length) below the tip. */
const LEVELS = [0, 0.4, 0.75];

const viewer = uniform(new THREE.Vector3());
const sunDirection = uniform(new THREE.Vector3(0, 1, 0));
const sunLight = uniform(new THREE.Color(1, 1, 1));

interface Chunk {
  mesh: THREE.Mesh<THREE.InstancedBufferGeometry>;
  count: number;
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
  private readonly material = grassMaterial(1);
  private readonly farMaterial = grassMaterial(FAR_WIDTH);
  private readonly clump = clumpGeometry();
  private focusKey = "";
  private queue: [number, number][] = [];
  /** Chunks being generated, and those finished (null: nothing grows there). */
  private readonly building = new Set<string>();
  private readonly finished: [number, number, GrassResult | null][] = [];
  /** Chunks known to be bare, so they aren't requested again. */
  private readonly bare = new Set<string>();
  private wanted = new Set<string>();

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
    while (this.building.size)
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
    const key = `${fx},${fz}`;
    if (key !== this.focusKey) {
      this.focusKey = key;
      this.rescan(camera);
    }
    for (let n = 0; n < apply && this.finished.length; n++) {
      const [cx, cz, r] = this.finished.shift()!;
      const k = `${cx},${cz}`;
      if (!this.wanted.has(k) || this.chunks.has(k)) continue;
      if (!r) {
        this.bare.add(k);
        continue;
      }
      const chunk = this.chunk(cx, cz, r);
      this.chunks.set(k, chunk);
      this.group.add(chunk.mesh);
    }
    while (this.building.size < inFlight && this.queue.length) {
      const [cx, cz] = this.queue.shift()!;
      const k = `${cx},${cz}`;
      if (this.chunks.has(k) || this.bare.has(k) || this.building.has(k))
        continue;
      this.building.add(k);
      void this.workers.run({ kind: "grass", cx, cz }).then((r) => {
        this.building.delete(k);
        this.finished.push([cx, cz, r]);
      });
    }
    // Draw only as many clumps (sorted by rank) as the nearest point needs.
    for (const c of this.chunks.values()) {
      const d = squareDistance(c.cx, c.cz, camera);
      const geo = c.mesh.geometry;
      geo.instanceCount =
        d > RANGE ? 0 : Math.ceil(c.count * Math.min(1, keep(d) + 0.02));
      const far = d > FAR_FROM;
      c.mesh.material = far ? this.farMaterial : this.material;
      geo.setDrawRange(0, far ? geo.index!.count / 2 : Infinity);
    }
  }

  private rescan(camera: THREE.Vector3): void {
    const reach = Math.ceil(RANGE / CHUNK) + 1;
    const fx = Math.floor(camera.x / CHUNK);
    const fz = Math.floor(camera.z / CHUNK);
    const wanted = new Set<string>();
    const list: [number, number, number][] = [];
    for (let j = -reach; j <= reach; j++)
      for (let i = -reach; i <= reach; i++) {
        const cx = fx + i;
        const cz = fz + j;
        const d = squareDistance(cx, cz, camera);
        if (d > RANGE + CHUNK / 2) continue;
        wanted.add(`${cx},${cz}`);
        list.push([cx, cz, d]);
      }
    for (const [k, c] of this.chunks)
      if (!wanted.has(k)) {
        this.group.remove(c.mesh);
        c.mesh.geometry.dispose();
        this.chunks.delete(k);
      }
    for (const k of this.bare) if (!wanted.has(k)) this.bare.delete(k);
    this.wanted = wanted;
    list.sort((a, b) => a[2] - b[2]);
    this.queue = list
      .filter(([cx, cz]) => !this.chunks.has(`${cx},${cz}`))
      .map(([cx, cz]) => [cx, cz]);
  }

  /** A generated chunk as a mesh: the clump geometry, instanced over its clumps. */
  private chunk(cx: number, cz: number, r: GrassResult): Chunk {
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
      Math.hypot(CHUNK * 0.71, (maxY - minY) / 2 + 1.5),
    );
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    return { mesh, count, cx, cz };
  }
}

/** Fraction of clumps kept at distance `d` (CPU mirror of the shader's thinning). */
function keep(d: number): number {
  return 1 - (1 - THIN_TO) * THREE.MathUtils.smoothstep(d, THIN_FROM, RANGE);
}

/** Horizontal distance from a point to chunk (cx, cz)'s square. */
function squareDistance(cx: number, cz: number, p: THREE.Vector3): number {
  const dx = Math.max(cx * CHUNK - p.x, 0, p.x - (cx + 1) * CHUNK);
  const dz = Math.max(cz * CHUNK - p.z, 0, p.z - (cz + 1) * CHUNK);
  return Math.hypot(dx, dz);
}

/**
 * One tuft of blades, 1 high: each a tapered, curving strip leaning out from the centre.
 * Attribute "blade": (fraction along the blade, random tone).
 */
function clumpGeometry(): THREE.BufferGeometry {
  const random = rng(7);
  const pos: number[] = [];
  const blade: number[] = [];
  const axis: number[] = [];
  const index: number[] = [];
  for (let b = 0; b < BLADES; b++) {
    const base = pos.length / 3;
    const a0 = random() * Math.PI * 2;
    const rr = 0.22 * Math.sqrt(random());
    const ox = Math.cos(a0) * rr;
    const oz = Math.sin(a0) * rr;
    // Leaning outwards, more for blades further out.
    const lean = a0 + (random() - 0.5) * 1.2;
    const amount = 0.15 + 0.5 * random() * (0.4 + rr / 0.22);
    const length = 0.55 + 0.45 * random();
    const width = 0.035 + 0.025 * random();
    const twist = (random() - 0.5) * 1.2;
    const tone = random();
    const dx = Math.cos(lean);
    const dz = Math.sin(lean);
    // Across the blade: perpendicular to its lean, twisted a little.
    const ax = Math.cos(lean + Math.PI / 2 + twist);
    const az = Math.sin(lean + Math.PI / 2 + twist);
    const at = (t: number) => {
      const out = amount * length * t * t;
      const up = length * t * (1 - 0.3 * amount * t);
      return [ox + dx * out, up, oz + dz * out];
    };
    for (const t of LEVELS) {
      const [cx, cy, cz] = at(t);
      const w = (width / 2) * (1 - t) ** 0.7;
      pos.push(
        cx! - ax * w,
        cy!,
        cz! - az * w,
        cx! + ax * w,
        cy!,
        cz! + az * w,
      );
      blade.push(t, tone, t, tone);
      axis.push(cx!, cz!, cx!, cz!);
    }
    const tip = at(1);
    pos.push(...tip);
    blade.push(1, tone);
    axis.push(tip[0]!, tip[2]!);
    for (let l = 0; l < LEVELS.length - 1; l++) {
      const k = base + l * 2;
      index.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
    const k = base + (LEVELS.length - 1) * 2;
    index.push(k, k + 1, k + 2);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("blade", new THREE.Float32BufferAttribute(blade, 2));
  geo.setAttribute("axis", new THREE.Float32BufferAttribute(axis, 2));
  geo.setIndex(index);
  return geo;
}

/** Grass material; `width` scales the blades' width (for the sparser far tufts). */
function grassMaterial(width: number): THREE.MeshStandardNodeMaterial {
  const root = attribute("root", "vec4");
  const info = attribute("info", "vec4");
  const ground = attribute("ground", "vec2");
  const blade = attribute("blade", "vec2");
  const t = blade.x;
  const tone = blade.y;
  const height = info.x;
  const rank = info.y;

  const dist = root.xz.distance(viewer.xz);
  const kept = float(1).sub(
    smoothstep(THIN_FROM, RANGE, dist).mul(1 - THIN_TO),
  );
  // Clumps beyond their rank shrink away; the rest widen to keep the cover, and all sink
  // into the ground towards the end of the range.
  const show = kept.sub(rank).mul(30).clamp(0, 1);
  const fade = float(1).sub(smoothstep(FADE_FROM, RANGE, dist));
  const h = height.mul(show).mul(fade);
  const widen = float(1).div(sqrt(kept));
  // Taller tufts spread wider, but less than in proportion.
  const spread = height.mul(0.4).add(0.3).mul(show).mul(fade).mul(widen);
  const c = cos(root.w);
  const s = sin(root.w);
  const p = positionGeometry;
  // Widened about each blade's own axis (attribute "axis": its centre line here).
  const axis = attribute("axis", "vec2");
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
