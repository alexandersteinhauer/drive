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
  time,
  transformNormalToView,
  uniform,
  varying,
  vec3,
} from "three/tsl";
import type { Node } from "three/webgpu";
import { Surface, type Landscape, type LandSample } from "./world/landscape";
import { hash2, rng } from "./world/noise";
import { meadowColor } from "./terrainMaterial";
import { UNDER_ROAD, variationAt } from "./terrainView";
import { forest } from "./vegetation";

/** Grass is streamed in square chunks (m) on a world-aligned grid. */
const CHUNK = 32;
/** Terrain vertex spacing at full detail (128 m tiles, 64 segments): blades sit on its triangles. */
const CELL = 2;
/** Clump grid spacing (m). */
const SPACING = 0.3;
/** Grass reaches this far from the camera, sinking into the ground over the last stretch. */
const RANGE = 75;
const FADE_FROM = 55;
/** Clumps thin out with distance to this fraction (widening to keep the cover). */
const THIN_FROM = 12;
const THIN_TO = 0.3;
/** CPU time per frame for building chunks (ms). */
const BUDGET = 1;

const BLADES = 14;
/** Beyond this distance, chunks draw only the first half of each tuft's blades, wider. */
const FAR_FROM = 20;
const FAR_WIDTH = 1.8;
/** Blade levels (fraction of its length) below the tip. */
const LEVELS = [0, 0.4, 0.75];

const viewer = uniform(new THREE.Vector3());
const sunDirection = uniform(new THREE.Vector3(0, 1, 0));
const sunLight = uniform(new THREE.Color(1, 1, 1));

const sample: LandSample = {
  height: 0,
  surface: Surface.Grass,
  roadDistance: Infinity,
};

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
  private job: { key: string; gen: Generator<void, Chunk | null> } | null =
    null;

  constructor(private readonly land: Landscape) {}

  setSun(light: THREE.DirectionalLight): void {
    sunDirection.value.copy(light.position).normalize();
    sunLight.value.copy(light.color).multiplyScalar(light.intensity);
  }

  /** Streams chunks around the camera and trims distant ones. */
  update(camera: THREE.Vector3, budget = BUDGET): void {
    viewer.value.copy(camera);
    const fx = Math.floor(camera.x / CHUNK);
    const fz = Math.floor(camera.z / CHUNK);
    const key = `${fx},${fz}`;
    if (key !== this.focusKey) {
      this.focusKey = key;
      this.rescan(camera);
    }
    const end = performance.now() + budget;
    while (performance.now() < end) {
      if (!this.job) {
        const next = this.queue.shift();
        if (!next) break;
        const k = `${next[0]},${next[1]}`;
        if (this.chunks.has(k)) continue;
        this.job = { key: k, gen: this.build(next[0], next[1]) };
      }
      const r = this.job.gen.next();
      if (r.done) {
        if (r.value) {
          this.chunks.set(this.job.key, r.value);
          this.group.add(r.value.mesh);
        }
        this.job = null;
      }
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
    if (this.job && !wanted.has(this.job.key)) this.job = null;
    list.sort((a, b) => a[2] - b[2]);
    this.queue = list
      .filter(([cx, cz]) => !this.chunks.has(`${cx},${cz}`))
      .map(([cx, cz]) => [cx, cz]);
  }

  private *build(cx: number, cz: number): Generator<void, Chunk | null> {
    const land = this.land;
    const edge = land.road.edge;
    const x0 = cx * CHUNK;
    const z0 = cz * CHUNK;
    // The terrain's own vertices (with a one-sample border for normals), so the blades sit
    // exactly on its triangles.
    const n = CHUNK / CELL;
    const g = n + 3;
    const hs = new Float32Array(g * g);
    const rs = new Float32Array(g * g);
    for (let j = 0; j < g; j++) {
      for (let i = 0; i < g; i++) {
        land.sample(x0 + (i - 1) * CELL, z0 + (j - 1) * CELL, sample);
        hs[j * g + i] = sample.height;
        rs[j * g + i] = Math.min(sample.roadDistance, 100);
      }
      if (j % 6 === 5) yield;
    }
    const v = n + 1;
    // Per vertex: height, normal x/y/z, road distance, variation, forest.
    const vert = new Float32Array(v * v * 7);
    for (let j = 0; j < v; j++)
      for (let i = 0; i < v; i++) {
        const at = (di: number, dj: number) =>
          hs[(j + 1 + dj) * g + i + 1 + di]!;
        const road = rs[(j + 1) * g + i + 1]!;
        const under =
          UNDER_ROAD *
          (1 - Math.min(1, Math.max(0, (road - edge + 1.2) / 1.2)));
        const nx = -(at(1, 0) - at(-1, 0));
        const ny = 2 * CELL;
        const nz = -(at(0, 1) - at(0, -1));
        const len = Math.hypot(nx, ny, nz);
        const x = x0 + i * CELL;
        const z = z0 + j * CELL;
        const o = (j * v + i) * 7;
        vert[o] = at(0, 0) - under;
        vert[o + 1] = nx / len;
        vert[o + 2] = ny / len;
        vert[o + 3] = nz / len;
        vert[o + 4] = road;
        vert[o + 5] = variationAt(x, z);
        vert[o + 6] = forest(x, z);
      }
    yield;

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
    // Per clump: x, y, z, yaw | height, rank, variation, forest | normal x, z.
    const out: number[] = [];
    for (let j = 0; j < cells; j++) {
      for (let i = 0; i < cells; i++) {
        const gx = Math.round(x0 / SPACING) + i;
        const gz = Math.round(z0 / SPACING) + j;
        const x = (gx + hash2(gx, gz, 41)) * SPACING;
        const z = (gz + hash2(gx, gz, 42)) * SPACING;
        if (x < x0 || x >= x0 + CHUNK || z < z0 || z >= z0 + CHUNK) continue;
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
          THREE.MathUtils.smoothstep(r, 0.7, 2.2) *
          (1 - THREE.MathUtils.smoothstep(slope, 0.08, 0.17)) *
          (1 - 0.85 * woods) *
          (0.75 + 0.25 * patch);
        if (hash2(gx, gz, 43) > density) continue;
        // Mown short along the verge, longer out in the meadow, longest in its patches.
        const field = THREE.MathUtils.smoothstep(r, 2, 7);
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
          hash2(gx, gz, 46),
          p[5]!,
          woods,
          p[1]!,
          p[3]!,
        );
      }
      if (j % 10 === 9) yield;
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
      root.set(out.slice(s, s + 4), k * 4);
      info.set(out.slice(s + 4, s + 8), k * 4);
      ground.set(out.slice(s + 8, s + 10), k * 2);
      minY = Math.min(minY, out[s + 1]!);
      maxY = Math.max(maxY, out[s + 1]!);
    });
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
  // Wind: slow waves rolling across the meadow, and a quicker flutter.
  const phase = root.x.mul(0.09).add(root.z.mul(0.06));
  const gust = sin(time.mul(1.1).add(phase))
    .mul(0.55)
    .add(sin(time.mul(2.7).add(phase.mul(2.3)).add(rank.mul(6))).mul(0.2))
    .add(0.4);
  const bend = t.mul(t).mul(h).mul(gust).mul(0.25);
  const m = new THREE.MeshStandardNodeMaterial({
    side: THREE.DoubleSide,
    roughness: 0.85,
    metalness: 0,
  });
  m.positionNode = vec3(
    root.x.add(lx).add(bend.mul(0.8)),
    root.y.add(ly).sub(bend.mul(bend).mul(0.6)),
    root.z.add(lz).add(bend.mul(0.6)),
  );

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
