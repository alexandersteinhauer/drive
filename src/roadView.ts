import * as THREE from "three/webgpu";
import { normalMap, uv as meshUv, vec3 } from "three/tsl";
import { HexTiling } from "./hexTile";
import { loadLayer } from "./terrainMaterial";
import { BatchPool } from "./batchPool";
import type { Landscape } from "./world/landscape";

/** Samples per road chunk mesh (2 m each). */
const CHUNK = 64;
const RADIUS = 2300;
/** Size of one asphalt texture tile on the road, in metres. */
const ASPHALT_TILE = 2.5;
const PAINT = new THREE.Color(0xe9e6da);
const DASH = 3; // samples on, then off, for the centre line
/** Marker posts: every this many samples on both sides, just beyond the shoulder. */
const POST_EVERY = 25;
const POST_OUT = 0.45;
const POST_WHITE = new THREE.Color(0xe8e6df);
const POST_BLACK = new THREE.Color(0x1a1a1a);
const REFLECTOR = new THREE.Color(0xd9781c);

/**
 * Builds road surface and markings in chunks along the road, around a focus point. Each
 * part (carriageway, shoulders, markings and posts) is one batched object across all chunks.
 */
export class RoadView {
  readonly group = new THREE.Group();
  /** Per chunk, its piece in each part's batch. */
  private readonly chunks = new Map<number, number[]>();
  private readonly parts: BatchPool[];

  constructor(private readonly land: Landscape) {
    const pool = (material: THREE.Material, name: string) => {
      const p = new BatchPool(material, {
        vertices: 65536,
        indices: 160_000,
        instances: 96,
      });
      p.mesh.name = name;
      p.mesh.receiveShadow = true;
      this.group.add(p.mesh);
      return p;
    };
    this.parts = [
      // The carriageway: ambientCG Asphalt015 (CC0).
      pool(asphaltMaterial(), "asphalt"),
      // Shoulders: ambientCG Gravel040 (CC0).
      pool(gravelMaterial(), "gravel"),
      // Markings and posts: plain vertex colours.
      pool(
        new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 }),
        "markings",
      ),
    ];
  }

  update(focus: THREE.Vector3): void {
    const road = this.land.road;
    const last = Math.floor((road.finalCount - 2) / CHUNK);
    const near = new Set<number>();
    for (let c = 0; c <= last; c++) {
      // Distance from the focus to the chunk's ends and middle; good enough for culling.
      const a = c * CHUNK;
      const ids = [
        a,
        Math.min(a + CHUNK / 2, road.finalCount - 1),
        Math.min(a + CHUNK, road.finalCount - 1),
      ];
      const d = Math.min(
        ...ids.map((i) =>
          Math.hypot(road.xs[i]! - focus.x, road.zs[i]! - focus.z),
        ),
      );
      if (d < RADIUS) near.add(c);
    }
    for (const [c, pieces] of this.chunks)
      if (!near.has(c)) {
        pieces.forEach((piece, k) => this.parts[k]!.remove(piece));
        this.chunks.delete(c);
      }
    for (const c of near) if (!this.chunks.has(c)) this.build(c);
  }

  private build(c: number): void {
    const road = this.land.road;
    const { halfWidth } = road.cfg;
    const edge = road.edge;
    const pos: number[] = [];
    const col: number[] = [];
    const uv: number[] = [];
    const index: number[] = [];
    const first = c * CHUNK;
    const end = Math.min(first + CHUNK, road.finalCount - 1);

    /** A strip between two lateral offsets over samples [from, to], lifted by `lift`. */
    const strip = (
      left: number,
      right: number,
      from: number,
      to: number,
      color: THREE.Color,
      lift: number,
    ) => {
      const start = pos.length / 3;
      for (let i = from; i <= to; i++) {
        const h = road.headings[i]!;
        const rx = Math.cos(h);
        const rz = Math.sin(h);
        for (const lat of [left, right]) {
          const x = road.xs[i]! + rx * lat;
          const z = road.zs[i]! + rz * lat;
          // Inside the shoulders the surface is the road plane; outside it, the blended terrain.
          const y =
            Math.abs(lat) <= edge
              ? road.heights[i]! + lat * road.slopes[i]!
              : this.land.height(x, z);
          pos.push(x, y + lift, z);
          col.push(color.r, color.g, color.b);
          // World-scale UVs: across the road, and along it by distance travelled.
          uv.push(lat / ASPHALT_TILE, (i * road.cfg.spacing) / ASPHALT_TILE);
        }
      }
      for (let k = 0; k < to - from; k++) {
        const a = start + k * 2;
        index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    };

    strip(-halfWidth, halfWidth, first, end, PAINT, 0);
    const asphaltEnd = [pos.length / 3, index.length] as const;
    strip(-edge, -halfWidth, first, end, PAINT, 0);
    strip(halfWidth, edge, first, end, PAINT, 0);
    const gravelEnd = [pos.length / 3, index.length] as const;
    strip(-halfWidth + 0.15, -halfWidth + 0.3, first, end, PAINT, 0.012);
    strip(halfWidth - 0.3, halfWidth - 0.15, first, end, PAINT, 0.012);
    for (let i = first; i < end; i++)
      if (i % (DASH * 3) === 0)
        strip(-0.07, 0.07, i, Math.min(i + DASH, end), PAINT, 0.012);

    /** An oriented box: centre, half sizes along the road, up and across, and a colour. */
    const box = (
      cx: number,
      cy: number,
      cz: number,
      along: number,
      up: number,
      across: number,
      h: number,
      color: THREE.Color,
    ) => {
      const f = [Math.sin(h), 0, -Math.cos(h)];
      const r = [Math.cos(h), 0, Math.sin(h)];
      const u = [0, 1, 0];
      const corner = (a: number, b: number, c: number) =>
        [0, 1, 2].map(
          (k) =>
            [cx, cy, cz][k]! +
            f[k]! * a * along +
            u[k]! * b * up +
            r[k]! * c * across,
        );
      // Six faces, four corners each (own vertices, so normals stay flat).
      const faces: [number, number, number][][] = [
        [
          [1, -1, -1],
          [1, 1, -1],
          [1, 1, 1],
          [1, -1, 1],
        ],
        [
          [-1, -1, 1],
          [-1, 1, 1],
          [-1, 1, -1],
          [-1, -1, -1],
        ],
        [
          [-1, 1, -1],
          [-1, 1, 1],
          [1, 1, 1],
          [1, 1, -1],
        ],
        [
          [-1, -1, 1],
          [1, -1, 1],
          [1, 1, 1],
          [-1, 1, 1],
        ],
        [
          [1, -1, -1],
          [-1, -1, -1],
          [-1, 1, -1],
          [1, 1, -1],
        ],
      ];
      for (const face of faces) {
        const start = pos.length / 3;
        for (const [a, b, c] of face) {
          pos.push(...corner(a, b, c));
          col.push(color.r, color.g, color.b);
          uv.push(0, 0);
        }
        index.push(start, start + 1, start + 2, start, start + 2, start + 3);
      }
    };
    for (let i = first; i < end; i++) {
      if (i % POST_EVERY !== 0) continue;
      const h = road.headings[i]!;
      for (const side of [-1, 1]) {
        const lat = side * (edge + POST_OUT);
        const x = road.xs[i]! + Math.cos(h) * lat;
        const z = road.zs[i]! + Math.sin(h) * lat;
        const ground = this.land.height(x, z);
        // A 1 m post with a black band near the top holding the reflectors.
        box(x, ground + 0.45, z, 0.06, 0.55, 0.05, h, POST_WHITE);
        box(x, ground + 0.86, z, 0.062, 0.1, 0.052, h, POST_BLACK);
        for (const end of [-1, 1])
          box(
            x + Math.sin(h) * end * 0.064,
            ground + 0.86,
            z - Math.cos(h) * end * 0.064,
            0.004,
            0.05,
            0.02,
            h,
            REFLECTOR,
          );
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.computeVertexNormals();
    // Split into the parts, each vertices and indices in a range of their own. A slot fits
    // a full chunk's carriageway or shoulders; markings vary, so their slots come in
    // power-of-two sizes.
    const full = [(CHUNK + 1) * 2, CHUNK * 6];
    const ends = [
      [0, 0],
      asphaltEnd,
      gravelEnd,
      [pos.length / 3, index.length],
    ];
    const pieces = this.parts.map((part, k) => {
      const [v0, i0] = ends[k]!;
      const [v1, i1] = ends[k + 1]!;
      const piece = new THREE.BufferGeometry();
      for (const name of ["position", "normal", "color", "uv"]) {
        const a = geo.getAttribute(name) as THREE.BufferAttribute;
        piece.setAttribute(
          name,
          new THREE.BufferAttribute(
            a.array.slice(v0! * a.itemSize, v1! * a.itemSize),
            a.itemSize,
          ),
        );
      }
      piece.setIndex(index.slice(i0, i1).map((i) => i - v0!));
      piece.computeBoundingSphere();
      const room =
        k < 2
          ? [full[0]! * (k + 1), full[1]! * (k + 1)]
          : [pow2(v1! - v0!), pow2(i1! - i0!)];
      return part.add(piece, room[0], room[1]);
    });
    geo.dispose();
    this.chunks.set(c, pieces);
  }
}

/** The smallest power of two of at least `n` (and 64). */
function pow2(n: number): number {
  return Math.max(64, 2 ** Math.ceil(Math.log2(n)));
}

/**
 * The carriageway: ambientCG Asphalt015 (CC0), hex-tiled (see hexTile.ts) so its 2.5 m tile
 * doesn't show as a repeating pattern.
 */
function asphaltMaterial(): THREE.MeshStandardNodeMaterial {
  const loader = new THREE.TextureLoader();
  const load = (name: string, srgb: boolean) => {
    const t = loader.load(`${import.meta.env.BASE_URL}textures/${name}`);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    // Sharp at grazing angles, where most of the road is seen from a chase camera.
    t.anisotropy = 8;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  const hex = new HexTiling(meshUv());
  const color = hex.sample(load("asphalt015_color.webp", true));
  const w = hex.weights(color);
  const m = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  // Slightly lighter and warmer: the scan reads cool and dark next to sunlit asphalt.
  m.colorNode = hex.blend(color, w).rgb.mul(vec3(1.1, 1.06, 1.0));
  m.roughnessNode = hex.blend(
    hex.sample(load("asphalt015_roughness.webp", false)),
    w,
  ).g;
  m.normalNode = normalMap(
    hex.normal(load("asphalt015_normal.webp", false), w).mul(0.5).add(0.5),
  );
  return m;
}

/** Shoulders: ambientCG Gravel040 (CC0), hex-tiled like the asphalt. */
function gravelMaterial(): THREE.MeshStandardNodeMaterial {
  const layer = loadLayer("gravel040");
  const hex = new HexTiling(meshUv());
  const color = hex.sample(layer.color);
  const w = hex.weights(color);
  const m = new THREE.MeshStandardNodeMaterial({
    roughness: 0.85,
    metalness: 0,
  });
  // Lighter than the scan, so the shoulder reads as dusty gravel next to the asphalt.
  m.colorNode = hex.blend(color, w).rgb.mul(vec3(1.35, 1.3, 1.2));
  m.normalNode = normalMap(hex.normal(layer.normal, w).mul(0.5).add(0.5));
  return m;
}
