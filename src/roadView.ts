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
/**
 * Guide posts (Leitpfosten, StVO signs 620-40 right and 620-41 left), every this many
 * samples (50 m) on both sides, their road face this far out from the carriageway edge.
 */
const POST_EVERY = 25;
const POST_OUT = 0.5;
const POST_WHITE = new THREE.Color(0xf1f0ea); // RAL 9016 traffic white
const POST_BLACK = new THREE.Color(0x1e1e1e); // RAL 9017 traffic black
const REFLECTOR = new THREE.Color(0xf6f6f3); // white retroreflectors
/**
 * The post's cross-section, in metres: u along the road (positive on the face with the
 * rectangular reflector), v outwards from the face towards the road. A blunted triangle,
 * 120 mm deep across the road, narrowing away from it.
 */
const POST_SECTION: [number, number][] = [
  [0.05, 0],
  [0.0125, 0.12],
  [-0.0125, 0.12],
  [-0.05, 0],
];
const POST_DEPTH = 0.12;
/** Heights above the ground at the road face (v = 0) and the outer side (v = depth). */
const POST_TOP = [0.95, 1.0] as const;
const BAND_BOTTOM = [0.5, 0.57] as const;
const BAND_HEIGHT = 0.25;
/** Buried below the ground, so it stands in slopes too. */
const POST_FOOT = -0.3;

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

    /** A triangle or quad of one colour, wound to face along `out`. */
    const face = (
      corners: THREE.Vector3[],
      out: THREE.Vector3,
      color: THREE.Color,
    ) => {
      const [a, b, d] = corners as [
        THREE.Vector3,
        THREE.Vector3,
        THREE.Vector3,
      ];
      const n = new THREE.Vector3()
        .subVectors(b, a)
        .cross(new THREE.Vector3().subVectors(d, a));
      const ordered = n.dot(out) < 0 ? corners.toReversed() : corners;
      const start = pos.length / 3;
      for (const p of ordered) {
        pos.push(p.x, p.y, p.z);
        col.push(color.r, color.g, color.b);
        uv.push(0, 0);
      }
      for (let k = 1; k < ordered.length - 1; k++)
        index.push(start, start + k, start + k + 1);
    };
    /** A disc of `radius` facing along `out`, centred at `at`, upright. */
    const disc = (
      at: THREE.Vector3,
      out: THREE.Vector3,
      radius: number,
      color: THREE.Color,
    ) => {
      const side = new THREE.Vector3(-out.z, 0, out.x).normalize();
      const corners = Array.from({ length: 12 }, (_, k) => {
        const t = (k / 12) * Math.PI * 2;
        return at
          .clone()
          .addScaledVector(side, Math.cos(t) * radius)
          .add(new THREE.Vector3(0, Math.sin(t) * radius, 0));
      });
      face(corners, out, color);
    };

    /**
     * A guide post at sample i on one side (1 right, -1 left of travel). Per StVO 620: 1 m
     * high, its top sloping down towards the road, a 250 mm black band parallel to it.
     * Facing approaching traffic on its own side, a 40 x 180 mm rectangular reflector
     * (620-40); facing the other way, two 60 mm round ones (620-41).
     */
    const post = (i: number, side: number) => {
      const h = road.headings[i]!;
      const forward = new THREE.Vector3(Math.sin(h), 0, -Math.cos(h));
      const right = new THREE.Vector3(Math.cos(h), 0, Math.sin(h));
      // Traffic on this side approaches from behind (right) or from ahead (left).
      const u = forward.clone().multiplyScalar(-side);
      const v = right.clone().multiplyScalar(side);
      const base = new THREE.Vector3(
        road.xs[i]!,
        0,
        road.zs[i]!,
      ).addScaledVector(v, halfWidth + POST_OUT);
      const centre = base.clone().addScaledVector(v, POST_DEPTH / 2);
      base.y = this.land.height(centre.x, centre.z);
      const at = (pu: number, pv: number, y: number) =>
        base
          .clone()
          .addScaledVector(u, pu)
          .addScaledVector(v, pv)
          .setY(base.y + y);
      const lerp = (range: readonly [number, number], pv: number) =>
        range[0] + ((range[1] - range[0]) * pv) / POST_DEPTH;
      /** The post between two heights (functions of v), with a cap on top if asked. */
      const slice = (
        lo: (pv: number) => number,
        hi: (pv: number) => number,
        color: THREE.Color,
        cap: boolean,
      ) => {
        POST_SECTION.forEach(([u0, v0], k) => {
          const [u1, v1] = POST_SECTION[(k + 1) % POST_SECTION.length]!;
          const out = u
            .clone()
            .multiplyScalar(v1 - v0)
            .addScaledVector(v, -(u1 - u0));
          // The section runs clockwise seen from above, so this points outwards; check.
          const mid = at((u0 + u1) / 2, (v0 + v1) / 2, 0).sub(
            at(0, POST_DEPTH / 2, 0),
          );
          if (out.dot(mid) < 0) out.negate();
          face(
            [
              at(u0, v0, lo(v0)),
              at(u1, v1, lo(v1)),
              at(u1, v1, hi(v1)),
              at(u0, v0, hi(v0)),
            ],
            out,
            color,
          );
        });
        if (cap)
          face(
            POST_SECTION.map(([pu, pv]) => at(pu, pv, hi(pv))),
            new THREE.Vector3(0, 1, 0),
            color,
          );
      };
      const bandLo = (pv: number) => lerp(BAND_BOTTOM, pv);
      const bandHi = (pv: number) => bandLo(pv) + BAND_HEIGHT;
      slice(() => POST_FOOT, bandLo, POST_WHITE, false);
      slice(bandLo, bandHi, POST_BLACK, false);
      slice(bandHi, (pv) => lerp(POST_TOP, pv), POST_WHITE, true);

      // Reflectors, centred on the front and back faces, just proud of them.
      const [[fu0, fv0], [fu1, fv1]] = [POST_SECTION[0]!, POST_SECTION[1]!];
      const along = new THREE.Vector3()
        .addScaledVector(u, fu1 - fu0)
        .addScaledVector(v, fv1 - fv0)
        .normalize();
      const front = u
        .clone()
        .multiplyScalar(fv1 - fv0)
        .addScaledVector(v, -(fu1 - fu0))
        .normalize();
      if (front.dot(u) < 0) front.negate();
      const mid = (fu0 + fu1) / 2;
      const midV = (fv0 + fv1) / 2;
      const lift = 0.002;
      const rect = at(mid, midV, 0).addScaledVector(front, lift);
      const corner = (a: number, y: number) =>
        rect
          .clone()
          .addScaledVector(along, a)
          .setY(base.y + y);
      face(
        [
          corner(-0.02, 0.568),
          corner(0.02, 0.568),
          corner(0.02, 0.748),
          corner(-0.02, 0.748),
        ],
        front,
        REFLECTOR,
      );
      // The back face mirrors the front across u = 0.
      const back = front.clone().addScaledVector(u, -2 * front.dot(u));
      for (const y of [0.58, 0.74])
        disc(
          at(-mid, midV, y).addScaledVector(back, lift),
          back,
          0.03,
          REFLECTOR,
        );
    };
    for (let i = first; i < end; i++)
      if (i % POST_EVERY === 0) for (const side of [-1, 1]) post(i, side);

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
