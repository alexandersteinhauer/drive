import * as THREE from "three/webgpu";
import { loadLayer } from "./terrainMaterial";
import type { Landscape } from "./world/landscape";

/** Samples per road chunk mesh (2 m each). */
const CHUNK = 64;
const RADIUS = 1500;
/** Size of one asphalt texture tile on the road, in metres. */
const ASPHALT_TILE = 2.5;
const PAINT = new THREE.Color(0xe9e6da);
const DASH = 3; // samples on, then off, for the centre line

/** Builds road surface and markings in chunks along the road, around a focus point. */
export class RoadView {
  readonly group = new THREE.Group();
  private readonly chunks = new Map<number, THREE.Mesh>();
  /** Markings: plain vertex colours. */
  private readonly material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.9,
  });
  /** The carriageway: ambientCG Asphalt015 (CC0). */
  private readonly asphalt = asphaltMaterial();
  /** Shoulders: ambientCG Gravel040 (CC0). */
  private readonly gravel = gravelMaterial();

  constructor(private readonly land: Landscape) {}

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
    for (const [c, mesh] of this.chunks)
      if (!near.has(c)) {
        this.group.remove(mesh);
        mesh.geometry.dispose();
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
    const asphaltIndices = index.length;
    strip(-edge, -halfWidth, first, end, PAINT, 0);
    strip(halfWidth, edge, first, end, PAINT, 0);
    const gravelIndices = index.length;
    strip(-halfWidth + 0.15, -halfWidth + 0.3, first, end, PAINT, 0.012);
    strip(halfWidth - 0.3, halfWidth - 0.15, first, end, PAINT, 0.012);
    for (let i = first; i < end; i++)
      if (i % (DASH * 3) === 0)
        strip(-0.07, 0.07, i, Math.min(i + DASH, end), PAINT, 0.012);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.computeVertexNormals();
    // The carriageway uses the asphalt material, everything else the vertex-colour one.
    geo.addGroup(0, asphaltIndices, 0);
    geo.addGroup(asphaltIndices, gravelIndices - asphaltIndices, 1);
    geo.addGroup(gravelIndices, index.length - gravelIndices, 2);
    const mesh = new THREE.Mesh(geo, [
      this.asphalt,
      this.gravel,
      this.material,
    ]);
    mesh.receiveShadow = true;
    this.group.add(mesh);
    this.chunks.set(c, mesh);
  }
}

function asphaltMaterial(): THREE.MeshStandardMaterial {
  const loader = new THREE.TextureLoader();
  const load = (name: string, srgb: boolean) => {
    const t = loader.load(`${import.meta.env.BASE_URL}textures/${name}`);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    // Sharp at grazing angles, where most of the road is seen from a chase camera.
    t.anisotropy = 8;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  return new THREE.MeshStandardMaterial({
    map: load("asphalt015_color.webp", true),
    normalMap: load("asphalt015_normal.webp", false),
    roughnessMap: load("asphalt015_roughness.webp", false),
    roughness: 1,
    // Slightly lighter and warmer: the scan reads cool and dark next to sunlit asphalt.
    color: new THREE.Color().setRGB(1.1, 1.06, 1.0),
  });
}

function gravelMaterial(): THREE.MeshStandardMaterial {
  const layer = loadLayer("gravel040");
  return new THREE.MeshStandardMaterial({
    map: layer.color,
    normalMap: layer.normal,
    roughness: 0.85,
    // Lighter than the scan, so the shoulder reads as dusty gravel next to the asphalt.
    color: new THREE.Color().setRGB(1.35, 1.3, 1.2),
  });
}
