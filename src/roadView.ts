import * as THREE from "three/webgpu";
import type { Landscape } from "./world/landscape";

/** Samples per road chunk mesh (2 m each). */
const CHUNK = 64;
const RADIUS = 1500;
const ASPHALT = new THREE.Color(0x3b3c3f);
const GRAVEL = new THREE.Color(0x8f8875);
const PAINT = new THREE.Color(0xe9e6da);
const DASH = 3; // samples on, then off, for the centre line

/** Builds road surface and markings in chunks along the road, around a focus point. */
export class RoadView {
  readonly group = new THREE.Group();
  private readonly chunks = new Map<number, THREE.Mesh>();
  private readonly material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.9,
  });

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
        }
      }
      for (let k = 0; k < to - from; k++) {
        const a = start + k * 2;
        index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    };

    strip(-halfWidth, halfWidth, first, end, ASPHALT, 0);
    strip(-edge - 0.4, -halfWidth, first, end, GRAVEL, 0);
    strip(halfWidth, edge + 0.4, first, end, GRAVEL, 0);
    strip(-halfWidth + 0.15, -halfWidth + 0.3, first, end, PAINT, 0.012);
    strip(halfWidth - 0.3, halfWidth - 0.15, first, end, PAINT, 0.012);
    for (let i = first; i < end; i++)
      if (i % (DASH * 3) === 0)
        strip(-0.07, 0.07, i, Math.min(i + DASH, end), PAINT, 0.012);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.receiveShadow = true;
    this.group.add(mesh);
    this.chunks.set(c, mesh);
  }
}
