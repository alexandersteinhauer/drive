import * as THREE from "three/webgpu";
import { Surface, type Landscape, type LandSample } from "./world/landscape";
import { hash2 } from "./world/noise";
import { terrainMaterial } from "./terrainMaterial";

const TILE = 128;
const RADIUS = 1400;
/** Vertex spacing per level of detail, and the distance (tile centre to camera) each is used up to. */
const LODS = [
  { segments: 64, within: 300 },
  { segments: 32, within: 700 },
  { segments: 16, within: Infinity },
];
const TREES_WITHIN = 900;
const SKIRT = 6;
/** Terrain under the road is pushed down so it never pokes through the road mesh. */
const UNDER_ROAD = 0.3;

interface Tile {
  key: string;
  lod: number;
  mesh: THREE.Mesh;
  trees?: THREE.InstancedMesh[];
}

const sample: LandSample = {
  height: 0,
  surface: Surface.Grass,
  roadDistance: Infinity,
};

/** Streams terrain tiles (with skirts to hide LOD cracks) and trees around a focus point. */
export class TerrainView {
  readonly group = new THREE.Group();
  private readonly tiles = new Map<string, Tile>();
  private readonly material: THREE.Material;
  private readonly trunkGeo = new THREE.CylinderGeometry(
    0.18,
    0.28,
    2.4,
    6,
  ).translate(0, 1.2, 0);
  private readonly crownGeo = new THREE.ConeGeometry(1.7, 5.5, 7).translate(
    0,
    4.6,
    0,
  );
  private readonly trunkMat = new THREE.MeshStandardMaterial({
    color: 0x5a4632,
    roughness: 1,
  });
  private readonly crownMat = new THREE.MeshStandardMaterial({
    color: 0x2f5a2a,
    roughness: 1,
    flatShading: true,
  });

  constructor(private readonly land: Landscape) {
    this.material = terrainMaterial(land.road.edge);
  }

  /** Builds missing tiles nearest-first until `budgetMs` is spent; drops far ones. */
  update(focus: THREE.Vector3, budgetMs: number): void {
    const start = performance.now();
    const fx = Math.floor(focus.x / TILE);
    const fz = Math.floor(focus.z / TILE);
    const reach = Math.ceil(RADIUS / TILE);
    const wanted: {
      key: string;
      tx: number;
      tz: number;
      lod: number;
      dist: number;
    }[] = [];
    for (let tx = fx - reach; tx <= fx + reach; tx++)
      for (let tz = fz - reach; tz <= fz + reach; tz++) {
        const dist = Math.hypot(
          (tx + 0.5) * TILE - focus.x,
          (tz + 0.5) * TILE - focus.z,
        );
        if (dist > RADIUS) continue;
        // Only build where the road can no longer change: its future lies further along -Z.
        if (-tz * TILE + 64 > this.land.road.finalProgress) continue;
        const lod = LODS.findIndex((l) => dist <= l.within);
        wanted.push({ key: `${tx},${tz}`, tx, tz, lod, dist });
      }
    const keep = new Set(wanted.map((w) => w.key));
    for (const [key, tile] of this.tiles)
      if (!keep.has(key)) this.drop(key, tile);

    wanted.sort((a, b) => a.dist - b.dist);
    for (const w of wanted) {
      const existing = this.tiles.get(w.key);
      if (existing && existing.lod === w.lod) continue;
      if (performance.now() - start > budgetMs) break;
      if (existing) this.drop(w.key, existing);
      this.build(w.key, w.tx, w.tz, w.lod, w.dist);
    }
  }

  private drop(key: string, tile: Tile): void {
    this.group.remove(tile.mesh);
    tile.mesh.geometry.dispose();
    for (const t of tile.trees ?? []) {
      this.group.remove(t);
      t.dispose();
    }
    this.tiles.delete(key);
  }

  private build(
    key: string,
    tx: number,
    tz: number,
    lod: number,
    dist: number,
  ): void {
    const n = LODS[lod]!.segments;
    const step = TILE / n;
    const x0 = tx * TILE;
    const z0 = tz * TILE;
    // Heights on an (n + 3)^2 grid with a one-sample border, for normals that match across tiles.
    const g = n + 3;
    const heights = new Float32Array(g * g);
    const nearRoad = new Float32Array(g * g);
    for (let j = 0; j < g; j++)
      for (let i = 0; i < g; i++) {
        this.land.sample(x0 + (i - 1) * step, z0 + (j - 1) * step, sample);
        heights[j * g + i] = sample.height;
        nearRoad[j * g + i] = sample.roadDistance;
      }

    const v = n + 1;
    const ring = 4 * n;
    const count = v * v + ring * 2;
    const pos = new Float32Array(count * 3);
    const nor = new Float32Array(count * 3);
    const variation = new Float32Array(count);
    const roadDist = new Float32Array(count);
    const edge = this.land.road.edge;
    const put = (k: number, i: number, j: number, drop: number) => {
      const h = heights[(j + 1) * g + i + 1]!;
      const hl = heights[(j + 1) * g + i]!;
      const hr = heights[(j + 1) * g + i + 2]!;
      const hd = heights[j * g + i + 1]!;
      const hu = heights[(j + 2) * g + i + 1]!;
      const road = nearRoad[(j + 1) * g + i + 1]!;
      pos[k * 3] = x0 + i * step;
      // Sunk under the road surface, rising to exactly road height at the shoulder edge,
      // where the road mesh ends and the terrain (with its own gravel verge) takes over.
      const under =
        UNDER_ROAD * (1 - Math.min(1, Math.max(0, (road - edge + 1.2) / 1.2)));
      pos[k * 3 + 1] = h - drop - under;
      pos[k * 3 + 2] = z0 + j * step;
      const nx = -(hr - hl);
      const ny = 2 * step;
      const nz = -(hu - hd);
      const len = Math.hypot(nx, ny, nz);
      nor[k * 3] = nx / len;
      nor[k * 3 + 1] = ny / len;
      nor[k * 3 + 2] = nz / len;
      variation[k] = variationAt(x0 + i * step, z0 + j * step);
      roadDist[k] = Math.min(road, 100);
    };
    for (let j = 0; j < v; j++)
      for (let i = 0; i < v; i++) put(j * v + i, i, j, 0);

    const index: number[] = [];
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const a = j * v + i;
        index.push(a, a + v, a + 1, a + 1, a + v, a + v + 1);
      }
    // Skirts: the boundary loop duplicated and dropped, stitched to the edge.
    const loop: [number, number][] = [];
    for (let i = 0; i < n; i++) loop.push([i, 0]);
    for (let j = 0; j < n; j++) loop.push([n, j]);
    for (let i = n; i > 0; i--) loop.push([i, n]);
    for (let j = n; j > 0; j--) loop.push([0, j]);
    const base = v * v;
    loop.forEach(([i, j], k) => {
      put(base + k, i, j, 0);
      put(base + ring + k, i, j, SKIRT);
    });
    for (let k = 0; k < ring; k++) {
      const a = base + k;
      const b = base + ((k + 1) % ring);
      const c = base + ring + k;
      const d = base + ring + ((k + 1) % ring);
      index.push(a, b, c, b, d, c);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    geo.setAttribute("variation", new THREE.BufferAttribute(variation, 1));
    geo.setAttribute("road", new THREE.BufferAttribute(roadDist, 1));
    geo.setIndex(index);
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    this.group.add(mesh);
    const tile: Tile = { key, lod, mesh };
    if (dist <= TREES_WITHIN) tile.trees = this.plantTrees(tx, tz);
    this.tiles.set(key, tile);
  }

  private plantTrees(tx: number, tz: number): THREE.InstancedMesh[] {
    const spacing = 8;
    const per = TILE / spacing;
    const matrices: THREE.Matrix4[] = [];
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const p = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const n = new THREE.Vector3();
    for (let j = 0; j < per; j++)
      for (let i = 0; i < per; i++) {
        const gx = tx * per + i;
        const gz = tz * per + j;
        const x = (gx + hash2(gx, gz, 11)) * spacing;
        const z = (gz + hash2(gx, gz, 12)) * spacing;
        // Clumpy forests: density from a low-frequency field.
        const density =
          0.5 +
          0.5 *
            Math.sin(x * 0.006 + 1.3) *
            Math.sin(z * 0.0047 + Math.sin(x * 0.002) * 2);
        if (hash2(gx, gz, 13) > density * density * 0.9) continue;
        this.land.sample(x, z, sample);
        if (sample.roadDistance < this.land.road.edge + 7) continue;
        if (this.land.normal(x, z, n).y < 0.8) continue;
        const size = 0.7 + hash2(gx, gz, 14) * 0.7;
        q.setFromAxisAngle(up, hash2(gx, gz, 15) * Math.PI * 2);
        m.compose(p.set(x, sample.height - 0.2, z), q, s.setScalar(size));
        matrices.push(m.clone());
      }
    if (matrices.length === 0) return [];
    return [
      [this.trunkGeo, this.trunkMat],
      [this.crownGeo, this.crownMat],
    ].map(([geo, mat]) => {
      const inst = new THREE.InstancedMesh(
        geo as THREE.BufferGeometry,
        mat as THREE.Material,
        matrices.length,
      );
      matrices.forEach((mm, k) => inst.setMatrixAt(k, mm));
      inst.castShadow = true;
      this.group.add(inst);
      return inst;
    });
  }
}

/** Large dry/green patches across the meadows (0 = lush, 1 = dry), plus a little noise. */
function variationAt(x: number, z: number): number {
  const dry = 0.5 + 0.5 * Math.sin(x * 0.011 + Math.sin(z * 0.007) * 3);
  return Math.min(1, dry * 0.9 + hash2(x, z, 3) * 0.1);
}
