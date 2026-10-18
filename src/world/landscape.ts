import type { Vector3 } from "three";
import type { Ground, RayHit } from "../sim/ground";
import { Simplex } from "./noise";
import { defaultRoad, Road, type RoadConfig, type RoadHit } from "./road";

export const Surface = { Asphalt: 0, Gravel: 1, Grass: 2 } as const;
export type Surface = (typeof Surface)[keyof typeof Surface];

/** Tire grip multiplier per surface. */
const GRIP = [1, 0.75, 0.6];
/** Embankments and cuttings blend into the hills over this range of distances past the road edge. */
const MIN_BLEND = 8;
const MAX_BLEND = 40;

export interface LandSample {
  height: number;
  surface: Surface;
  /** Distance from the road centreline (Infinity if far away). */
  roadDistance: number;
}

const roadHit: RoadHit = {
  distance: 0,
  lateral: 0,
  height: 0,
  slope: 0,
  index: 0,
};
const sampleOut: LandSample = {
  height: 0,
  surface: Surface.Grass,
  roadDistance: Infinity,
};

/** Rolling hills with an endless road cut into them. This is also the physics ground. */
export class Landscape implements Ground {
  readonly road: Road;
  private readonly noise: Simplex;

  constructor(seed = 1, roadConfig: RoadConfig = defaultRoad) {
    this.noise = new Simplex(seed);
    this.road = new Road(
      { ...roadConfig, seed: roadConfig.seed + seed },
      (x, z) => this.hills(x, z),
    );
  }

  /** Terrain before the road is cut in. */
  hills(x: number, z: number): number {
    const n = this.noise;
    return (
      60 * n.fbm(x / 1100, z / 1100, 3) +
      12 * n.fbm(x / 260 + 31.7, z / 260, 3) +
      1.2 * n.noise(x / 45, z / 45)
    );
  }

  sample(x: number, z: number, out: LandSample = sampleOut): LandSample {
    const hills = this.hills(x, z);
    const road = this.road;
    const edge = road.edge;
    out.roadDistance = Infinity;
    out.surface = Surface.Grass;
    out.height = hills;
    if (!road.query(x, z, edge + MAX_BLEND, roadHit)) return out;
    out.roadDistance = roadHit.distance;
    if (roadHit.distance <= edge) {
      out.height = roadHit.height + roadHit.lateral * roadHit.slope;
      out.surface =
        Math.abs(roadHit.lateral) <= road.cfg.halfWidth
          ? Surface.Asphalt
          : Surface.Gravel;
      return out;
    }
    const lateral = Math.max(-edge, Math.min(edge, roadHit.lateral));
    const edgeHeight = roadHit.height + lateral * roadHit.slope;
    const blend = Math.min(
      MAX_BLEND,
      MIN_BLEND + 1.5 * Math.abs(hills - edgeHeight),
    );
    const t = Math.min(1, (roadHit.distance - edge) / blend);
    out.height = edgeHeight + (hills - edgeHeight) * t * t * (3 - 2 * t);
    return out;
  }

  height(x: number, z: number): number {
    return this.sample(x, z).height;
  }

  normal(x: number, z: number, out: Vector3): Vector3 {
    const e = 0.25;
    const dx = this.height(x + e, z) - this.height(x - e, z);
    const dz = this.height(x, z + e) - this.height(x, z - e);
    return out.set(-dx, 2 * e, -dz).normalize();
  }

  raycast(
    origin: Vector3,
    dir: Vector3,
    maxDistance: number,
    out: RayHit,
  ): boolean {
    const gap = (t: number) =>
      origin.y +
      dir.y * t -
      this.height(origin.x + dir.x * t, origin.z + dir.z * t);
    if (dir.y >= -1e-3 || gap(maxDistance) > 0) return false;
    // The ray is close to vertical: step by the vertical gap until it closes.
    let t = 0;
    for (let i = 0; i < 6; i++) {
      const g = gap(t);
      if (Math.abs(g) < 1e-5) break;
      t = Math.min(maxDistance, Math.max(0, t + g / -dir.y));
    }
    const x = origin.x + dir.x * t;
    const z = origin.z + dir.z * t;
    out.distance = t;
    this.normal(x, z, out.normal);
    out.grip = GRIP[this.sample(x, z).surface]!;
    return true;
  }
}
