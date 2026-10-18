import type { Vector3 } from "three";

export interface RayHit {
  distance: number;
  normal: Vector3;
}

/** Anything the car can drive on. Heights are along +Y. */
export interface Ground {
  /** Casts a ray; on hit fills `out` and returns true. */
  raycast(
    origin: Vector3,
    dir: Vector3,
    maxDistance: number,
    out: RayHit,
  ): boolean;
  height(x: number, z: number): number;
  normal(x: number, z: number, out: Vector3): Vector3;
}

export class FlatGround implements Ground {
  constructor(readonly level = 0) {}

  raycast(
    origin: Vector3,
    dir: Vector3,
    maxDistance: number,
    out: RayHit,
  ): boolean {
    if (dir.y >= -1e-6) return false;
    const distance = Math.max(0, (origin.y - this.level) / -dir.y);
    if (distance > maxDistance) return false;
    out.distance = distance;
    out.normal.set(0, 1, 0);
    return true;
  }

  height(): number {
    return this.level;
  }

  normal(_x: number, _z: number, out: Vector3): Vector3 {
    return out.set(0, 1, 0);
  }
}
