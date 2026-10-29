import type { Vector3 } from "three";

export interface RayHit {
  distance: number;
  normal: Vector3;
  /** Tire grip multiplier of the surface (1 = dry asphalt). */
  grip: number;
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
  /** Trees, rocks and the like standing on it, if it has any (see `Obstacles`). */
  obstaclesNear?: Obstacles["obstaclesNear"];
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
    out.grip = 1;
    return true;
  }

  height(): number {
    return this.level;
  }

  normal(_x: number, _z: number, out: Vector3): Vector3 {
    return out.set(0, 1, 0);
  }
}

/**
 * Something solid standing on the ground (a tree trunk, a rock): a vertical capsule, the
 * segment from (x, bottom, z) to (x, top, z) grown by `radius`.
 */
export interface Obstacle {
  x: number;
  z: number;
  bottom: number;
  top: number;
  radius: number;
}

/** A ground with obstacles on it. */
export interface Obstacles {
  /**
   * Fills `out` with the obstacles that come within `range` (m) of (x, z), horizontally,
   * and returns how many. `out` grows as needed; entries past the count are stale.
   */
  obstaclesNear(x: number, z: number, range: number, out: Obstacle[]): number;
}
