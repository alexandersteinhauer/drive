import type { Obstacle } from "../sim/ground";
import type { Landscape } from "./landscape";
import {
  FIRST_ROCK,
  PLANT_STEP,
  ROCK_STEP,
  plantAt,
  rocksAt,
  type Plant,
} from "./plants";

/**
 * What the car can crash into: tree trunks and rocks, from the same placement functions
 * the vegetation is drawn from (plants.ts). Bushes are driven through.
 *
 * Trunk radius at the foot of each tree kind (m, at scale 1), measured from the models;
 * they stand as capsules up to well above the car.
 */
const TRUNK = [0.23, 0.21, 0.33, 0.31, 0.28, 0.22, 0.23, 0.24];
const TRUNK_HEIGHT = 6;
/**
 * Rocks (kinds FIRST_ROCK..), at scale 1: the centre of their footprint (model x, z),
 * their mean half-width and height above their origin, from the models' bounds. A rock is
 * a capsule of that radius whose top is the rock's.
 */
const ROCK: [number, number, number, number][] = [
  [-0.11, 0.34, 1.45, 1.99],
  [-0.18, 0.08, 1.3, 1.85],
  [-0.11, 0.85, 1.55, 2.0],
];
/** Widest obstacle there is (m): a cell's obstacles stay within this of the cell. */
const MAX_RADIUS = 1.75 * 2.2;
/** Cells cached at most; past that the cache starts over. */
const MAX_CELLS = 50_000;

const plant: Plant = {
  x: 0,
  y: 0,
  z: 0,
  yaw: 0,
  scale: 1,
  kind: 0,
  tree: true,
};
const NONE: readonly Obstacle[] = [];

export class ObstacleField {
  /** Obstacles per tree cell and per rock cell, by cell key. */
  private readonly trees = new Map<number, readonly Obstacle[]>();
  private readonly rocks = new Map<number, readonly Obstacle[]>();

  constructor(private readonly land: Landscape) {}

  near(x: number, z: number, range: number, out: Obstacle[]): number {
    let n = 0;
    // Trees stand within a cell of their grid point (jittered by up to a cell).
    const t = range + 1;
    for (
      let gz = Math.floor((z - t) / PLANT_STEP) - 1;
      gz <= Math.floor((z + t) / PLANT_STEP);
      gz++
    )
      for (
        let gx = Math.floor((x - t) / PLANT_STEP) - 1;
        gx <= Math.floor((x + t) / PLANT_STEP);
        gx++
      )
        n = take(this.treeCell(gx, gz), x, z, range, out, n);
    // A rock group spreads up to ~6 m from its centre, which is anywhere in its cell.
    const r = range + 6 + MAX_RADIUS;
    for (
      let gz = Math.floor((z - r) / ROCK_STEP);
      gz <= Math.floor((z + r) / ROCK_STEP);
      gz++
    )
      for (
        let gx = Math.floor((x - r) / ROCK_STEP);
        gx <= Math.floor((x + r) / ROCK_STEP);
        gx++
      )
        n = take(this.rockCell(gx, gz), x, z, range, out, n);
    return n;
  }

  private treeCell(gx: number, gz: number): readonly Obstacle[] {
    const key = gx * 2 ** 24 + gz;
    let list = this.trees.get(key);
    if (!list) {
      if (this.trees.size >= MAX_CELLS) this.trees.clear();
      list = this.treeAt(gx, gz);
      this.trees.set(key, list);
    }
    return list;
  }

  private rockCell(gx: number, gz: number): readonly Obstacle[] {
    const key = gx * 2 ** 24 + gz;
    let list = this.rocks.get(key);
    if (!list) {
      if (this.rocks.size >= MAX_CELLS) this.rocks.clear();
      list = this.rocksAt(gx, gz);
      this.rocks.set(key, list);
    }
    return list;
  }

  private treeAt(gx: number, gz: number): readonly Obstacle[] {
    if (!plantAt(this.land, gx, gz, true, plant)) return NONE;
    const s = plant.scale;
    return [
      {
        x: plant.x,
        z: plant.z,
        bottom: plant.y - 1,
        top: plant.y + TRUNK_HEIGHT * s,
        radius: TRUNK[plant.kind]! * s,
      },
    ];
  }

  private rocksAt(gx: number, gz: number): readonly Obstacle[] {
    const list: Obstacle[] = [];
    rocksAt(this.land, gx, gz, (x, y, z, yaw, s, kind) => {
      const [cx, cz, half, height] = ROCK[kind - FIRST_ROCK]!;
      const c = Math.cos(yaw);
      const n = Math.sin(yaw);
      const radius = half * s;
      const top = y + height * s;
      list.push({
        x: x + (c * cx + n * cz) * s,
        z: z + (-n * cx + c * cz) * s,
        bottom: Math.min(y, top - radius) - 1,
        top: top - radius,
        radius,
      });
    });
    return list.length ? list : NONE;
  }
}

/** Appends the obstacles of `list` within `range` of (x, z) to `out` from `n`; returns the new count. */
function take(
  list: readonly Obstacle[],
  x: number,
  z: number,
  range: number,
  out: Obstacle[],
  n: number,
): number {
  for (const o of list) {
    const reach = range + o.radius;
    const dx = o.x - x;
    const dz = o.z - z;
    if (dx * dx + dz * dz <= reach * reach) out[n++] = o;
  }
  return n;
}
