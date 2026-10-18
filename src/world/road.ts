import { rng } from "./noise";

/**
 * An endless road, generated forward in fixed-spacing samples.
 *
 * Plan view: a sequence of straights and circular arcs joined by clothoid transitions
 * (curvature ramps linearly), like real road design. The heading stays within
 * `maxHeading` of the main direction (-Z), so the road always progresses and
 * never loops back on itself.
 *
 * Profile: terrain height along the road, grade-limited, then Gaussian-smoothed.
 * Curves are banked. Samples become "final" (queryable) once the smoothing window
 * ahead of them exists.
 */
export interface RoadConfig {
  seed: number;
  spacing: number;
  halfWidth: number;
  shoulder: number;
  minRadius: number;
  maxRadius: number;
  maxHeading: number;
  maxGrade: number;
  /** Gaussian smoothing of the height profile, in samples. */
  smoothing: number;
  /** Bank slope per unit curvature, and its limit. */
  bankFactor: number;
  maxBank: number;
}

export const defaultRoad: RoadConfig = {
  seed: 7,
  spacing: 2,
  halfWidth: 3.5,
  shoulder: 1.0,
  minRadius: 90,
  maxRadius: 700,
  maxHeading: (55 * Math.PI) / 180,
  maxGrade: 0.08,
  smoothing: 15,
  bankFactor: 6,
  maxBank: 0.06,
};

export interface RoadHit {
  /** Distance from the centreline. */
  distance: number;
  /** Signed offset from the centreline, positive to the right of travel. */
  lateral: number;
  /** Centreline height and cross slope (height change per metre to the right). */
  height: number;
  slope: number;
  /** Fractional sample index. */
  index: number;
}

interface Piece {
  length: number;
  k0: number;
  k1: number;
}

const CELL = 64;
const cellKey = (cx: number, cz: number) => cx * 1_000_003 + cz;

export class Road {
  readonly xs: number[] = [];
  readonly zs: number[] = [];
  readonly headings: number[] = [];
  readonly curvatures: number[] = [];
  /** Final heights and cross slopes, valid for indices < finalCount. */
  readonly heights: number[] = [];
  readonly slopes: number[] = [];
  finalCount = 0;

  private readonly clamped: number[] = [];
  private readonly random: () => number;
  private readonly pieces: Piece[] = [];
  private pieceS = 0;
  private lastWasCurve = true;
  private readonly cells = new Map<number, number[]>();
  private readonly weights: number[];

  constructor(
    readonly cfg: RoadConfig,
    /** Terrain height before the road is cut in. */
    private readonly terrainHeight: (x: number, z: number) => number,
  ) {
    this.random = rng(cfg.seed);
    const w = Math.ceil(cfg.smoothing * 3);
    this.weights = Array.from({ length: 2 * w + 1 }, (_, i) =>
      Math.exp(-(((i - w) / cfg.smoothing) ** 2) / 2),
    );
    // Start with a straight so the car spawns somewhere calm.
    this.pieces.push({ length: 120, k0: 0, k1: 0 });
    this.push(0, 0, 0, 0);
  }

  get edge(): number {
    return this.cfg.halfWidth + this.cfg.shoulder;
  }

  /** Progress (distance along -Z) of the last final sample; nothing final lies beyond the road's future. */
  get finalProgress(): number {
    return this.finalCount > 0 ? -this.zs[this.finalCount - 1]! : 0;
  }

  /** Generates until final samples reach `progress` along -Z. */
  ensure(progress: number): void {
    while (this.finalProgress < progress) this.extend();
  }

  private extend(): void {
    const { spacing } = this.cfg;
    const i = this.xs.length - 1;
    const k0 = this.curvatures[i]!;
    const k1 = this.curvatureAt(spacing);
    const km = (k0 + k1) / 2;
    const h0 = this.headings[i]!;
    const hm = h0 + (km * spacing) / 2;
    this.push(
      this.xs[i]! + Math.sin(hm) * spacing,
      this.zs[i]! - Math.cos(hm) * spacing,
      h0 + km * spacing,
      k1,
    );
    this.finalize();
  }

  private push(x: number, z: number, heading: number, curvature: number): void {
    const i = this.xs.length;
    this.xs.push(x);
    this.zs.push(z);
    this.headings.push(heading);
    this.curvatures.push(curvature);
    const raw = this.terrainHeight(x, z);
    const step = this.cfg.maxGrade * this.cfg.spacing;
    const prev = i > 0 ? this.clamped[i - 1]! : raw;
    this.clamped.push(Math.min(prev + step, Math.max(prev - step, raw)));
  }

  /** Advances along the plan and returns the curvature there. */
  private curvatureAt(advance: number): number {
    this.pieceS += advance;
    while (this.pieces.length === 0 || this.pieceS > this.pieces[0]!.length) {
      if (this.pieces.length > 0) this.pieceS -= this.pieces.shift()!.length;
      if (this.pieces.length === 0) this.plan();
    }
    const p = this.pieces[0]!;
    return p.k0 + ((p.k1 - p.k0) * this.pieceS) / p.length;
  }

  private plan(): void {
    const r = this.random;
    const { maxHeading, minRadius, maxRadius } = this.cfg;
    const heading = this.headings[this.headings.length - 1]!;
    if (this.lastWasCurve && r() < 0.6) {
      this.pieces.push({ length: 20 + r() * 220, k0: 0, k1: 0 });
      this.lastWasCurve = false;
      return;
    }
    // Turn back towards the main direction more often the further off it we are.
    let sign = r() < 0.5 + (0.5 * heading) / maxHeading ? -1 : 1;
    let room = sign > 0 ? maxHeading - heading : maxHeading + heading;
    if (room < 0.15) {
      sign = -sign;
      room = sign > 0 ? maxHeading - heading : maxHeading + heading;
    }
    const radius = minRadius * Math.pow(maxRadius / minRadius, r() ** 1.3);
    const angle = Math.min(room, 0.25 + r() * 1.05);
    const k = sign / radius;
    let transition = Math.min(90, Math.max(30, radius * 0.3));
    let arc = angle * radius - transition;
    if (arc < 0) {
      transition = angle * radius;
      arc = 0;
    }
    this.pieces.push({ length: transition, k0: 0, k1: k });
    if (arc > 0) this.pieces.push({ length: arc, k0: k, k1: k });
    this.pieces.push({ length: transition, k0: k, k1: 0 });
    this.lastWasCurve = true;
  }

  private finalize(): void {
    const w = (this.weights.length - 1) / 2;
    const { bankFactor, maxBank } = this.cfg;
    while (this.finalCount + w < this.xs.length) {
      const i = this.finalCount;
      let sum = 0;
      let norm = 0;
      for (let j = -w; j <= w; j++) {
        if (i + j < 0) continue;
        const wt = this.weights[j + w]!;
        sum += wt * this.clamped[i + j]!;
        norm += wt;
      }
      this.heights.push(sum / norm);
      // Bank into the curve: a right-hand curve (positive curvature) lowers the right side.
      this.slopes.push(
        -Math.max(
          -maxBank,
          Math.min(maxBank, this.curvatures[i]! * bankFactor),
        ),
      );
      this.finalCount++;
      if (i > 0) this.index(i - 1);
    }
  }

  /** Adds segment i -> i+1 to the spatial grid. */
  private index(i: number): void {
    const x0 = Math.floor(Math.min(this.xs[i]!, this.xs[i + 1]!) / CELL);
    const x1 = Math.floor(Math.max(this.xs[i]!, this.xs[i + 1]!) / CELL);
    const z0 = Math.floor(Math.min(this.zs[i]!, this.zs[i + 1]!) / CELL);
    const z1 = Math.floor(Math.max(this.zs[i]!, this.zs[i + 1]!) / CELL);
    for (let cx = x0; cx <= x1; cx++)
      for (let cz = z0; cz <= z1; cz++) {
        const key = cellKey(cx, cz);
        const list = this.cells.get(key);
        if (list) list.push(i);
        else this.cells.set(key, [i]);
      }
  }

  /** Nearest point on the final road within `maxDistance` (up to one grid cell, 64 m). */
  query(x: number, z: number, maxDistance: number, out: RoadHit): boolean {
    const cx = Math.floor(x / CELL);
    const cz = Math.floor(z / CELL);
    const xs = this.xs;
    const zs = this.zs;
    let best = maxDistance * maxDistance;
    let bi = -1;
    let bt = 0;
    for (let dx = -1; dx <= 1; dx++)
      for (let dz = -1; dz <= 1; dz++) {
        const list = this.cells.get(cellKey(cx + dx, cz + dz));
        if (!list) continue;
        for (const i of list) {
          const ax = xs[i]!;
          const az = zs[i]!;
          const sx = xs[i + 1]! - ax;
          const sz = zs[i + 1]! - az;
          const t = Math.min(
            1,
            Math.max(0, ((x - ax) * sx + (z - az) * sz) / (sx * sx + sz * sz)),
          );
          const px = x - ax - sx * t;
          const pz = z - az - sz * t;
          const d = px * px + pz * pz;
          if (d < best) {
            best = d;
            bi = i;
            bt = t;
          }
        }
      }
    if (bi < 0) return false;
    const sx = xs[bi + 1]! - xs[bi]!;
    const sz = zs[bi + 1]! - zs[bi]!;
    out.distance = Math.sqrt(best);
    out.lateral =
      ((x - xs[bi]!) * -sz + (z - zs[bi]!) * sx) / Math.hypot(sx, sz);
    out.height =
      this.heights[bi]! + (this.heights[bi + 1]! - this.heights[bi]!) * bt;
    out.slope =
      this.slopes[bi]! + (this.slopes[bi + 1]! - this.slopes[bi]!) * bt;
    out.index = bi + bt;
    return true;
  }

  /** Index of the final sample nearest to (x, z), searching everything; for spawning. */
  nearestIndex(x: number, z: number): number {
    let best = Infinity;
    let bi = 0;
    for (let i = 0; i < this.finalCount; i++) {
      const d = (this.xs[i]! - x) ** 2 + (this.zs[i]! - z) ** 2;
      if (d < best) {
        best = d;
        bi = i;
      }
    }
    return bi;
  }
}
