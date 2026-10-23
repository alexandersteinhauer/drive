import type { Road } from "./world/road";

/** CSS size of the map. */
const W = 240;
const H = 110;
/** Where the car sits, as a fraction of the height from the top: most of the map is road ahead. */
const CAR_Y = 0.78;
/** Metres of road ahead shown at rest, plus this many per m/s of speed. */
const RANGE = 260;
const RANGE_PER_SPEED = 7;
/** Draw every n-th road sample (2 m apart). */
const STRIDE = 2;

/**
 * Heading-up minimap at the bottom centre: the road's centreline around the car, so the
 * driver sees curves before they come into view. Zooms out with speed.
 */
export class Minimap {
  private readonly canvas = document.createElement("canvas");
  private readonly ctx: CanvasRenderingContext2D;
  private readonly dpr = Math.min(devicePixelRatio, 2);
  /** Road sample nearest the car, tracked locally from frame to frame. */
  private index = -1;
  private range = RANGE;

  constructor(
    private readonly road: Road,
    parent: HTMLElement,
  ) {
    this.canvas.id = "minimap";
    this.canvas.width = W * this.dpr;
    this.canvas.height = H * this.dpr;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    parent.append(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
  }

  get visible(): boolean {
    return !this.canvas.hidden;
  }

  set visible(on: boolean) {
    this.canvas.hidden = !on;
  }

  /**
   * Redraws around the car at (x, z), with the map's up along the horizontal direction
   * (dx, dz) (normalised by the caller). `speed` is in m/s.
   */
  update(
    x: number,
    z: number,
    dx: number,
    dz: number,
    speed: number,
    dt: number,
  ): void {
    if (this.canvas.hidden) return;
    const road = this.road;
    const n = road.finalCount;
    if (n < 2) return;
    this.track(x, z);

    const target = RANGE + RANGE_PER_SPEED * Math.abs(speed);
    this.range += (target - this.range) * (1 - Math.exp(-1.5 * dt));
    const cx = W / 2;
    const cy = H * CAR_Y;
    const s = (H * CAR_Y) / this.range;
    // Right of travel, in world XZ.
    const rx = -dz;
    const rz = dx;
    // Generous bounds: a sample is drawn while within the map's (rotated) rectangle plus margin.
    const reach = (Math.hypot(W, H) / s) * 0.75;

    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    let first = this.index;
    while (first > 0 && dist2(road, first, x, z) < reach * reach)
      first -= STRIDE;
    first = Math.max(0, first);
    ctx.beginPath();
    for (let i = first; i < n; i += STRIDE) {
      const px = road.xs[i]! - x;
      const pz = road.zs[i]! - z;
      const u = cx + (px * rx + pz * rz) * s;
      const v = cy - (px * dx + pz * dz) * s;
      if (i === first) ctx.moveTo(u, v);
      else ctx.lineTo(u, v);
      if (i > this.index && px * px + pz * pz > reach * reach) break;
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgb(0 0 0 / 35%)";
    ctx.lineWidth = 7;
    ctx.stroke();
    ctx.strokeStyle = "rgb(255 255 255 / 92%)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // The car: a white dot.
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "rgb(0 0 0 / 55%)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fill();
  }

  /** Walks the tracked index downhill in distance to the car; a full search when lost. */
  private track(x: number, z: number): void {
    const road = this.road;
    const last = road.finalCount - 1;
    if (this.index < 0 || this.index > last)
      this.index = road.nearestIndex(x, z);
    let i = this.index;
    let d = dist2(road, i, x, z);
    for (;;) {
      const back = i > 0 ? dist2(road, i - 1, x, z) : Infinity;
      const ahead = i < last ? dist2(road, i + 1, x, z) : Infinity;
      if (ahead < d) {
        i++;
        d = ahead;
      } else if (back < d) {
        i--;
        d = back;
      } else break;
    }
    // Far from where we were (a reset): search everything once.
    if (d > 200 * 200) i = road.nearestIndex(x, z);
    this.index = i;
  }
}

function dist2(road: Road, i: number, x: number, z: number): number {
  return (road.xs[i]! - x) ** 2 + (road.zs[i]! - z) ** 2;
}
