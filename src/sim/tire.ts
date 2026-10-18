/**
 * Tire model: normalized combined-slip Pacejka "magic formula" with
 * relaxation-length slip dynamics.
 *
 * Slip states are kept in the direction of the force they produce:
 *  - kx: longitudinal slip ratio, positive when the wheel spins faster than the ground (drive)
 *  - ky: lateral slip (tan of slip angle), positive when the force points to the wheel's right
 *
 * Relaxation lengths make the slips lag the kinematics by the distance travelled,
 * which keeps the model stable and well defined down to zero speed. Below a small
 * speed an extra damping term stops the tire "spring" from oscillating at standstill.
 */
export interface TireConfig {
  /** Peak friction coefficients at nominal load. */
  muX: number;
  muY: number;
  /** Slip ratio and lateral slip (tan of angle) at which peak force occurs. */
  peakSlipX: number;
  peakSlipY: number;
  /** Magic formula shape; sliding friction tends to sin(C * pi / 2) of peak. */
  shapeC: number;
  shapeE: number;
  /** Relaxation lengths in metres. */
  relaxX: number;
  relaxY: number;
  /** Fractional loss of mu per unit of load above nominal. */
  loadSensitivity: number;
  nominalLoad: number;
  /** Low speed damping, in slip units per m/s of slip velocity. */
  lowSpeedDampingX: number;
  lowSpeedDampingY: number;
  /** Contact speed (m/s) above which low speed damping is fully off. */
  lowSpeedThreshold: number;
  rollingResistance: number;
}

const MAX_NORMALIZED_SLIP = 25;

export class TireCurve {
  readonly b: number;

  constructor(
    readonly c: number,
    readonly e: number,
  ) {
    // Solve for B so the curve peaks at normalized slip 1.
    const target = Math.tan(Math.PI / (2 * c));
    let lo = 0;
    let hi = 100;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (mid - e * (mid - Math.atan(mid)) < target) lo = mid;
      else hi = mid;
    }
    this.b = (lo + hi) / 2;
  }

  /** Normalized force (peak 1 at rho = 1) for normalized slip rho >= 0. */
  evaluate(rho: number): number {
    const x = this.b * rho;
    return Math.sin(this.c * Math.atan(x - this.e * (x - Math.atan(x))));
  }

  /** f(rho) / rho, finite at rho = 0. */
  ratio(rho: number): number {
    return rho < 1e-6 ? this.c * this.b : this.evaluate(rho) / rho;
  }
}

export class TireState {
  kx = 0;
  ky = 0;
  fx = 0;
  fy = 0;

  reset(): void {
    this.kx = this.ky = this.fx = this.fy = 0;
  }

  /** Lets the slip states decay while the wheel is in the air. */
  relax(dt: number): void {
    const decay = Math.exp(-dt * 20);
    this.kx *= decay;
    this.ky *= decay;
    this.fx = this.fy = 0;
  }

  /**
   * Advances slip states and computes forces.
   * @param vx contact patch velocity along the wheel heading (m/s)
   * @param vy contact patch velocity to the wheel's right (m/s)
   * @param wheelSpeed omega * radius (m/s)
   */
  update(
    cfg: TireConfig,
    curve: TireCurve,
    load: number,
    /** Surface grip multiplier. */
    grip: number,
    vx: number,
    vy: number,
    wheelSpeed: number,
    dt: number,
  ): void {
    const ax = Math.abs(vx);
    const slipX = wheelSpeed - vx;
    if (ax > 1e-4) {
      const tx = slipX / ax;
      const ty = -vy / ax;
      this.kx = tx + (this.kx - tx) * Math.exp((-ax * dt) / cfg.relaxX);
      this.ky = ty + (this.ky - ty) * Math.exp((-ax * dt) / cfg.relaxY);
    } else {
      this.kx += (slipX * dt) / cfg.relaxX;
      this.ky -= (vy * dt) / cfg.relaxY;
    }
    const n = Math.hypot(this.kx / cfg.peakSlipX, this.ky / cfg.peakSlipY);
    if (n > MAX_NORMALIZED_SLIP) {
      const s = MAX_NORMALIZED_SLIP / n;
      this.kx *= s;
      this.ky *= s;
    }

    const blend = Math.max(0, 1 - Math.hypot(vx, vy) / cfg.lowSpeedThreshold);
    const nx = (this.kx + blend * cfg.lowSpeedDampingX * slipX) / cfg.peakSlipX;
    const ny = (this.ky - blend * cfg.lowSpeedDampingY * vy) / cfg.peakSlipY;
    const rho = Math.hypot(nx, ny);

    const loadFactor = Math.min(
      1.3,
      Math.max(0.6, 1 - cfg.loadSensitivity * (load / cfg.nominalLoad - 1)),
    );
    const scale = load * grip * loadFactor * curve.ratio(rho);
    this.fx = cfg.muX * scale * nx;
    this.fy = cfg.muY * scale * ny;
  }
}
