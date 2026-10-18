import { Quaternion, Vector3 } from "three";
import type { AxleConfig, CarConfig, CombustionConfig } from "./config";
import type { Ground, RayHit } from "./ground";
import { TireCurve, TireState } from "./tire";

const GRAVITY = 9.81;
const RPM = 30 / Math.PI; // rad/s -> rpm

export interface CarInput {
  throttle: number;
  brake: number;
  /** -1 full left, +1 full right. */
  steer: number;
  handbrake: number;
}

export class Wheel {
  /** Suspension top mount, relative to the centre of mass in body space. */
  readonly mount = new Vector3();
  readonly contactPoint = new Vector3();
  readonly normal = new Vector3(0, 1, 0);
  readonly tire = new TireState();
  contact = false;
  /** Current suspension length (mount to wheel centre). */
  length: number;
  compression = 0;
  load = 0;
  omega = 0;
  /** Accumulated rotation, for rendering. */
  spin = 0;
  steer = 0;
  driveTorque = 0;
  absFactor = 1;

  constructor(
    readonly axle: AxleConfig,
    /** -1 left, +1 right. */
    readonly side: number,
    mountY: number,
  ) {
    this.mount.set(side * axle.halfTrack, mountY, axle.z);
    this.length = axle.restLength;
  }

  reset(): void {
    this.tire.reset();
    this.contact = false;
    this.length = this.axle.restLength;
    this.compression = this.load = this.omega = this.driveTorque = 0;
    this.absFactor = 1;
  }
}

// Scratch vectors, reused every step to avoid allocation.
const up = new Vector3();
const fwd = new Vector3();
const right = new Vector3();
const p = new Vector3();
const r = new Vector3();
const f = new Vector3();
const vc = new Vector3();
const tf = new Vector3();
const tl = new Vector3();
const down = new Vector3();
const hit: RayHit = { distance: 0, normal: new Vector3() };
const qInv = new Quaternion();
const wb = new Vector3();
const lb = new Vector3();
const tb = new Vector3();

export class Car {
  readonly position = new Vector3();
  readonly orientation = new Quaternion();
  readonly velocity = new Vector3();
  readonly angularVelocity = new Vector3();
  /** State before the last step, for render interpolation. */
  readonly prevPosition = new Vector3();
  readonly prevOrientation = new Quaternion();

  readonly wheels: Wheel[];
  readonly curve: TireCurve;

  /** -1 reverse, 0 neutral, 1..n forward. Electric cars only use -1 and 1. */
  gear = 1;
  /** Engine or motor speed (rad/s). */
  engineOmega = 0;
  /** Signed forward speed (m/s). */
  speed = 0;

  private readonly force = new Vector3();
  private readonly torque = new Vector3();
  private readonly corners: Vector3[];
  private readonly invInertia: Vector3;
  private shiftTimer = 0;
  private targetGear = 1;
  private reverseTimer = 0;
  private sinceShift = 0;
  private reflectedInertia = 0;
  private tcsFactor = 1;
  private throttle = 0;

  constructor(
    readonly cfg: CarConfig,
    readonly ground: Ground,
  ) {
    const wheelbase = cfg.rear.z - cfg.front.z;
    const frontShare = cfg.rear.z / wheelbase;
    const mountY = (a: AxleConfig, share: number) => {
      const staticLoad = (cfg.mass * GRAVITY * share) / 2;
      return (
        a.wheelRadius - cfg.cgHeight + a.restLength - staticLoad / a.springRate
      );
    };
    const fy = mountY(cfg.front, frontShare);
    const ry = mountY(cfg.rear, 1 - frontShare);
    this.wheels = [
      new Wheel(cfg.front, -1, fy),
      new Wheel(cfg.front, 1, fy),
      new Wheel(cfg.rear, -1, ry),
      new Wheel(cfg.rear, 1, ry),
    ];
    this.curve = new TireCurve(cfg.tire.shapeC, cfg.tire.shapeE);
    this.invInertia = new Vector3(
      1 / cfg.inertia[0],
      1 / cfg.inertia[1],
      1 / cfg.inertia[2],
    );
    const [cx, cy, cz] = cfg.body.center;
    const [hx, hy, hz] = cfg.body.halfExtents;
    this.corners = [];
    for (const sx of [-1, 1])
      for (const sy of [-1, 1])
        for (const sz of [-1, 1])
          this.corners.push(
            new Vector3(cx + sx * hx, cy + sy * hy, cz + sz * hz),
          );
    this.reset(new Vector3(0, 0, 0), 0);
  }

  get rpm(): number {
    return this.engineOmega * RPM;
  }

  get wheelbase(): number {
    return this.cfg.rear.z - this.cfg.front.z;
  }

  /** Places the car at rest at ground position `at` facing `yaw` (radians, 0 = -Z). */
  reset(at: Vector3, yaw: number): void {
    this.position.set(
      at.x,
      this.ground.height(at.x, at.z) + this.cfg.cgHeight,
      at.z,
    );
    this.orientation.setFromAxisAngle(new Vector3(0, 1, 0), yaw);
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);
    this.prevPosition.copy(this.position);
    this.prevOrientation.copy(this.orientation);
    for (const w of this.wheels) w.reset();
    this.gear = this.targetGear = 1;
    this.shiftTimer = this.reverseTimer = 0;
    const pt = this.cfg.powertrain;
    this.engineOmega = pt.type === "combustion" ? pt.engine.idleRpm / RPM : 0;
  }

  step(dt: number, input: CarInput): void {
    const { cfg } = this;
    this.prevPosition.copy(this.position);
    this.prevOrientation.copy(this.orientation);

    up.set(0, 1, 0).applyQuaternion(this.orientation);
    fwd.set(0, 0, -1).applyQuaternion(this.orientation);
    right.set(1, 0, 0).applyQuaternion(this.orientation);
    this.speed = this.velocity.dot(fwd);

    const { throttle, brake } = this.selectDirection(dt, input);
    this.throttle = throttle;
    this.updateSteering(input.steer);

    this.force.set(0, -cfg.mass * GRAVITY, 0);
    this.torque.set(0, 0, 0);

    for (const w of this.wheels) this.updateSuspension(w, dt);
    this.antiRoll(this.wheels[0]!, this.wheels[1]!);
    this.antiRoll(this.wheels[2]!, this.wheels[3]!);
    for (const w of this.wheels) this.updateTire(w, dt);

    this.updateDrivetrain(dt, throttle);
    for (const w of this.wheels)
      this.updateWheelSpin(w, dt, brake, input.handbrake);

    // Aerodynamic drag.
    const v = this.velocity.length();
    this.force.addScaledVector(
      this.velocity,
      -0.5 * cfg.airDensity * cfg.dragArea * v,
    );

    this.bodyCollision();
    this.integrate(dt);
    this.updateGearbox(dt);
  }

  private selectDirection(
    dt: number,
    input: CarInput,
  ): { throttle: number; brake: number } {
    const stopped = Math.abs(this.speed) < 0.5;
    const wantReverse =
      this.gear >= 0 && input.brake > 0.1 && input.throttle < 0.1;
    const wantForward =
      this.gear < 0 && input.throttle > 0.1 && input.brake < 0.1;
    this.reverseTimer =
      stopped && (wantReverse || wantForward) ? this.reverseTimer + dt : 0;
    if (this.reverseTimer > 0.25) {
      this.gear = this.targetGear = wantReverse ? -1 : 1;
      this.shiftTimer = this.reverseTimer = 0;
    }
    return this.gear < 0
      ? { throttle: input.brake, brake: input.throttle }
      : { throttle: input.throttle, brake: input.brake };
  }

  private updateSteering(steer: number): void {
    const { lock, ackermann, limitAssist } = this.cfg.steering;
    const L = this.wheelbase;
    let hi = lock;
    let lo = -lock;
    if (limitAssist) {
      // Velocity of the front axle centre in body axes.
      r.set(0, 0, this.cfg.front.z).applyQuaternion(this.orientation);
      vc.crossVectors(this.angularVelocity, r).add(this.velocity);
      const vf = vc.dot(fwd);
      if (vf > 2) {
        const zeroSlip = clamp(Math.atan2(vc.dot(right), vf), -lock, lock);
        const t = this.cfg.tire;
        const window =
          (L * t.muY * GRAVITY) / (vf * vf) + 1.1 * Math.atan(t.peakSlipY);
        hi = clamp(Math.max(0, zeroSlip) + window, 0, lock);
        lo = clamp(Math.min(0, zeroSlip) - window, -lock, 0);
      }
    }
    const delta = steer >= 0 ? steer * hi : -steer * lo;
    const mag = Math.abs(delta);
    for (const w of this.wheels) {
      if (!w.axle.steered) continue;
      if (mag < 1e-5) {
        w.steer = 0;
        continue;
      }
      const radius = L / Math.tan(mag);
      const inner = w.side === Math.sign(delta);
      const ideal = Math.atan(
        L / (radius + (inner ? -1 : 1) * w.axle.halfTrack),
      );
      w.steer = Math.sign(delta) * (mag + ackermann * (ideal - mag));
    }
  }

  private updateSuspension(w: Wheel, dt: number): void {
    const a = w.axle;
    p.copy(w.mount).applyQuaternion(this.orientation).add(this.position);
    down.copy(up).negate();
    const prev = w.compression;
    w.contact = this.ground.raycast(p, down, a.restLength + a.wheelRadius, hit);
    if (!w.contact) {
      w.length = a.restLength;
      w.compression = 0;
      w.load = 0;
      return;
    }
    w.length = Math.max(0, hit.distance - a.wheelRadius);
    w.compression = a.restLength - (hit.distance - a.wheelRadius);
    w.contactPoint.copy(p).addScaledVector(down, hit.distance);
    w.normal.copy(hit.normal);
    const rate = (w.compression - prev) / dt;
    let load =
      a.springRate * w.compression +
      rate * (rate > 0 ? a.bumpDamping : a.reboundDamping);
    if (w.compression > a.bumpStop)
      load += 10 * a.springRate * (w.compression - a.bumpStop);
    w.load = Math.max(0, load) * Math.max(0, up.dot(w.normal));
  }

  private antiRoll(left: Wheel, rightWheel: Wheel): void {
    const k = left.axle.antiRoll * (left.compression - rightWheel.compression);
    if (left.contact) left.load = Math.max(0, left.load + k);
    if (rightWheel.contact) rightWheel.load = Math.max(0, rightWheel.load - k);
  }

  private updateTire(w: Wheel, dt: number): void {
    if (!w.contact) {
      w.tire.relax(dt);
      return;
    }
    const n = w.normal;
    // Wheel heading, projected onto the ground plane.
    tf.copy(fwd)
      .multiplyScalar(Math.cos(w.steer))
      .addScaledVector(right, Math.sin(w.steer));
    tf.addScaledVector(n, -n.dot(tf)).normalize();
    tl.crossVectors(tf, n);

    r.subVectors(w.contactPoint, this.position);
    vc.crossVectors(this.angularVelocity, r).add(this.velocity);
    w.tire.update(
      this.cfg.tire,
      this.curve,
      w.load,
      vc.dot(tf),
      vc.dot(tl),
      w.omega * w.axle.wheelRadius,
      dt,
    );

    f.copy(n)
      .multiplyScalar(w.load)
      .addScaledVector(tf, w.tire.fx)
      .addScaledVector(tl, w.tire.fy);
    this.applyForce(f, r);
  }

  private gearRatio(pt: CombustionConfig, gear: number): number {
    const g = pt.gearbox;
    if (gear > 0) return g.ratios[gear - 1]! * g.final;
    if (gear < 0) return -g.reverse * g.final;
    return 0;
  }

  private engineTorque(
    pt: CombustionConfig,
    omega: number,
    pedal: number,
  ): number {
    const e = pt.engine;
    const rpm = omega * RPM;
    const governor =
      e.idleThrottle * clamp((1.3 * e.idleRpm - rpm) / (0.3 * e.idleRpm), 0, 1);
    // Progressive pedal map: small inputs give proportionally more torque, like a real throttle body.
    const open = 1 - (1 - pedal) * (1 - pedal);
    const t = rpm > e.limiterRpm ? 0 : governor + open * (1 - governor);
    const friction = e.frictionTorque + e.frictionPerRpm * rpm;
    return t * interpolate(e.torqueCurve, rpm) - (1 - t) * friction;
  }

  private updateDrivetrain(dt: number, throttle: number): void {
    const driven = this.wheels.filter((w) => w.axle.driven);
    if (this.cfg.tcs) {
      const spin = Math.max(...driven.map((w) => (w.contact ? w.tire.kx : 0)));
      const slipping = spin > 1.5 * this.cfg.tire.peakSlipX && this.speed > -1;
      this.tcsFactor = clamp(this.tcsFactor + (slipping ? -8 : 4) * dt, 0, 1);
      throttle *= this.tcsFactor;
    }
    const pt = this.cfg.powertrain;
    const total =
      pt.type === "electric"
        ? this.electricTorque(driven, throttle)
        : this.combustionTorque(pt, driven, dt, throttle);
    this.splitDiff(driven, total, dt);
  }

  /** Returns the total torque at the driven wheels. The motor is rigidly geared to them. */
  private electricTorque(driven: Wheel[], throttle: number): number {
    const m = this.cfg.powertrain as Extract<
      CarConfig["powertrain"],
      { type: "electric" }
    >;
    const dir = this.gear < 0 ? -1 : 1;
    const avgOmega = driven.reduce((s, w) => s + w.omega, 0) / driven.length;
    // Motor speed, positive when turning in the selected direction.
    const omega = avgOmega * m.ratio * dir;
    this.engineOmega = Math.abs(omega);
    this.reflectedInertia = m.inertia * m.ratio * m.ratio;
    const limit = (torque: number, power: number) =>
      Math.min(torque, power / Math.max(Math.abs(omega), 1));
    const drive =
      omega * RPM > m.maxRpm ? 0 : throttle * limit(m.maxTorque, m.maxPower);
    // Regen fades out near standstill so the car rolls to a stop instead of reversing.
    const regen =
      (1 - throttle) *
      limit(m.regenTorque, m.regenPower) *
      smoothstep(0, 60, omega);
    return (drive - regen) * m.ratio * dir * m.efficiency;
  }

  /**
   * Engine and clutch. The clutch torque that keeps engine and wheels locked together is
   * solved for implicitly; if it exceeds the clutch capacity, the clutch slips at capacity.
   * Returns the total torque at the driven wheels.
   */
  private combustionTorque(
    pt: CombustionConfig,
    driven: Wheel[],
    dt: number,
    throttle: number,
  ): number {
    const e = pt.engine;
    const n = driven.length;
    const shifting = this.shiftTimer > 0;
    const ratio = shifting ? 0 : this.gearRatio(pt, this.gear);
    const eff = pt.gearbox.efficiency;
    const idle = e.idleRpm / RPM;
    const avgOmega = driven.reduce((s, w) => s + w.omega, 0) / n;
    const engineTorque = this.engineTorque(
      pt,
      this.engineOmega,
      shifting ? 0 : throttle,
    );
    this.reflectedInertia = 0;

    let clutch = 0;
    if (ratio !== 0) {
      const engagement = smoothstep(
        e.engageRpm / RPM,
        e.launchRpm / RPM,
        this.engineOmega,
      );
      const capacity = engagement * e.clutchTorque;
      const wheelInertia = driven.reduce((s, w) => s + w.axle.wheelInertia, 0);
      // External torque on the driven wheels, from the tires.
      const external = driven.reduce(
        (s, w) => s - w.tire.fx * w.axle.wheelRadius,
        0,
      );
      const mismatch = this.engineOmega - ratio * avgOmega;
      const lockTorque =
        (mismatch / dt +
          engineTorque / e.inertia -
          (ratio * external) / wheelInertia) /
        (1 / e.inertia + (eff * ratio * ratio) / wheelInertia);
      clutch = clamp(lockTorque, -capacity, capacity);
    }

    if (shifting) {
      // Rev-match towards the next gear while the clutch is open.
      const target = Math.max(
        idle,
        avgOmega * this.gearRatio(pt, this.targetGear),
      );
      this.engineOmega += (target - this.engineOmega) * Math.min(1, dt * 25);
    } else {
      this.engineOmega += ((engineTorque - clutch) / e.inertia) * dt;
    }
    this.engineOmega = clamp(
      this.engineOmega,
      idle,
      (e.limiterRpm * 1.05) / RPM,
    );
    return clutch * ratio * eff;
  }

  private splitDiff(driven: Wheel[], total: number, dt: number): void {
    if (driven.length !== 2) {
      for (const w of driven) w.driveTorque = total / driven.length;
      return;
    }
    const [l, rw] = driven as [Wheel, Wheel];
    const d = this.cfg.diff;
    const dOmega = l.omega - rw.omega;
    const powered = total * (this.gear < 0 ? -1 : 1) >= 0;
    const capacity =
      d.preload + (powered ? d.powerRamp : d.coastRamp) * Math.abs(total);
    const inertia = l.axle.wheelInertia + this.reflectedInertia / 2;
    const lock =
      Math.sign(dOmega) *
      Math.min(capacity, (Math.abs(dOmega) * inertia) / (2 * dt));
    l.driveTorque = total / 2 - lock;
    rw.driveTorque = total / 2 + lock;
  }

  private updateWheelSpin(
    w: Wheel,
    dt: number,
    brake: number,
    handbrake: number,
  ): void {
    const a = w.axle;
    const inertia = a.wheelInertia + (a.driven ? this.reflectedInertia / 2 : 0);
    w.omega += ((w.driveTorque - w.tire.fx * a.wheelRadius) / inertia) * dt;

    if (this.cfg.abs) {
      const locking =
        w.contact &&
        brake > 0 &&
        w.tire.kx < -1.2 * this.cfg.tire.peakSlipX &&
        Math.abs(this.speed) > 2;
      w.absFactor = clamp(w.absFactor + (locking ? -12 : 8) * dt, 0.3, 1);
    }
    let friction =
      a.brakeTorque * brake * w.absFactor +
      this.cfg.tire.rollingResistance * w.load * a.wheelRadius;
    if (!a.steered) friction += this.cfg.handbrakeTorque * handbrake;
    const dw = (friction * dt) / inertia;
    w.omega = Math.abs(w.omega) <= dw ? 0 : w.omega - Math.sign(w.omega) * dw;
    w.spin += w.omega * dt;
  }

  /** Penalty contacts for the body box corners, for crashes and rollovers. */
  private bodyCollision(): void {
    for (const c of this.corners) {
      r.copy(c).applyQuaternion(this.orientation);
      p.copy(r).add(this.position);
      const depth = this.ground.height(p.x, p.z) - p.y;
      if (depth <= 0) continue;
      const n = this.ground.normal(p.x, p.z, hit.normal);
      vc.crossVectors(this.angularVelocity, r).add(this.velocity);
      const vn = vc.dot(n);
      const fn = Math.max(0, 150000 * depth - 8000 * vn);
      f.copy(vc).addScaledVector(n, -vn); // tangential velocity
      const vt = f.length();
      if (vt > 1e-6) f.multiplyScalar(-Math.min(0.6 * fn, 4000 * vt) / vt);
      f.addScaledVector(n, fn);
      this.applyForce(f, r);
    }
  }

  private applyForce(force: Vector3, arm: Vector3): void {
    this.force.add(force);
    this.torque.x += arm.y * force.z - arm.z * force.y;
    this.torque.y += arm.z * force.x - arm.x * force.z;
    this.torque.z += arm.x * force.y - arm.y * force.x;
  }

  private integrate(dt: number): void {
    this.velocity.addScaledVector(this.force, dt / this.cfg.mass);
    this.position.addScaledVector(this.velocity, dt);

    // Angular dynamics in body space: I dw/dt = tau - w x Iw.
    const [ix, iy, iz] = this.cfg.inertia;
    qInv.copy(this.orientation).invert();
    wb.copy(this.angularVelocity).applyQuaternion(qInv);
    tb.copy(this.torque).applyQuaternion(qInv);
    lb.set(wb.x * ix, wb.y * iy, wb.z * iz);
    tb.sub(r.crossVectors(wb, lb));
    wb.addScaledVector(tb.multiply(this.invInertia), dt);
    this.angularVelocity.copy(wb).applyQuaternion(this.orientation);

    const w = this.angularVelocity;
    const q = this.orientation;
    const h = dt / 2;
    const { x, y, z, w: qw } = q;
    q.set(
      x + h * (w.x * qw + w.y * z - w.z * y),
      y + h * (w.y * qw + w.z * x - w.x * z),
      z + h * (w.z * qw + w.x * y - w.y * x),
      qw - h * (w.x * x + w.y * y + w.z * z),
    ).normalize();
  }

  private updateGearbox(dt: number): void {
    const pt = this.cfg.powertrain;
    if (pt.type !== "combustion") return;
    const g = pt.gearbox;
    this.sinceShift += dt;
    if (this.shiftTimer > 0) {
      this.shiftTimer -= dt;
      if (this.shiftTimer <= 0) this.gear = this.targetGear;
      return;
    }
    if (this.gear < 1 || this.sinceShift < 0.6) return;
    const driven = this.wheels.filter((w) => w.axle.driven);
    const avgOmega = driven.reduce((s, w) => s + w.omega, 0) / driven.length;
    const rpmIn = (gear: number) => avgOmega * this.gearRatio(pt, gear) * RPM;
    const t = this.throttle;
    const upRpm = g.shiftUpRpmLight + (g.shiftUpRpm - g.shiftUpRpmLight) * t;
    const downRpm =
      g.shiftDownRpmLight + (g.shiftDownRpm - g.shiftDownRpmLight) * t;
    let next = this.gear;
    if (rpmIn(this.gear) > upRpm && this.gear < g.ratios.length)
      next = this.gear + 1;
    else if (
      this.gear > 1 &&
      rpmIn(this.gear) < downRpm &&
      rpmIn(this.gear - 1) < 0.95 * g.shiftUpRpm
    )
      next = this.gear - 1;
    if (next !== this.gear) {
      this.targetGear = next;
      this.shiftTimer = g.shiftTime;
      this.sinceShift = 0;
    }
  }
}

function smoothstep(lo: number, hi: number, v: number): number {
  const t = clamp((v - lo) / (hi - lo), 0, 1);
  return t * t * (3 - 2 * t);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function interpolate(table: [number, number][], x: number): number {
  const first = table[0]!;
  if (x <= first[0]) return first[1];
  for (let i = 1; i < table.length; i++) {
    const [x1, y1] = table[i]!;
    if (x <= x1) {
      const [x0, y0] = table[i - 1]!;
      return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
  }
  return table[table.length - 1]![1];
}
