import type { MappedEngineConfig, SimulatedEngineConfig } from "./config";
import { EngineSim } from "./engineSim";

const RPM = 30 / Math.PI; // rad/s -> rpm

/** The driven side of the clutch, seen from the crank. */
export interface DriveLoad {
  /** Inertia of the driven wheels through the gearing (kg m^2). */
  inertia: number;
  /** Speed the gearing turns the clutch at (rad/s). */
  speed: number;
  /** Torque on the driven side from the tyres, through the gearing (N m). */
  torque: number;
  /** Torque the clutch can pass now (N m). */
  capacity: number;
}

/** A combustion engine on the car's side of the clutch. */
export interface Powerplant {
  /** Crank speed (rad/s). */
  readonly omega: number;
  /**
   * Advances `dt` at `throttle` (pedal, 0..1) against `load` (null: clutch open), and returns
   * the clutch torque passed to the driven side (N m at the crank). While the clutch is out
   * for a shift, `revTarget` is the crank speed the next gear will turn.
   */
  step(
    dt: number,
    throttle: number,
    load: DriveLoad | null,
    revTarget: number | null,
  ): number;
  /** Back to idle (after putting the car back on the road). */
  reset(): void;
  dispose(): void;
}

/**
 * An engine from its torque curve: the clutch torque that keeps engine and wheels locked
 * together is solved for implicitly; beyond the clutch's capacity it slips at capacity.
 */
export class MappedEngine implements Powerplant {
  omega: number;

  constructor(readonly cfg: MappedEngineConfig) {
    this.omega = cfg.idleRpm / RPM;
  }

  /** Torque at the crank, net of friction. */
  private torque(pedal: number): number {
    const e = this.cfg;
    const rpm = this.omega * RPM;
    const governor =
      e.idleThrottle * clamp((1.3 * e.idleRpm - rpm) / (0.3 * e.idleRpm), 0, 1);
    // Progressive pedal map: small inputs give proportionally more torque, like a real throttle body.
    const open = 1 - (1 - pedal) * (1 - pedal);
    const t = rpm > e.limiterRpm ? 0 : governor + open * (1 - governor);
    const friction = e.frictionTorque + e.frictionPerRpm * rpm;
    return t * interpolate(e.torqueCurve, rpm) - (1 - t) * friction;
  }

  step(
    dt: number,
    throttle: number,
    load: DriveLoad | null,
    revTarget: number | null,
  ): number {
    const e = this.cfg;
    const idle = e.idleRpm / RPM;
    const engineTorque = this.torque(throttle);
    let clutch = 0;
    if (load) {
      const mismatch = this.omega - load.speed;
      const lockTorque =
        (mismatch / dt +
          engineTorque / e.inertia -
          load.torque / load.inertia) /
        (1 / e.inertia + 1 / load.inertia);
      clutch = clamp(lockTorque, -load.capacity, load.capacity);
    }
    if (revTarget !== null) {
      // Rev-match towards the next gear while the clutch is open.
      const target = Math.max(idle, revTarget);
      this.omega += (target - this.omega) * Math.min(1, dt * 25);
    } else {
      this.omega += ((engineTorque - clutch) / e.inertia) * dt;
    }
    this.omega = clamp(this.omega, idle, (e.limiterRpm * 1.05) / RPM);
    return clutch;
  }

  reset(): void {
    this.omega = this.cfg.idleRpm / RPM;
  }

  dispose(): void {}
}

/** Engine speed below which it has stalled, and at which the starter lets go. */
const STALL_RPM = 300;
/** Idling time before a new engine is handed over (s). */
const SETTLE = 1.5;
const CATCH_RPM = 600;

/**
 * The simulated engine (engine/): stepped at its own rate against the load, with the clutch
 * solved inside its constraint solver. If it stalls, the driver turns the key: the starter
 * cranks it until it runs.
 */
export class SimulatedEngine implements Powerplant {
  readonly sim: EngineSim;
  private stepDebt = 0;
  private cranking = false;
  /** Throttle as the driver's foot has it (eased: a foot, not a switch). */
  private pedal = 0;

  constructor(readonly cfg: SimulatedEngineConfig) {
    this.sim = new EngineSim(cfg.spec);
    this.sim.ignition = true;
    this.reset();
    // Let it settle into its idle: just started, it stumbles for a second.
    this.sim.setLoad(1, 0, 0, 0);
    this.sim.step(Math.round(SETTLE * this.sim.frequency));
  }

  get omega(): number {
    return Math.max(0, this.sim.speed);
  }

  step(
    dt: number,
    throttle: number,
    load: DriveLoad | null,
    revTarget: number | null,
  ): number {
    const rpm = this.sim.speed * RPM;
    if (this.cranking) this.cranking = rpm < CATCH_RPM;
    else this.cranking = rpm < STALL_RPM;
    this.sim.starter = this.cranking;

    // Blip the throttle to match revs on a downshift.
    let target = throttle;
    if (revTarget !== null)
      target = clamp((revTarget * RPM - rpm) / 600, 0, 0.7);
    const tau = target < this.pedal ? 0.03 : 0.06;
    this.pedal += (target - this.pedal) * (1 - Math.exp(-dt / tau));
    this.sim.throttle = this.pedal;

    if (load)
      this.sim.setLoad(load.inertia, load.speed, load.torque, load.capacity);
    else this.sim.setLoad(1, 0, 0, 0);
    this.stepDebt += dt * this.sim.frequency;
    const steps = Math.floor(this.stepDebt);
    this.stepDebt -= steps;
    this.sim.step(steps);
    return load ? this.sim.clutchTorque : 0;
  }

  /** The pedal as the engine has it (eased), 0..1. */
  get throttle(): number {
    return this.pedal;
  }

  reset(): void {
    const idle = this.cfg.idleRpm / RPM;
    if (this.sim.speed < idle) this.sim.speed = idle;
  }

  dispose(): void {
    this.sim.dispose();
  }
}

export function createPowerplant(
  cfg: MappedEngineConfig | SimulatedEngineConfig,
): Powerplant {
  return cfg.type === "simulated"
    ? new SimulatedEngine(cfg)
    : new MappedEngine(cfg);
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
