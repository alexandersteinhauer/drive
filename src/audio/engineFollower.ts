import type { EngineSim } from "../sim/engineSim";

const RPM = 30 / Math.PI;

/**
 * Runs a simulated engine alongside the car for its sound, held to the car's engine speed
 * by a dynamometer (so it works against it at the car's throttle: load and overrun sound
 * different). With the clutch out for a gear change it revs down on its own.
 */
export class EngineFollower {
  private throttle = 0;
  private dynoRpm: number;
  private overrun = 0;
  private stepDebt = 0;

  constructor(
    readonly sim: EngineSim,
    idleRpm: number,
  ) {
    sim.speed = idleRpm / RPM;
    sim.ignition = true;
    this.dynoRpm = idleRpm;
  }

  /** Simulates `dt` (s) towards the car's rpm and throttle; returns the exhaust signal. */
  advance(
    dt: number,
    rpm: number,
    throttle: number,
    engaged: boolean,
  ): { samples: Float32Array[]; overrun: number } {
    // A foot, not a switch.
    const tau = throttle < this.throttle ? 0.05 : 0.12;
    this.throttle += (throttle - this.throttle) * (1 - Math.exp(-dt / tau));
    this.sim.throttle = this.throttle;

    // The clutch lets go once the foot is (nearly) off, so the revs don't flare; running
    // free, the engine must not sink below the car's (idle) speed.
    const engineRpm = this.sim.speed * RPM;
    const hold = engaged || this.throttle > 0.1 || engineRpm < rpm;
    if (!hold) this.dynoRpm = engineRpm;
    else if (engaged) {
      // The clutch takes up over a few tenths of a second; small changes pass at once.
      const max = 8000 * dt;
      this.dynoRpm += Math.max(-max, Math.min(max, rpm - this.dynoRpm));
    }
    this.sim.dyno(hold, Math.max(0, this.dynoRpm) / RPM);

    if (hold) {
      // Follows the airflow rather than the pedal (plate gamma 2).
      const s = Math.min(1, Math.max(0, this.throttle));
      this.overrun = 1 - Math.sin((s * s * Math.PI) / 2);
    } else this.overrun *= Math.exp(-dt / 0.2);

    this.stepDebt += dt * this.sim.frequency;
    const steps = Math.floor(this.stepDebt);
    this.stepDebt -= steps;
    this.sim.step(steps);
    return { samples: this.sim.takeAudio(), overrun: this.overrun };
  }
}
