import type { CarInput } from "./sim/car";

const STEER_IN = 3.5; // full lock in ~0.3 s
const STEER_OUT = 6;
const PEDAL_IN = 8;
const PEDAL_OUT = 10;
const DEADZONE = 0.08;

/** Keyboard (with smoothing) and gamepad, merged into one CarInput. */
export class Input {
  private readonly keys = new Set<string>();
  private readonly state: CarInput = {
    throttle: 0,
    brake: 0,
    steer: 0,
    handbrake: 0,
  };
  private readonly pressed = new Set<string>();
  private padResetHeld = false;

  constructor(target: Window) {
    target.addEventListener("keydown", (e) => {
      if (!e.repeat) this.pressed.add(e.code);
      this.keys.add(e.code);
    });
    target.addEventListener("keyup", (e) => this.keys.delete(e.code));
    target.addEventListener("blur", () => this.keys.clear());
  }

  /** Forgets held keys and unconsumed presses. */
  reset(): void {
    this.keys.clear();
    this.pressed.clear();
  }

  /** True while `code` is held. */
  held(code: string): boolean {
    return this.keys.has(code);
  }

  /** True once per press of `code` since the last call. */
  take(code: string): boolean {
    return this.pressed.delete(code);
  }

  update(dt: number): CarInput {
    const k = (...codes: string[]) => codes.some((c) => this.keys.has(c));
    const s = this.state;
    const steerTarget =
      (k("KeyD", "ArrowRight") ? 1 : 0) - (k("KeyA", "ArrowLeft") ? 1 : 0);
    const returning =
      steerTarget === 0 || Math.sign(steerTarget) !== Math.sign(s.steer);
    s.steer = approach(
      s.steer,
      steerTarget,
      (returning ? STEER_OUT : STEER_IN) * dt,
    );
    s.throttle = pedal(s.throttle, k("KeyW", "ArrowUp"), dt);
    s.brake = pedal(s.brake, k("KeyS", "ArrowDown"), dt);
    s.handbrake = k("Space") ? 1 : 0;

    const pad = navigator.getGamepads?.().find((g) => g?.connected);
    if (!pad) return s;
    const axis = pad.axes[0] ?? 0;
    const padSteer =
      Math.abs(axis) < DEADZONE
        ? 0
        : (Math.sign(axis) * (Math.abs(axis) - DEADZONE)) / (1 - DEADZONE);
    const button = (i: number) => pad.buttons[i]?.value ?? 0;
    const resetHeld = button(3) > 0.5;
    if (resetHeld && !this.padResetHeld) this.pressed.add("KeyR");
    this.padResetHeld = resetHeld;
    return {
      steer:
        steerTarget !== 0
          ? s.steer
          : Math.sign(padSteer) * Math.abs(padSteer) ** 1.4,
      throttle: Math.max(s.throttle, button(7)),
      brake: Math.max(s.brake, button(6)),
      handbrake: Math.max(s.handbrake, button(0), button(5)),
    };
  }
}

function approach(v: number, target: number, step: number): number {
  return v < target ? Math.min(target, v + step) : Math.max(target, v - step);
}

function pedal(v: number, held: boolean, dt: number): number {
  return approach(v, held ? 1 : 0, (held ? PEDAL_IN : PEDAL_OUT) * dt);
}
