import { describe, expect, test } from "bun:test";
import { Vector3 } from "three";
import { Car, type CarInput } from "./car";
import { electricCoupe, sportsCoupe, type CarConfig } from "./config";
import { FlatGround } from "./ground";

const DT = 1 / 480;
const idle: CarInput = { throttle: 0, brake: 0, steer: 0, handbrake: 0 };

function makeCar(cfg: CarConfig = sportsCoupe): Car {
  return new Car(structuredClone(cfg), new FlatGround());
}

/** Runs until `until` returns true or `maxTime` passes; returns elapsed time. */
function run(
  car: Car,
  maxTime: number,
  input: (t: number) => CarInput,
  until?: (t: number) => boolean,
): number {
  let t = 0;
  while (t < maxTime) {
    car.step(DT, input(t));
    t += DT;
    if (
      !Number.isFinite(car.position.x + car.velocity.x + car.angularVelocity.y)
    )
      throw new Error(`non-finite state at t=${t.toFixed(3)}`);
    if (until?.(t)) break;
  }
  return t;
}

const kmh = (car: Car) => car.speed * 3.6;
const upright = (car: Car) =>
  new Vector3(0, 1, 0).applyQuaternion(car.orientation).y;

function accelerateTo(car: Car, target: number): number {
  return run(
    car,
    60,
    () => ({ ...idle, throttle: 1 }),
    () => kmh(car) >= target,
  );
}

describe("car", () => {
  test("settles at rest without creeping", () => {
    const car = makeCar();
    run(car, 4, () => idle);
    const start = car.position.clone();
    run(car, 4, () => idle);
    console.log(
      `rest: height=${car.position.y.toFixed(3)} drift=${car.position.distanceTo(start).toExponential(2)} m`,
    );
    expect(car.position.distanceTo(start)).toBeLessThan(0.01);
    expect(car.velocity.length()).toBeLessThan(0.01);
    expect(Math.abs(car.position.y - sportsCoupe.cgHeight)).toBeLessThan(0.03);
    for (const w of car.wheels) expect(w.contact).toBe(true);
  });

  test("0-100 km/h and top speed are plausible", () => {
    const car = makeCar();
    run(car, 2, () => idle);
    const t100 = accelerateTo(car, 100);
    const lateral = Math.abs(car.position.x);
    run(car, 60, () => ({ ...idle, throttle: 1 }));
    console.log(
      `0-100: ${t100.toFixed(2)} s, top: ${kmh(car).toFixed(0)} km/h, gear ${car.gear}, ${car.rpm.toFixed(0)} rpm, drift ${lateral.toFixed(3)} m`,
    );
    expect(t100).toBeGreaterThan(4);
    expect(t100).toBeLessThan(7.5);
    expect(kmh(car)).toBeGreaterThan(220);
    expect(kmh(car)).toBeLessThan(300);
    expect(lateral).toBeLessThan(0.5);
  });

  test("brakes from 100 km/h in a straight line", () => {
    const car = makeCar();
    accelerateTo(car, 100);
    const start = car.position.clone();
    const t = run(
      car,
      20,
      () => ({ ...idle, brake: 1 }),
      () => car.speed < 0.1,
    );
    const distance = car.position.distanceTo(start);
    console.log(
      `100-0: ${distance.toFixed(1)} m in ${t.toFixed(2)} s, heading drift ${car.angularVelocity.y.toFixed(4)}`,
    );
    expect(distance).toBeGreaterThan(30);
    expect(distance).toBeLessThan(45);
  });

  test("holds a steady corner near the grip limit", () => {
    const car = makeCar();
    accelerateTo(car, 70);
    // Hold ~70 km/h with a simple speed controller and a fixed steering input.
    let sumA = 0;
    let n = 0;
    run(car, 12, (t) => {
      const err = 70 - kmh(car);
      if (t > 6) {
        sumA += car.speed * car.angularVelocity.y;
        n++;
      }
      return {
        ...idle,
        steer: 0.45,
        throttle: clamp01(0.3 + err * 0.1),
        brake: clamp01(-err * 0.05),
      };
    });
    const g = Math.abs(sumA / n) / 9.81;
    console.log(
      `steady corner: ${g.toFixed(2)} g at ${kmh(car).toFixed(0)} km/h, roll up.y=${upright(car).toFixed(3)}`,
    );
    expect(g).toBeGreaterThan(0.6);
    expect(g).toBeLessThan(1.2);
    expect(upright(car)).toBeGreaterThan(0.95);
  });

  test("survives a handbrake slide and stays upright", () => {
    const car = makeCar();
    accelerateTo(car, 80);
    run(car, 1.5, () => ({ ...idle, steer: 1, handbrake: 1 }));
    run(
      car,
      8,
      () => ({ ...idle, brake: 1 }),
      () => Math.abs(car.speed) < 0.3,
    );
    console.log(
      `after slide: speed ${kmh(car).toFixed(1)} km/h, up.y=${upright(car).toFixed(3)}`,
    );
    expect(upright(car)).toBeGreaterThan(0.95);
    expect(Math.abs(car.speed)).toBeLessThan(1);
  });

  test("power oversteer can be provoked from a corner", () => {
    const cfg = structuredClone(sportsCoupe);
    cfg.esc.enabled = false;
    const car = makeCar(cfg);
    accelerateTo(car, 50);
    car.gear = 2;
    let maxSlip = 0;
    run(
      car,
      3,
      () => ({ ...idle, steer: 0.6, throttle: 1 }),
      () => {
        const side = car.velocity
          .clone()
          .applyQuaternion(car.orientation.clone().invert());
        maxSlip = Math.max(maxSlip, Math.abs(Math.atan2(side.x, -side.z)));
        return false;
      },
    );
    console.log(
      `power oversteer: max body slip ${((maxSlip * 180) / Math.PI).toFixed(1)} deg`,
    );
    expect(maxSlip).toBeGreaterThan((5 * Math.PI) / 180);
  });

  test("reverses when braking from standstill", () => {
    const car = makeCar();
    run(car, 2, () => idle);
    run(car, 4, () => ({ ...idle, brake: 1 }));
    console.log(`reverse: gear ${car.gear}, speed ${kmh(car).toFixed(1)} km/h`);
    expect(car.gear).toBe(-1);
    expect(car.speed).toBeLessThan(-3);
  });
});

/** Full braking from 100 km/h with a steering input; returns distance, lock fraction and how far the car turned. */
function brakeTest(abs: boolean, steer: number) {
  const cfg = structuredClone(electricCoupe);
  cfg.abs = abs;
  // Isolate ABS from stability control.
  cfg.esc.enabled = false;
  const car = makeCar(cfg);
  accelerateTo(car, 100);
  const start = car.position.clone();
  let locked = 0;
  let fast = 0;
  let turned = 0;
  run(
    car,
    20,
    () => ({ ...idle, brake: 1, steer }),
    () => {
      if (car.speed > 3) {
        fast++;
        if (car.wheels.some((w) => w.omega === 0)) locked++;
      }
      // How far the car turns while braking: locked fronts can't steer it.
      turned += Math.abs(car.angularVelocity.y) * DT;
      return car.speed < 0.3;
    },
  );
  return {
    distance: car.position.distanceTo(start),
    locked: locked / fast,
    turned: (turned * 180) / Math.PI,
  };
}

describe("abs", () => {
  test("stops shorter than locked wheels", () => {
    const on = brakeTest(true, 0);
    const off = brakeTest(false, 0);
    console.log(
      `abs straight: ${on.distance.toFixed(1)} m vs ${off.distance.toFixed(1)} m locked`,
    );
    expect(on.locked).toBe(0);
    expect(off.locked).toBeGreaterThan(0.5);
    expect(on.distance).toBeLessThan(off.distance - 3);
  });

  test("keeps the car steerable while braking in a turn", () => {
    const on = brakeTest(true, 0.5);
    const off = brakeTest(false, 0.5);
    console.log(
      `abs turning: car turned ${on.turned.toFixed(0)} deg vs ${off.turned.toFixed(0)} deg locked`,
    );
    expect(on.locked).toBe(0);
    expect(on.turned).toBeGreaterThan(1.5 * off.turned);
  });
});

/**
 * A quick keyboard lane change at 100 km/h, off the throttle: full right for 0.5 s, full left
 * for 0.5 s, then straight. Keyboard steering ramps like Input does. Returns max body slip in degrees.
 */
function laneChange(cfg: CarConfig, esc: boolean): number {
  const config = structuredClone(cfg);
  config.esc.enabled = esc;
  const car = makeCar(config);
  accelerateTo(car, 100);
  let steer = 0;
  let maxSlip = 0;
  run(car, 4, (t) => {
    const target = t < 0.5 ? 1 : t < 1 ? -1 : 0;
    const rate = (target > steer ? steer < 0 : steer > 0) ? 6 : 3.5;
    steer =
      target > steer
        ? Math.min(target, steer + rate * DT)
        : Math.max(target, steer - rate * DT);
    const v = car.velocity
      .clone()
      .applyQuaternion(car.orientation.clone().invert());
    if (car.speed > 5)
      maxSlip = Math.max(maxSlip, Math.abs(Math.atan2(v.x, -v.z)));
    return { ...idle, steer };
  });
  return (maxSlip * 180) / Math.PI;
}

describe("stability control", () => {
  for (const [name, cfg] of [
    ["electric", electricCoupe],
    ["petrol", sportsCoupe],
  ] as const)
    test(`keeps the ${name} car straight through a lane change that spins it without`, () => {
      const on = laneChange(cfg, true);
      const off = laneChange(cfg, false);
      console.log(
        `esc lane change (${name}): body slip ${on.toFixed(1)} deg vs ${off.toFixed(1)} deg without`,
      );
      expect(off).toBeGreaterThan(30);
      expect(on).toBeLessThan(8);
    });
});

describe("electric car", () => {
  test("settles at rest without creeping", () => {
    const car = makeCar(electricCoupe);
    run(car, 4, () => idle);
    const start = car.position.clone();
    run(car, 4, () => idle);
    expect(car.position.distanceTo(start)).toBeLessThan(0.01);
    expect(Math.abs(car.position.y - electricCoupe.cgHeight)).toBeLessThan(
      0.03,
    );
  });

  test("accelerates without gears to a motor-limited top speed", () => {
    const car = makeCar(electricCoupe);
    const t100 = accelerateTo(car, 100);
    run(car, 60, () => ({ ...idle, throttle: 1 }));
    console.log(
      `ev 0-100: ${t100.toFixed(2)} s, top: ${kmh(car).toFixed(0)} km/h, ${car.rpm.toFixed(0)} rpm`,
    );
    expect(car.gear).toBe(1);
    expect(t100).toBeGreaterThan(4);
    expect(t100).toBeLessThan(7);
    expect(kmh(car)).toBeGreaterThan(200);
    expect(kmh(car)).toBeLessThan(225);
  });

  test("lift-off regen slows the car and fades out near standstill", () => {
    const car = makeCar(electricCoupe);
    accelerateTo(car, 100);
    const v0 = car.speed;
    run(car, 1, () => idle);
    const decel = (v0 - car.speed) / 9.81;
    run(
      car,
      60,
      () => idle,
      () => car.speed < 0.2,
    );
    run(car, 3, () => idle);
    console.log(
      `ev regen: ${decel.toFixed(2)} g at 100 km/h, final speed ${car.speed.toFixed(3)} m/s`,
    );
    expect(decel).toBeGreaterThan(0.1);
    expect(decel).toBeLessThan(0.3);
    expect(car.speed).toBeGreaterThanOrEqual(-0.05);
    expect(car.speed).toBeLessThan(0.2);
  });

  test("brakes and reverses", () => {
    const car = makeCar(electricCoupe);
    accelerateTo(car, 100);
    const start = car.position.clone();
    run(
      car,
      20,
      () => ({ ...idle, brake: 1 }),
      () => car.speed < 0.1,
    );
    const distance = car.position.distanceTo(start);
    run(car, 4, () => ({ ...idle, brake: 1 }));
    console.log(
      `ev 100-0: ${distance.toFixed(1)} m, reverse ${kmh(car).toFixed(1)} km/h`,
    );
    expect(distance).toBeLessThan(45);
    expect(car.gear).toBe(-1);
    expect(car.speed).toBeLessThan(-3);
  });
});

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}
