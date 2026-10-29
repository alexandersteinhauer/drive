import { describe, expect, test } from "bun:test";
import { Vector3 } from "three";
import { Car, FATAL_ENERGY, type CarInput } from "./car";
import { bmwI4, sportsCoupe, type CarConfig } from "./config";
import {
  FlatGround,
  type Obstacle,
  type Obstacles,
  type RayHit,
} from "./ground";

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

/** Flat ground with a different surface grip, like grass. */
class SlipperyGround extends FlatGround {
  constructor(readonly grip: number) {
    super();
  }
  override raycast(
    origin: Vector3,
    dir: Vector3,
    maxDistance: number,
    out: RayHit,
  ): boolean {
    const hit = super.raycast(origin, dir, maxDistance, out);
    out.grip = this.grip;
    return hit;
  }
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
  const cfg = structuredClone(bmwI4);
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

/**
 * Full throttle from a standstill while weaving with keyboard steering, full lock each way every
 * 0.6 s. Returns the final speed in km/h.
 */
function weave(cfg: CarConfig): number {
  const car = makeCar(cfg);
  run(car, 2, () => idle);
  let steer = 0;
  run(car, 10, (t) => {
    const target = Math.floor(t / 0.6) % 2 ? -1 : 1;
    const rate = Math.sign(target) !== Math.sign(steer) ? 6 : 3.5;
    steer =
      target > steer
        ? Math.min(target, steer + rate * DT)
        : Math.max(target, steer - rate * DT);
    return { ...idle, throttle: 1, steer };
  });
  return kmh(car);
}

describe("stability control", () => {
  for (const [name, cfg] of [
    ["electric", bmwI4],
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

  test("lets the car accelerate while weaving at full throttle", () => {
    // Tight low-speed turns have large body slip at the centre of mass without any sliding.
    const speed = weave(bmwI4);
    console.log(`esc weave: ${speed.toFixed(0)} km/h after 10 s`);
    expect(speed).toBeGreaterThan(50);
  });
});

/** Seconds to 20 km/h from a standstill at full throttle, steering held at `steer`. */
function launch(tcs: boolean, grip: number, steer: number): number {
  const cfg = structuredClone(bmwI4);
  cfg.tcs = tcs;
  const car = new Car(cfg, new SlipperyGround(grip));
  run(car, 2, () => ({ ...idle, handbrake: 1 }));
  let s = 0;
  return run(
    car,
    10,
    () => {
      s = Math.min(steer, s + 3.5 * DT);
      return { ...idle, throttle: 1, steer: s };
    },
    () => kmh(car) >= 20,
  );
}

describe("traction control", () => {
  // Full lock on grass is left out: it is grip-limited below 20 km/h.
  for (const [surface, grip, steer] of [
    ["asphalt", 1, 0],
    ["asphalt", 1, 0.5],
    ["asphalt", 1, 1],
    ["grass", 0.6, 0],
    ["grass", 0.6, 0.5],
  ] as const)
    test(`does not slow a launch on ${surface} with steering ${steer}`, () => {
      const on = launch(true, grip, steer);
      const off = launch(false, grip, steer);
      console.log(
        `tcs launch ${surface} steer ${steer}: 0-20 km/h in ${on.toFixed(2)} s vs ${off.toFixed(2)} s without`,
      );
      expect(on).toBeLessThan(off * 1.05);
    });
});

describe("bmw i4 edrive40", () => {
  test("settles at rest without creeping", () => {
    const car = makeCar(bmwI4);
    run(car, 4, () => idle);
    const start = car.position.clone();
    run(car, 4, () => idle);
    expect(car.position.distanceTo(start)).toBeLessThan(0.01);
    expect(Math.abs(car.position.y - bmwI4.cgHeight)).toBeLessThan(0.03);
  });

  test("matches the published acceleration and limited top speed", () => {
    const car = makeCar(bmwI4);
    run(car, 2, () => idle);
    const t100 = accelerateTo(car, 100);
    run(car, 60, () => ({ ...idle, throttle: 1 }));
    console.log(
      `i4 0-100: ${t100.toFixed(2)} s (BMW: 5.7), top: ${kmh(car).toFixed(1)} km/h (BMW: 190, limited), ${car.rpm.toFixed(0)} rpm`,
    );
    expect(car.gear).toBe(1);
    expect(t100).toBeGreaterThan(5.5);
    expect(t100).toBeLessThan(6.5);
    expect(kmh(car)).toBeGreaterThan(187);
    expect(kmh(car)).toBeLessThan(191);
  });

  test("lift-off regen slows the car and fades out near standstill", () => {
    const car = makeCar(bmwI4);
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
      `i4 regen: ${decel.toFixed(2)} g at 100 km/h, final speed ${car.speed.toFixed(3)} m/s`,
    );
    expect(decel).toBeGreaterThan(0.1);
    expect(decel).toBeLessThan(0.3);
    expect(car.speed).toBeGreaterThanOrEqual(-0.05);
    expect(car.speed).toBeLessThan(0.2);
  });

  test("brakes and reverses", () => {
    const car = makeCar(bmwI4);
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
      `i4 100-0: ${distance.toFixed(1)} m, reverse ${kmh(car).toFixed(1)} km/h`,
    );
    expect(distance).toBeLessThan(45);
    expect(car.gear).toBe(-1);
    expect(car.speed).toBeLessThan(-3);
  });
});

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/** Isolate TCS from stability control and differential locking. */
function tcsCar(cfg: CarConfig = bmwI4, ground = new FlatGround()): Car {
  const config = structuredClone(cfg);
  config.esc.enabled = false;
  config.diff = { preload: 0, powerRamp: 0, coastRamp: 0 };
  return new Car(config, ground);
}

function tractionLaunchSlip(cfg: CarConfig, dir: -1 | 1, tcs: boolean): number {
  const car = tcsCar(cfg, new SlipperyGround(0.2));
  car.cfg.tcs = tcs;
  car.gear = dir;
  run(car, 1, () => ({ ...idle, [dir > 0 ? "throttle" : "brake"]: 1 }));
  expect(dir * car.speed).toBeGreaterThan(0.5);
  const driven = car.wheels.filter((w) => w.axle.driven);
  for (const w of driven) expect(w.tcsTorque).toBe(0);
  return Math.max(
    ...driven.map((w) => dir * (w.omega * w.axle.wheelRadius - w.vx)),
  );
}

describe("traction control direction and wheel-local slip", () => {
  for (const dir of [1, -1] as const) {
    test(`brakes the overspinning wheel, not the slower wheel, in ${dir > 0 ? "forward" : "reverse"}`, () => {
      const car = tcsCar();
      run(car, 1, () => idle);
      car.gear = dir;
      car.velocity.z = -3 * dir;
      const [left, right] = car.wheels.slice(2);
      left!.omega = (dir * 20) / left!.axle.wheelRadius;
      right!.omega = (dir * 3) / right!.axle.wheelRadius;
      car.step(DT, { ...idle, [dir > 0 ? "throttle" : "brake"]: 0.3 });
      expect(left!.contact).toBe(true);
      expect(right!.contact).toBe(true);
      expect(left!.tcsTorque).toBeGreaterThan(500);
      expect(right!.tcsTorque).toBe(0);
    });
  }

  for (const dir of [1, -1] as const)
    test(`does not brake against equal wheel speeds in a locked differential in ${dir > 0 ? "forward" : "reverse"}`, () => {
      const car = makeCar(bmwI4);
      car.gear = dir;
      car.velocity.z = -3 * dir;
      car.angularVelocity.y = -0.8 * dir;
      for (const w of car.wheels) w.omega = (4 * dir) / w.axle.wheelRadius;
      car.step(DT, { ...idle, [dir > 0 ? "throttle" : "brake"]: 0.3 });
      const inner = car.wheels[3]!;
      expect(
        dir * (inner.omega * inner.axle.wheelRadius - inner.vx),
      ).toBeGreaterThan(1);
      for (const w of car.wheels) expect(w.tcsTorque).toBe(0);
    });

  for (const [name, cfg] of [
    ["coupe", sportsCoupe],
    ["i4", bmwI4],
  ] as const)
    for (const dir of [1, -1] as const)
      test(`cuts ${name} torque when both wheels spin in ${dir > 0 ? "forward" : "reverse"}`, () => {
        expect(tractionLaunchSlip(cfg, dir, true)).toBeLessThan(
          tractionLaunchSlip(cfg, dir, false) * 0.25,
        );
      });

  test("does not brake a rolling wheel just because the other wheel is locked by the service brake", () => {
    const car = tcsCar();
    run(car, 1, () => idle);
    car.velocity.z = -3;
    const [rolling, locked] = car.wheels.slice(2);
    rolling!.omega = 3 / rolling!.axle.wheelRadius;
    locked!.omega = 0;
    car.step(DT, { ...idle, brake: 0.5 });
    expect(rolling!.tcsTorque).toBe(0);
    expect(locked!.tcsTorque).toBe(0);
  });

  for (const dir of [1, -1] as const)
    for (const steer of [-1, 1] as const)
      test(`does not brake normal wheel-speed differences in a ${dir > 0 ? "forward" : "reverse"} ${steer > 0 ? "right" : "left"} circle`, () => {
        const car = makeCar({
          ...structuredClone(bmwI4),
          esc: { ...bmwI4.esc, enabled: false },
        });
        run(car, 1, () => idle);
        car.gear = dir;
        let samples = 0;
        let maxUnwarrantedTorque = 0;
        // A tight, steady ~3 m/s circle. Each wheel has a different *ground* speed;
        // neither is spinning if its own circumference speed matches its local contact vx.
        run(
          car,
          9,
          () => {
            const speed = dir * car.speed;
            const drive = Math.max(0, Math.min(1, 0.1 + (3 - speed) * 0.35));
            const service = Math.max(0, Math.min(1, (speed - 3) * 0.35));
            return {
              ...idle,
              steer,
              throttle: dir > 0 ? drive : service,
              brake: dir > 0 ? service : drive,
            };
          },
          (t) => {
            if (t < 5) return false;
            const driven = car.wheels.filter((w) => w.axle.driven);
            const slips = driven.map(
              (w) => dir * (w.omega * w.axle.wheelRadius - w.vx),
            );
            if (Math.max(...slips) < 0.3) {
              samples++;
              maxUnwarrantedTorque = Math.max(
                maxUnwarrantedTorque,
                ...driven.map((w) => w.tcsTorque),
              );
            }
            return false;
          },
        );
        expect(dir * car.speed).toBeGreaterThan(1.5);
        expect(samples).toBeGreaterThan(500);
        expect(maxUnwarrantedTorque).toBeLessThan(25);
      });
});

describe("resting suspension and reset", () => {
  for (const [name, cfg] of [
    ["coupe", sportsCoupe],
    ["i4", bmwI4],
  ] as const)
    test(`${name} starts in static equilibrium without a suspension hop`, () => {
      const car = makeCar(cfg);
      car.step(DT, idle);
      // A car spawned at rest has no compression-rate impulse; one step must not
      // generate tens of times its static corner load or launch it vertically.
      expect(Math.abs(car.velocity.y)).toBeLessThan(1e-10);
      for (const wheel of car.wheels) {
        expect(wheel.contact).toBe(true);
        expect(wheel.load).toBeLessThan((cfg.mass * 9.81) / 2);
      }
      let rise = car.position.y - cfg.cgHeight;
      run(
        car,
        0.5,
        () => idle,
        () => {
          rise = Math.max(rise, car.position.y - cfg.cgHeight);
          return false;
        },
      );
      expect(rise).toBeLessThan(1e-10);
    });

  for (const [name, cfg] of [
    ["coupe", sportsCoupe],
    ["i4", bmwI4],
  ] as const)
    for (const vy of [-0.5, 0.5])
      test(`${name} suspension uses current ${vy < 0 ? "bump" : "rebound"} velocity, independent of timestep`, () => {
        for (const dt of [1 / 240, 1 / 960]) {
          const car = makeCar(cfg);
          const frontShare = cfg.rear.z / car.wheelbase;
          car.velocity.y = vy;
          car.step(dt, idle);
          for (const [i, w] of car.wheels.entries()) {
            const damping = vy < 0 ? w.axle.bumpDamping : w.axle.reboundDamping;
            const share = i < 2 ? frontShare : 1 - frontShare;
            const staticLoad = (cfg.mass * 9.81 * share) / 2;
            expect(w.load).toBeCloseTo(staticLoad - damping * vy, 8);
          }
        }
      });

  for (const [name, cfg] of [
    ["coupe", sportsCoupe],
    ["i4", bmwI4],
  ] as const)
    test(`${name} reset clears driving history but preserves assist switches`, () => {
      const config = structuredClone(cfg);
      const ground = new SlipperyGround(0.2);
      const used = new Car(config, ground);
      const fresh = new Car(structuredClone(cfg), new SlipperyGround(0.2));
      used.escMode = fresh.escMode = "sport";
      used.absEnabled = fresh.absEnabled = false;
      run(used, 2, () => ({ ...idle, throttle: 1, steer: 0.7 }));
      used.reset(new Vector3(0, 0, 0), 0);
      expect(used.escMode).toBe("sport");
      expect(used.absEnabled).toBe(false);
      for (let i = 0; i < 480; i++) {
        const drive = { ...idle, throttle: 1 };
        used.step(DT, drive);
        fresh.step(DT, drive);
        if (i % 40 !== 0) continue;
        expect(used.position.distanceTo(fresh.position)).toBeLessThan(1e-5);
        expect(used.velocity.distanceTo(fresh.velocity)).toBeLessThan(1e-5);
        expect(
          used.angularVelocity.distanceTo(fresh.angularVelocity),
        ).toBeLessThan(1e-5);
        expect(Math.abs(used.engineOmega - fresh.engineOmega)).toBeLessThan(
          1e-4,
        );
        for (let w = 0; w < 4; w++)
          expect(
            Math.abs(used.wheels[w]!.omega - fresh.wheels[w]!.omega),
          ).toBeLessThan(1e-4);
      }
    });
});

/** Flat ground with one tree trunk on it. */
class TreeGround extends FlatGround implements Obstacles {
  constructor(readonly tree: Obstacle) {
    super();
  }
  obstaclesNear(x: number, z: number, range: number, out: Obstacle[]): number {
    const t = this.tree;
    if (Math.hypot(t.x - x, t.z - z) > range + t.radius) return 0;
    out[0] = t;
    return 1;
  }
}

describe("obstacles", () => {
  const trunk = (x: number): Obstacle => ({
    x,
    z: -12,
    bottom: -1,
    top: 6,
    radius: 0.3,
  });
  /** Rolls the car at `speed` km/h into a trunk `offset` m to the side of its centreline. */
  function crash(speed: number, offset = 0, throttle = 0) {
    const car = new Car(structuredClone(bmwI4), new TreeGround(trunk(offset)));
    car.reset(new Vector3(0, 0, 0), 0);
    run(car, 0.5, () => idle);
    const v = speed / 3.6;
    car.velocity.set(0, 0, -v);
    for (const w of car.wheels) w.omega = v / w.axle.wheelRadius;
    let fatalAt = -1;
    let t = 0;
    run(
      car,
      4,
      () => ({ ...idle, throttle }),
      (time) => {
        t = time;
        if (car.fatal && fatalAt < 0) fatalAt = time;
        return fatalAt >= 0;
      },
    );
    return { car, fatalAt, t };
  }

  test("a slow hit stops the car at the trunk, survivably", () => {
    const { car, fatalAt } = crash(30);
    expect(fatalAt).toBe(-1);
    expect(Math.abs(car.speed)).toBeLessThan(2);
    // The bumper (about 2.4 m ahead of the centre of mass) stops short of the trunk's axis.
    const front = car.position.z - 2.42;
    expect(front).toBeGreaterThan(-12 + 0.3 - 0.4);
    expect(upright(car)).toBeGreaterThan(0.95);
    expect(car.impactEnergy).toBeGreaterThan(0);
  });

  test("a fast hit is fatal, at the moment of impact", () => {
    const { car, fatalAt } = crash(80);
    expect(fatalAt).toBeGreaterThan(0);
    // Within a fraction of a second of the bumper reaching the trunk.
    expect(fatalAt).toBeLessThan((12 - 2.7) / (80 / 3.6) + 0.15);
    expect(car.impactEnergy).toBeGreaterThanOrEqual(
      FATAL_ENERGY * car.cfg.mass,
    );
    // A pole impact's peak deceleration, as felt in the car (filtered).
    expect(car.impactPeak / 9.81).toBeGreaterThan(30);
    expect(car.impactPeak / 9.81).toBeLessThan(100);
  });

  test("the threshold lies between 55 and 75 km/h head-on", () => {
    expect(crash(55).fatalAt).toBe(-1);
    expect(crash(75).fatalAt).toBeGreaterThan(0);
  });

  test("pushing against a trunk at full throttle never adds up to a fatal impact", () => {
    const { car } = crash(20, 0, 1);
    const settled = car.impactEnergy;
    run(car, 6, () => ({ ...idle, throttle: 1 }));
    expect(car.fatal).toBe(false);
    expect(car.impactEnergy).toBeLessThan(settled + 0.01 * car.cfg.mass);
    expect(car.position.z - 2.42).toBeGreaterThan(-12 + 0.3 - 0.4);
  });

  test("passing beside a trunk does not touch it", () => {
    const { car, fatalAt } = crash(80, 1.4);
    expect(fatalAt).toBe(-1);
    expect(car.impactEnergy).toBe(0);
    expect(car.impactPeak).toBe(0);
  });
});
