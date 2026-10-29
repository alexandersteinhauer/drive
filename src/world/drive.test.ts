import { expect, test } from "bun:test";
import { Quaternion, Vector3 } from "three";
import { Car } from "../sim/car";
import { bmwI4, vwBeetle, type CarConfig } from "../sim/config";
import { Landscape } from "./landscape";

const DT = 1 / 480;

/** Pure-pursuit autopilot along the right-hand lane; returns stats. */
function drive(cfg: CarConfig, seconds: number, targetKmh: number) {
  const land = new Landscape(1);
  const road = land.road;
  road.ensure(6000);
  // The autopilot computes exact wheel angles, so it drives without the steering assist.
  const config = structuredClone(cfg);
  config.steering.limitAssist = false;
  const car = new Car(config, land);
  const start = 20;
  const h0 = road.headings[start]!;
  car.reset(
    new Vector3(
      road.xs[start]! + Math.cos(h0) * 1.75,
      0,
      road.zs[start]! + Math.sin(h0) * 1.75,
    ),
    -h0,
  );

  const inv = new Quaternion();
  const local = new Vector3();
  let index = start;
  let maxOff = 0;
  let maxSpeed = 0;
  let airborne = 0;
  let maxTilt = 0;
  let escTime = 0;
  const t0 = performance.now();
  for (let t = 0; t < seconds; t += DT) {
    // Index nearest to the car, searched locally.
    let best = Infinity;
    for (
      let i = Math.max(0, index - 20);
      i < Math.min(road.finalCount, index + 20);
      i++
    ) {
      const d =
        (road.xs[i]! - car.position.x) ** 2 +
        (road.zs[i]! - car.position.z) ** 2;
      if (d < best) {
        best = d;
        index = i;
      }
    }
    const h = road.headings[index]!;
    const lateral =
      (car.position.x - road.xs[index]!) * Math.cos(h) +
      (car.position.z - road.zs[index]!) * Math.sin(h);
    if (t > 2) maxOff = Math.max(maxOff, Math.abs(lateral - 1.75));
    const ahead = Math.min(
      road.finalCount - 1,
      index + Math.round(Math.max(6, car.speed * 0.9) / 2),
    );
    const ha = road.headings[ahead]!;
    local
      .set(
        road.xs[ahead]! + Math.cos(ha) * 1.75,
        car.position.y,
        road.zs[ahead]! + Math.sin(ha) * 1.75,
      )
      .sub(car.position)
      .applyQuaternion(inv.copy(car.orientation).invert());
    const ld2 = local.x * local.x + local.z * local.z;
    const delta = Math.atan((2 * car.wheelbase * local.x) / ld2);
    const err = targetKmh - car.speed * 3.6;
    car.step(DT, {
      steer: Math.max(-1, Math.min(1, delta / car.cfg.steering.lock)),
      throttle: Math.max(0, Math.min(1, err * 0.15)),
      brake: Math.max(0, Math.min(1, -err * 0.08)),
      handbrake: 0,
    });
    maxSpeed = Math.max(maxSpeed, car.speed * 3.6);
    if (car.wheels.every((w) => !w.contact)) airborne += DT;
    if (car.escActive) escTime += DT;
    maxTilt = Math.max(
      maxTilt,
      Math.acos(new Vector3(0, 1, 0).applyQuaternion(car.orientation).y),
    );
    expect(Number.isFinite(car.position.x)).toBe(true);
  }
  const wall = performance.now() - t0;
  const distance = (index - start) * road.cfg.spacing;
  return {
    distance,
    maxOff,
    maxSpeed,
    airborne,
    maxTilt,
    msPerStep: wall / (seconds / DT),
    escShare: escTime / seconds,
  };
}

for (const [name, cfg] of [
  ["beetle", vwBeetle],
  ["electric", bmwI4],
] as const)
  test(`${name} car follows the road over the hills`, () => {
    const r = drive(cfg, 90, 90);
    console.log(
      `${name}: ${(r.distance / 1000).toFixed(2)} km, max lane offset ${r.maxOff.toFixed(2)} m, ` +
        `max ${r.maxSpeed.toFixed(0)} km/h, airborne ${r.airborne.toFixed(2)} s, max tilt ${((r.maxTilt * 180) / Math.PI).toFixed(1)} deg, ` +
        `${(r.msPerStep * 1000).toFixed(1)} us/step, esc active ${(r.escShare * 100).toFixed(1)}%`,
    );
    expect(r.distance).toBeGreaterThan(1800);
    expect(r.maxOff).toBeLessThan(1.5);
    expect(r.airborne).toBeLessThan(0.05);
    // A simulated engine (engine/) runs ~21 of its steps per car step.
    const simulated =
      cfg.powertrain.type === "combustion" &&
      cfg.powertrain.engine.type === "simulated";
    expect(r.msPerStep).toBeLessThan(simulated ? 0.6 : 0.1);
    // Normal driving must not trigger stability control.
    expect(r.escShare).toBeLessThan(0.01);
  }, 60_000);
