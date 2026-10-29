import { expect, test } from "bun:test";
import { EngineSim } from "../engineSim";
import { f154cd } from "./f154cd";

const RPM = Math.PI / 30;

/**
 * The simulated F154 CD on the dynamometer at full throttle against Ferrari's figures:
 * 770 Nm at 3000 rpm and 720 cv (530 kW) at 8000, with up to ~1.5 bar of boost.
 */
test("makes the published torque and power", () => {
  const sim = new EngineSim(f154cd);
  sim.ignition = true;
  sim.speed = 2000 * RPM;
  const at = (rpm: number) => {
    sim.dyno(true, rpm * RPM);
    sim.throttle = 1;
    sim.step(Math.round(1.5 * sim.frequency));
    let torque = 0;
    let boost = 0;
    for (let i = 0; i < 10; i++) {
      sim.step(Math.round(0.02 * sim.frequency));
      torque += sim.dynoTorque / 10;
      boost = Math.max(boost, sim.boost / 1e5);
    }
    return { torque, boost, cv: (torque * rpm * RPM) / 735.5 };
  };
  // Up through the revs as in the car: the simulated gases can settle differently when the
  // speed jumps (at 5000 rpm, ~5 % less torque straight from 3000).
  const low = at(2000);
  const peak = at(3000);
  at(4000);
  const mid = at(5000);
  at(6000);
  at(7000);
  // Just below the hard cut at 8000.
  const top = at(7950);
  sim.dispose();
  console.log(
    `F154 CD: 2000 rpm ${low.torque.toFixed(0)} Nm, 3000 ${peak.torque.toFixed(0)} Nm at ${peak.boost.toFixed(2)} bar, 5000 ${mid.torque.toFixed(0)} Nm at ${mid.boost.toFixed(2)} bar, 7950 ${top.cv.toFixed(0)} cv`,
  );
  expect(low.torque).toBeLessThan(peak.torque);
  expect(peak.torque).toBeGreaterThan(730);
  expect(peak.torque).toBeLessThan(810);
  expect(mid.torque).toBeGreaterThan(730);
  expect(mid.torque).toBeLessThan(810);
  expect(top.cv).toBeGreaterThan(690);
  expect(top.cv).toBeLessThan(760);
  expect(peak.boost).toBeLessThan(1.6);
});
