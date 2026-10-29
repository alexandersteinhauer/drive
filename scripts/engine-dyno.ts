// Runs a simulated engine on the dynamometer at full throttle, held at each speed in turn,
// and prints its torque, power and boost:
//   bun scripts/engine-dyno.ts [vw1200|f154cd] [throttle]
import { EngineSim, loadEngineModule } from "../src/sim/engineSim";
import type { EngineSpec } from "../src/sim/engineSpec";
import { f154cd } from "../src/sim/engines/f154cd";
import { vw1200 } from "../src/sim/engines/vw1200";

const engines: Record<string, { spec: EngineSpec; rpms: number[] }> = {
  vw1200: {
    spec: vw1200,
    rpms: [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
  },
  f154cd: {
    spec: f154cd,
    rpms: [1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 7500, 7950],
  },
};
const name = process.argv[2] ?? "f154cd";
const throttle = Number(process.argv[3] ?? 1);
const engine = engines[name];
if (!engine)
  throw new Error(`usage: engine-dyno.ts <${Object.keys(engines).join("|")}>`);
await loadEngineModule(() =>
  Bun.file(new URL("../src/sim/engine.wasm", import.meta.url)).arrayBuffer(),
);
const RPM = Math.PI / 30;
const sim = new EngineSim(engine.spec);
sim.ignition = true;
sim.speed = engine.rpms[0]! * RPM;
for (const rpm of engine.rpms) {
  sim.dyno(true, rpm * RPM);
  sim.throttle = throttle;
  // Long enough for the turbo to spool and the plenum to fill.
  sim.step(Math.round(1.5 * sim.frequency));
  let torque = 0;
  let boost = 0;
  let manifold = 0;
  const n = 20;
  for (let i = 0; i < n; i++) {
    sim.step(Math.round(0.02 * sim.frequency));
    torque += sim.dynoTorque / n;
    boost += sim.boost / n;
    manifold += sim.manifoldPressure / n;
  }
  const kw = (torque * rpm * RPM) / 1000;
  console.log(
    `${String(rpm).padStart(5)} rpm ${torque.toFixed(0).padStart(4)} Nm ${kw.toFixed(0).padStart(4)} kW ${(kw / 0.7355).toFixed(0).padStart(4)} cv  boost ${(boost / 1e5).toFixed(2)} bar  manifold ${(manifold / 1e5).toFixed(2)} bar  turbo ${((sim.turboSpeed * 30) / Math.PI / 1000).toFixed(0)}k rpm`,
  );
}
sim.dispose();
