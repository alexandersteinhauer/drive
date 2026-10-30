// Drives the 488 Pista as the game does (src/sim/car.ts) with the reference exhaust
// (engine/src/duct.rs) and writes what its tailpipes radiate to the listener, appending as it
// goes: <out>.f32 (Pa at the fluid rate, see <out>.json) and <out>.exits.f32.
//   DUCT_THREADS=6 bun engine/scripts/reference-drive.ts <out-prefix> <idle|drive> [reference.json]
// (the reference config defaults to pista-reference.json: its ducts, physics and boost merged
// into the spec), then uv run engine/scripts/reference_sound.py <out-prefix> with
// DIAMETER=0.085. WASM=1: through the WebAssembly module instead of the native library.
// drive: idle 2 s, full throttle up through the gears until 3 s into 7th, lift, brake to a
// stop (the gearbox downshifting), then 2 s more (holding the brake there, the game selects
// reverse: cut the sound at the stop, as the log shows it).
import { appendFileSync, writeFileSync } from "node:fs";
const [out, scenario, refArg] = process.argv.slice(2) as [string, string, string | undefined];
const here = new URL(".", import.meta.url).pathname;
const repo = `${here}/../..`;
const refPath = refArg ?? `${here}/pista-reference.json`;
const { Car } = await import(`${repo}/src/sim/car.ts`);
const { ferrari488Pista } = await import(`${repo}/src/sim/config.ts`);
const { loadEngineModule } = await import(`${repo}/src/sim/engineSim.ts`);
const { FlatGround } = await import(`${repo}/src/sim/ground.ts`);
if (!process.env.WASM) await (await import("./native-engine.ts")).useNativeEngine(repo);
await loadEngineModule(() => Bun.file(`${repo}/src/sim/engine.wasm`).arrayBuffer());

const ref = JSON.parse(await Bun.file(refPath).text());
const config = structuredClone(ferrari488Pista);
const spec = config.powertrain.engine.spec;
const merge = (a: any, b: any) => {
  for (const k in b) {
    if (b[k] && typeof b[k] === "object" && !Array.isArray(b[k])) {
      a[k] ??= {};
      merge(a[k], b[k]);
    } else a[k] = b[k];
  }
};
merge(spec, ref);
// The car steps at 1/480 s: whole engine steps, and fluid substeps the pipes' own.
spec.simulationFrequency = Number(process.env.FREQ ?? 19200);
spec.fluidSteps = Number(process.env.FLUID ?? 12);
const rate = spec.simulationFrequency * spec.fluidSteps;
writeFileSync(`${out}.json`, JSON.stringify({ rate, scenario, outlets: 2 }));
writeFileSync(`${out}.f32`, new Uint8Array());
writeFileSync(`${out}.exits.f32`, new Uint8Array());
writeFileSync(`${out}.log`, "");

const started = performance.now();
const car = new Car(config, new FlatGround());
const sim = (car as any).powerplant.sim;
sim.collectAudio = false;
const x = sim.module.exports;
const take = () => {
  const n = x.engine_duct_sound_len(sim.e);
  const sound = sim.module.f32(x.engine_duct_sound_ptr(sim.e), n).slice();
  const m = x.engine_duct_exits_len(sim.e);
  const exits = sim.module.f32(x.engine_duct_exits_ptr(sim.e), m).slice();
  x.engine_duct_clear(sim.e);
  appendFileSync(`${out}.f32`, new Uint8Array(sound.buffer));
  appendFileSync(`${out}.exits.f32`, new Uint8Array(exits.buffer));
};
take(); // the engine's settling, dropped below by the WAV's start
const settled = performance.now();

const DT = 1 / 480;
const FRAME = 1 / 60;
let t = 0;
let pedal = 0;
let brake = 0;
let phase = "idle";
let seventh = -1;
let stopped = -1;
let nextLog = 0;
const log = (s: string) => {
  console.log(s);
  appendFileSync(`${out}.log`, s + "\n");
};
log(`settled in ${((settled - started) / 1000).toFixed(0)} s`);
for (;;) {
  if (scenario === "idle") {
    if (t >= 8) break;
  } else {
    if (phase === "idle" && t >= 2) phase = "throttle";
    if (phase === "throttle" && car.gear >= 7 && seventh < 0) seventh = t;
    if (phase === "throttle" && ((seventh >= 0 && t >= seventh + 3) || t >= 45)) phase = "lift";
    if (phase === "lift" && t >= (seventh >= 0 ? seventh + 3.7 : 45.7)) phase = "brake";
    if (phase === "brake" && car.speed < 0.3) {
      phase = "stopped";
      stopped = t;
    }
    if (phase === "stopped" && t >= stopped + 2) break;
  }
  const wantPedal = phase === "throttle" ? 1 : 0;
  const wantBrake = phase === "brake" ? 0.8 : phase === "stopped" ? 0.3 : 0;
  pedal = Math.max(0, Math.min(1, pedal + Math.sign(wantPedal - pedal) * 8 * FRAME));
  brake = Math.max(0, Math.min(1, brake + Math.sign(wantBrake - brake) * 8 * FRAME));
  if (Math.abs(pedal - wantPedal) < 8 * FRAME) pedal = wantPedal;
  if (Math.abs(brake - wantBrake) < 8 * FRAME) brake = wantBrake;
  for (let i = 0; i < FRAME / DT; i++) car.step(DT, { throttle: pedal, brake, steer: 0, handbrake: 0 });
  t += FRAME;
  take();
  if (!Number.isFinite(car.rpm) || !Number.isFinite(sim.boost)) {
    log(`not finite at ${t.toFixed(2)} s: stopping`);
    process.exit(1);
  }
  if (t >= nextLog) {
    log(
      `${t.toFixed(2)} ${phase} ${car.rpm.toFixed(0)} rpm gear ${car.gear} ${(car.speed * 3.6).toFixed(1)} km/h ` +
        `pedal ${pedal.toFixed(2)} brake ${brake.toFixed(2)} boost ${(sim.boost / 1e5).toFixed(2)} ` +
        `wall ${((performance.now() - started) / 1000).toFixed(0)} s`,
    );
    nextLog += 0.25;
  }
}
log(`done: ${t.toFixed(2)} s simulated in ${((performance.now() - started) / 1000).toFixed(0)} s`);
