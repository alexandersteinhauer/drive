// Drives the Beetle on flat ground with its simulated engine, as the game does, and writes
// what it sounds like: idle, a launch through the gears, lifting off, cruising, braking.
//   bun scripts/engine-drive.ts [out.wav]
// Prints the engine, gearbox and clutch every 0.25 s.
import { writeFileSync } from "node:fs";
import { Loudness } from "../src/audio/loudness";
import { Car } from "../src/sim/car";
import { vwBeetle } from "../src/sim/config";
import { EngineModule, loadEngineModule } from "../src/sim/engineSim";
import { FlatGround } from "../src/sim/ground";
import { SimulatedEngine } from "../src/sim/powerplant";

const out = process.argv[2] ?? "engine-drive.wav";
const wasm = () =>
  Bun.file(new URL("../src/sim/engine.wasm", import.meta.url)).arrayBuffer();
await loadEngineModule(wasm);

const DT = 1 / 480;
const FRAME = 1 / 60;
const RATE = 44100;
// [from, to) seconds: throttle held, brake held.
const throttleHeld: [number, number][] = [
  [2, 24],
  [27, 34],
  [34.5, 40],
];
const brakeHeld: [number, number][] = [[42, 50]];
const total = 52;
const held = (spans: [number, number][], t: number) =>
  spans.some(([a, b]) => t >= a && t < b);

const car = new Car(structuredClone(vwBeetle), new FlatGround());
const engine = car.powerplant as SimulatedEngine;
engine.sim.collectAudio = true;
const spec =
  vwBeetle.powertrain.type === "combustion" &&
  vwBeetle.powertrain.engine.type === "simulated"
    ? vwBeetle.powertrain.engine.spec
    : null!;

// The synthesizer as the worklet runs it.
const audio = await EngineModule.load(await wasm());
const x = audio.exports;
const synth = x.synth_new(
  1,
  spec.simulationFrequency,
  spec.audio.hfGain,
  spec.audio.noise,
  spec.audio.jitter,
);
const wav = audio.write(
  new Uint8Array(
    await Bun.file(
      `public/audio/impulse/${spec.exhausts[0]!.impulseResponse}`,
    ).arrayBuffer(),
  ),
);
x.synth_set_impulse_wav(
  synth,
  0,
  wav.ptr,
  wav.len,
  spec.exhausts[0]!.impulseVolume,
);
x.synth_set_gain(synth, 0.003);
const loudness = new Loudness(RATE);
const pcm: number[] = [];

let pedal = 0;
let brake = 0;
let overrun = 0;
let t = 0;
let nextLog = 0;
while (t < total) {
  pedal = Math.max(
    0,
    Math.min(1, pedal + (held(throttleHeld, t) ? 8 : -10) * FRAME),
  );
  brake = Math.max(
    0,
    Math.min(1, brake + (held(brakeHeld, t) ? 8 : -10) * FRAME),
  );
  for (let i = 0; i < FRAME / DT; i++)
    car.step(DT, { throttle: pedal, brake, steer: 0, handbrake: 0 });
  t += FRAME;

  const samples = engine.sim.takeAudio()[0]!;
  if (car.clutchEngagement > 0.5) {
    const s = car.engineThrottle;
    overrun = 1 - Math.sin((s * s * Math.PI) / 2);
  } else overrun *= Math.exp(-FRAME / 0.2);
  const block = audio.write(new Uint8Array(samples.buffer));
  x.synth_write(synth, block.ptr, samples.length, 1);
  audio.free(block);
  const n = x.synth_input_available(synth);
  const ptr = x.synth_render(synth, n);
  const rendered = audio.f32(ptr, x.synth_rendered()).slice();
  loudness.process(rendered, overrun);
  for (const v of rendered) pcm.push(v);

  if (t >= nextLog) {
    console.log(
      `${t.toFixed(2).padStart(5)}s ${car.rpm.toFixed(0).padStart(4)} rpm gear ${car.gear} ` +
        `${(car.speed * 3.6).toFixed(1).padStart(5)} km/h pedal ${pedal.toFixed(2)} brake ${brake.toFixed(2)} ` +
        `clutch ${car.clutchEngagement.toFixed(2)} engine throttle ${car.engineThrottle.toFixed(2)}`,
    );
    nextLog += 0.25;
  }
}

const data = Buffer.alloc(pcm.length * 2);
pcm.forEach((v, i) =>
  data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, v)) * 32767), i * 2),
);
const header = Buffer.alloc(44);
header.write("RIFF", 0);
header.writeUInt32LE(36 + data.length, 4);
header.write("WAVEfmt ", 8);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);
header.writeUInt16LE(1, 22);
header.writeUInt32LE(RATE, 24);
header.writeUInt32LE(RATE * 2, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write("data", 36);
header.writeUInt32LE(data.length, 40);
writeFileSync(out, Buffer.concat([header, data]));
console.log(`wrote ${out}`);
