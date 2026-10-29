// Drives a car on flat ground with its simulated engine, as the game does, and writes what
// it sounds like: idle, a launch through the gears, lifting off, cruising, braking.
//   bun scripts/engine-drive.ts [out.wav] [beetle|pista]
// RECORDED=1: through engine-sim's exhaust impulse response instead of the computed
// acoustics, as the game plays it with ?exhaust=impulse.
// Prints the engine, gearbox, clutch and the sound's level (before the loudness stage)
// every 0.25 s.
import { writeFileSync } from "node:fs";
import { Loudness } from "../src/audio/loudness";
import { Car } from "../src/sim/car";
import {
  ferrari488Pista,
  vwBeetle,
  withImpulseExhaust,
} from "../src/sim/config";
import { EngineModule, loadEngineModule } from "../src/sim/engineSim";
import { audioChannels } from "../src/sim/engineSpec";
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

const cars = { beetle: vwBeetle, pista: ferrari488Pista };
const carName = (process.argv[3] ?? "beetle") as keyof typeof cars;
if (!cars[carName]) throw new Error(`no car ${carName}`);
const config = process.env.RECORDED
  ? withImpulseExhaust(cars[carName])
  : structuredClone(cars[carName]);
const car = new Car(config, new FlatGround());
const engine = car.powerplant as SimulatedEngine;
engine.sim.collectAudio = true;
const spec = structuredClone(engine.cfg.spec);

// The synthesizer as the worklet runs it: the voice (engine/src/voice.rs), or engine-sim's.
const audio = await EngineModule.load(await wasm());
const x = audio.exports;
const json = audio.write(new TextEncoder().encode(JSON.stringify(spec)));
const voice = !!spec.audio.voice;
const synth = voice
  ? x.voice_for_engine(json.ptr, json.len)
  : x.synth_for_engine(json.ptr, json.len);
if (!spec.audio.acoustics)
  for (const [i, exhaust] of spec.exhausts.entries()) {
    const wav = audio.write(
      new Uint8Array(
        await Bun.file(
          `public/audio/impulse/${exhaust.impulseResponse}`,
        ).arrayBuffer(),
      ),
    );
    (voice ? x.voice_set_impulse_wav : x.synth_set_impulse_wav)(
      synth,
      i,
      wav.ptr,
      wav.len,
      exhaust.impulseVolume,
    );
  }
if (!voice) x.synth_set_gain(synth, 0.003);
const channels = audioChannels(spec);
const loudness = new Loudness(RATE);
const pcm: number[] = [];

let pedal = 0;
let brake = 0;
let t = 0;
let nextLog = 0;
let power = 0;
let peak = 0;
let clipped = 0;
let powerSamples = 0;
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

  const perChannel = engine.sim.takeAudio();
  const steps = perChannel[0]!.length;
  const samples = new Float32Array(steps * channels);
  perChannel.forEach((c, k) => samples.set(c, k * steps));
  const block = audio.write(new Uint8Array(samples.buffer));
  (voice ? x.voice_write : x.synth_write)(synth, block.ptr, steps, channels);
  audio.free(block);
  const n = (voice ? x.voice_input_available : x.synth_input_available)(synth);
  const ptr = (voice ? x.voice_render : x.synth_render)(synth, n);
  const rendered = audio.f32(ptr, x.synth_rendered()).slice();
  for (const v of rendered) {
    power += v * v;
    peak = Math.max(peak, Math.abs(v));
    if (Math.abs(v) > 0.999) clipped++;
  }
  powerSamples += rendered.length;
  loudness.process(rendered);
  for (const v of rendered) pcm.push(v);

  if (t >= nextLog) {
    console.log(
      `${t.toFixed(2).padStart(5)}s ${car.rpm.toFixed(0).padStart(4)} rpm gear ${car.gear} ` +
        `${(car.speed * 3.6).toFixed(1).padStart(5)} km/h pedal ${pedal.toFixed(2)} brake ${brake.toFixed(2)} ` +
        `tcs ${(car as unknown as { tcsFactor: number }).tcsFactor.toFixed(2)} clutch ${car.clutchEngagement.toFixed(2)} engine throttle ${car.engineThrottle.toFixed(2)} ` +
        `${(10 * Math.log10(power / Math.max(1, powerSamples) + 1e-12)).toFixed(0)} dB ` +
        `peak ${(20 * Math.log10(peak + 1e-12)).toFixed(0)} dB${clipped ? ` CLIPPED ${clipped}` : ""}`,
    );
    power = powerSamples = peak = clipped = 0;
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
