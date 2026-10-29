// Offline check of the wasm build: drives a scripted run, times the simulation against
// real time and writes a WAV to listen to.
//   bun engine-sim/render.ts [out.wav] [simulationFrequency] [engine]
// GAIN=<g> overrides the fixed output gain; RAW=1 skips the loudness compression.
import { writeFileSync } from "node:fs";
// @ts-expect-error: Emscripten glue, built by build.sh
import createModule from "./.cache/node/engine-sim.mjs";
import { Loudness } from "../src/audio/loudness";

const out = process.argv[2] ?? "engine.wav";
const frequency = Number(process.argv[3] ?? 0);
const engine = process.argv[4] ?? "beetle";
const rate = 44100;
const block = 128;

const m = await createModule();
const path = m.stringToNewUTF8(`/engines/${engine}.mr`);
let t0 = performance.now();
const status = m._es_init(path, frequency);
if (status !== 0) throw new Error(`es_init failed: ${status}`);
if (process.env.GAIN) m._es_set_gain(Number(process.env.GAIN));
console.log(`init ${(performance.now() - t0).toFixed(0)} ms`);

// `DYNO=1`: wide-open-throttle torque at fixed speeds instead of the drive.
if (process.env.DYNO) {
  const buf = 4410;
  for (const rpm of [1000, 1500, 2000, 2500, 3000, 3600, 4000, 4500]) {
    m._es_set_controls(rpm, 1, 1);
    for (let i = 0; i < 25; i++) m._es_render(buf); // let the dyno settle, then measure
    let sum = 0;
    for (let i = 0; i < 10; i++) {
      m._es_render(buf);
      sum += m._es_dyno_torque();
    }
    const nm = sum / 10;
    console.log(
      `${rpm} rpm: ${nm.toFixed(1)} Nm, ${((nm * rpm * Math.PI) / 30 / 735.5).toFixed(1)} PS`,
    );
  }
  process.exit(0);
}

// [seconds, rpm, throttle]: idle, blip, pull through the rev range under load, overrun,
// part-throttle cruise.
const script = [
  [0, 850, 0.0],
  [2.5, 850, 0.0],
  [3.0, 2500, 0.5],
  [3.6, 900, 0.0],
  [5.0, 1200, 1.0],
  [11.0, 4400, 1.0],
  [11.2, 4400, 0.0],
  [15.0, 1500, 0.0],
  [16.0, 2500, 0.25],
  [19.0, 2600, 0.25],
];
const at = (t) => {
  for (let i = 1; i < script.length; i++) {
    const [t1, r1, th1] = script[i];
    const [t0, r0, th0] = script[i - 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [
        r0 + (r1 - r0) * f,
        f < 1 ? th0 + (th1 - th0) * (t1 - t0 < 0.3 ? f : 0) : th1,
      ];
    }
  }
  return script.at(-1).slice(1);
};

const total = Math.round(script.at(-1)[0] * rate);
const pcm = new Int16Array(total);
const loudness = new Loudness(rate);
const buf = new Float32Array(block);
t0 = performance.now();
let log = 0;
for (let i = 0; i < total; i += block) {
  const t = i / rate;
  const [rpm, throttle] = at(t);
  m._es_set_controls(rpm, throttle, 1);
  const ptr = m._es_render(block) >> 2;
  const n = Math.min(block, total - i);
  buf.set(m.HEAPF32.subarray(ptr, ptr + block));
  if (!process.env.RAW) loudness.process(buf, m._es_overrun());
  for (let j = 0; j < n; j++)
    pcm[i + j] = Math.max(-1, Math.min(1, buf[j]!)) * 32767;
  if (t >= log) {
    console.log(
      `t=${t.toFixed(1)}s target=${rpm.toFixed(0)} rpm=${m._es_engine_rpm().toFixed(0)} thr=${throttle.toFixed(2)} dyno=${m._es_dyno_torque().toFixed(1)} Nm`,
    );
    log += 1;
  }
}
const elapsed = (performance.now() - t0) / 1000;
console.log(
  `simulated ${(total / rate).toFixed(1)} s in ${elapsed.toFixed(2)} s (${((elapsed / (total / rate)) * 100).toFixed(0)} % of real time)`,
);

const header = Buffer.alloc(44);
header.write("RIFF", 0);
header.writeUInt32LE(36 + pcm.byteLength, 4);
header.write("WAVEfmt ", 8);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);
header.writeUInt16LE(1, 22);
header.writeUInt32LE(rate, 24);
header.writeUInt32LE(rate * 2, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write("data", 36);
header.writeUInt32LE(pcm.byteLength, 40);
writeFileSync(out, Buffer.concat([header, Buffer.from(pcm.buffer)]));
console.log(`wrote ${out}`);
