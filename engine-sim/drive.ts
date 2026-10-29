// Drives the game's Beetle simulation with a keyboard-like throttle pattern and renders the
// engine sound it would make, the way the game feeds it (rpm and throttle per 60 Hz frame).
//   bun engine-sim/drive.ts [out.wav]
// Prints a level every 50 ms around throttle changes, to find drops and gaps.
import { writeFileSync } from "node:fs";
// @ts-expect-error: Emscripten glue, built by build.sh
import createModule from "./.cache/node/engine-sim.mjs";
import { Loudness } from "../src/audio/loudness";
import { Car } from "../src/sim/car";
import { vwBeetle } from "../src/sim/config";
import { FlatGround } from "../src/sim/ground";

const out = process.argv[2] ?? "drive.wav";
const rate = 44100;
const block = 128;
const DT = 1 / 480;
const FRAME = 1 / 60;

// Throttle key held during these [from, to) seconds.
const held: [number, number][] = [
  [1, 14],
  [16, 22],
  [22.4, 26],
];
const total = 30;

const car = new Car(structuredClone(vwBeetle), new FlatGround());
const m = await createModule();
const path = m.stringToNewUTF8("/engines/beetle.mr");
if (m._es_init(path, 0) !== 0) throw new Error("init failed");

const pcm = new Int16Array(Math.round(total * rate));
const loudness = new Loudness(rate);
const buf = new Float32Array(block);
let pedal = 0;
let simT = 0;
let frameT = 0;
let rpm = 850;
let throttle = 0;
let engaged = true;
// AudioParam setTargetAtTime smoothing, as in EngineSound.update.
let rpmS = 850;
const log: string[] = [];
let lastGear = 1;

for (let i = 0; i < pcm.length; i += block) {
  const t = i / rate;
  // Game frames due by now: physics at 480 Hz, controls sampled per frame.
  while (frameT <= t) {
    const down = held.some(([a, b]) => frameT >= a && frameT < b);
    pedal = Math.max(0, Math.min(1, pedal + (down ? 8 : -10) * FRAME));
    while (simT < frameT + FRAME) {
      car.step(DT, { throttle: pedal, brake: 0, steer: 0, handbrake: 0 });
      simT += DT;
    }
    rpm = car.rpm;
    throttle = car.engineThrottle;
    engaged = !car.clutchOpen;
    if (car.gear !== lastGear)
      log.push(`${frameT.toFixed(2)}s gear ${lastGear}->${car.gear}`);
    lastGear = car.gear;
    frameT += FRAME;
  }
  const dt = block / rate;
  rpmS += (rpm - rpmS) * (1 - Math.exp(-dt / 0.01));
  m._es_set_controls(rpmS, throttle, engaged ? 1 : 0);
  const ptr = m._es_render(block) >> 2;
  buf.set(m.HEAPF32.subarray(ptr, ptr + block));
  if (!process.env.RAW) loudness.process(buf, m._es_overrun());
  const n = Math.min(block, pcm.length - i);
  for (let j = 0; j < n; j++)
    pcm[i + j] = Math.max(-1, Math.min(1, buf[j]!)) * 32767;
  if (i % (block * 17) === 0)
    log.push(
      `${t.toFixed(2)}s rpm=${rpm.toFixed(0)} sim=${m._es_engine_rpm().toFixed(0)} thr=${throttle.toFixed(2)} speed=${(car.speed * 3.6).toFixed(0)} km/h`,
    );
}

// Levels in 50 ms windows.
const win = rate / 20;
const levels: string[] = [];
for (let s = 0; s + win <= pcm.length; s += win) {
  let sum = 0;
  for (let j = s; j < s + win; j++) sum += (pcm[j]! / 32768) ** 2;
  levels.push((10 * Math.log10(sum / win + 1e-12)).toFixed(0));
}
console.log(log.filter((l) => l.includes("gear")).join("\n"));
if (process.env.VERBOSE) console.log(log.join("\n"));
for (let s = 0; s < levels.length; s += 20)
  console.log(
    `${(s / 20).toFixed(0).padStart(2)}s ${levels.slice(s, s + 20).join(" ")}`,
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
