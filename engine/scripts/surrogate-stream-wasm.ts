// Replay actual game-rate flow through the SAME streaming WASM adapter used by the worklet.
// Pressure output is Pa (fullScale=1), before Loudness/directivity; no signal realignment.
// bun engine/scripts/surrogate-stream-wasm.ts model.json native-flow.f32 native-controls.f32
//     <native-rate> <audio-rate> out.f32 [--bench]
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const [modelPath, flowPath, controlPath, inputText, outputText, outPath, flag] =
  process.argv.slice(2);
if (!outPath || (flag && flag !== "--bench"))
  throw Error("model flow controls native-rate audio-rate out [--bench]");
const model = readFileSync(modelPath!);
const metadata = JSON.parse(model.toString()) as {
  cylinders: number;
  widths: number[];
};
const source = readFileSync(flowPath!),
  conditioning = readFileSync(controlPath!);
const flow = new Float32Array(
  source.buffer,
  source.byteOffset,
  source.byteLength / 4,
);
const controls = new Float32Array(
  conditioning.buffer,
  conditioning.byteOffset,
  conditioning.byteLength / 4,
);
const nc = metadata.cylinders,
  ng = metadata.widths.length;
const frames = flow.length / nc,
  inputRate = Number(inputText),
  outputRate = Number(outputText);
if (
  !Number.isInteger(frames) ||
  controls.length !== frames * ng ||
  ng !== 5 ||
  !(inputRate > 0) ||
  !(outputRate > 0)
)
  throw Error("invalid model/native input dimensions or rates");
const wasm = fileURLToPath(
  new URL("../../src/sim/engine.wasm", import.meta.url),
);
const { instance } = await WebAssembly.instantiate(readFileSync(wasm), {});
const x = instance.exports as Record<string, any> & {
  memory: WebAssembly.Memory;
};
const mp = x.alloc(model.length);
new Uint8Array(x.memory.buffer, mp, model.length).set(model);
const s = x.surrogate_voice_new(mp, model.length, inputRate, outputRate, 1);
x.dealloc(mp, model.length);
if (!s) throw Error("model rejected by streaming adapter");
// Small game-like packets avoid timing a huge queued-input memmove unlike actual playback.
const packet = Math.ceil(inputRate / 60),
  channels = nc + ng;
const ip = x.alloc(packet * channels * 4),
  rendered = new Float32Array(
    Math.ceil((frames / inputRate) * outputRate) + 1024,
  );
function replay(): number {
  let written = 0;
  for (let i = 0; i < frames; i += packet) {
    const n = Math.min(packet, frames - i);
    const planar = new Float32Array(x.memory.buffer, ip, n * channels);
    for (let t = 0; t < n; t++) {
      for (let c = 0; c < nc; c++) planar[c * n + t] = flow[(i + t) * nc + c]!;
      for (let c = 0; c < ng; c++)
        planar[(nc + c) * n + t] = controls[(i + t) * ng + c]!;
    }
    x.surrogate_voice_write(s, ip, n, channels);
    if (x.surrogate_voice_failed(s))
      throw Error(`invalid native input at frame ${i}`);
    while (x.surrogate_voice_input_available(s) > 0) {
      const want = Math.min(128, x.surrogate_voice_input_available(s));
      const p = x.surrogate_voice_render(s, want),
        nout = x.synth_rendered();
      if (nout !== want || x.surrogate_voice_failed(s))
        throw Error("stream output failed");
      rendered.set(new Float32Array(x.memory.buffer, p, nout), written);
      written += nout;
    }
  }
  return written;
}
replay();
const timing: number[] = [];
let count = 0;
for (let i = 0; i < (flag ? 5 : 1); i++) {
  x.surrogate_voice_reset(s);
  const t = performance.now();
  count = replay();
  timing.push(performance.now() - t);
}
const sorted = [...timing].sort((a, b) => a - b),
  median = sorted[Math.floor(sorted.length / 2)]!;
writeFileSync(outPath, new Uint8Array(rendered.buffer, 0, count * 4));
console.log(
  JSON.stringify(
    {
      nativeFrames: frames,
      audioFrames: count,
      inputRate,
      outputRate,
      inputSeconds: frames / inputRate,
      outputSeconds: count / outputRate,
      medianMs: median,
      streamPercentRealtime: median / ((frames / inputRate) * 10),
      minMs: sorted[0],
      maxMs: sorted.at(-1),
      includes:
        "JS packet copy, model, input and output resampling; excludes model load, loudness and directivity",
    },
    null,
    2,
  ),
);
x.dealloc(ip, packet * channels * 4);
x.surrogate_voice_free(s);
