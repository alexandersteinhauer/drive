// Experimental learned exhaust decoder, through the exact WASM artifact shipped by the
// game. Does not replace the game's voice. No per-clip gain, alignment or noise is applied.
// bun engine/scripts/surrogate-wasm.ts model.json flow.f32 controls.f32 out.f32 [--bench]
// Inputs/outputs: frame-interleaved f32 at the model's rate (not the audio device's rate).
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const [modelPath, flowPath, controlsPath, outPath, flag] = process.argv.slice(2);
if (!outPath || (flag && flag !== "--bench")) throw Error("model.json flow.f32 controls.f32 out.f32 [--bench]");
const wasmPath = fileURLToPath(new URL("../../src/sim/engine.wasm", import.meta.url));
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports as Record<string, any> & { memory: WebAssembly.Memory };
const modelBytes = readFileSync(modelPath);
const model = JSON.parse(modelBytes.toString());
const ptr = w.alloc(modelBytes.length);
new Uint8Array(w.memory.buffer, ptr, modelBytes.length).set(modelBytes);
const s = w.surrogate_new(ptr, modelBytes.length);
w.dealloc(ptr, modelBytes.length);
if (!s) throw Error("invalid surrogate model");
const block = w.surrogate_block_size();
const inputs = w.surrogate_inputs(s), controls = w.surrogate_controls(s), outputs = w.surrogate_outputs(s);
const flows = readFileSync(flowPath), ctls = readFileSync(controlsPath);
if (flows.length % (4 * inputs)) throw Error("incomplete flow frame");
const frames = flows.length / (4 * inputs);
if (ctls.length !== frames * controls * 4) throw Error("control/flow frame count mismatch");
const padded = Math.ceil(frames / block) * block;
const fp = w.alloc(padded * inputs * 4), cp = w.alloc(Math.max(4, padded * controls * 4)), op = w.alloc(padded * outputs * 4);
new Uint8Array(w.memory.buffer, fp, padded * inputs * 4).fill(0);
new Uint8Array(w.memory.buffer, cp, Math.max(4, padded * controls * 4)).fill(0);
new Uint8Array(w.memory.buffer, fp, flows.length).set(flows);
new Uint8Array(w.memory.buffer, cp, ctls.length).set(ctls);
const render = () => {
  for (let i = 0; i < padded; i += block) {
    if (w.surrogate_process(s, fp + 4 * i * inputs, cp + 4 * i * controls, op + 4 * i * outputs) !== 0) {
      throw Error(`invalid input or transformed feature at frame ${i}`);
    }
  }
};
// Warm the WASM/JIT on a complete trajectory, reset outside every timed interval.
render();
const milliseconds: number[] = [];
for (let i = 0; i < (flag ? 7 : 1); i++) {
  w.surrogate_reset(s);
  const t = performance.now(); render(); milliseconds.push(performance.now() - t);
}
writeFileSync(outPath, new Uint8Array(w.memory.buffer, op, frames * outputs * 4));
const sorted = [...milliseconds].sort((a, b) => a - b), median = sorted[Math.floor(sorted.length / 2)];
console.log(JSON.stringify({ frames, rate: model.rate, inputs, controls, outputs, block, modelBytes: modelBytes.length,
  medianMs: median, minMs: sorted[0], maxMs: sorted.at(-1), decoderPercentRealtime: median / (frames / model.rate * 10) }, null, 2));
w.dealloc(fp, padded * inputs * 4); w.dealloc(cp, Math.max(4, padded * controls * 4)); w.dealloc(op, padded * outputs * 4);
w.surrogate_delete(s);
