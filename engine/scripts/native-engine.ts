// The engine crate built natively (engine/target/release/libengine_sim.dylib) in place of the
// WebAssembly module, through Bun's FFI, with EngineModule's interface: for the reference
// simulation's long renders, which run several times faster natively.
//   import { useNativeEngine } from "./native-engine.ts"; await useNativeEngine(repo);
// before loadEngineModule(), after `cargo build --release` in engine/.
import { dlopen, FFIType as T, toArrayBuffer } from "bun:ffi";

export async function useNativeEngine(repo: string): Promise<void> {
  const { EngineModule } = await import(`${repo}/src/sim/engineSim.ts`);
  const p = T.ptr, f = T.f64, u = T.u32, z = T.u64_fast;
  const sig: Record<string, [T[], T]> = {
    alloc: [[z], p],
    dealloc: [[p, z], T.void],
    engine_new: [[p, z], p],
    engine_free: [[p], T.void],
    engine_step: [[p, u], T.void],
    engine_frequency: [[p], f],
    engine_set_throttle: [[p, f], T.void],
    engine_set_ignition: [[p, u], T.void],
    engine_set_starter: [[p, u], T.void],
    engine_set_spark_cut: [[p, f], T.void],
    engine_set_spark_retard: [[p, f], T.void],
    engine_set_boost_limit: [[p, f], T.void],
    engine_set_dyno: [[p, u, u, f], T.void],
    engine_dyno_torque: [[p], f],
    engine_filtered_dyno_torque: [[p], f],
    engine_speed: [[p], f],
    engine_set_speed: [[p, f], T.void],
    engine_set_load: [[p, f, f, f, f], T.void],
    engine_clutch_torque: [[p], f],
    engine_load_speed: [[p], f],
    engine_boost: [[p], f],
    engine_turbo_speed: [[p], f],
    engine_manifold_pressure: [[p], f],
    engine_set_audio: [[p, u], T.void],
    engine_audio_len: [[p], z],
    engine_audio_channels: [[p], z],
    engine_audio_ptr: [[p, z], p],
    engine_audio_clear: [[p], T.void],
    engine_duct_sound_len: [[p], z],
    engine_duct_sound_ptr: [[p], p],
    engine_duct_exits_len: [[p], z],
    engine_duct_exits_ptr: [[p], p],
    engine_duct_clear: [[p], T.void],
  };
  // A library older than the sources would run the old simulation, silently ignoring
  // spec fields it doesn't know.
  const { statSync, readdirSync } = await import("node:fs");
  const path = `${repo}/engine/target/release/libengine_sim.dylib`;
  const built = statSync(path).mtimeMs;
  const stale = readdirSync(`${repo}/engine/src`).filter(
    (f) => statSync(`${repo}/engine/src/${f}`).mtimeMs > built,
  );
  if (stale.length)
    throw new Error(`${path} is older than engine/src/${stale.join(", ")}: cargo build --release --lib in engine/`);
  const lib = dlopen(
    path,
    Object.fromEntries(Object.entries(sig).map(([k, [args, returns]]) => [k, { args, returns }])),
  );
  const exports: Record<string, unknown> = {};
  for (const k of Object.keys(sig)) exports[k] = (lib.symbols as any)[k];
  const native = Object.create(EngineModule.prototype);
  Object.assign(native, {
    exports,
    write(data: Uint8Array) {
      const ptr = (exports.alloc as (n: number) => number)(Math.max(1, data.length));
      new Uint8Array(toArrayBuffer(ptr, 0, Math.max(1, data.length))).set(data);
      return { ptr, len: Math.max(1, data.length) };
    },
    free(block: { ptr: number; len: number }) {
      (exports.dealloc as (p: number, n: number) => void)(block.ptr, block.len);
    },
    f32(ptr: number, length: number) {
      if (!ptr || length === 0) return new Float32Array(0);
      return new Float32Array(toArrayBuffer(ptr, 0, length * 4));
    },
  });
  EngineModule.load = async () => native;
}
