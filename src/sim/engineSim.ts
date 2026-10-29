import type { EngineSpec } from "./engineSpec";

/**
 * The simulated combustion engine (engine/, Rust compiled to WebAssembly): load the module
 * once with `loadEngineModule`, then create engines from specs synchronously.
 */

interface Exports {
  memory: WebAssembly.Memory;
  alloc(len: number): number;
  dealloc(ptr: number, len: number): void;
  engine_new(ptr: number, len: number): number;
  engine_free(e: number): void;
  engine_step(e: number, steps: number): void;
  engine_frequency(e: number): number;
  engine_set_throttle(e: number, pedal: number): void;
  engine_set_ignition(e: number, on: number): void;
  engine_set_starter(e: number, on: number): void;
  engine_set_spark_cut(e: number, cut: number): void;
  engine_set_spark_retard(e: number, retard: number): void;
  engine_set_boost_limit(e: number, limit: number): void;
  engine_set_dyno(
    e: number,
    enabled: number,
    hold: number,
    speed: number,
  ): void;
  engine_dyno_torque(e: number): number;
  engine_filtered_dyno_torque(e: number): number;
  engine_speed(e: number): number;
  engine_set_speed(e: number, speed: number): void;
  engine_set_load(
    e: number,
    inertia: number,
    speed: number,
    torque: number,
    capacity: number,
  ): void;
  engine_clutch_torque(e: number): number;
  engine_load_speed(e: number): number;
  engine_boost(e: number): number;
  engine_turbo_speed(e: number): number;
  engine_manifold_pressure(e: number): number;
  engine_set_audio(e: number, on: number): void;
  engine_audio_len(e: number): number;
  engine_audio_channels(e: number): number;
  engine_audio_ptr(e: number, channel: number): number;
  engine_audio_clear(e: number): void;
  synth_new(
    channels: number,
    inputRate: number,
    hfGain: number,
    noise: number,
    jitter: number,
  ): number;
  synth_for_engine(ptr: number, len: number): number;
  acoustic_impulses(ptr: number, len: number): number;
  synth_free(s: number): void;
  synth_set_impulse_wav(
    s: number,
    channel: number,
    ptr: number,
    len: number,
    volume: number,
  ): number;
  synth_set_gain(s: number, gain: number): void;
  synth_set_input_rate(s: number, rate: number): void;
  synth_write(s: number, ptr: number, steps: number, channels: number): void;
  synth_input_available(s: number): number;
  synth_drop_input(s: number, keep: number): void;
  synth_render(s: number, frames: number): number;
  synth_rendered(): number;
  voice_for_engine(ptr: number, len: number): number;
  voice_free(v: number): void;
  voice_set_impulse_wav(
    v: number,
    index: number,
    ptr: number,
    len: number,
    volume: number,
  ): number;
  voice_set_input_rate(v: number, rate: number): void;
  voice_write(v: number, ptr: number, steps: number, channels: number): void;
  voice_input_available(v: number): number;
  voice_drop_input(v: number, keep: number): void;
  voice_render(v: number, frames: number): number;
}

/** One instance of the WebAssembly module (its own memory). */
export class EngineModule {
  readonly exports: Exports;

  constructor(instance: WebAssembly.Instance) {
    this.exports = instance.exports as unknown as Exports;
  }

  static async load(bytes: BufferSource): Promise<EngineModule> {
    const module = await WebAssembly.compile(bytes);
    return new EngineModule(await WebAssembly.instantiate(module, {}));
  }

  /** Copies `data` into the module's memory; free with `free`. */
  write(data: Uint8Array): { ptr: number; len: number } {
    const ptr = this.exports.alloc(data.length);
    new Uint8Array(this.exports.memory.buffer, ptr, data.length).set(data);
    return { ptr, len: data.length };
  }

  free(block: { ptr: number; len: number }): void {
    this.exports.dealloc(block.ptr, block.len);
  }

  f32(ptr: number, length: number): Float32Array {
    return new Float32Array(this.exports.memory.buffer, ptr, length);
  }
}

let shared: Promise<EngineModule> | null = null;
let loaded: EngineModule | null = null;

/** Loads the module for the game's simulation, once (from `bytes` if given). */
export function loadEngineModule(
  bytes?: () => Promise<BufferSource>,
): Promise<EngineModule> {
  shared ??= (async () =>
    (loaded = await EngineModule.load(
      bytes
        ? await bytes()
        : await (
            await fetch(new URL("./engine.wasm", import.meta.url))
          ).arrayBuffer(),
    )))();
  return shared;
}

/** The loaded module; cars with simulated engines need it before they are made. */
export function engineModule(): EngineModule {
  if (!loaded)
    throw new Error("engine module not loaded: await loadEngineModule()");
  return loaded;
}

/** A simulated engine. Its crank speed is positive forwards (rad/s). */
export class EngineSim {
  private readonly e: number;
  readonly frequency: number;

  constructor(
    readonly spec: EngineSpec,
    readonly module: EngineModule = engineModule(),
  ) {
    const json = module.write(new TextEncoder().encode(JSON.stringify(spec)));
    this.e = module.exports.engine_new(json.ptr, json.len);
    module.free(json);
    if (!this.e) throw new Error("engine spec rejected");
    this.frequency = module.exports.engine_frequency(this.e);
    this.collectAudio = false;
  }

  dispose(): void {
    this.module.exports.engine_free(this.e);
  }

  step(steps: number): void {
    this.module.exports.engine_step(this.e, steps);
  }

  set throttle(pedal: number) {
    this.module.exports.engine_set_throttle(this.e, pedal);
  }

  set ignition(on: boolean) {
    this.module.exports.engine_set_ignition(this.e, on ? 1 : 0);
  }

  set starter(on: boolean) {
    this.module.exports.engine_set_starter(this.e, on ? 1 : 0);
  }

  /** Fraction of sparks skipped, spread evenly over the cylinders (0..1). */
  set sparkCut(cut: number) {
    this.module.exports.engine_set_spark_cut(this.e, cut);
  }

  /** Spark retard from the map (rad): the ECU's fast torque reduction. */
  set sparkRetard(retard: number) {
    this.module.exports.engine_set_spark_retard(this.e, retard);
  }

  /** Cap on the turbo's boost target (gauge Pa). */
  set boostLimit(limit: number) {
    this.module.exports.engine_set_boost_limit(this.e, limit);
  }

  /** Crank speed, rad/s. */
  get speed(): number {
    return this.module.exports.engine_speed(this.e);
  }

  set speed(v: number) {
    this.module.exports.engine_set_speed(this.e, v);
  }

  /** Holds the crank at `speed` (rad/s) with a dynamometer, or releases it. */
  dyno(enabled: boolean, speed = 0, hold = true): void {
    this.module.exports.engine_set_dyno(
      this.e,
      enabled ? 1 : 0,
      hold ? 1 : 0,
      speed,
    );
  }

  /** Torque the dynamometer absorbs, averaged over the last engine cycle (N m). */
  get dynoTorque(): number {
    return this.module.exports.engine_filtered_dyno_torque(this.e);
  }

  /**
   * The driven side, seen from the crank, for the next steps: inertia (kg m^2), speed (rad/s),
   * the torque on it (N m, forwards positive) and the torque the clutch can pass.
   */
  setLoad(
    inertia: number,
    speed: number,
    torque: number,
    capacity: number,
  ): void {
    this.module.exports.engine_set_load(
      this.e,
      inertia,
      speed,
      torque,
      capacity,
    );
  }

  /** Turbocharger boost over the atmosphere (Pa); 0 without one. */
  get boost(): number {
    return this.module.exports.engine_boost(this.e);
  }

  /** Turbocharger shaft speed (rad/s). */
  get turboSpeed(): number {
    return this.module.exports.engine_turbo_speed(this.e);
  }

  /** Pressure behind the throttle (Pa). */
  get manifoldPressure(): number {
    return this.module.exports.engine_manifold_pressure(this.e);
  }

  /** Clutch torque on the driven side, averaged over the steps since `setLoad` (N m). */
  get clutchTorque(): number {
    return this.module.exports.engine_clutch_torque(this.e);
  }

  /** Whether to keep the exhaust signal for `takeAudio` (off: it is dropped). */
  set collectAudio(on: boolean) {
    this.module.exports.engine_set_audio(this.e, on ? 1 : 0);
  }

  /** The exhaust signal since the last call, per exhaust system (one value per step). */
  takeAudio(): Float32Array[] {
    const x = this.module.exports;
    const n = x.engine_audio_len(this.e);
    const channels = x.engine_audio_channels(this.e);
    const out: Float32Array[] = [];
    for (let c = 0; c < channels; c++)
      out.push(this.module.f32(x.engine_audio_ptr(this.e, c), n).slice());
    x.engine_audio_clear(this.e);
    return out;
  }
}
