// Types for the Emscripten build of engine-sim (engine-sim/build.sh, bridge.cpp).
export interface EngineSimModule {
  HEAPF32: Float32Array;
  stringToNewUTF8(s: string): number;
  _free(ptr: number): void;
  /** Compiles the engine script; 0 on success. */
  _es_init(scriptPath: number, simulationFrequency: number): number;
  _es_set_controls(rpm: number, throttle: number, engaged: number): void;
  _es_set_volume(volume: number): void;
  /** Simulates and returns a pointer to `frames` float samples at 44.1 kHz. */
  _es_render(frames: number): number;
  /** How closed the throttle is while the dyno holds the engine, 0..1. */
  _es_overrun(): number;
  _es_engine_rpm(): number;
  _es_dyno_torque(): number;
}

export default function createModule(options?: {
  instantiateWasm?: (
    imports: WebAssembly.Imports,
    done: (instance: WebAssembly.Instance) => void,
  ) => object;
  print?: (s: string) => void;
  printErr?: (s: string) => void;
}): Promise<EngineSimModule>;
