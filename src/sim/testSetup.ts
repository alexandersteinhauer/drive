// Preloaded by bun test (bunfig.toml): cars with simulated engines need the WebAssembly module.
import { loadEngineModule } from "./engineSim";

await loadEngineModule(() =>
  Bun.file(new URL("./engine.wasm", import.meta.url)).arrayBuffer(),
);
