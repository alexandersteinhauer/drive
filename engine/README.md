# engine

The game's combustion engine: a Rust port of [engine-sim](https://github.com/ange-yaghi/engine-sim)
by Ange Yaghi (MIT). It simulates the crankshaft, pistons and rods as constrained rigid bodies,
the gas in every cylinder, runner, intake and exhaust, the valvetrain, ignition and combustion,
at 10 kHz, and synthesizes the sound from the exhaust pulses.

- `src/engine.rs`: the engine and its simulation step; `src/spec.rs`: the engine as data
  (built in TypeScript, `src/sim/engines/`); `src/gas.rs`, `src/solver.rs`, `src/function.rs`:
  gas dynamics, constraint solver, sampled curves; `src/synth.rs`: the sound;
  `src/wasm.rs`: the WebAssembly interface.
- `./build.sh` compiles it to `../src/sim/engine.wasm` (needs the `wasm32-unknown-unknown`
  target: `rustup target add wasm32-unknown-unknown`).
- `reference/build.sh` builds the original C++ natively and records a run of the Beetle;
  `cargo test --release` then checks the port reproduces it exactly, step for step and sample
  for sample (`bun run test-engine` does both).

Differences from engine-sim: no script compiler (engines are data), no UI, single-threaded
synthesizer; rand() is a small fixed generator so runs repeat.
