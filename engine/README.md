# engine

The game's combustion engine: a Rust port of [engine-sim](https://github.com/ange-yaghi/engine-sim)
by Ange Yaghi (MIT). It simulates the crankshaft, pistons and rods as constrained rigid bodies,
the gas in every cylinder, runner, intake and exhaust, the valvetrain, ignition and combustion,
at 10 kHz, and synthesizes the sound from the exhaust pulses.

- `src/engine.rs`: the engine and its simulation step; `src/spec.rs`: the engine as data
  (built in TypeScript, `src/sim/engines/`); `src/gas.rs`, `src/solver.rs`, `src/function.rs`:
  gas dynamics, constraint solver, sampled curves; `src/synth.rs`: the sound;
  `src/acoustics.rs`: the exhaust's acoustics (below);
  `src/wasm.rs`: the WebAssembly interface.
- `./build.sh` compiles it to `../src/sim/engine.wasm` (needs the `wasm32-unknown-unknown`
  target: `rustup target add wasm32-unknown-unknown`).
- `reference/build.sh` builds the original C++ natively and records a run of the Beetle;
  `cargo test --release --features reference` then checks the port reproduces it exactly,
  step for step and sample for sample (`bun run test-engine` does both). The `reference`
  feature keeps engine-sim's arithmetic where the game uses a faster equivalent (`x^(1/1.4)`
  in subsonic flow, accurate to an ulp).

## Exhaust acoustics

engine-sim's gas model lumps each pipe and the muffler into single volumes: right for flow and
back-pressure, but they carry no sound waves, so engine-sim takes a muffler's character from a
recorded impulse response. `src/acoustics.rs` computes it instead, from the exhaust's
geometry (`audio.acoustics` in the spec): a linear acoustic network of pipes (plane waves at
the hot gas's speed of sound, wall losses, absorption for packed pipes), orifices (perforated
cones and caps), volumes and radiating tailpipe ends, solved per frequency into one impulse
response per cylinder, from its exhaust valve to a listener behind the car. The engine then
emits each cylinder's pulse at the valve, and the synthesizer convolves them (partitioned FFT
convolution). Tests check it against the expansion chamber's textbook transmission loss.

Differences from engine-sim: no script compiler (engines are data), no UI, single-threaded
synthesizer; rand() is a small fixed generator so runs repeat. In the game the engine drives
the car: `Engine::set_load` puts the driven wheels (through the gearing) on the far side of
the clutch, and the clutch is solved with the rest of the constraints (src/sim/powerplant.ts).
`examples/bench.rs` times it (~9 % of real time natively, ~10 % in WebAssembly).
