# engine

The game's combustion engine: a Rust port of [engine-sim](https://github.com/ange-yaghi/engine-sim)
by Ange Yaghi (MIT). It simulates the crankshaft, pistons and rods, the gas in every cylinder, runner, intake and exhaust, the valvetrain, ignition and combustion,
at 10 kHz, and synthesizes the sound from the exhaust pulses.

- `src/engine.rs`: the engine and its simulation step; `src/spec.rs`: the engine as data
  (built in TypeScript, `src/sim/engines/`); `src/crank.rs`: the crank mechanism (below);
  `src/gas.rs`, `src/solver.rs`, `src/function.rs`: gas dynamics, engine-sim's constraint
  solver, sampled curves; `src/synth.rs`: the sound;
  `src/acoustics.rs`: the exhaust's acoustics (below);
  `src/wasm.rs`: the WebAssembly interface.
- `./build.sh` compiles it to `../src/sim/engine.wasm` (needs the `wasm32-unknown-unknown`
  target: `rustup target add wasm32-unknown-unknown`).
- `reference/build.sh` builds the original C++ natively and records a run of the Beetle;
  `cargo test --release --features reference` then checks the port reproduces it exactly,
  step for step and sample for sample (`bun run test-engine` does both), with engine-sim's
  mechanism (`Engine::constrained`). The `reference` feature keeps engine-sim's arithmetic
  where the game uses a faster equivalent (`x^(1/1.4)` in subsonic flow, accurate to an ulp).

## Crank mechanism

engine-sim makes the crank, each rod and each piston a rigid body and joins them with
constraints (pins, the cylinder wall), solved each step by Gauss-Seidel to a loose tolerance,
so the joints give a little and the solve is most of a step's cost. The game's `Crank`
(`src/crank.rs`) has one coordinate instead, the crank angle: the pistons and rods follow from
it exactly, and the crank moves by the mechanism's Lagrangian, its inertia varying with the
angle and the gas pressures acting through the pistons' strokes, integrated symplectically
(energy holds to 0.06 % over 100 s of free spin). The wall's reaction on each piston, which
sets its friction, comes from the rod's and the piston's equations of motion. What acts on
the crank with a limited torque (bearing friction, clutch, drag, dyno, starter) is solved
exactly. `tests/mechanism.rs` checks it against engine-sim's mechanism: torque within 0.5 N m
on the dyno, speed within 0.6 % revving freely.

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
the clutch: slipping, it passes its capacity; locked, the two turn as one (src/sim/powerplant.ts).
`examples/bench.rs` times it (~6.5 % of real time natively, ~7.5 % in WebAssembly in V8;
`CONSTRAINED=1` for engine-sim's mechanism).
