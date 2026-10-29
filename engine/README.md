# engine

The game's combustion engine: a Rust port of [engine-sim](https://github.com/ange-yaghi/engine-sim)
by Ange Yaghi (MIT). It simulates the crankshaft, pistons and rods, the gas in every cylinder, runner, intake and exhaust, the valvetrain, ignition and combustion,
at 10 kHz, and makes the sound from its exhaust valves' flow.

- `src/engine.rs`: the engine and its simulation step; `src/spec.rs`: the engine as data
  (built in TypeScript, `src/sim/engines/`); `src/crank.rs`: the crank mechanism (below);
  `src/gas.rs`, `src/solver.rs`, `src/function.rs`: gas dynamics, engine-sim's constraint
  solver, sampled curves; `src/voice.rs`: the sound (below); `src/synth.rs`: engine-sim's
  synthesizer; `src/acoustics.rs`: the exhaust's acoustics (below); `src/turbo.rs`: turbochargers (below);
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

## Sound

The game's cars sound through the voice (`src/voice.rs`, `audio.voice` in the spec), which
replaced engine-sim's synthesizer (`src/synth.rs`, kept for the reference test) after a
comparison with recordings of a real 488 Pista (a dynamometer, an onboard run, cold
starts). engine-sim filters an exhaust signal sampled from its gas dynamics; at 10 kHz and
behind its 1.9 kHz anti-aliasing filter that had no high frequencies, its pressure source
thumped with every swing of a turbine's back-pressure, and the simulated runners' sound above
a few hundred Hz depends on the fluid step (0D volumes ringing numerically), so it isn't a
prediction to listen to. The voice takes from the simulation what it predicts (each exhaust
valve's flow, the cylinders' burns and their variation, the turbo's state) and models the
rest directly:

- The source: each valve's flow over the step, as the volume velocity at the acoustics'
  density (the engine's control stream, `Engine::write_voice`), interpolated to 44.1 kHz:
  so it launches the wave the real flow does, c m' / A, whatever the runner's
  back-pressure. Every blowdown's strength varies at random beyond the simulation's
  (cycle-to-cycle variation, more at light load).
- The exhaust's acoustics (below), from each port to the listener. A turbocharger's
  wastegates (`acoustics.wastegate`) are a way round the turbines, opened as the engine's
  are: the manifolds' responses are computed at several openings and blended, so the
  pulses get past louder as the boost controller opens them at high revs, and the exhaust
  goes quiet on a closed throttle (the opening wastegate changes the spectrum: Kabral et
  al., ETC 2017). Valves (`acoustics.valves`, a silencer's bypass) open with revs and
  load, crossfading between the responses with them shut and open.
- Where the pulses turn nonlinear (`acoustics.nonlinear`: a turbo engine's primaries, where
  they are near a bar, or each port's pipe without a turbine), they steepen as they travel,
  crests outrunning troughs into shocks (Burgers'). Each slice of pipe warps the wave's time
  by its pressure and takes each sample as the mean of the folded wave over its interval,
  counting the stretches running backwards negative: Whitham's equal-area rule, which puts
  the shocks where the fold's lobes cancel, keeps the mean, and takes the steepened excess
  away, so a wave saturates (up to far beyond an exhaust's pulses). Loud pulses gain the
  high harmonics no linear pipe makes, and only loud ones (Selamet et al., SAE 880082): an
  exhaust's rasp, and its brightening with load.
- Turbulence: the valve jets', carried through the acoustics, and the tailpipe jets'
  (Lighthill's U^8), pulsing with their flow, both near their physical estimates.
- The engine bay (the manifolds' shells radiating the pressure inside, through the walls'
  mass law), valves seating and injectors ticking, and the turbocharger: compressor noise,
  its blade-pass whine, the blow-off valve (the F154's recirculate: faint).

Levels are pascals at the listener (the chase camera) over a full scale per car. The 488
Pista's exhaust lengths, areas and losses that the published figures leave open were fitted
to a dynamometer recording of the real car by what an exhaust's sound is made of: its
harmonics (energy on the engine's half orders, crank-angle domain), how much of each band
they are, and the band levels up to 5 kHz. A first fit to band levels alone filled the bands
with noise (the recording's highs are partly the dynamometer cell's): the engine sounded like
air rushing in rather than an exhaust. Now the highs are the exhaust's own harmonics, from
its shocks; the primaries' losses to their large pulses are alike at every frequency (the
wall losses' growth with frequency would take those harmonics away again). At full throttle
3000-7500 rpm, 500-1500 Hz is 83-98 % harmonics (the real car 78-86 %; the synthesizer with
computed acoustics fitted to band levels 17-57 %); lifting off at 6000 rpm drops the level
by ~20 dB (the real car's recording 6-15, perhaps compressed; the synthesizer's rose by 16);
firing to firing, the pulses' energy varies about as much as the real car's (15-18 %). The
voice costs ~3 % of real time (the synthesizer ~6 %).

### Exhaust acoustics

engine-sim's gas model lumps each pipe and the muffler into single volumes: right for flow and
back-pressure, but they carry no sound waves, so engine-sim takes a muffler's character from a
recorded impulse response. `src/acoustics.rs` computes it instead, from the exhaust's
geometry (`audio.acoustics` in the spec): a linear acoustic network of pipes (plane waves at
the hot gas's speed of sound, wall losses, absorption for packed pipes), orifices (perforated
cones and caps), volumes and radiating tailpipe ends, solved per frequency into impulse
responses from each exhaust valve to a listener behind the car (or split at a junction, for
the voice's nonlinear stretch). Tests check it against the expansion chamber's textbook
transmission loss. A choked blowdown is a flow the pipe's waves can't push back on: the ports
are rigid ends (no volume behind them).

The mean flow damps the exhaust, most at the low frequencies where its modes boom: each
restriction resists a small change of its flow by its pressure loss's slope (quasi-steady
theory, valid below a Strouhal number of ~1: Durrieu, Hirschberg, Peters et al., JASA 110,
2001). Orifices and the turbine lose their jet, m' / (Cd^2 A^2); the tailpipe ends theirs,
m' / A^2, which takes ~4 M of a wave's energy per reflection as measured for open pipes with
flow (Peters et al., J. Fluid Mech. 256, 1993); pipes add turbulent wall friction. A test
checks a pipe's damped mode against the sum of these losses. The voice computes its responses
at one flow (`voice.acousticFlow`, the engine working hard).

## Physics beyond engine-sim's

engine-sim's thermodynamics are simplified in ways that make an engine far too efficient
(the F154 reached ~50 % brake efficiency at a realistic burn, where the ideal fuel-air cycle
is ~45 % and real engines ~35 %), which its example engines make up for with burns two to
three times too slow. `physics` in the spec (opt-in; without it the port is engine-sim's,
step for step) corrects them:

- `realGas` (gas.rs): the heat capacity ratio falls with temperature, gamma(T) = 1.3678 -
  8.13e-5 (T - 300) (Gatowski et al., SAE 841359, as fitted to equilibrium burned gas in
  Klein, "Single-zone cylinder pressure modeling", Linköping 2004), with the internal energy
  its exact integral; and gas flowing between volumes carries its enthalpy, not just its
  internal energy (engine-sim leaves out the flow work: a cylinder blowing down didn't cool).
  The turbocharger uses the same properties for the hot exhaust.
- `wallTemperature`: Woschni's heat transfer to the cylinder walls (SAE 670931, 1967; SI
  form and constants from Klein), in place of engine-sim's fixed 100 W/m^2 K.
- The mean piston speed (which sets the flame's turbulence) is 2 x stroke x revolutions
  per second, not engine-sim's average of 256 sampled slots, which reads 20-40 % low when a
  cycle takes a whole number of steps (5000, 6000, 7500 rpm at 10 kHz) and slowed the burn.

The engines then use gasoline's lower heating value (44 MJ/kg, Heywood) and a flame speed
calibrated to the textbook criterion for best torque, half the charge burned ~8 degrees
after top dead centre (Heywood), with a 10-90 % burn of ~20-33 degrees. The Beetle then makes
VW's published torque curve at VW's published distributor timing; the Ferrari Ferrari's at
the published ~1.5 bar of boost with knock-limited timing (below). `ignition.fullLoad` adds
a second timing curve by load, blended by manifold pressure as an ECU's spark table is:
best-torque timing at part load, knock-limited under boost (the knock limit is an estimate
from the CRC AVFL-20 dynamometer data; the simulation doesn't model knock).
`examples/calibrate.rs` measures and fits all of it on the dynamometer.

What stays estimated: the knock limit, the turbine and port sizes (fitted to the published
torque and boost), and the spark efficiency the ECU's torque control uses (measured on the
simulated engine; about 1.5 times as sensitive to retard as Ayala et al.'s generalised curve,
SAE 2006-01-0229). Below 3000 rpm the turbos spool later than the real engine's twin-scroll
units, whose use of the individual exhaust pulses the model leaves out.

## Turbochargers

engine-sim has none; `src/turbo.rs` adds one (`turbo` in the spec), kept to the shaft's power
balance. The exhaust leaves each collector through the turbine and a wastegate beside it,
two openings sharing the flow; the turbine takes the isentropic work of its share (the
gas's temperature, the pressure drop to the atmosphere) at its efficiency; the compressor
does Euler's work, `0.9 U^2` per kg at its wheel's tip speed U, on the air the engine draws
through the throttle, which sets its pressure ratio; the shaft's inertia and bearing losses
make the lag. A PI boost controller opens the wastegate as the shaft passes the speed of
the boost the ECU targets (a map by rpm, which the game can cap per gear), which also lowers
the back-pressure the engine pushes against; a blow-off valve
caps the charge at the manifold's pressure plus its spring as the throttle closes, and an
intercooler cools the charge. The throttle then draws from the charge instead of the
atmosphere. Without a turbo the engine is engine-sim's, step for step.
`Engine::set_spark_retard` and `set_spark_cut` are the ECU's fast torque path: the game's
traction and stability control realise their torque requests as Bosch's torque-based ECUs do
(ME7): the throttle for lasting cuts, spark retard (up to 30 degrees) to bridge the time the
charge takes to fall, and cylinder cut only beyond that (src/sim/powerplant.ts); dual-clutch
upshifts cut briefly.
`scripts/engine-dyno.ts` prints an engine's full-throttle torque, power and boost.

Differences from engine-sim: no script compiler (engines are data), no UI, single-threaded
synthesizer; rand() is a small fixed generator so runs repeat. In the game the engine drives
the car: `Engine::set_load` puts the driven wheels (through the gearing) on the far side of
the clutch: slipping, it passes its capacity; locked, the two turn as one (src/sim/powerplant.ts).
`examples/bench.rs` times it (~6.5 % of real time natively, ~7.5 % in WebAssembly in V8;
`CONSTRAINED=1` for engine-sim's mechanism).
