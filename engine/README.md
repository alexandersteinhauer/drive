# engine

The game's combustion engine: a Rust port of [engine-sim](https://github.com/ange-yaghi/engine-sim)
by Ange Yaghi (MIT). It simulates the crankshaft, pistons and rods, the gas in every cylinder, runner, intake and exhaust, the valvetrain, ignition and combustion,
at 10 kHz, and makes the sound from its exhaust valves' flow.

- `src/engine.rs`: the engine and its simulation step; `src/spec.rs`: the engine as data
  (built in TypeScript, `src/sim/engines/`); `src/crank.rs`: the crank mechanism (below);
  `src/gas.rs`, `src/solver.rs`, `src/function.rs`: gas dynamics, engine-sim's constraint
  solver, sampled curves; `src/voice.rs`: the sound (below); `src/synth.rs`: engine-sim's
  synthesizer; `src/acoustics.rs`: the exhaust's acoustics (below); `src/turbo.rs`: turbochargers (below);
  `src/duct.rs`: the exhaust's gas dynamics for the reference simulation (below);
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

## Reference simulation

The voice approximates what the exhaust does to the valves' flow (linear acoustics, a
steepener, losses fitted to a recording). The reference simulation computes it instead,
offline: `src/duct.rs` replaces the exhaust runners and collectors with pipes and volumes
(`ducts` in the spec), and the sound is the pressure their open ends radiate.
`examples/reference.rs` runs an engine with them on the dynamometer and writes it;
`scripts/reference-drive.ts` drives the game's car with them (through the native library,
`scripts/native-engine.ts`), the 488 Pista's exhaust from `scripts/pista-reference.json`; and
`scripts/reference_sound.py` makes what was recorded the sound heard. Natively, with the
`parallel` feature, the pipes are stepped on `DUCT_THREADS` threads (`src/pool.rs`: the pipes
split among the threads once, longest processing time first, each thread taking its own every
stage (its pipes' data staying in its core's cache), its pipes' ends and volute channels with
them, the threads spinning between stages, the nodes after them in order, so the result is the
same to the bit): 3.9 times as fast on 8 threads of an M-series Mac (18 cores; past 8 hardly
faster, a stage being ~100 microseconds of work).

- Each pipe is a column of finite volumes of the one-dimensional Euler equations with the
  gas's composition, fifth order in space (WENO-Z, a cell's values at both its faces from one
  set of smoothness indicators, by variable along the pipe) and third in time (SSP
  Runge-Kutta), HLLC fluxes, in the cylinders' real gas (so energy passes between them exactly). Cells are sized
  for 30 to the wavelength at 3 kHz in each pipe's gas: numerical damping ~0.02 dB/m there.
- A pipe is a run of sections: straight, cones (the area varying cell to cell, the walls
  pushing on the gas by p dA), a catalyst's monolith (its channels' open area and hydraulic
  diameter), packed. Lumped volumes are kept to real junctions: at kilohertz a volume isn't
  compact (a 0.4 L one feeding a 63 mm pipe cuts off near 700 Hz), where cones and volutes
  are ducts.
- Where the section steps, each side's state lies on its own wave curve, mass and total
  enthalpy pass, and the momentum balance is Borda and Carnot's out of the narrower side
  (losing (u1 - u2)^2 / 2 of head) and isentropic into it; the step's wall takes the
  difference between the sides' fluxes. (A Riemann solver across the step instead would need
  a pressure jump of rho c times the velocity's to pass the flow.)
- Walls: friction and heat transfer from the local Reynolds number (Haaland; Colburn, or
  laminar flow's constants for round and square channels); a packing's flow resistivity
  (Darcy).
- A pipe's end meets a node (a cylinder, a junction or chamber's volume, the atmosphere)
  through an opening: a half Riemann problem, the end's state on its gas's wave curve, passing
  what the opening does, quasi-steady and compressible (as a flow bench defines the port's
  discharge coefficient). The exhaust valve opens by the port's flow-bench curve. Into the
  pipe, the jet through the opening mixes out to the subsonic state with its mass flow and
  total enthalpy (at most sonic): out of a valve into a wider pipe it doesn't expand on
  supersonically, as through a nozzle's diverging part. Through the pipe's own section the
  inflow is isentropic, its face sonic at the critical pressure once it chokes (finding it
  from its mass flux is ill-conditioned near sonic, and rang at kilohertz there); a wall is
  exact, nothing passing. Each end's and each step's solution is found within a bracket about
  its last (0.1 % either side, widened fourfold until it holds; else over the whole range),
  by Illinois's regula falsi till a step moves it less than 1e-10 of the pressure (~1e-5 Pa;
  against bisection to 1e-15 it is then within 1e-8, where waiting for the bracket itself to
  close took a third more evaluations); a step's over its upstream end's pressure, whose wave
  curve gives the mass flow directly. (Searching over the mass flow instead, each trial
  inverted the wave curve: twice the work.) The states carrying a given mass flux (into a pipe
  through a narrower opening; into a step's narrower side) by Newton's method over the
  velocity: the flux is concave in it below sonic, so from rest each step lands below the
  root, closer.
- Where a strong rarefaction outruns the high-order scheme (a stage would leave a cell
  without internal energy), the substep is done again in halves, at the last with
  first-order fluxes, which HLLC keeps positive.
- Turbines: a quasi-two-dimensional volute (Galindo, Tiseira, Fajardo and García-Cuevas,
  Energy Convers. Manage. 85, 2014): each scroll a tapered duct to its tongue, every cell of it
  discharging through its own stator channel (`nozzles`) into the stator and rotor's space, a
  rotor nozzle from there (reaction 0.5: each nozzle half the expansion), the
  stage's isentropic work at the turbo's efficiency taken in between, the wastegate's ports in
  the volutes' feeds, opened by the boost controller; the shaft gets the power. A lumped
  turbine (one nozzle between small volumes) let the kilohertz through almost whole; the
  volute's averaging over its 360 degrees takes 24 dB at 3 kHz off them here, as measured
  turbines do (17-33 dB at 1-2 kHz: Kabral et al., ETC 2017; Tiikoja et al., SAE
  2011-01-1525).
- Orifices between nodes (turbine rotors, wastegates) and a volute's stator channels: the gas
  in the opening moves as a slug, d m/dt = Cd A / l (dp - m|m| / (2 rho Cd^2 A^2 Y^2)), its
  length l the passage and its end corrections, Y the compressible expansion factor of the
  pressure ratio: once steady exactly the quasi-steady compressible (and choked) flow; at
  any time at most what the opening chokes at (a wastegate closing on its flow stops its gas
  as fast, rather than the loss, ~ m^2 / A, growing without bound). The
  quasi-steady flow alone, ~ sqrt(dp), has infinite conductance at no flow: between small
  volumes (a 50 cc rotor space, a 30 cc turbine outlet) it chattered at idle, 6-9.5 kHz up
  30 dB at the turbine's outlet, heard as bit crushing; with the slug's mass it is 60 dB down.
  `tests/duct.rs` checks the steady flow against the orifice's, a vessel's Helmholtz
  frequency through a hole, and a valve closing on its flow.
- Perforated walls (a silencer's core): each cell's holes a slug of gas driven by the
  pressure across it against its jet's loss and viscous drag, its mass the holes' gas over
  the wall and their end corrections.
- Flaps (a silencer's bypass valves, `ducts.flaps`): a pipe's end shut or opening by the
  crank speed or the pedal, whichever opens it more, as the voice's valves.
- Open ends radiate: a resistance in parallel with the end correction's mass, the
  monopole's power at low frequency (within 1 % of Levine and Schwinger's reflection up to
  ka = 0.8, ~1.1 kHz for an 80 mm tailpipe; too reflective above, 0.57 for 0.50 at ka = 1.5).
  The listener hears that power as an unflanged pipe beams it (`scripts/reference_sound.py`:
  Levine and Schwinger's power gain at the listener's angle, +2.6 dB at ka = 1 and +8.3 dB at
  ka = 3, 20 degrees off the axis; without the jet's flow, which quietens the axis itself).
- Combustion varies from cycle to cycle (`physics.cycleVariation`): each burn's flame speed
  scaled by a normal factor, sigma 0.1 for the F154, which gives the literature's
  coefficients of variation (IMEP 1.6-2.6 %, peak pressure 9-11 % at full load; IMEP ~9 % at
  idle: Heywood 9.4).
  Each cylinder draws its burns' randomness from its own sequence, three numbers every
  ignition event, so a burn's depends on the cylinder and its count alone (without cycle
  variation, engine-sim's single sequence, as tests/reference.rs needs).

`examples/abtrace.rs` runs the engine on the dynamometer with the reference exhaust or the
game's lumped one, recording each step's state (each cylinder's pressure, temperature and
exhaust valve flow; torque, boost, turbine power, wastegate) and, with the ducts, each open
end's sound before directivity: the dyno holds the crank to the program's speed and the burns'
randomness is the cylinders' own, so the two engines run the same crank motion and burns
(peak pressures correlated 0.99-1.0 cycle by cycle), and differ only by their exhausts.

`tests/duct.rs` checks it against exact solutions: Sod's shock tube, the speed of sound and
reflections off closed and open ends, a choked nozzle's flow, Darcy's friction, a
perforation's Helmholtz resonance, a pulse through a cone (pressure as A^-1/2), a sudden
expansion's loss (Borda-Carnot, within 0.5 %), and conservation of mass and energy; and that
shock tubes across steps, a catalyst heating its gas, and a blowdown into a wider pipe (its
end subsonic) stay bounded.

The pipes carry no turbulence: the tailpipe jets' mixing noise is worked out afterwards
(`scripts/reference_sound.py`) from the simulated exit flow (each open end's velocity, density and temperature, recorded with
the sound), quasi-steadily, by Stone, Groesbeck and Zola's method for static round jets, hot
or cold (NASA TM-82712, 1981: eq. A1 for the level, Table I for the 1/3-octave spectrum by
angle and Strouhal number), which agrees within 1-3 dB with NASA's SHJAR measurements of hot
subsonic jets (Khavaran and Bridges, NASA/TM-2009-215608) at 150-300 m/s. (Tam's
similarity spectra fit its shapes; his own level fits, made for supersonic and cold jets, have
heating quieten a slow jet, where measurements show it louder.) Quasi-steady is marginal:
an engine's pulses come within a factor of a few of the noise's peak frequency.

Not yet modelled: turbulence noise from the flow inside (at the catalyst, the perforations),
the frequency-dependent wall losses of the acoustic boundary layer (Kirchhoff's,
~0.4-1 dB/m at 1 kHz in these pipes), the walls' own temperature (set per pipe), the
intake's sound.

### Learned exhaust surrogate (experimental)

The reference is also a teacher for a per-car real-time approximation. This pipeline is
experimental: **the voice remains the default**, and a good fit to the reference is not
independent evidence of fidelity to a real car. The Pista can already audition the pilot:
`?car=pista&exhaust=surrogate`, or Escape → Pista exhaust → Learned (experimental).
The original mode is `?car=pista&exhaust=original`. Other cars keep their existing sound.

The shipped pilot (`public/audio/surrogate/pista-pilot.json`, provenance alongside) predicts
only the two tailpipes' radiated pressure before directivity. It does **not** yet include jet,
mechanical or turbo noise, and it was trained before the sonic-boundary correction. It has
known transient/high-band errors, not production fidelity. The actual game engine stays at
its original frequency/fluid steps and keeps the same torque/boost physics; a separate
physical kg/s + five-control stream feeds `src/surrogate_stream.rs`. That adapter resamples
to the 20 kHz model and the audio device, divides by a fixed 16 Pa/full-scale calibration,
then uses the game's existing loudness/output controls. It cannot recover source bandwidth
missing from the game's 10 kHz control stream. Training also used the reference config's
cycle variation and flat boost target, whereas the game keeps its original settings; this
is an additional provisional input-distribution gap, not a hidden driving-physics change.
Model load/init/runtime failure creates a new original-voice worklet and changes the simulation stream atomically, with a visible warning.
Cached audio nodes are muted/suspended and their histories cleared when cars switch.

- `examples/abtrace.rs`: paired game/reference traces, crank trajectory held by the dyno,
  separate burn seeds for training and held-out drives. Records physical pressures before
  listener directivity, and continuous exit states, not separate fictitious event targets.
- `examples/superpose.rs`: a deep clone of one warm state, then independent single/pair
  burn perturbations; `scripts/superpose_analyse.py` measures the second difference,
  `y(a+b) - y(a) - y(b) + y(base)`. An unchanged branch must replay bit for bit. Small
  perturbations test a local linearization, not full-amplitude superposition.
- `scripts/surrogate_fit.py`: the first learned baseline, a control-gated bank of causal
  FIRs with signed-power flow features. It is regularized regression, not a raw-waveform
  neural network. Controls are game-available rpm, manifold pressure, wastegate, throttle,
  and the flap actuator's state; the targets are each outlet's reference pressure. Gates
  act on the current output, so already-running tails respond to changed controls.
  Pressure filters are constrained to zero DC. Matrix-free least squares has an explicitly
  tested adjoint; diagonal preconditioning preserves the original ridge penalty.
- `src/surrogate.rs`: the same fitted model in Rust and WASM, with shared input FFTs,
  summed frequency-domain MIMO convolution, bounded normalized gates, input validation
  before state changes, and no allocations while processing. It does not use `voice.rs`
  or `acoustics.rs`, resample, add noise, normalize levels or realign phase.
- `src/jet.rs`: the reference's Stone mixing noise at runtime, from each outlet's continuous
  velocity/density/temperature. Its 1/3-octave filters have analytically normalized white-noise
  gain (no whole-clip normalization). Levels match the Python formula to roundoff; random
  samples need not. New band levels ramp causally over 250 us, rather than using the future
  control frame. `examples/jetnoise.rs` renders 4 kHz exit states at 48 kHz.

Run the pressure-only pilot end to end (native data, fitting, held-out metrics and A/B WAVs,
then native and WASM replay/benchmarks), writing large temporary files outside the repo:

```sh
bash engine/scripts/surrogate-pilot.sh /absolute/path/to/pilot
```

Or use the individual tools. `scripts/test_surrogate_fit.py` checks causality, indexing,
identification on an independent synthetic trajectory, and preservation of the regularized
objective under preconditioning. A local steady-state fit alone is not an acceptance test:
hold out entire trajectories and random seeds, check shifts/lift/reapply/flaps/thermal
history, compare game-input vs reference-input fits, inspect harmonic and band errors, and
measure the actual WASM decoder. The optional exit-state fit also needs noise-level and
modulation validation; do not feed reference-only exit states to a deployed student.

Training is before `reference_sound.py`'s whole-record zero-phase directivity. That filter
is not a causal runtime model. Listener directivity/resampling and their explicit latency
remain separate from identifying the causal exhaust. The pressure pilot is 20 kHz (below
10 kHz bandwidth); its WAVs are not a claim to reproduce the reference's full 24 kHz band.

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
