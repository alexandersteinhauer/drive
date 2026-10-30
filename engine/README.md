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
  gas's composition, WENO-Z reconstruction (a cell's values at both faces from one set of
  smoothness indicators, by variable along the pipe), third-order SSP Runge-Kutta and HLLC
  fluxes, in the cylinders' real gas. WENO's reconstruction is fifth order, but it acts on
  primitives inferred from conservative cell averages, and incoming boundary ghosts use a
  lower-order constant extension: general nonlinear fifth-order convergence is **not**
  established. Cells are sized for 30 to the wavelength at 3 kHz in each pipe's gas; the
  linear propagation test gives numerical damping ~0.02 dB/m there.
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
  exact, nothing passing. For a restricted inlet, the critical pressure equates the
  opening's supply with the pipe's sonic mass capacity at the incoming total enthalpy.
  The wave's feasibility at that state selects sonic versus subsonic inflow, not a switch
  on the interior Mach number. The boundary therefore cannot jump at Mach 1 or silently
  cap velocity while passing the wrong valve mass. A source-based lower bracket also
  handles an interior withdrawal beyond its rarefaction escape speed.
  Each end's and each step's solution is bracketed near its last (0.1 % either side,
  widened fourfold; otherwise the full range) and solved by Illinois's regula falsi to
  relative pressure tolerance 1e-13. Near rest, `log1p`/`expm1` retain small pressure and
  kinetic heads instead of subtracting stagnation-sized quantities. A step's root is its
  upstream end pressure, whose wave curve gives the mass flow directly. Inverting a
  restricted entry's mass flux at fixed static pressure uses bracketed Newton iterations
  with d(rho v)/dv = rho (1 + v^2/(cp T)); a contraction along an isentrope has a different,
  decreasing-density derivative.
- After solving both ends from interior-only traces, the gas entering through each end
  supplies its ghost cells. A constant extension replaces extrapolation of downstream gas
  through an inlet; only stencils reaching outside are rebuilt. This is an explicitly
  lower-order boundary approximation, with conservation, positivity and linear acoustic
  mesh tests, not a claim to a globally fifth-order boundary scheme.
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
end subsonic) stay bounded. Sonic-inlet regressions additionally check the opening's
mass and total enthalpy, continuity through the interior Mach-1 crossing, the rarefaction
bracket's domain, and a frozen incoming-ghost positivity failure. The prescribed-inlet
linear-wave test refines both mesh and time: relative L2 errors at 10/5/2.5/1.25 mm were
2.75/0.187/0.065/0.052 %, with no rejected steps. This is linear boundary consistency
evidence, not nonlinear full-engine convergence.

**Remaining validation limit:** the 6000 rpm cold-start/perturbed-burn runs stay bounded
with zero retries and identical crank/burn schedules; serial and 8-thread outputs replay
bit for bit. However, very small burn perturbations (1e-7 to 1e-4) do not yet demonstrate a
linear waveform limit. Their response norms grew by factors 6.47, 4.97 and 2.38 for
successive tenfold perturbations, rather than ten. The residual mechanism is not fully
localized. Do not infer an LTI Jacobian from those data, label it physical turbulence, or
claim the remaining difference is only roundoff. These corrections address demonstrated
boundary/algebra defects; they do not prove complete physical fidelity or resolve the
remaining frozen-gamma/full-area, transition-friction and laminar-heat approximations.

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

A short actual-gearbox reference capture can stop one second after reaching a selected
gear, without scripting the engine speed:

```sh
OUT=/absolute/path/to/shift-capture
DUCT_THREADS=8 FREQ=20000 FLUID=12 THROUGH_GEAR=4 MAX_SECONDS=12 \
  bun engine/scripts/reference-drive.ts "$OUT" drive
DIAMETER=0.084 DISTANCE=5 uv run engine/scripts/reference_sound.py "$OUT" 20
```

`MAX_SECONDS` bounds the driving program, excluding the engine's settling period.
The capture metadata records the actual settling sample count and gear-change times, so
playback cuts need not assume a fixed settling length. The standard reference configuration
is merged unchanged; the simulated gearbox, spark cuts/retard and boost response drive the
sound. The 240 kHz fluid rate also divides exactly into the 4 kHz jet-noise control clock.

### Learned exhaust surrogate (experimental)

The reference is also a teacher for a per-car real-time approximation. The Pista now
uses the data-coverage model by default: `?car=pista`. The original, pilot and preview
selectors and the pause-menu exhaust section have been removed. Old `exhaust` URL values
or saved Pista settings do not change its sound. Other cars keep their existing sound.

The default model (`public/audio/surrogate/pista-coverage-candidate.json`, provenance
alongside) remains experimental. It predicts only the two tailpipes' radiated pressure
before directivity, not jet, mechanical, intake, bay or turbo noise. It uses the corrected
reference, stock upstream game settings, native 10 kHz/four-substep inputs and the exact
runtime resampling path: six gates, 512 taps, ridge damping 1. It is pressure-only
regression, not a neural network. A good fit to the reference is not independent evidence
of fidelity to a real car.

On two fresh final tests, 200–400 Hz errors were +0.78/+3.97 dB versus the previous
preview's +12.02/+10.62 dB, and pooled normalized waveform MSE improved from 0.248 to
0.123. Median absolute band-level error worsened from 1.06 to 1.83 dB, so the full
automatic promotion gate still **failed**; the user explicitly chose this trade-off and
then made the candidate the default. Residual bass and higher-band errors remain
disclosed. No EQ, per-clip gain, new noise or physics changes conceal any limitation.

The actual game engine stays at its original frequency/fluid steps and keeps the same
torque/boost physics; a separate physical kg/s + five-control stream feeds
`src/surrogate_stream.rs`. That adapter resamples to the 20 kHz model and the audio device,
divides by a fixed 16 Pa/full-scale calibration, then uses the game's existing
loudness/output controls. It cannot recover source bandwidth missing from the game's
10 kHz control stream. Model load/init/runtime failure creates a new original-voice
worklet and changes the simulation stream atomically, with a visible warning. This is a
failure fallback, not a selectable Pista sound. Cached audio nodes are muted/suspended
and their histories cleared when cars switch.

The original pilot (`pista-pilot.json`) and previous preview (`pista-retrained.json`),
with their provenance, are retained for reproducibility only. The pilot predates the
sonic-boundary correction and has additional training-rate and source-setting mismatches;
its limitations are not a description of the new default model.

- `examples/abtrace.rs`: paired game/reference traces, crank trajectory held by the dyno,
  separate burn seeds for training, validation and test drives. `EXHAUST_ONLY=1` swaps only
  the exhaust geometry; it does not merge the reference config's combustion/boost overrides
  into the deployed source engine. Metadata records the complete simulation spec, native
  rate, fluid steps, seed and actual actuator state. Matching commands/seeds does not make
  the two exhausts' instantaneous cylinder/valve states equal. Records physical pressures
  before listener directivity and continuous exit states, not fictitious event targets.
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
  tested adjoint; diagonal preconditioning preserves the original ridge penalty. The native
  source rate and model rate are independent: raw flow and all five controls undergo the
  same f32 linear interpolation as `SurrogateStream` **before** nonlinear features. Traces
  are tagged at step end; teacher decimation uses that fixed timestamp, not a fitted shift.
  Optional segment balance changes training residual weights only, not deployment gain.
  Iteration-limit stops are reported as such, not as optimizer convergence. Training's
  operator partitions input time into overlap-add blocks and shares their FFTs across
  gates/features; its overlapping-window adjoint preserves the full-record convolution
  and original ridge objective. Direct/dense tests and full-FFT comparisons check that this
  is a computation optimization, not a different model or loss.
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

The deployment-matched pipeline uses the game's 10 kHz/four-substep engine as the source,
while the teacher adds the corrected exhaust with 24 fluid substeps (240 kHz radiation).
Training, validation and test have distinct trajectories and seeds. Model fitting uses
validation for comparison; the final test is evaluated with frozen weights.

The previous retrained preview's fixed fit settings can be reproduced with
`GATES=6 TAPS=512 DAMPING=1 ITERATIONS=400 PREEMPHASIS=0.9` before the command above.
Its provenance also stores the independent confirmation program and all seeds; the shorter
model was selected on validation, not on confirmation labels.

Or use the individual tools. `scripts/test_surrogate_fit.py` checks causality, indexing,
identification on an independent synthetic trajectory, preservation of the regularized
objective under preconditioning, exact native/model interpolation, actual flap state,
fixed teacher decimation timestamps and frozen-model evaluation. `scripts/surrogate_select.py`
uses the fitter's canonical segment boundaries and one common warm-up exclusion. It
supports a raw/preemphasized waveform criterion or `--criterion spectrum` (worst then mean
absolute global band-level error, with a waveform-correlation floor). Never supply final
confirmation results to select parameters. The first waveform-first round exposed a bass
regression and was rejected; the second spectral-validation round used a new independent
confirmation trajectory, and its remaining bass regression is disclosed above.
`scripts/surrogate-stream-wasm.ts` replays the exported native-rate input through the actual
streaming adapter, rather than bypassing resampling with precomputed model-rate features. A local steady-state fit alone is not an acceptance test:
hold out entire trajectories and random seeds, check shifts/lift/reapply/flaps/thermal
history, compare game-input vs reference-input fits, inspect harmonic and band errors, and
measure the actual WASM decoder. The optional exit-state fit also needs noise-level and
modulation validation; do not feed reference-only exit states to a deployed student.

Training is before `reference_sound.py`'s whole-record zero-phase directivity. That filter
is not a causal runtime model. Listener directivity/resampling and their explicit latency
remain separate from identifying the causal exhaust. The pressure pilot is 20 kHz (below
10 kHz bandwidth); its WAVs are not a claim to reproduce the reference's full 24 kHz band.

### Data-coverage learning curve

`surrogate-data-plan.json` records a separate, reproducible coverage study rather than
silently changing the shipped preview. Its nested training stages contain 25.15, 38.15,
68.2 and 81.2 seconds across one, two, four and five independently initialized runs.
The additions include 4,300–5,200 rpm load holds, throttle lift/reapplication, descending
RPM histories, broader load/shift coverage, and one repeated program with a different
combustion seed. New nearby-RPM validation and two new test programs use distinct seeds.
The previously inspected confirmation drive is explicitly **diagnostic**, not a fresh test.

Acquisition keeps the same corrected exhaust, stock upstream game spec, 10 kHz/four-step
source and 24-substep teacher. With the old deployment-matched acquisition available:

```sh
uv run engine/scripts/surrogate_acquire.py "$OUT" \
  --renderer "$BUILT_ABTRACE" --spec "$OLD/spec.json" --baseline "$OLD/data"
```

The source spec must match the current exported game spec before acquisition. The runner
pins the binary, spec and reference configuration; validates complete metadata, physical
sample grids and byte counts before reuse; and hashes completed files. It writes an
immutable `manifest.json` plus separate capture completion records. Without `--baseline`,
it renders the baseline/diagnostic programs too. Large captures stay outside the repo.

`surrogate_study.py` fits one shared model across independent trajectories. Each run has
its own zero-start convolution and preemphasis state; captures are **not concatenated**.
Keep architecture, gates, ridge and training loss fixed for a data-only comparison:

```sh
uv run engine/scripts/surrogate_study.py fit "$OUT/manifest.json" "$OUT/baseline" \
  --stage baseline --gates-from public/audio/surrogate/pista-retrained.json \
  --iterations 6000 --atol 1e-7 --btol 1e-7
# Repeat for targeted, coverage and replicated stages after their captures complete.
```

The residuals are scaled by `sqrt(baseline_effective_samples / total_effective_samples)`
while the original-coefficient ridge stays fixed. Thus adding duplicate recordings does
not silently weaken regularization. Training excludes one filter-length warmup per run;
all comparisons use a common 1,024-sample exclusion by default. Added data must improve
unseen validation conditions, not just training error. An iteration ceiling is not proof
of convergence; compare the reported LSQR stops and residuals before interpreting the
learning curve. Current and longer-history models must not be conflated in this study.

Only choose a stage from validation results, freeze its exact weights/provenance, then
use `evaluate --split test` on the reserved programs. No run-specific alignment, gain,
playback EQ, extra noise or reference-physics adjustment is part of this experiment.
Test results do not automatically replace the game's default model. The explicit selection
and no-regression gates are in `scripts/surrogate-data-rules.json`.

The completed study is recorded in `scripts/surrogate-data-results.json`. Every stage
reached LSQR tolerance; adding the first 13 s of targeted coverage reduced the two
validation drives' 200–400 Hz errors from the old preview's +5.23/+11.38 dB to
+0.21/+0.44 dB. The larger 68.2/81.2 s fits improved waveform scores further, but their
worst validation bass errors were slightly larger. The pre-recorded spectral ranking
therefore selected the 38.15 s model before any final-test scoring.

On two fresh frozen-model tests, bass error fell from +12.02/+10.62 dB to
+0.78/+3.97 dB, and pooled normalized waveform MSE fell from 0.248 to 0.123.
However, median absolute band-level error worsened from 1.06 to 1.83 dB, so the full
no-regression gate **failed**. The weights are saved as
`public/audio/surrogate/pista-coverage-candidate.json`, with provenance alongside. The user
subsequently requested this candidate and then made it the Pista default, removing the
sound selector. The original voice remains a failure fallback; pilot and previous-preview
weights remain archived. The user override does not turn the failed spectral check into a pass.
The candidate still uses 512 taps (25.6 ms history), the original calibration and loss,
and no playback EQ. It fixes much, not all, of the bass generalization problem.

Acquisition and multi-run fitting have separate synthetic tests in `test_surrogate_acquire.py` and
`test_surrogate_study.py`.

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
