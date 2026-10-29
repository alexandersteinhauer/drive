//! The engine's sound, made from the simulation's exhaust valve flows: in place of
//! engine-sim's synthesizer (synth.rs), which filters an exhaust signal sampled from the gas
//! dynamics, this takes from the simulation what it predicts and models the rest directly.
//! Measured against recordings of a real 488 Pista (engine/README.md, "Sound"):
//!
//! - The simulated runners' sound above a few hundred Hz depends on the fluid step (0D
//!   volumes ringing numerically): not a prediction to listen to. Each valve's flow over a
//!   step is, below that: the source the exhaust's acoustics (acoustics.rs) take. The engine
//!   writes it (Engine::write_voice), this interpolates it to the audio rate.
//! - Loud pulses steepen as they travel, into shocks (Steepener): the high harmonics, and
//!   the brightness that grows with load, no linear exhaust has.
//! - Above ~2 kHz an exhaust is mostly turbulence: the valve jets', carried through the
//!   exhaust, and the tailpipe jets', pulsing with the flow. Flow noise dominates
//!   high-performance turbocharged engines' tailpipe noise (Wiemeler et al., SAE
//!   2008-01-0404); pulses dominate at low speed and flow noise at high (Pang et al., SAE
//!   2005-01-2352).
//! - Combustion varies from cycle to cycle (Ozdor, Dulger & Sher, SAE 940987), which
//!   breaks the pulses' harmonics up into a real engine's rough tone; the simulation's
//!   only varies with its flame.
//! - A car is more than its exhaust: its engine bay (thin manifolds and turbine housings
//!   radiating the pulses inside, the valvetrain, direct injectors) and its turbochargers
//!   (compressor noise and whine, the blow-off valve).
//!
//! Per audio sample, from the control stream resampled from the simulation's steps (the
//! exhaust's convolutions a block at a time):
//!
//! ```text
//! valve flow q_i x this blowdown's strength ─> manifold ─> steepening ─> rest of exhaust ─> + tail jet noise
//!   + valve jet noise ~ |q|^1.5                (per port)  (per bank)    (valves blended)    (from the flow out)
//!                                                   └─> engine bay (shells, mass law)
//! valve events ─> valve seating, injector ticks ─┐
//! turbo state ─> whoosh, whine, blow-off ────────┴─> heard directly, as late as the exhaust's lead
//! ```
//!
//! Without a junction to steepen at (no turbine), each port's pipe steepens its pulses and
//! the responses run port to listener; without computed acoustics, recorded responses take
//! each exhaust's flow. Levels follow physical estimates (each spec level a multiple of one),
//! in pascals at the listener; `full_scale` maps them to the output.

use crate::spec::{EngineSpec, VoiceSpec};
use crate::synth::{HighPass, Partitioned, BLOCK};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// The state the engine writes after the cylinders' valve flows, one value each per step.
#[derive(Clone, Copy)]
pub enum Global {
    /// Crank speed (rad/s).
    CrankSpeed,
    /// Mass flow out of the tailpipes (kg/s) and the exhaust's temperature (K), averaged.
    ExhaustFlow,
    ExhaustTemperature,
    /// Air drawn through the throttle (kg/s), averaged.
    AirFlow,
    /// Pressure behind the throttle (Pa) and the pedal (0..1).
    Manifold,
    Throttle,
    /// Turbocharger: charge pressure (Pa), shaft speed (rad/s), air vented by the blow-off
    /// valve (kg/s), wastegate opening (0..1).
    Charge,
    TurboSpeed,
    Vented,
    Wastegate,
}
pub const GLOBALS: usize = 10;

const RATE: f64 = 44100.0;
/// The computed responses: length (samples) and bandwidth; split (Split), the first
/// part's length.
const IMPULSE_LENGTH: usize = 4096;
const MANIFOLD_LENGTH: usize = 2048;
const MAX_FREQUENCY: f64 = 18000.0;
const TAPER_FROM: f64 = 15000.0;
const RHO_AIR: f64 = 1.2;
const C_AIR: f64 = 343.0;
/// Listener distance without computed acoustics (the chase camera's).
const LISTENER: f64 = 5.0;
/// A valve's volume flow (m^3/s) at which its jet's noise is `port_noise` times the flow
/// (it grows as the flow to the 1.5: the jet's power as its velocity cubed).
const PORT_FLOW: f64 = 0.15;
/// Lighthill's constant: a jet of velocity U and area A radiates K rho U^8 A / c^5.
const LIGHTHILL: f64 = 3e-5;
/// Peak Strouhal number of jet noise, f D / U; above, a hot turbulent exhaust's noise
/// keeps up to about here (Hz), then falls away.
const JET_STROUHAL: f64 = 0.2;
const TAIL_ROLL_OFF: f64 = 4000.0;
/// The acoustics' responses' lead ahead of their first arrival (acoustics::PRE_DELAY).
const PRE_DELAY: usize = 128;
/// Exhaust gas: heat capacity ratio and specific gas constant (J/kg K).
const GAMMA_EXHAUST: f64 = 1.35;
const R_EXHAUST: f64 = 287.0;
/// Without computed acoustics, the pulses steepen down each cylinder's primary pipe, at
/// this temperature (K), into what the recorded responses take over.
const PRIMARY_TEMPERATURE: f64 = 1000.0;
/// The engine bay's walls (thin steel, then the engine cover) pass the manifolds' sound
/// falling 6 dB per octave above this (Hz), as a wall's mass law has it.
const BAY_WALL: f64 = 300.0;
/// The output fades in over this long at first (s).
const FADE_IN: f64 = 0.3;
/// The wastegate openings (of their full area) the manifolds' responses are computed at.
const WASTEGATE_OPENINGS: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
/// How much more combustion varies with the manifold empty than at full load.
const LIGHT_LOAD_VARIATION: f64 = 2.0;
/// Exhaust valve actuators' time constant (s).
const VALVE_TIME: f64 = 0.15;
/// Valve seating at 1000 rpm and an injector's tick (Pa at the listener, before their
/// levels): quiet next to any exhaust but its idle.
const VALVE_PA: f64 = 0.02;
const INJECTOR_PA: f64 = 0.01;
/// Compressor noise and whine at a tip Mach number of one and 0.3 kg/s, and a blow-off
/// valve venting 0.1 kg/s at a tip Mach number of one (Pa at the listener).
const WHOOSH_PA: f64 = 0.05;
const WHISTLE_PA: f64 = 0.01;
const BLOW_OFF_PA: f64 = 0.3;

#[derive(Clone, Copy, Default)]
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
}

impl Biquad {
    fn rbj(b: [f64; 3], a: [f64; 3]) -> Self {
        Biquad {
            b: [(b[0] / a[0]) as f32, (b[1] / a[0]) as f32, (b[2] / a[0]) as f32],
            a: [(a[1] / a[0]) as f32, (a[2] / a[0]) as f32],
            z: [0.0; 2],
        }
    }

    fn terms(f: f64, q: f64) -> (f64, f64) {
        let w = 2.0 * PI * f.clamp(1.0, 0.49 * RATE) / RATE;
        (w.cos(), w.sin() / (2.0 * q))
    }

    fn low_pass(f: f64, q: f64) -> Self {
        let (c, a) = Self::terms(f, q);
        Self::rbj([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0], [1.0 + a, -2.0 * c, 1.0 - a])
    }

    fn high_pass(f: f64, q: f64) -> Self {
        let (c, a) = Self::terms(f, q);
        Self::rbj([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0], [1.0 + a, -2.0 * c, 1.0 - a])
    }

    /// Unity gain at `f`.
    fn band_pass(f: f64, q: f64) -> Self {
        let (c, a) = Self::terms(f, q);
        Self::rbj([a, 0.0, -a], [1.0 + a, -2.0 * c, 1.0 - a])
    }

    /// New coefficients, keeping the state.
    fn retune(&mut self, other: Biquad) {
        self.b = other.b;
        self.a = other.a;
    }

    #[inline]
    fn f(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }

    /// Clears a decayed state before it turns subnormal (slow on some hardware).
    fn flush(&mut self) {
        for z in &mut self.z {
            if z.abs() < 1e-20 {
                *z = 0.0;
            }
        }
    }
}

/// White noise, uniform in -1..1 (xorshift32).
struct Noise(u32);

impl Noise {
    #[inline]
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x as i32 as f32 * (1.0 / 2_147_483_648.0)
    }

    /// Standard normal (Box-Muller).
    fn gauss(&mut self) -> f64 {
        let u1 = (self.next() as f64 * 0.5 + 0.5).max(1e-12);
        let u2 = self.next() as f64 * 0.5 + 0.5;
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Uniform noise's standard deviation is 1/sqrt(3): this makes it one.
const UNIT_NOISE: f32 = 1.732_050_8;

/// Through two filters in turn.
#[inline]
fn chain(filters: &mut [Biquad; 2], x: f32) -> f32 {
    let y = filters[0].f(x);
    filters[1].f(y)
}

#[inline]
fn catmull_rom(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    y1 + 0.5 * t * (y2 - y0 + t * (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3 + t * (3.0 * (y1 - y2) + y3 - y0)))
}

/// A convolution shared by the ports whose responses are the same.
struct Group {
    convolution: Partitioned,
    input: Vec<f32>,
    /// The valve jets' noise into it, before shaping.
    noise: f32,
    noise_filter: Biquad,
}

/// Nonlinear propagation down a pipe (Burgers'): a wave's crest travels faster than its
/// trough, at c + (gamma + 1)/2 u for a particle velocity u = p / (rho c), so over a length
/// L a sample of pressure p arrives kappa p = L (gamma + 1)/2 p / (rho c^3) earlier. How
/// large exhaust pulses (a tenth of a bar or more) gain the high harmonics no linear pipe
/// makes, the more the larger (Selamet et al., SAE 880082; the brassiness of loud
/// trombones: Msallam et al., Acustica 86, 2000). Solved as Pestorius's algorithm does
/// (JASA 1973): the length in slices, each one warping the wave's time a little, and where
/// a crest would overtake the trough ahead, a shock: the two merge, and the energy the
/// steepening put into the jump is lost, as a real shock's is. So a wave saturates however
/// loud: beyond a few shock-forming distances, a sawtooth whose amplitude no longer grows
/// with the source's.
struct Steepener {
    slices: Vec<Slice>,
    /// Takes out the mean and the infrasound first: the back-pressure the mean flow keeps
    /// up is no wave, and would only shift every sample alike, crowding the delay out.
    mean: [Biquad; 2],
}

/// One slice of Steepener's pipe, SLICE_DELAY samples late.
struct Slice {
    /// Samples per pascal earlier, and the most (within the delay).
    kappa: f64,
    /// The last input's arrival (samples) and value; inputs so far; the next output's time.
    arrival: f64,
    value: f32,
    time: f64,
    next: f64,
    out: VecDeque<f32>,
}

/// Below this (Hz), a Steepener's input is taken out (4th-order Butterworth).
const STEEPENER_HIGH_PASS: f64 = 20.0;
/// Slices of a Steepener, and each one's delay (samples).
const SLICES: usize = 8;
const SLICE_DELAY: f64 = 8.0;

impl Slice {
    fn push(&mut self, p: f32) {
        let early = (self.kappa * p as f64).clamp(-(SLICE_DELAY - 1.0), SLICE_DELAY - 1.0);
        let a = self.time + SLICE_DELAY - early;
        if a <= self.arrival {
            // Overtaking the sample ahead: a shock. They go on as one, at its arrival,
            // with their mean.
            self.value = 0.5 * (self.value + p);
        } else {
            while self.next <= a {
                let f = ((self.next - self.arrival) / (a - self.arrival)) as f32;
                self.out.push_back(self.value + (p - self.value) * f);
                self.next += 1.0;
            }
            self.arrival = a;
            self.value = p;
        }
        self.time += 1.0;
    }

    fn pop(&mut self) -> f32 {
        self.out.pop_front().unwrap_or(self.value)
    }
}

impl Steepener {
    fn new(length: f64, temperature: f64) -> Self {
        let c = (GAMMA_EXHAUST * R_EXHAUST * temperature).sqrt();
        let rho = crate::units::ATM / (R_EXHAUST * temperature);
        let kappa = length * (GAMMA_EXHAUST + 1.0) / 2.0 / (rho * c * c * c) * RATE;
        let slice = || Slice {
            kappa: kappa / SLICES as f64,
            arrival: SLICE_DELAY - 1.0,
            value: 0.0,
            time: 0.0,
            next: 0.0,
            out: VecDeque::new(),
        };
        Steepener {
            slices: (0..SLICES).map(|_| slice()).collect(),
            mean: [Biquad::high_pass(STEEPENER_HIGH_PASS, 0.54), Biquad::high_pass(STEEPENER_HIGH_PASS, 1.31)],
        }
    }

    /// The whole pipe's base delay (samples).
    fn delay() -> usize {
        SLICES * SLICE_DELAY as usize
    }

    /// A block through the pipe, delay() late.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (o, &x) in output.iter_mut().zip(input) {
            let mut v = chain(&mut self.mean, x);
            for s in &mut self.slices {
                s.push(v);
                v = s.pop();
            }
            *o = v;
        }
        for f in &mut self.mean {
            f.flush();
        }
    }
}

/// A cylinder's pulses steepening down its own pipe (without one junction to do it at):
/// its flow as the pressure of the wave it launches there, rho c / A, and back.
struct Port {
    input: Vec<f32>,
    steepener: Steepener,
    impedance: f32,
}

/// Each exhaust system past the junction its pulses steepen from: the pressure there,
/// steepened, then on to the listener.
struct Split {
    input: Vec<f32>,
    /// None with recorded responses (the ports' pipes steepen).
    steepener: Option<Steepener>,
    convolution: Partitioned,
    /// Recorded responses: the last steepened sample, for the radiation's derivative.
    last: f32,
    /// The engine bay's share, through its walls (a one-pole low-pass's state).
    bay: f32,
}

#[derive(Clone, Copy)]
enum Click {
    Valve,
    Injector,
}

pub struct Voice {
    spec: VoiceSpec,
    cylinders: usize,
    channels: usize,
    port_group: Vec<usize>,
    groups: Vec<Group>,
    /// Each group's exhaust system.
    group_exhaust: Vec<usize>,
    /// Per exhaust system, with nonlinear propagation from a junction (else the groups'
    /// responses reach the listener); per cylinder, with it from each port.
    splits: Vec<Split>,
    ports: Vec<Port>,
    /// Recorded responses (no acoustics): they take the tailpipes' radiation, the
    /// derivative of their flow, which computed ones include.
    recorded: bool,
    listener: f64,
    /// Tailpipes: how many, and each one's area and diameter.
    outlets: f64,
    outlet_area: f64,
    outlet_diameter: f64,
    compressor_radius: f64,

    /// The control stream: frames of `channels` values, the first one already behind.
    input: Vec<f32>,
    /// Where the next audio sample falls in it (frames), and frames per audio sample.
    position: f64,
    step: f64,
    input_rate: f64,
    audio_rate: f64,
    /// The frame events were last looked for in.
    frame: usize,

    /// Each cylinder's current blowdown strength.
    strength: Vec<f32>,
    noise: Noise,
    /// Valve and injector clicks to come: (audio sample, kind, amplitude).
    pending: Vec<(f64, Click, f32)>,
    clock: f64,
    valve_ring: [Biquad; 2],
    injector_ring: [Biquad; 2],
    whoosh_filter: [Biquad; 2],
    blow_off_filter: [Biquad; 2],
    whistle_phase: f64,
    flutter_phase: f64,
    /// Tailpipe flow: the acoustic volume velocity (integral of the radiated pressure) and
    /// the jet noise's shaping.
    tail_volume: f64,
    tail_filters: [Biquad; 2],

    /// The wastegate openings the manifolds' responses are for (none without wastegates),
    /// and where the engine's was among them at the end of the last block.
    wastegates: Vec<f64>,
    wastegate: f64,
    /// Exhaust valves: the crank speeds and pedal positions they open between, and how far
    /// open they were at the end of the last block.
    valves: Option<([f64; 2], [f64; 2])>,
    valve: f64,
    /// The block being made: samples so far, its controls' sums, the sources heard
    /// directly (and the last block's, now due).
    filled: usize,
    controls: [f64; GLOBALS],
    direct: Vec<f32>,
    /// The direct sources, delayed to the exhaust's lead.
    direct_delay: VecDeque<f32>,
    exhaust: Vec<f32>,
    /// The engine bay's sound for the block (apart: the tailpipes' flow is the exhaust's).
    bay: Vec<f32>,
    tmp: Vec<f32>,
    hearing: HighPass,
    /// Fading in at first (0..1): the voice starts against an exhaust already flowing,
    /// which its filters, from rest, would hear as a step.
    fade: f32,
    out: VecDeque<f32>,
}

impl Voice {
    /// The voice for `spec` (None without `audio.voice`); with acoustics, computes their
    /// responses (~0.1 s).
    pub fn new(spec: &EngineSpec) -> Option<Voice> {
        let v = spec.audio.voice.clone()?;
        let port_exhaust: Vec<usize> = spec.banks.iter().flat_map(|b| b.cylinders.iter().map(|c| c.exhaust)).collect();
        let cylinders = port_exhaust.len();
        let group = |sets: &[Vec<f32>]| Group {
            convolution: Partitioned::blending(sets),
            input: vec![0.0; BLOCK],
            noise: 0.0,
            noise_filter: Biquad::high_pass(500.0, 0.7),
        };
        let exhausts = spec.exhausts.len();
        let band = |length| crate::acoustics::Band {
            sample_rate: RATE,
            length,
            max_frequency: MAX_FREQUENCY,
            taper_from: TAPER_FROM,
        };
        // The responses are the continuous ones sampled (per second): as a discrete
        // convolution's, times the sample interval.
        let discrete = |r: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
            r.into_iter().map(|r| r.into_iter().map(|h| h / RATE as f32).collect()).collect()
        };
        let one_set = |r: Vec<Vec<f32>>| -> Vec<Vec<Vec<f32>>> { r.into_iter().map(|r| vec![r]).collect() };
        let mut splits = Vec::new();
        let mut ports = Vec::new();
        let mut wastegates: Vec<f64> = Vec::new();
        let (groups, port_group, recorded, listener, outlets, outlet_diameter) = match &spec.audio.acoustics {
            Some(a) => {
                let responses = match &a.nonlinear {
                    Some(nl) if nl.nodes.is_empty() => {
                        let t = nl.temperature;
                        let impedance = (crate::units::ATM / (R_EXHAUST * t) * (GAMMA_EXHAUST * R_EXHAUST * t).sqrt()) as f32;
                        for b in &spec.banks {
                            for _ in &b.cylinders {
                                ports.push(Port {
                                    input: vec![0.0; BLOCK],
                                    steepener: Steepener::new(nl.length, t),
                                    impedance: impedance / b.head.exhaust_runner_area as f32,
                                });
                            }
                        }
                        one_set(discrete(crate::acoustics::impulse_responses_to(
                            a,
                            RATE,
                            IMPULSE_LENGTH,
                            v.acoustic_flow,
                            MAX_FREQUENCY,
                            TAPER_FROM,
                        )))
                    }
                    Some(nl) => {
                        let (manifolds, rest) = crate::acoustics::split_responses(
                            a,
                            &band(IMPULSE_LENGTH),
                            v.acoustic_flow,
                            &nl.nodes,
                            &port_exhaust,
                        );
                        // With valves, the rest of the way also with them open (the
                        // manifolds, behind their turbines, hardly notice).
                        let opened = a.valves.as_ref().map(|_| {
                            let open = a.opened();
                            discrete(
                                crate::acoustics::split_responses(
                                    &open,
                                    &band(IMPULSE_LENGTH),
                                    v.acoustic_flow,
                                    &nl.nodes,
                                    &port_exhaust,
                                )
                                .1,
                            )
                        });
                        for (k, r) in discrete(rest).into_iter().enumerate() {
                            let sets = match &opened {
                                Some(o) => vec![r, o[k].clone()],
                                None => vec![r],
                            };
                            splits.push(Split {
                                input: vec![0.0; BLOCK],
                                steepener: Some(Steepener::new(nl.length, nl.temperature)),
                                convolution: Partitioned::blending(&sets),
                                last: 0.0,
                                bay: 0.0,
                            });
                        }
                        // The manifolds ring briefly: their responses, shortened (faded out
                        // over their last eighth). With wastegates, at each of their
                        // openings: an open one lets the pulses past the turbine.
                        let fade = MANIFOLD_LENGTH / 8;
                        let shorten = |r: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
                            discrete(r)
                                .into_iter()
                                .map(|r| {
                                    (0..MANIFOLD_LENGTH)
                                        .map(|i| {
                                            let w = (MANIFOLD_LENGTH - i).min(fade) as f32 / fade as f32;
                                            r[i] * (0.5 - 0.5 * (PI as f32 * w).cos())
                                        })
                                        .collect()
                                })
                                .collect()
                        };
                        let mut by_opening = vec![shorten(manifolds)];
                        if !a.wastegate.is_empty() {
                            wastegates = WASTEGATE_OPENINGS.to_vec();
                            for &opening in &WASTEGATE_OPENINGS[1..] {
                                let (m, _) = crate::acoustics::split_responses(
                                    &a.with_wastegate(opening),
                                    &band(IMPULSE_LENGTH),
                                    v.acoustic_flow,
                                    &nl.nodes,
                                    &port_exhaust,
                                );
                                by_opening.push(shorten(m));
                            }
                        }
                        (0..by_opening[0].len())
                            .map(|port| by_opening.iter().map(|set| set[port].clone()).collect())
                            .collect()
                    }
                    None => one_set(discrete(crate::acoustics::impulse_responses_to(
                        a,
                        RATE,
                        IMPULSE_LENGTH,
                        v.acoustic_flow,
                        MAX_FREQUENCY,
                        TAPER_FROM,
                    ))),
                };
                // Ports whose responses are the same (a symmetric manifold's) share one
                // convolution.
                let mut distinct: Vec<usize> = Vec::new();
                let mut port_group = Vec::new();
                for (i, sets) in responses.iter().enumerate() {
                    let peak = sets.iter().flatten().fold(0.0f32, |m, x| m.max(x.abs()));
                    let same = distinct.iter().position(|&j| {
                        responses[j].iter().flatten().zip(sets.iter().flatten()).all(|(a, b)| (a - b).abs() <= 1e-5 * peak)
                    });
                    port_group.push(same.unwrap_or_else(|| {
                        distinct.push(i);
                        distinct.len() - 1
                    }));
                }
                let groups: Vec<Group> = distinct.iter().map(|&i| group(&responses[i])).collect();
                let d = a.outlets.iter().map(|o| o.diameter).sum::<f64>() / a.outlets.len().max(1) as f64;
                (groups, port_group, false, a.listener_distance, a.outlets.len() as f64, d)
            }
            None => {
                // The pulses steepen down each cylinder's primary pipe, the recorded
                // responses (set_impulse_response) taking each exhaust's flow after.
                let t = PRIMARY_TEMPERATURE;
                let impedance = (crate::units::ATM / (R_EXHAUST * t) * (GAMMA_EXHAUST * R_EXHAUST * t).sqrt()) as f32;
                for b in &spec.banks {
                    for c in &b.cylinders {
                        ports.push(Port {
                            input: vec![0.0; BLOCK],
                            steepener: Steepener::new(c.primary_length, t),
                            impedance: impedance / b.head.exhaust_runner_area as f32,
                        });
                    }
                }
                let groups = spec.exhausts.iter().map(|_| group(&[vec![1.0]])).collect();
                for _ in 0..exhausts {
                    splits.push(Split {
                        input: vec![0.0; BLOCK],
                        steepener: None,
                        convolution: Partitioned::new(&[]),
                        last: 0.0,
                        bay: 0.0,
                    });
                }
                (groups, port_exhaust.clone(), true, LISTENER, spec.exhausts.len() as f64, 0.08)
            }
        };
        let group_exhaust: Vec<usize> = (0..groups.len())
            .map(|g| port_exhaust[port_group.iter().position(|&x| x == g).unwrap_or(0)])
            .collect();
        // The direct sources wait for the exhaust: its responses' leads and the steepening.
        let steepening = if splits.iter().all(|s| s.steepener.is_none()) && ports.is_empty() { 0 } else { Steepener::delay() };
        let delay = match (&spec.audio.acoustics, splits.is_empty()) {
            (Some(_), true) => PRE_DELAY + steepening,
            (Some(_), false) => 2 * PRE_DELAY + steepening,
            (None, _) => steepening,
        };
        Some(Voice {
            cylinders,
            channels: cylinders + GLOBALS,
            port_group,
            groups,
            group_exhaust,
            splits,
            ports,
            wastegates,
            wastegate: 0.0,
            valves: spec.audio.acoustics.as_ref().and_then(|a| a.valves.as_ref()).map(|v| (v.speed, v.throttle)),
            valve: 0.0,
            recorded,
            listener,
            outlets: outlets.max(1.0),
            outlet_area: PI * outlet_diameter * outlet_diameter / 4.0,
            outlet_diameter,
            compressor_radius: spec.turbo.as_ref().map_or(0.0, |t| t.compressor_radius),
            input: vec![0.0; cylinders + GLOBALS],
            position: 1.0,
            step: spec.simulation_frequency / RATE,
            input_rate: spec.simulation_frequency,
            audio_rate: RATE,
            frame: usize::MAX,
            strength: vec![1.0; cylinders],
            noise: Noise(0x2545_f491),
            pending: Vec::new(),
            clock: 0.0,
            valve_ring: [Biquad::band_pass(2600.0, 6.0), Biquad::band_pass(5400.0, 8.0)],
            injector_ring: [Biquad::band_pass(7200.0, 10.0), Biquad::band_pass(11500.0, 10.0)],
            whoosh_filter: [Biquad::high_pass(3000.0, 0.6), Biquad::low_pass(12000.0, 0.6)],
            blow_off_filter: [Biquad::band_pass(1800.0, 0.6), Biquad::high_pass(400.0, 0.7)],
            whistle_phase: 0.0,
            flutter_phase: 0.0,
            tail_volume: 0.0,
            tail_filters: [Biquad::high_pass(200.0, 0.6), Biquad::low_pass(TAIL_ROLL_OFF, 0.5)],
            filled: 0,
            controls: [0.0; GLOBALS],
            direct: vec![0.0; BLOCK],
            direct_delay: std::iter::repeat_n(0.0, delay).collect(),
            exhaust: vec![0.0; BLOCK],
            bay: vec![0.0; BLOCK],
            tmp: vec![0.0; BLOCK],
            hearing: HighPass::new(4, 20.0, RATE),
            fade: 0.0,
            // The output runs a block behind the input (the convolution's).
            out: std::iter::repeat_n(0.0, BLOCK).collect(),
            spec: v,
        })
    }

    /// The control stream's channels per step.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// A recorded impulse response (16-bit samples) for exhaust system `index`, without
    /// computed acoustics.
    pub fn set_impulse_response(&mut self, index: usize, samples: &[i16], volume: f32) {
        let end = samples.iter().rposition(|s| (*s as i32).abs() > 100).map_or(0, |i| i + 1).min(10000);
        let impulse: Vec<f32> = samples[..end].iter().map(|&s| volume * s as f32 / i16::MAX as f32).collect();
        if let Some(s) = self.splits.get_mut(index) {
            s.convolution = Partitioned::new(&impulse);
        }
    }

    pub fn set_input_rate(&mut self, rate: f64) {
        self.input_rate = rate;
        self.step = rate / self.audio_rate;
    }

    /// Appends one step's frame (`channels` values).
    pub fn write_input(&mut self, frame: &[f32]) {
        self.input.extend_from_slice(&frame[..self.channels]);
    }

    fn frames(&self) -> usize {
        self.input.len() / self.channels
    }

    /// Audio samples the buffered input can make (interpolation needs two frames ahead).
    pub fn input_available(&self) -> usize {
        let last = self.frames() as f64 - 2.0;
        if last <= self.position {
            0
        } else {
            ((last - self.position) / self.step).ceil() as usize
        }
    }

    /// Drops buffered input beyond `keep` audio samples' worth (the game ran ahead).
    pub fn drop_input(&mut self, keep: usize) {
        let available = self.input_available();
        if available > keep {
            self.position += (available - keep) as f64 * self.step;
            self.compact();
        }
    }

    /// Forgets frames the interpolation no longer needs.
    fn compact(&mut self) {
        let behind = (self.position as usize).saturating_sub(1);
        if behind > 0 {
            self.input.drain(..behind * self.channels);
            self.position -= behind as f64;
            self.frame = self.frame.wrapping_sub(behind);
        }
    }

    /// Renders up to `out.len()` samples (-1..1); returns how many there was input for.
    pub fn render(&mut self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.input_available());
        for o in &mut out[..n] {
            self.source_sample();
            if self.filled == BLOCK {
                self.finish_block();
            }
            *o = self.out.pop_front().unwrap_or(0.0);
        }
        self.compact();
        n
    }

    /// Looks for valve events between frame `f` and the next.
    fn events(&mut self, f: usize) {
        let ch = self.channels;
        let omega = self.input[(f + 1) * ch + self.cylinders + Global::CrankSpeed as usize] as f64;
        // Combustion varies more at light load (COV of IMEP rising as the manifold
        // empties: Amin, Elmaihy & Shahin 2023), most at idle.
        let manifold = self.input[(f + 1) * ch + self.cylinders + Global::Manifold as usize] as f64;
        let variation = self.spec.variation * (1.0 + LIGHT_LOAD_VARIATION * (1.0 - manifold / crate::units::ATM).clamp(0.0, 1.0));
        for i in 0..self.cylinders {
            let (a, b) = (self.input[f * ch + i], self.input[(f + 1) * ch + i]);
            if a == 0.0 && b != 0.0 {
                // A blowdown begins: its strength varies with the cycle's combustion.
                let s = variation;
                self.strength[i] = (s * self.noise.gauss() - 0.5 * s * s).exp() as f32;
            } else if a != 0.0 && b == 0.0 && omega > 10.0 {
                // The exhaust valves seat; the intake valves ~210 degrees on, after the
                // injector's tick early in the intake stroke.
                let seat = (VALVE_PA * self.spec.valvetrain * (omega / 104.7).powf(1.5)) as f32;
                let samples = |deg: f64| deg * PI / 180.0 / omega * RATE;
                self.pending.push((self.clock, Click::Valve, seat));
                self.pending.push((self.clock + samples(210.0), Click::Valve, seat));
                self.pending.push((self.clock + samples(60.0), Click::Injector, (INJECTOR_PA * self.spec.injectors) as f32));
            }
        }
    }

    /// One audio sample's sources, into the block being made.
    fn source_sample(&mut self) {
        let f = self.position as usize;
        if f != self.frame {
            self.frame = f;
            self.events(f);
        }
        let t = (self.position - f as f64) as f32;
        let ch = self.channels;
        let k = self.filled;
        let base = (f - 1) * ch;
        for i in 0..self.cylinders {
            let y = |j: usize| self.input[base + j * ch + i];
            let q = catmull_rom(y(0), y(1), y(2), y(3), t) * self.strength[i];
            let g = &mut self.groups[self.port_group[i]];
            match self.ports.get_mut(i) {
                Some(p) => p.input[k] = q,
                None => g.input[k] += q,
            }
            let jet = q.abs() * (q.abs() / PORT_FLOW as f32).sqrt();
            g.noise += self.noise.next() * UNIT_NOISE * jet;
        }
        let port_noise = self.spec.port_noise as f32;
        for g in &mut self.groups {
            g.input[k] += port_noise * g.noise_filter.f(g.noise);
            g.noise = 0.0;
        }
        let mut controls = [0.0f64; GLOBALS];
        for (j, c) in controls.iter_mut().enumerate() {
            let (a, b) = (self.input[base + ch + self.cylinders + j], self.input[base + 2 * ch + self.cylinders + j]);
            *c = (a + (b - a) * t) as f64;
            self.controls[j] += *c;
        }

        let mut direct = 0.0f32;
        // Valve seating and injector ticks: impulses ringing the head's resonances.
        let (mut valve, mut injector) = (0.0f32, 0.0f32);
        let clock = self.clock;
        self.pending.retain(|&(at, kind, a)| {
            if at > clock {
                return true;
            }
            match kind {
                Click::Valve => valve += a,
                Click::Injector => injector += a,
            }
            false
        });
        // Band-pass resonators have a gain of Q/(pi f) per unit impulse area... drive them
        // with an impulse of the click's peak amplitude.
        direct += self.valve_ring[0].f(valve) + self.valve_ring[1].f(0.6 * valve);
        direct += self.injector_ring[0].f(injector) + self.injector_ring[1].f(0.5 * injector);
        // The turbocharger.
        if self.compressor_radius > 0.0 {
            let mach = controls[Global::TurboSpeed as usize] * self.compressor_radius / C_AIR;
            let air = (controls[Global::AirFlow as usize].max(0.0) / 0.3).sqrt();
            if self.spec.whoosh > 0.0 {
                let a = (self.spec.whoosh * WHOOSH_PA * mach.powi(3) * air) as f32;
                let w = self.noise.next() * UNIT_NOISE;
                direct += a * chain(&mut self.whoosh_filter, w) * 2.0;
            }
            let blades = self.spec.compressor_blades;
            if self.spec.whistle > 0.0 && blades > 0.0 {
                let f = controls[Global::TurboSpeed as usize] / (2.0 * PI) * blades;
                self.whistle_phase = (self.whistle_phase + f / RATE).fract();
                if f < 0.45 * RATE {
                    let a = self.spec.whistle * WHISTLE_PA * mach * mach * air;
                    direct += (a * (2.0 * PI * self.whistle_phase).sin()) as f32;
                }
            }
            let vented = controls[Global::Vented as usize];
            if self.spec.blow_off > 0.0 && vented > 0.0 {
                // Venting what the compressor still pumps, the louder the faster its jet:
                // the pressure the compressor makes over the atmosphere, about its tip Mach
                // number squared (nothing as it idles). With the flutter of a compressor
                // near surge.
                self.flutter_phase = (self.flutter_phase + 18.0 / RATE).fract();
                let flutter = 0.65 + 0.35 * (2.0 * PI * self.flutter_phase).sin();
                let a = (self.spec.blow_off * BLOW_OFF_PA * vented / 0.1 * mach * mach * flutter) as f32;
                let w = self.noise.next() * UNIT_NOISE;
                direct += a * chain(&mut self.blow_off_filter, w) * 2.0;
            }
        }
        self.direct[k] = direct;
        self.clock += 1.0;
        self.position += self.step;
        self.filled += 1;
    }

    /// Convolves the block, adds the tailpipes' jet noise and the direct sources (a block
    /// late, matching the responses' lead) and queues the result.
    fn finish_block(&mut self) {
        self.exhaust.iter_mut().for_each(|v| *v = 0.0);
        let (wastegate0, wastegate1) = if self.wastegates.is_empty() {
            (0.0, 0.0)
        } else {
            let opening = self.controls[Global::Wastegate as usize] / BLOCK as f64;
            let w = &self.wastegates;
            let x = (1..w.len())
                .find(|&i| opening <= w[i])
                .map_or((w.len() - 1) as f64, |i| (i - 1) as f64 + (opening - w[i - 1]) / (w[i] - w[i - 1]));
            (self.wastegate, x.max(0.0))
        };
        self.wastegate = wastegate1;
        for (i, p) in self.ports.iter_mut().enumerate() {
            let z = p.impedance;
            p.input.iter_mut().for_each(|v| *v *= z);
            p.steepener.process(&p.input, &mut self.tmp);
            let g = &mut self.groups[self.port_group[i]];
            for (x, &y) in g.input.iter_mut().zip(&self.tmp) {
                *x += y / z;
            }
        }
        for (i, g) in self.groups.iter_mut().enumerate() {
            g.convolution.process_blended(&g.input, &mut self.tmp, wastegate0, wastegate1);
            let target = match self.splits.get_mut(self.group_exhaust[i]) {
                Some(s) => &mut s.input,
                None => &mut self.exhaust,
            };
            for (e, &y) in target.iter_mut().zip(&self.tmp) {
                *e += y;
            }
            g.input.iter_mut().for_each(|v| *v = 0.0);
            g.noise_filter.flush();
        }
        let (valve0, valve1) = match self.valves {
            Some((speed, throttle)) => {
                let step = |x: f64, [from, at]: [f64; 2]| {
                    let x = ((x - from) / (at - from).max(1e-9)).clamp(0.0, 1.0);
                    x * x * (3.0 - 2.0 * x)
                };
                let mean = |g: Global| self.controls[g as usize] / BLOCK as f64;
                let open = step(mean(Global::CrankSpeed), speed).max(step(mean(Global::Throttle), throttle));
                // The actuators take a few tenths of a second.
                let next = self.valve + (open - self.valve) * (BLOCK as f64 / RATE / VALVE_TIME).min(1.0);
                (self.valve, next)
            }
            None => (0.0, 0.0),
        };
        self.valve = valve1;
        // The tailpipe's radiation (recorded responses): a monopole's, rho / (4 pi r) dQ/dt.
        let radiation = (RHO_AIR / (4.0 * PI * self.listener) * RATE * self.spec.impulse_gain) as f32;
        for s in &mut self.splits {
            // The engine bay: the thin manifolds and turbine housings radiate the pressure
            // inside them, less their walls' transmission loss (mass law: falling 6 dB per
            // octave), heard before the tailpipes' (a shorter way to the listener). Not
            // with recorded responses, whose pipe is no particular one.
            if self.spec.bay > 0.0 && !self.recorded {
                let bay = self.spec.bay as f32;
                let a = (1.0 - (-2.0 * PI * BAY_WALL / RATE).exp()) as f32;
                for (e, &p) in self.bay.iter_mut().zip(&s.input) {
                    s.bay += a * (p - s.bay);
                    *e += bay * s.bay;
                }
                if s.bay.abs() < 1e-20 {
                    s.bay = 0.0;
                }
            }
            let mut warped = std::mem::take(&mut self.tmp);
            match &mut s.steepener {
                Some(st) => st.process(&s.input, &mut warped),
                None => warped.copy_from_slice(&s.input),
            }
            if self.recorded {
                for w in warped.iter_mut() {
                    let d = (*w - s.last) * radiation;
                    s.last = *w;
                    *w = d;
                }
            }
            s.convolution.process_blended(&warped, &mut s.input, valve0, valve1);
            for (e, &y) in self.exhaust.iter_mut().zip(&s.input) {
                *e += y;
            }
            s.input.iter_mut().for_each(|v| *v = 0.0);
            self.tmp = warped;
        }
        let mean = |c: f64| c / BLOCK as f64;
        let flow = mean(self.controls[Global::ExhaustFlow as usize]).max(0.0);
        let hot = mean(self.controls[Global::ExhaustTemperature as usize]).max(300.0);
        self.controls = [0.0; GLOBALS];

        // The tailpipe jets: their mean velocity, and the sound's own volume velocity.
        if self.spec.tail_noise > 0.0 {
            // The gas has cooled along the exhaust by the tailpipe.
            let t_tail = 300.0 + 0.5 * (hot - 300.0);
            let rho = crate::units::ATM / (287.0 * t_tail);
            let area = self.outlet_area * self.outlets;
            let u_mean = flow / (rho * area);
            let peak = (JET_STROUHAL * u_mean / self.outlet_diameter).max(50.0);
            self.tail_filters[0].retune(Biquad::high_pass(peak, 0.6));
            // White noise through the band keeps about this much of its power.
            let band = (0.5 * RATE / (TAIL_ROLL_OFF * PI / 2.0)).sqrt();
            let to_volume = 4.0 * PI * self.listener / RHO_AIR / RATE;
            let leak = 1.0 - 2.0 * PI * 10.0 / RATE;
            // p_rms^2 = W rho c / (4 pi r^2), W = K rho U^8 A / c^5 (Lighthill).
            let jet = self.spec.tail_noise * LIGHTHILL * rho * area / C_AIR.powi(5) * RHO_AIR * C_AIR
                / (4.0 * PI * self.listener * self.listener);
            for e in self.exhaust.iter_mut() {
                self.tail_volume = self.tail_volume * leak + *e as f64 * to_volume;
                let u = (u_mean + self.tail_volume / area).max(0.0);
                let u2 = u * u;
                let p = (jet * u2 * u2 * u2 * u2).sqrt() * band;
                let w = self.noise.next() * UNIT_NOISE;
                *e += (p as f32) * chain(&mut self.tail_filters, w);
            }
        }
        let scale = (1.0 / self.spec.full_scale) as f32;
        self.direct_delay.extend(self.direct.iter());
        let fade_step = (1.0 / (FADE_IN * RATE)) as f32;
        for k in 0..BLOCK {
            let direct = self.direct_delay.pop_front().unwrap_or(0.0);
            let v = self.hearing.f(self.exhaust[k] + self.bay[k] + direct) * scale;
            self.fade = (self.fade + fade_step).min(1.0);
            let f = self.fade * self.fade * (3.0 - 2.0 * self.fade);
            self.out.push_back(v * f);
        }
        self.bay.iter_mut().for_each(|v| *v = 0.0);
        for f in self
            .valve_ring
            .iter_mut()
            .chain(&mut self.injector_ring)
            .chain(&mut self.whoosh_filter)
            .chain(&mut self.blow_off_filter)
            .chain(&mut self.tail_filters)
        {
            f.flush();
        }
        self.filled = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The amplitude of `x`'s component at `f` (Hz).
    fn amplitude(x: &[f32], f: f64) -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let w = 2.0 * PI * f * i as f64 / RATE;
            re += v as f64 * w.cos();
            im += v as f64 * w.sin();
        }
        2.0 * re.hypot(im) / x.len() as f64
    }

    fn steepened(amplitude_pa: f64, f: f64) -> Vec<f32> {
        let mut s = Steepener::new(2.0, 900.0);
        let n = 44100;
        let input: Vec<f32> = (0..n).map(|i| (amplitude_pa * (2.0 * PI * f * i as f64 / RATE).sin()) as f32).collect();
        let mut out = vec![0.0f32; n];
        for (i, o) in input.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
            s.process(i, o);
        }
        // Past the start-up, a whole number of periods.
        out[4410..4410 + 44100 / 2].to_vec()
    }

    #[test]
    fn small_waves_pass_unchanged_and_large_ones_gain_harmonics() {
        // A quiet wave (1 Pa) travels linearly: no second harmonic to speak of.
        let quiet = steepened(1.0, 400.0);
        let h1 = amplitude(&quiet, 400.0);
        assert!((h1 - 1.0).abs() < 0.01, "fundamental {h1}");
        assert!(amplitude(&quiet, 800.0) < 1e-3);
        // A pulse of a tenth of a bar steepens on the way: Burgers's second harmonic, for a
        // distance short of the shock's, is kappa-dependent: B2 / B1 = x / (2 x_shock).
        let loud = steepened(1e4, 400.0);
        let (b1, b2) = (amplitude(&loud, 400.0), amplitude(&loud, 800.0));
        let s = Steepener::new(2.0, 900.0);
        let kappa: f64 = s.slices.iter().map(|x| x.kappa).sum();
        let shock = RATE / (2.0 * PI * 400.0 * kappa * 1e4); // in units of the 2 m
        let expected = 1.0 / (2.0 * shock);
        assert!((b2 / b1 - expected).abs() < 0.2 * expected, "B2/B1 {} against {expected}", b2 / b1);
    }

    #[test]
    fn shocks_take_energy_and_never_stall() {
        // Far beyond the shock-forming distance (a bar of pressure at 500 Hz), a pulse train
        // with its mean: the output keeps time and loses energy to its shocks.
        let mut s = Steepener::new(2.5, 900.0);
        let n = 44100;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let ph = (500.0 * i as f64 / RATE).fract();
                (1e5 * (-ph * 12.0).exp() + 3e4) as f32
            })
            .collect();
        let mut out = vec![0.0f32; n];
        for (i, o) in input.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
            s.process(i, o);
        }
        let ac = |x: &[f32]| {
            let m = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
            (x.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
        };
        assert!(ac(&out[4410..]) < ac(&input[4410..]));
        for sl in &s.slices {
            assert!(sl.out.len() < 2 * SLICE_DELAY as usize + 2, "queue {}", sl.out.len());
        }
    }

    #[test]
    fn interpolation_is_exact_for_lines() {
        for t in [0.0f32, 0.25, 0.5, 0.9] {
            assert!((catmull_rom(1.0, 2.0, 3.0, 4.0, t) - (2.0 + t)).abs() < 1e-6);
        }
    }

    #[test]
    fn a_still_engine_is_silent() {
        let spec: EngineSpec = serde_json::from_str(include_str!("../tests/vw1200.json")).unwrap();
        let mut spec = spec;
        spec.audio.voice = Some(VoiceSpec {
            full_scale: 1.0,
            acoustic_flow: 0.0,
            variation: 0.05,
            port_noise: 1.0,
            tail_noise: 1.0,
            bay: 1.0,
            valvetrain: 1.0,
            injectors: 1.0,
            whoosh: 1.0,
            whistle: 1.0,
            compressor_blades: 10.0,
            blow_off: 1.0,
            impulse_gain: 1.0,
        });
        let mut v = Voice::new(&spec).unwrap();
        let frame = vec![0.0f32; v.channels()];
        for _ in 0..2000 {
            v.write_input(&frame);
        }
        let mut out = vec![1.0f32; 4096];
        let n = v.render(&mut out);
        assert!(n > 4000);
        assert!(out[..n].iter().all(|&x| x == 0.0));
    }
}
