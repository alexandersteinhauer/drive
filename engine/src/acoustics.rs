//! The exhaust system's acoustics: what a pressure pulse at each exhaust valve sounds like at
//! a listener behind the car. engine-sim lumps every pipe and the muffler into single gas
//! volumes, which carry the flow but no sound waves, and takes the muffler's character from
//! a recorded impulse response; this computes that response from the exhaust's geometry
//! instead, as a linear acoustic network (the transfer-matrix method used for mufflers):
//!
//! - pipes: plane waves at the speed of sound of the gas inside (hot exhaust carries sound
//!   ~1.5x faster than air), with visco-thermal wall losses, and absorption for packed ones;
//! - orifices: perforated cones, baffles and caps, as a mass of gas plus a resistance;
//! - volumes: chambers small against the wavelength (a cylinder behind its open valve);
//! - outlets: tailpipe ends radiating into the open (unflanged pipe radiation load), heard as
//!   monopoles at `listener_distance`.
//!
//! Solved per frequency for the pressure at every junction (nodal admittance), then turned
//! into impulse responses at the audio rate.

use crate::spec::AcousticsSpec;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub const ZERO: C = C { re: 0.0, im: 0.0 };
    pub fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    pub fn j(im: f64) -> C {
        C { re: 0.0, im }
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn inv(self) -> C {
        let d = self.re * self.re + self.im * self.im;
        C::new(self.re / d, -self.im / d)
    }
    pub fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    pub fn exp(self) -> C {
        let e = self.re.exp();
        C::new(e * self.im.cos(), e * self.im.sin())
    }
    pub fn sin(self) -> C {
        C::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }
    pub fn cos(self) -> C {
        C::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
    }
}
impl std::ops::Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
}
impl std::ops::Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
}
impl std::ops::Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl std::ops::Div for C {
    type Output = C;
    fn div(self, o: C) -> C {
        self * o.inv()
    }
}
impl std::ops::Neg for C {
    type Output = C;
    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }
}
impl std::ops::AddAssign for C {
    fn add_assign(&mut self, o: C) {
        *self = *self + o;
    }
}
impl std::ops::SubAssign for C {
    fn sub_assign(&mut self, o: C) {
        *self = *self - o;
    }
}

/// In-place radix-2 FFT (`inverse`: unscaled inverse).
pub fn fft(data: &mut [C], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * std::f64::consts::PI / len as f64;
        let w_len = C::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut w = C::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = data[start + k];
                let v = data[start + k + len / 2] * w;
                data[start + k] = u + v;
                data[start + k + len / 2] = u - v;
                w = w * w_len;
            }
        }
        len <<= 1;
    }
}

/// The gas: exhaust products are near air in molar mass, with a lower heat capacity ratio.
struct Gas {
    density: f64,
    c: f64,
    viscosity: f64,
}

const GAMMA_EXHAUST: f64 = 1.35;
const R_SPECIFIC: f64 = 287.0;
const PRANDTL: f64 = 0.7;
const AMBIENT: f64 = 293.15;
/// Samples ahead of the first arrival.
const PRE_DELAY: f64 = 128.0;

fn gas(temperature: f64, gamma: f64) -> Gas {
    let density = crate::units::ATM / (R_SPECIFIC * temperature);
    // Sutherland's law for air.
    let mu = 1.716e-5 * (temperature / 273.15).powf(1.5) * (273.15 + 110.4) / (temperature + 110.4);
    Gas { density, c: (gamma * R_SPECIFIC * temperature).sqrt(), viscosity: mu / density }
}

enum Element {
    /// Admittance matrix [[y11, y12], [y12, y22]] between two nodes.
    TwoPort { a: usize, b: usize, y: [C; 3] },
    /// Admittance to ambient pressure.
    Shunt { node: usize, y: C },
}

/// Impulse responses (one per port, at `sample_rate`, `length` samples) of the radiated
/// pressure to a unit volume-velocity impulse at that port, scaled by `spec.gain`.
pub fn impulse_responses(spec: &AcousticsSpec, sample_rate: f64, length: usize) -> Vec<Vec<f32>> {
    let mut nodes: HashMap<String, usize> = HashMap::new();
    let node = |name: &str, nodes: &mut HashMap<String, usize>| {
        let n = nodes.len();
        nodes.entry(name.to_string()).or_insert(n);
    };
    for p in &spec.pipes {
        node(&p.a, &mut nodes);
        node(&p.b, &mut nodes);
    }
    for o in &spec.orifices {
        node(&o.a, &mut nodes);
        node(&o.b, &mut nodes);
    }
    for v in &spec.volumes {
        node(&v.node, &mut nodes);
    }
    for o in &spec.outlets {
        node(&o.node, &mut nodes);
    }
    for p in &spec.ports {
        node(p, &mut nodes);
    }
    let n = nodes.len();
    let index = |name: &str| nodes[name];
    let ports: Vec<usize> = spec.ports.iter().map(|p| index(p)).collect();

    // Room for the tail beyond `length` to decay before it would wrap around.
    let n_fft = length.next_power_of_two() * 4;
    let bins = n_fft / 2;
    let mut spectra = vec![vec![C::ZERO; n_fft]; ports.len()];
    let air = gas(AMBIENT, 1.4);

    let mut elements = Vec::new();
    let mut matrix = vec![C::ZERO; n * n];
    let mut rhs = vec![C::ZERO; n * ports.len()];
    for bin in 1..=bins {
        let omega = 2.0 * std::f64::consts::PI * bin as f64 * sample_rate / n_fft as f64;
        elements.clear();
        for p in &spec.pipes {
            let g = gas(p.temperature, GAMMA_EXHAUST);
            let radius = p.diameter / 2.0;
            let area = std::f64::consts::PI * radius * radius;
            // Kirchhoff's visco-thermal attenuation, times `damping` (packing, roughness).
            let alpha = (omega * g.viscosity / 2.0).sqrt() / (radius * g.c)
                * (1.0 + (GAMMA_EXHAUST - 1.0) / PRANDTL.sqrt())
                * p.damping
                // Absorptive lining (a perforated pipe in packing): rising with frequency.
                + p.absorption * omega / (2.0 * std::f64::consts::PI * 1000.0);
            let k = C::new(omega / g.c, -alpha);
            let kl = k.scale(p.length);
            let y0 = area / (g.density * g.c);
            let denominator = C::j(1.0 / y0) * kl.sin();
            let y11 = kl.cos() / denominator;
            let y12 = -denominator.inv();
            elements.push(Element::TwoPort { a: index(&p.a), b: index(&p.b), y: [y11, y12, y11] });
        }
        for o in &spec.orifices {
            let g = gas(o.temperature, GAMMA_EXHAUST);
            let z0 = g.density * g.c / o.area;
            let z = C::new(o.resistance * z0, omega * g.density * o.length / o.area);
            let y = z.inv();
            elements.push(Element::TwoPort { a: index(&o.a), b: index(&o.b), y: [y, -y, y] });
        }
        for v in &spec.volumes {
            let g = gas(v.temperature, GAMMA_EXHAUST);
            let y = C::j(omega * v.volume / (g.density * g.c * g.c));
            elements.push(Element::Shunt { node: index(&v.node), y });
        }
        let mut outlet_terms = Vec::new();
        for o in &spec.outlets {
            let radius = o.diameter / 2.0;
            let area = std::f64::consts::PI * radius * radius;
            let ka = omega / air.c * radius;
            let z = C::new(0.25 * ka * ka, 0.6133 * ka).scale(air.density * air.c / area);
            elements.push(Element::Shunt { node: index(&o.node), y: z.inv() });
            // Radiated pressure per unit node pressure: j w rho U / (4 pi r), U = p / z.
            let radiation = C::j(omega * air.density / (4.0 * std::f64::consts::PI * spec.listener_distance)) / z;
            outlet_terms.push((index(&o.node), radiation));
        }

        // Y p = q, one right-hand side per port.
        matrix.iter_mut().for_each(|m| *m = C::ZERO);
        for e in &elements {
            match *e {
                Element::TwoPort { a, b, y } => {
                    matrix[a * n + a] += y[0];
                    matrix[a * n + b] += y[1];
                    matrix[b * n + a] += y[1];
                    matrix[b * n + b] += y[2];
                }
                Element::Shunt { node, y } => matrix[node * n + node] += y,
            }
        }
        rhs.iter_mut().for_each(|r| *r = C::ZERO);
        let m = ports.len();
        for (i, &p) in ports.iter().enumerate() {
            rhs[p * m + i] = C::new(1.0, 0.0);
        }
        solve(&mut matrix, &mut rhs, n, m);

        // Heard without the time sound takes to reach the listener, which would only add
        // latency; a few samples' delay leave room for the ringing of the band limit ahead
        // of a sharp arrival.
        let delay = C::j(-omega * PRE_DELAY / sample_rate).exp();
        for i in 0..m {
            let mut h = C::ZERO;
            for &(node, radiation) in &outlet_terms {
                h += rhs[node * m + i] * radiation;
            }
            spectra[i][bin] = h * delay;
        }
    }

    spectra
        .into_iter()
        .map(|mut s| {
            // A real response: Hermitian spectrum; the Nyquist bin real.
            s[bins] = C::new(s[bins].re, 0.0);
            for bin in 1..bins {
                s[n_fft - bin] = C::new(s[bin].re, -s[bin].im);
            }
            fft(&mut s, true);
            let scale = spec.gain * sample_rate / n_fft as f64;
            // Fade the last eighth out, so a truncated tail doesn't click.
            let fade = length / 8;
            (0..length)
                .map(|i| {
                    let w = if i + fade >= length {
                        let t = (length - i) as f64 / fade as f64;
                        0.5 - 0.5 * (std::f64::consts::PI * t).cos()
                    } else {
                        1.0
                    };
                    (s[i].re * scale * w) as f32
                })
                .collect()
        })
        .collect()
}

/// Gaussian elimination with partial pivoting: solves `a x = b` for `m` columns of `b`
/// (row-major, `b` is n x m), leaving x in `b`.
fn solve(a: &mut [C], b: &mut [C], n: usize, m: usize) {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .unwrap();
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            for k in 0..m {
                b.swap(col * m + k, pivot * m + k);
            }
        }
        let inv = a[col * n + col].inv();
        for row in col + 1..n {
            let f = a[row * n + col] * inv;
            if f == C::ZERO {
                continue;
            }
            for k in col..n {
                let v = a[col * n + k];
                a[row * n + k] -= f * v;
            }
            for k in 0..m {
                let v = b[col * m + k];
                b[row * m + k] -= f * v;
            }
        }
    }
    for col in (0..n).rev() {
        let inv = a[col * n + col].inv();
        for k in 0..m {
            let mut v = b[col * m + k];
            for j in col + 1..n {
                v -= a[col * n + j] * b[j * m + k];
            }
            b[col * m + k] = v * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{AcousticOutlet, AcousticPipe};

    fn pipe(a: &str, b: &str, length: f64, diameter: f64) -> AcousticPipe {
        AcousticPipe {
            a: a.into(),
            b: b.into(),
            length,
            diameter,
            temperature: AMBIENT,
            damping: 1.0,
            absorption: 0.0,
        }
    }

    #[test]
    fn fft_round_trips() {
        let mut d: Vec<C> = (0..64).map(|i| C::new((i as f64 * 0.3).sin(), 0.0)).collect();
        let original = d.clone();
        fft(&mut d, false);
        fft(&mut d, true);
        for (a, b) in d.iter().zip(&original) {
            assert!((a.re / 64.0 - b.re).abs() < 1e-12);
        }
    }

    #[test]
    fn a_pipe_delays_by_its_length() {
        // 1 m of cold pipe with an open end: the first arrival comes after 1 m / c.
        let spec = AcousticsSpec {
            pipes: vec![pipe("port", "end", 1.0, 0.03)],
            orifices: vec![],
            volumes: vec![],
            ports: vec!["port".into()],
            outlets: vec![AcousticOutlet { node: "end".into(), diameter: 0.03 }],
            listener_distance: 1.0,
            gain: 1.0,
        };
        let ir = &impulse_responses(&spec, 44100.0, 4096)[0];
        let c = gas(AMBIENT, GAMMA_EXHAUST).c;
        let expected = 1.0 / c * 44100.0 + PRE_DELAY;
        let peak = ir.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap().0;
        assert!((peak as f64 - expected).abs() < 6.0, "peak at {peak}, expected {expected:.0}");
        // Causal: nothing well before the sound could get there (band-limiting leaves some
        // ringing just ahead of a sharp arrival).
        let early: f32 = ir[..(expected as usize - 60)].iter().map(|v| v * v).sum();
        let total: f32 = ir.iter().map(|v| v * v).sum();
        assert!(early < 1e-3 * total, "energy before arrival {:.2e}", early / total);
    }

    #[test]
    fn an_expansion_chamber_attenuates_as_theory_says() {
        // Transmission loss of an expansion chamber (area ratio m, length L) is
        // 10 log10(1 + (m - 1/m)^2 sin^2(kL) / 4): its peak at kL = pi/2 should show in the
        // ratio of responses with and without the chamber.
        let (d, dc, l) = (0.03, 0.12, 0.5);
        let m: f64 = (dc / d) * (dc / d);
        // Anechoic ends, as the theory assumes: long pipes damped enough that nothing comes
        // back (the model keeps their characteristic impedance, so they don't reflect).
        let anechoic = |a: &str, b: &str| AcousticPipe { damping: 60.0, ..pipe(a, b, 3.0, d) };
        let with = |chamber: bool| {
            let mut pipes = vec![
                anechoic("port", "dead end"),
                pipe("port", "in", 0.3, d),
                pipe("out", "tail", 0.3, d),
                anechoic("tail", "end"),
            ];
            pipes.push(if chamber { pipe("in", "out", l, dc) } else { pipe("in", "out", l, d) });
            AcousticsSpec {
                pipes,
                orifices: vec![],
                volumes: vec![],
                ports: vec!["port".into()],
                outlets: vec![AcousticOutlet { node: "end".into(), diameter: d }],
                listener_distance: 1.0,
                gain: 1.0,
            }
        };
        let spectrum = |spec: &AcousticsSpec| {
            let mut ir: Vec<C> = impulse_responses(spec, 44100.0, 1 << 15)[0]
                .iter()
                .map(|&v| C::new(v as f64, 0.0))
                .collect();
            fft(&mut ir, false);
            ir
        };
        let (a, b) = (spectrum(&with(true)), spectrum(&with(false)));
        // Point by point against the theory, below where the damped ends stop being
        // anechoic enough.
        let c = gas(AMBIENT, GAMMA_EXHAUST).c;
        let n = a.len() as f64;
        let mut sum = 0.0;
        let mut count = 0.0;
        for f in (50..700).step_by(10) {
            let i = (f as f64 * n / 44100.0) as usize;
            let kl = 2.0 * std::f64::consts::PI * f as f64 / c * l;
            let theory = 10.0 * (1.0 + 0.25 * (m - 1.0 / m).powi(2) * kl.sin().powi(2)).log10();
            let ours = 20.0 * (b[i].abs() / a[i].abs()).log10();
            sum += (ours - theory).powi(2);
            count += 1.0;
        }
        let rms = (sum / count).sqrt();
        assert!(rms < 1.5, "{rms:.2} dB rms from the theory");
    }
}
