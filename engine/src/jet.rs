//! Reference tailpipe turbulent-mixing noise, in pascals at the listener. No engine-bay,
//! mechanical, turbine or valve noise. This ports `scripts/reference_sound.py`:
//! Stone, Groesbeck & Zola, NASA TM-82712 (1981), eq. A1 and Table I, for a static round
//! jet. The exit state is quasi-steady, not a model of turbulence inside the exhaust.
//!
//! Supply each outlet's [velocity (m/s), density (kg/m^3), temperature (K)] at 4 kHz.
//! The Stone band RMS pressures are interpolated, not the velocity or the noise samples.
//! After the first (immediate) target, a target is reached one control period later:
//! 250 us causal interpolation delay relative to targets timestamped at control ticks,
//! or 125 us relative to the Python reference's centered 250 us block averages when
//! those averages are supplied at the beginning of their blocks. Filters start at rest;
//! allow a warm-up when measuring stationary spectra.
//!
//! Each band uses the same third-order Butterworth *bandpass* as scipy.signal.butter
//! (six poles after the lowpass-to-bandpass transform), prewarped and bilinear-mapped.
//! Its white-noise variance is calculated from its poles/residues, not from the rendered
//! signal. Thus every band has the prescribed expected RMS, with no fitted level or
//! whole-realization normalization. Independent deterministic PCG streams have variance
//! one; their uniform rather than Gaussian samples change higher moments, not the PSD.

use std::f64::consts::PI;
use std::sync::OnceLock;

pub const CONTROL_RATE: f64 = 4_000.0;
const P_REF: f64 = 2.0e-5; // acoustic reference pressure, Pa
const T_INF: f64 = 298.15; // model's ambient air, K
const RHO_INF: f64 = 1.184; // kg/m^3
const C_INF: f64 = 346.1; // m/s
const RHO_ISA: f64 = 1.225; // Stone eq. A1 reference conditions
const C_ISA: f64 = 340.3;

/// The Python reference's exact base-2 1/3-octave centers (24.803 .. 20158.737 Hz),
/// retaining centers below 0.45 of the audio rate. The highest band edge is additionally
/// capped at 0.49 of the rate, as in the reference.
pub fn band_centers(sample_rate: f64) -> Result<Vec<f64>, String> {
    if !sample_rate.is_finite() || sample_rate < CONTROL_RATE {
        return Err("jet sample rate must be finite and at least 4000 Hz".into());
    }
    Ok((-16..14)
        .map(|i| 1_000.0 * 2.0f64.powf(i as f64 / 3.0))
        .filter(|&f| f < 0.45 * sample_rate)
        .collect())
}

fn validate_geometry(diameter: f64, distance: f64, off_axis_deg: f64) -> Result<(), String> {
    if !diameter.is_finite() || diameter <= 0.0 || !distance.is_finite() || distance <= 0.0 {
        return Err("jet diameter and listener distance must be positive and finite".into());
    }
    if !off_axis_deg.is_finite() || !(0.0..=180.0).contains(&off_axis_deg) {
        return Err(
            "jet listener angle must be finite and between 0 and 180 degrees off the outlet axis"
                .into(),
        );
    }
    Ok(())
}

fn validate_exit(exit: [f64; 3]) -> Result<(), String> {
    if exit.iter().any(|v| !v.is_finite()) || exit[1] <= 0.0 || exit[2] <= 0.0 {
        return Err(
            "jet exit velocity must be finite; density and temperature must be positive and finite"
                .into(),
        );
    }
    Ok(())
}

struct StoneTable {
    log_s: Vec<f64>,
    angles: Vec<f64>,
    rows: Vec<Vec<f64>>,
}

impl StoneTable {
    fn get() -> &'static Self {
        static TABLE: OnceLock<StoneTable> = OnceLock::new();
        TABLE.get_or_init(|| {
            let mut lines = include_str!("../scripts/stone_table1.csv").lines();
            let angles = lines
                .next()
                .expect("Stone table header")
                .split(',')
                .skip(1)
                .map(|v| {
                    if v == "th0_110" {
                        110.0
                    } else {
                        v[2..].parse().expect("Stone table angle")
                    }
                })
                .collect::<Vec<_>>();
            let mut log_s = Vec::new();
            let mut rows = Vec::new();
            for line in lines.filter(|l| !l.trim().is_empty()) {
                let mut values = line
                    .split(',')
                    .map(|v| v.parse::<f64>().expect("Stone table entry"));
                log_s.push(values.next().expect("Stone log Strouhal"));
                let row = values.collect::<Vec<_>>();
                assert_eq!(row.len(), angles.len());
                rows.push(row);
            }
            StoneTable {
                log_s,
                angles,
                rows,
            }
        })
    }

    /// np.interp's endpoint clamping and angle-then-log-Strouhal linear interpolation.
    fn lookup(&self, corrected_angle: f64, log_s: f64) -> f64 {
        let (a, wa) = bracket(&self.angles, corrected_angle);
        let (s, ws) = bracket(&self.log_s, log_s);
        let low = lerp(self.rows[s][a], self.rows[s][a + 1], wa);
        let high = lerp(self.rows[s + 1][a], self.rows[s + 1][a + 1], wa);
        lerp(low, high, ws)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn bracket(grid: &[f64], x: f64) -> (usize, f64) {
    if x <= grid[0] {
        return (0, 0.0);
    }
    if x >= grid[grid.len() - 1] {
        return (grid.len() - 2, 1.0);
    }
    let hi = grid.partition_point(|&v| v <= x);
    let lo = hi - 1;
    (lo, (x - grid[lo]) / (grid[hi] - grid[lo]))
}

/// One jet's 1/3-octave SPLs (dB re 20 uPa), with exactly the reference's ambient
/// conditions, Table I interpolation/clamping, and V < 1 m/s cutoff (including inflow).
/// `off_axis_deg` is measured from the *outlet* axis; Stone uses the inlet axis.
/// `frequencies_hz` are arbitrary positive finite band centers, not rounded nominal Hz.
/// Invalid thermodynamic states are errors, never silently clipped surrogate outputs.
pub fn stone_band_levels(
    diameter: f64,
    distance: f64,
    off_axis_deg: f64,
    exit: [f64; 3],
    frequencies_hz: &[f64],
) -> Result<Option<Vec<f64>>, String> {
    validate_geometry(diameter, distance, off_axis_deg)?;
    validate_exit(exit)?;
    if frequencies_hz.iter().any(|&f| !f.is_finite() || f <= 0.0) {
        return Err("jet band centers must be positive and finite".into());
    }
    let Some(parameters) = stone_parameters(diameter, distance, off_axis_deg, exit)? else {
        return Ok(None);
    };
    frequencies_hz
        .iter()
        .map(|&f| parameters.level(f))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

struct StoneParameters {
    overall: f64,
    corrected_angle: f64,
    strouhal_scale: f64,
}

fn stone_parameters(
    diameter: f64,
    distance: f64,
    off_axis_deg: f64,
    exit: [f64; 3],
) -> Result<Option<StoneParameters>, String> {
    let [velocity, density, temperature] = exit;
    if velocity < 1.0 {
        return Ok(None);
    }
    let theta = 180.0 - off_axis_deg;
    let va = velocity / C_INF;
    let va35 = va.powf(3.5);
    let omega = 3.0 * va35 / (0.6 + va35) - 1.0;
    let mc = 0.62 * va;
    let area = PI * diameter * diameter / 4.0;
    // NASA TM-82712 eq. A1: geometric spreading A/R^2, density exponent omega,
    // velocity exponent 7.5 in acoustic power, and convective directivity at theta.
    let overall = 141.0
        + 10.0 * ((RHO_INF / RHO_ISA).powi(2) * (C_INF / C_ISA).powi(4)).log10()
        + 10.0 * (area / distance.powi(2)).log10()
        + 10.0 * omega * (density / RHO_INF).log10()
        + 75.0 * va.log10()
        - 15.0 * ((1.0 + mc * theta.to_radians().cos()).powi(2) + 0.04 * mc.powi(2)).log10();
    let corrected_angle = theta * va.powf(0.1);
    // Temperature-corrected Strouhal number S = f D/V (Tj/Tinf)^[0.4(1+cos theta')].
    let strouhal_scale = diameter / velocity
        * (temperature / T_INF).powf(0.4 * (1.0 + corrected_angle.to_radians().cos()));
    if !overall.is_finite()
        || !corrected_angle.is_finite()
        || !strouhal_scale.is_finite()
        || strouhal_scale <= 0.0
    {
        return Err("jet state or geometry exceeds the numerical range of Stone's formula".into());
    }
    Ok(Some(StoneParameters {
        overall,
        corrected_angle,
        strouhal_scale,
    }))
}

impl StoneParameters {
    fn level(&self, frequency: f64) -> Result<f64, String> {
        let log_s = (frequency * self.strouhal_scale).log10();
        let level = self.overall + StoneTable::get().lookup(self.corrected_angle, log_s);
        if !log_s.is_finite() || !level.is_finite() {
            return Err("jet band exceeds the numerical range of Stone's formula".into());
        }
        Ok(level)
    }
}

/// Streaming, independently summed outlet jets. No allocations after construction on
/// valid input. At 48 kHz call `set_exit_states` once, then `next_sample` twelve times,
/// and repeat at 4 kHz. Noninteger audio/control ratios may use a phase accumulator.
/// `next_sample` before the first state is silent but still advances the noise filters.
pub struct Jet {
    sample_rate: f64,
    diameters: Vec<f64>,
    distance: f64,
    off_axis_deg: f64,
    centers: Vec<f64>,
    bands: Vec<NoiseBand>, // outlet-major, then band
    targets: Vec<f64>,
    interpolation: f64,
    initialized: bool,
}

impl Jet {
    pub fn new(
        sample_rate: f64,
        diameters: &[f64],
        distance: f64,
        off_axis_deg: f64,
    ) -> Result<Self, String> {
        let centers = band_centers(sample_rate)?;
        if diameters.is_empty() {
            return Err("at least one jet outlet is required".into());
        }
        for &diameter in diameters {
            validate_geometry(diameter, distance, off_axis_deg)?;
        }
        // Parse the embedded table here, never in a real-time control update.
        StoneTable::get();
        let mut prototypes = Vec::with_capacity(centers.len());
        for &fc in &centers {
            prototypes.push(ButterBand::new(fc, sample_rate)?);
        }
        let count = diameters
            .len()
            .checked_mul(centers.len())
            .ok_or("jet band count overflow")?;
        let mut bands = Vec::with_capacity(count);
        for o in 0..diameters.len() {
            for (b, filter) in prototypes.iter().enumerate() {
                bands.push(NoiseBand {
                    filter: filter.clone(),
                    rng: Pcg::new((o * centers.len() + b) as u64),
                    start: 0.0,
                    target: 0.0,
                });
            }
        }
        Ok(Self {
            sample_rate,
            diameters: diameters.to_vec(),
            distance,
            off_axis_deg,
            centers,
            bands,
            targets: vec![0.0; count],
            interpolation: 1.0,
            initialized: false,
        })
    }

    pub fn centers_hz(&self) -> &[f64] {
        &self.centers
    }
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
    pub fn outlets(&self) -> usize {
        self.diameters.len()
    }

    /// [velocity m/s, density kg/m^3, temperature K] per outlet, at 4 kHz.
    /// The complete update is validated before any audible state changes. Negative
    /// velocities and velocities below 1 m/s target silence, as in the Python reference.
    pub fn set_exit_states(&mut self, exits: &[[f64; 3]]) -> Result<(), String> {
        if exits.len() != self.diameters.len() {
            return Err("jet exit count does not match diameters".into());
        }
        for &exit in exits {
            validate_exit(exit)?;
        }
        for (o, &exit) in exits.iter().enumerate() {
            let target = &mut self.targets[o * self.centers.len()..(o + 1) * self.centers.len()];
            if let Some(p) =
                stone_parameters(self.diameters[o], self.distance, self.off_axis_deg, exit)?
            {
                for (amp, &fc) in target.iter_mut().zip(&self.centers) {
                    *amp = P_REF * 10.0f64.powf(p.level(fc)? / 20.0);
                    if !amp.is_finite() {
                        return Err("jet band amplitude is not finite".into());
                    }
                }
            } else {
                target.fill(0.0);
            }
        }
        for (band, &target) in self.bands.iter_mut().zip(&self.targets) {
            band.start = if self.initialized {
                lerp(band.start, band.target, self.interpolation)
            } else {
                target
            };
            band.target = target;
        }
        self.interpolation = if self.initialized { 0.0 } else { 1.0 };
        self.initialized = true;
        Ok(())
    }

    /// One audio-rate sample, Pa at the listener. Different outlets add incoherently
    /// in expectation, because their mixing-noise streams are independent.
    pub fn next_sample(&mut self) -> f64 {
        let t = self.interpolation;
        let mut output = 0.0;
        for band in &mut self.bands {
            output += band.filter.next(band.rng.unit_variance()) * lerp(band.start, band.target, t);
        }
        self.interpolation = (t + CONTROL_RATE / self.sample_rate).min(1.0);
        output
    }
}

struct NoiseBand {
    filter: ButterBand,
    rng: Pcg,
    start: f64,
    target: f64,
}

/// PCG-XSH-RR 64/32, one distinct odd-increment stream per outlet/band. A discrete
/// uniform midpoint on [-sqrt(3),sqrt(3)] has mean zero and variance 1 - 2^-64.
/// That rounding-scale difference is below f64 precision, not a calibration parameter.
struct Pcg {
    state: u64,
    increment: u64,
}
impl Pcg {
    fn new(stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            increment: stream.wrapping_mul(2).wrapping_add(1),
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(0x853c_49e6_748f_ea9b);
        rng.next_u32();
        rng
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.increment);
        let x = (((old >> 18) ^ old) >> 27) as u32;
        x.rotate_right((old >> 59) as u32)
    }
    fn unit_variance(&mut self) -> f64 {
        ((self.next_u32() as f64 + 0.5) * (1.0 / 4294967296.0) - 0.5) * 12.0f64.sqrt()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}
impl Complex {
    const ONE: Self = Self { re: 1.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn div(self, b: Self) -> Self {
        let d = b.norm2();
        Self::new(
            (self.re * b.re + self.im * b.im) / d,
            (self.im * b.re - self.re * b.im) / d,
        )
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn sqrt(self) -> Self {
        let r = self.re.hypot(self.im);
        Self::new(
            ((r + self.re) * 0.5).max(0.0).sqrt(),
            ((r - self.re) * 0.5).max(0.0).sqrt().copysign(self.im),
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}
impl Biquad {
    // Numerator b0 * (1 - z^-2): one zero each at DC and Nyquist per section.
    // Other SOS zero pairings (including scipy's) give the identical total filter.
    fn next(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = -self.a1 * y + self.z2;
        self.z2 = -self.b0 * x - self.a2 * y;
        y
    }
    #[cfg(test)]
    fn response(&self, q: Complex) -> Complex {
        Complex::ONE.sub(q.mul(q)).scale(self.b0).div(
            Complex::ONE
                .add(q.scale(self.a1))
                .add(q.mul(q).scale(self.a2)),
        )
    }
}

#[derive(Clone, Debug)]
struct ButterBand {
    sections: [Biquad; 3],
    white_noise_gain: f64,
}
impl ButterBand {
    fn new(center: f64, sample_rate: f64) -> Result<Self, String> {
        let edge = 2.0f64.powf(1.0 / 6.0);
        let lo = (PI * (center / edge) / sample_rate).tan();
        let hi = (PI * (center * edge).min(0.49 * sample_rate) / sample_rate).tan();
        let bw = hi - lo;
        let w02 = lo * hi;
        let mut positive = Vec::with_capacity(3);
        let mut real = Vec::with_capacity(2);
        // Third-order normalized Butterworth prototype: -1, -1/2 +/- i sqrt(3)/2.
        // Transform s -> (s^2+w0^2)/(bw*s), then z=(1+s)/(1-s); the factors 2 fs
        // cancel because lo/hi above were prewarped in units of 2 fs.
        for p in [
            Complex::new(-1.0, 0.0),
            Complex::new(-0.5, 3.0f64.sqrt() * 0.5),
            Complex::new(-0.5, -3.0f64.sqrt() * 0.5),
        ] {
            let a = p.scale(0.5 * bw);
            let root = a.mul(a).sub(Complex::new(w02, 0.0)).sqrt();
            for s in [a.add(root), a.sub(root)] {
                let z = Complex::ONE.add(s).div(Complex::ONE.sub(s));
                if z.im > 0.0 {
                    positive.push(z);
                } else if z.im == 0.0 {
                    real.push(z);
                }
            }
        }
        let mut pairs = positive.iter().map(|&p| (p, p.conj())).collect::<Vec<_>>();
        // Very near Nyquist, prewarping can turn the -1 prototype's conjugate pair
        // into two real poles. Pair those too (e.g. the top band at 45 kHz audio).
        if real.len() == 2 {
            pairs.push((real[0], real[1]));
        }
        if pairs.len() != 3
            || pairs
                .iter()
                .flat_map(|&(a, b)| [a, b])
                .any(|p| !p.norm2().is_finite() || p.norm2() >= 1.0)
        {
            return Err(
                "jet Butterworth poles cannot be represented stably at this sample rate".into(),
            );
        }
        pairs.sort_by(|&(a, b), &(c, d)| {
            a.norm2()
                .max(b.norm2())
                .total_cmp(&c.norm2().max(d.norm2()))
        });
        let w_center = 2.0 * w02.sqrt().atan();
        let q = Complex::new(w_center.cos(), -w_center.sin());
        let numerator = Complex::ONE.sub(q.mul(q));
        let mut sections = [Biquad::default(); 3];
        let mut poles = [Complex::default(); 6];
        let mut gain = 1.0;
        for (i, &(p, other)) in pairs.iter().enumerate() {
            let a1 = -p.add(other).re;
            let a2 = p.mul(other).re;
            // Each section has unit magnitude at the warped band center. Distributing
            // the gain this way keeps intermediate signals well scaled at low Hz.
            let denominator = Complex::ONE.add(q.scale(a1)).add(q.mul(q).scale(a2));
            let b0 = (denominator.norm2() / numerator.norm2()).sqrt();
            sections[i] = Biquad {
                b0,
                a1,
                a2,
                z1: 0.0,
                z2: 0.0,
            };
            poles[2 * i] = p;
            poles[2 * i + 1] = other;
            gain *= b0;
        }
        let variance = impulse_energy(&poles, gain);
        if !variance.is_finite() || variance <= 0.0 {
            return Err("jet Butterworth white-noise gain is not finite and positive".into());
        }
        Ok(Self {
            sections,
            white_noise_gain: variance.sqrt(),
        })
    }
    fn next(&mut self, mut x: f64) -> f64 {
        for section in &mut self.sections {
            x = section.next(x);
        }
        x / self.white_noise_gain
    }
    #[cfg(test)]
    fn response(&self, frequency: f64, sample_rate: f64) -> Complex {
        let w = 2.0 * PI * frequency / sample_rate;
        let q = Complex::new(w.cos(), -w.sin());
        self.sections
            .iter()
            .fold(Complex::ONE, |h, s| h.mul(s.response(q)))
            .scale(1.0 / self.white_noise_gain)
    }
}

/// Exact infinite impulse energy for H(z) = k (z^2-1)^3 / product(z-p_i).
/// Partial fractions give h[0]=k, h[n>=1]=sum_i r_i p_i^(n-1), so
/// sum_n h[n]^2 = k^2 + sum_ij r_i r_j/(1-p_i p_j). Complex conjugate terms
/// make the result real. This is the discrete H2 norm / unit-white-noise variance.
fn impulse_energy(poles: &[Complex; 6], gain: f64) -> f64 {
    let mut residues = [Complex::default(); 6];
    for (i, &p) in poles.iter().enumerate() {
        let mut denominator = Complex::ONE;
        for (j, &other) in poles.iter().enumerate() {
            if i != j {
                denominator = denominator.mul(p.sub(other));
            }
        }
        let numerator = p.mul(p).sub(Complex::ONE);
        residues[i] = numerator
            .mul(numerator)
            .mul(numerator)
            .scale(gain)
            .div(denominator);
    }
    let mut energy = gain * gain;
    for (i, &pi) in poles.iter().enumerate() {
        for (j, &pj) in poles.iter().enumerate() {
            energy += residues[i]
                .mul(residues[j])
                .div(Complex::ONE.sub(pi.mul(pj)))
                .re;
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;
    const FS: f64 = 48_000.0;

    #[test]
    fn stone_table_interpolation_and_endpoint_clamping() {
        let t = StoneTable::get();
        assert_eq!(t.lookup(140.0, -0.3), -8.9);
        assert_eq!(t.lookup(-1_000.0, -1_000.0), -85.0);
        assert_eq!(t.lookup(1_000.0, 1_000.0), -152.5);
        assert!((t.lookup(115.0, -0.55) - (-14.7 - 14.0 - 13.7 - 12.4) / 4.0).abs() < 1e-12);
        // The first and last log-S intervals are 1.8 decades, not 0.1.
        assert!((t.lookup(110.0, -2.7) - (-85.0 - 40.5) / 2.0).abs() < 1e-12);
        assert!((t.lookup(250.0, 2.7) - (-107.5 - 152.5) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn inflow_cutoff_and_invalid_inputs() {
        let f = band_centers(FS).unwrap();
        for v in [-100.0, 0.0, 0.999999] {
            assert!(stone_band_levels(0.085, 5.0, 20.0, [v, 1.184, 298.15], &f)
                .unwrap()
                .is_none());
        }
        assert!(
            stone_band_levels(0.085, 5.0, 20.0, [1.0, 1.184, 298.15], &f)
                .unwrap()
                .is_some()
        );
        for exit in [
            [0.0, 0.0, 298.15],
            [0.0, 1.184, -1.0],
            [f64::NAN, 1.184, 298.15],
        ] {
            assert!(stone_band_levels(0.085, 5.0, 20.0, exit, &f).is_err());
        }
        assert!(stone_band_levels(0.0, 5.0, 20.0, [10.0, 1.0, 300.0], &f).is_err());
        assert!(stone_band_levels(0.085, f64::NAN, 20.0, [10.0, 1.0, 300.0], &f).is_err());
        assert!(stone_band_levels(0.085, 5.0, 181.0, [10.0, 1.0, 300.0], &f).is_err());
        assert!(stone_band_levels(0.085, 5.0, 20.0, [10.0, 1.0, 300.0], &[0.0]).is_err());
        assert!(Jet::new(FS, &[], 5.0, 20.0).is_err());
        assert!(Jet::new(FS, &[0.085, -0.01], 5.0, 20.0).is_err());
        assert!(Jet::new(0.0, &[0.085], 5.0, 20.0).is_err());
        assert!(Jet::new(f64::INFINITY, &[0.085], 5.0, 20.0).is_err());
        assert_eq!(f.len(), 30);
        assert_eq!(band_centers(44_100.0).unwrap().len(), 29);
    }

    #[test]
    fn analytic_white_noise_gain_matches_impulse_energy() {
        let mut maximum_error: f64 = 0.0;
        for fs in [4_000.0, 9_000.0, 44_100.0, 45_000.0, FS, 96_000.0] {
            for fc in band_centers(fs).unwrap() {
                let mut f = ButterBand::new(fc, fs).unwrap();
                // Include both 100 center periods and 30 time constants of the
                // slowest pole: prewarping near Nyquist can leave a long real pole.
                let radius = f
                    .sections
                    .iter()
                    .map(|s| {
                        let discriminant = s.a1 * s.a1 - 4.0 * s.a2;
                        if discriminant < 0.0 {
                            s.a2.sqrt()
                        } else {
                            0.5 * (s.a1.abs() + discriminant.sqrt())
                        }
                    })
                    .fold(0.0_f64, f64::max);
                let n = (100.0 * fs / fc).max(-30.0 / radius.ln()).ceil() as usize;
                let mut energy = f.next(1.0).powi(2);
                for _ in 1..n {
                    energy += f.next(0.0).powi(2);
                }
                maximum_error = maximum_error.max((energy - 1.0).abs());
                assert!(
                    (energy - 1.0).abs() < 3e-7,
                    "fs={fs}, fc={fc}, energy={energy}"
                );
            }
        }
        eprintln!("maximum relative impulse-energy normalization error {maximum_error:.3e}");
    }

    #[test]
    fn causal_amplitude_interpolation_and_rejected_update_are_deterministic() {
        let mut a = Jet::new(FS, &[0.085, 0.085], 5.0, 20.0).unwrap();
        let mut b = Jet::new(FS, &[0.085, 0.085], 5.0, 20.0).unwrap();
        let low = [[100.0, 1.0, 300.0]; 2];
        let high = [[200.0, 0.6, 600.0]; 2];
        for j in [&mut a, &mut b] {
            j.set_exit_states(&low).unwrap();
        }
        let first = a.bands[0].target;
        assert_eq!(a.bands[0].start, first);
        for _ in 0..12 {
            assert_eq!(a.next_sample(), b.next_sample());
        }
        // The first outlet can be evaluated successfully before the second overflows;
        // scratch targets are allowed to change, but audible state must not change.
        assert!(a
            .set_exit_states(&[[200.0, 0.6, 600.0], [1e300, 1.0, 300.0]])
            .is_err());
        assert!(a.set_exit_states(&low[..1]).is_err());
        for _ in 0..12 {
            assert_eq!(a.next_sample(), b.next_sample());
        }
        a.set_exit_states(&high).unwrap();
        assert_eq!(a.interpolation, 0.0);
        assert_eq!(a.bands[0].start, first);
        let end = a.bands[0].target;
        for _ in 0..6 {
            a.next_sample();
        }
        assert!((a.interpolation - 0.5).abs() < 1e-14);
        assert!((lerp(a.bands[0].start, end, a.interpolation) - (first + end) / 2.0).abs() < 1e-14);
        for _ in 0..6 {
            a.next_sample();
        }
        assert!((a.interpolation - 1.0).abs() < 1e-14);
        a.set_exit_states(&[[0.0, 1.0, 300.0]; 2]).unwrap();
        for _ in 0..13 {
            a.next_sample();
        }
        assert!(a.bands.iter().all(|b| b.target == 0.0));
        assert_eq!(a.next_sample(), 0.0);
    }

    #[test]
    fn synthesized_stationary_spectrum_and_overall_level() {
        let diameters = [0.085, 0.062];
        let exits = [[150.0, 1.184, 298.15], [300.0, 0.5, 900.0]];
        let mut jet = Jet::new(FS, &diameters, 5.0, 20.0).unwrap();
        jet.set_exit_states(&exits).unwrap();
        let amplitudes = jet.bands.iter().map(|b| b.target).collect::<Vec<_>>();
        let expected_overall = amplitudes.iter().map(|v| v * v).sum::<f64>();
        let source = jet
            .centers
            .iter()
            .map(|&fc| ButterBand::new(fc, FS).unwrap())
            .collect::<Vec<_>>();
        let mut analysis = [4, 10, 16, 23].map(|b| ButterBand::new(jet.centers[b], FS).unwrap());
        // Independent synthesis bands overlap. Predict analysis-filtered power from
        // the complete bank's PSD, rather than incorrectly treating nominal band
        // amplitudes as the powers a second overlapping filterbank would measure.
        let mut expected_spectral = [0.0; 4];
        let quadrature_points = 65_536;
        for i in 0..quadrature_points {
            let f = (i as f64 + 0.5) * (FS * 0.5) / quadrature_points as f64;
            let mut psd = 0.0;
            for (b, filter) in source.iter().enumerate() {
                let power = amplitudes[b].powi(2) + amplitudes[source.len() + b].powi(2);
                psd += power * filter.response(f, FS).norm2();
            }
            for (power, filter) in expected_spectral.iter_mut().zip(&analysis) {
                // Analysis filters have unity passband gain, not unit noise variance.
                *power += psd * filter.response(f, FS).norm2() * filter.white_noise_gain.powi(2)
                    / quadrature_points as f64;
            }
        }
        let warmup = (2.0 * FS) as usize;
        let count = (12.0 * FS) as usize;
        let mut overall = 0.0;
        let mut spectral = [0.0; 4];
        for i in 0..warmup + count {
            let y = jet.next_sample();
            for (power, filter) in spectral.iter_mut().zip(&mut analysis) {
                let z = filter.next(y) * filter.white_noise_gain;
                if i >= warmup {
                    *power += z * z / count as f64;
                }
            }
            if i >= warmup {
                overall += y * y / count as f64;
            }
        }
        let overall_error = 10.0 * (overall / expected_overall).log10();
        assert!(
            overall_error.abs() < 0.4,
            "overall error {overall_error} dB"
        );
        for (b, (&measured, &expected)) in spectral.iter().zip(&expected_spectral).enumerate() {
            let error = 10.0 * (measured / expected).log10();
            assert!(error.abs() < 0.9, "analysis band {b} error {error} dB");
        }
        eprintln!(
            "jet total {:.6} dB, expected {:.6} dB; spectral errors {:?} dB",
            10.0 * (overall / P_REF.powi(2)).log10(),
            10.0 * (expected_overall / P_REF.powi(2)).log10(),
            spectral
                .iter()
                .zip(&expected_spectral)
                .map(|(a, b)| 10.0 * (a / b).log10())
                .collect::<Vec<_>>()
        );
    }

    // Generated by evaluating the unmodified `table` and `bands` function ASTs from
    // scripts/reference_sound.py with NumPy, including its exact base-2 centers.
    #[test]
    fn stone_matches_python_hot_cold_threshold_and_corrected_angles() {
        let frequencies = band_centers(FS).unwrap();
        let cases: [([f64; 3], f64, [f64; 30]); 6] = [
            (
                [150.0, 1.184, 298.15],
                20.0,
                [
                    39.19165215894432,
                    42.433126009029046,
                    45.72504034881634,
                    49.20788378267379,
                    52.75516506428083,
                    56.254030955833024,
                    59.76549293558216,
                    63.017824768791606,
                    65.7163351724087,
                    67.75050956461769,
                    69.43782969204656,
                    71.00347703806139,
                    72.16691413751903,
                    72.96956596602969,
                    72.99819795083457,
                    72.10966786430568,
                    70.71374773215402,
                    69.07221249184983,
                    67.33602451309626,
                    65.58670919180861,
                    63.83739387052095,
                    62.088078549233295,
                    60.33876322794565,
                    58.58944790665799,
                    56.840132585370334,
                    55.09081726408267,
                    53.34150194279502,
                    51.592186621507366,
                    49.84287130021971,
                    48.09355597893206,
                ],
            ),
            (
                [300.0, 0.5, 900.0],
                20.0,
                [
                    55.86150789726968,
                    59.226563268227586,
                    62.5916186391855,
                    65.95667401014342,
                    69.16327945541596,
                    72.33100112876257,
                    75.87462674142583,
                    79.73193755493138,
                    82.9837795010911,
                    85.93134786246536,
                    88.62697341823177,
                    90.51220409665953,
                    92.01204629105156,
                    93.44658906438437,
                    94.62806124446669,
                    95.46174446347914,
                    95.9476387214217,
                    95.51353272677916,
                    94.02107250560866,
                    92.1596119281613,
                    90.27587514145505,
                    88.3921383547488,
                    86.50840156804254,
                    84.62466478133629,
                    82.74092799463004,
                    80.85719120792379,
                    78.97345442121754,
                    77.08971763451129,
                    75.20598084780504,
                    73.32224406109879,
                ],
            ),
            (
                [80.0, 0.7, 700.0],
                70.0,
                [
                    27.76122689039578,
                    30.1129774383457,
                    32.47770682101948,
                    34.771594624055496,
                    36.97914792559136,
                    39.24456397204113,
                    41.37784237505929,
                    43.150511378059605,
                    44.52043372595596,
                    45.606081563680135,
                    46.549935478206194,
                    47.333179992714406,
                    47.91505117967065,
                    48.295549039074906,
                    48.535626302833045,
                    48.55242477522751,
                    48.452081443339516,
                    48.165789928742036,
                    47.639764482274416,
                    46.97503509960064,
                    46.272631776384685,
                    45.44351634292512,
                    44.47672696892338,
                    43.40925093114565,
                    42.30547428037772,
                    41.2016976296098,
                    40.03284826428009,
                    38.828728281624166,
                    37.690367677306,
                    36.52048831631232,
                ],
            ),
            (
                [1.0, 1.184, 298.15],
                0.0,
                [
                    -97.70253830537314,
                    -98.32886724409549,
                    -99.03127056731144,
                    -99.78358511286815,
                    -100.71197404291846,
                    -101.74104963674476,
                    -102.84482628751269,
                    -103.94860293828062,
                    -105.07905185965889,
                    -106.28317184231481,
                    -107.45993289058441,
                    -108.59141180762667,
                    -109.7955317902826,
                    -110.99965177293852,
                    -112.20377175559445,
                    -113.40789173825037,
                    -114.61201172090628,
                    -115.81613170356222,
                    -117.02025168621813,
                    -118.22437166887407,
                    -119.42849165152998,
                    -120.6326116341859,
                    -121.83673161684183,
                    -123.04085159949776,
                    -124.24497158215368,
                    -125.4490915648096,
                    -126.65321154746553,
                    -127.85733153012146,
                    -129.06145151277738,
                    -130.2655714954333,
                ],
            ),
            (
                [600.0, 0.4, 1000.0],
                0.0,
                [
                    82.27821134948955,
                    84.77770602696398,
                    87.2772007044384,
                    89.77669538191282,
                    92.27619005938725,
                    94.77568473686168,
                    97.27517941433611,
                    99.77467409181054,
                    102.27416876928497,
                    104.7736634467594,
                    107.01532523350055,
                    109.02219187126043,
                    111.0290585090203,
                    113.03592514678017,
                    115.03791057065511,
                    116.21725724033837,
                    116.15996072532175,
                    115.37952362088105,
                    114.10477291464835,
                    112.43987308569815,
                    110.35533573553364,
                    108.04562137816471,
                    105.73590702079576,
                    103.42619266342683,
                    101.1164783060579,
                    98.80676394868897,
                    96.49704959132,
                    94.18733523395107,
                    91.87762087658214,
                    89.5679065192132,
                ],
            ),
            (
                [250.0, 0.8, 500.0],
                180.0,
                [
                    46.20796437684458,
                    48.69032927386602,
                    51.19261214793211,
                    53.59420835822222,
                    55.90909207855581,
                    58.31000162506993,
                    60.62557200917951,
                    63.01777780934392,
                    65.22533111087978,
                    67.44158815876534,
                    69.72234355747611,
                    71.69164855473319,
                    73.25820689688634,
                    74.4421727317389,
                    75.43518564482916,
                    76.31674815646576,
                    76.99693734055039,
                    77.47575319708305,
                    77.76498945940537,
                    77.92926492749243,
                    77.82892159560444,
                    77.69010707669955,
                    77.26239962736031,
                    76.64682924325074,
                    75.94442592003477,
                    75.2136284837036,
                    74.29599810826606,
                    73.27768106905253,
                    72.1739044182846,
                    71.07012776751667,
                ],
            ),
        ];
        let mut maximum_error: f64 = 0.0;
        for (exit, angle, expected) in cases {
            let actual = stone_band_levels(0.085, 5.0, angle, exit, &frequencies)
                .unwrap()
                .unwrap();
            for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                maximum_error = maximum_error.max((a - b).abs());
                assert!(
                    (a - b).abs() < 3e-10,
                    "exit={exit:?}, angle={angle}, band={i}, actual={a}, expected={b}"
                );
            }
        }
        eprintln!("Stone maximum Python discrepancy {maximum_error:.3e} dB over 180 bands");
    }

    // scipy.signal.butter(3,[lo,hi],btype="band",fs=48000,output="sos")
    // evaluated with scipy.signal.sosfreqz. Compare complex response, not just level.
    #[test]
    fn butterworth_matches_scipy_bandpass_complex_response() {
        let fixtures = [
            (
                24.803141437003124,
                12.401570718501562,
                -0.0011223627548763646,
                -0.003503651392107503,
            ),
            (
                24.803141437003124,
                22.097086912079615,
                -0.49999999999860045,
                0.49999999999638334,
            ),
            (
                24.803141437003124,
                24.803141437003124,
                0.9999999999764979,
                4.0679568855973445e-07,
            ),
            (
                24.803141437003124,
                27.840584941885613,
                -0.4999999999726413,
                -0.5000000000912463,
            ),
            (
                24.803141437003124,
                37.20471215550469,
                -0.011458644006877908,
                0.018134331277799745,
            ),
            (250.0, 125.0, -0.0011226514837731588, -0.0035043087601205358),
            (
                250.0,
                222.72467953508482,
                -0.4999999999999856,
                0.500000000000161,
            ),
            (250.0, 250.0, 0.9999999991455552, 4.133398165478848e-05),
            (
                250.0,
                280.6155120773433,
                -0.49999999999991734,
                -0.5000000000003599,
            ),
            (250.0, 375.0, -0.01145406942864333, 0.018129466194970183),
            (
                2519.8420997897465,
                1259.9210498948732,
                -0.0011524495904034027,
                -0.0035718949840738887,
            ),
            (
                2519.8420997897465,
                2244.9240966187463,
                -0.500000000000003,
                0.5000000000000029,
            ),
            (
                2519.8420997897465,
                2519.8420997897465,
                0.9999910529022333,
                0.004230143673536219,
            ),
            (
                2519.8420997897465,
                2828.42712474619,
                -0.5000000000000047,
                -0.49999999999999756,
            ),
            (
                2519.8420997897465,
                3779.7631496846197,
                -0.010991444640169289,
                0.01763337246929454,
            ),
            (
                20158.73679831797,
                10079.368399158984,
                -0.008787215549679466,
                -0.015147141845720927,
            ),
            (
                20158.73679831797,
                17959.392772949966,
                -0.4999999999999993,
                0.49999999999999983,
            ),
            (
                20158.73679831797,
                20158.73679831797,
                0.7709025527011254,
                0.6357914416992425,
            ),
            (
                20158.73679831797,
                22627.416997969518,
                -0.49999999999999756,
                -0.4999999999999983,
            ),
            (
                20158.73679831797,
                23760.0,
                -0.0007113743478759194,
                0.0025096996740549588,
            ),
        ];
        for (fc, frequency, re, im) in fixtures {
            let filter = ButterBand::new(fc, FS).unwrap();
            let h = filter
                .response(frequency, FS)
                .scale(filter.white_noise_gain);
            assert!(
                (h.re - re).abs() < 3e-8 && (h.im - im).abs() < 3e-8,
                "fc={fc}, f={frequency}, h={h:?}, expected {re}+{im}i"
            );
        }
    }
}
