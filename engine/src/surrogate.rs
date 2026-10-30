//! Experimental causal, control-gated FIR surrogate, evaluated at the model's own rate.
//!
//! For feature `f = power_index * cylinders + cylinder_index`,
//! `phi_f[n] = sign(q[n]) * abs(q[n] / flow_scale)^power`. The raw output is
//! `y_o[n] = sum_g weight_g(controls[n]) * (bias_go + sum_f sum_k h_gof[k] phi_f[n-k])`.
//! Thus current controls mix already-running tails, not just newly arriving flow. There
//! is no resampling, output decoding, directivity, clipping or additional sound source here.
//!
//! Uniform partitioned overlap-save shares each feature's FFT/history across every gate
//! and output, and sums features in frequency space before one inverse FFT per gate/output.
//! No future blocks are read; tap zero is the current sample, with zero initial history.
use crate::synth::BLOCK;
use serde::{Deserialize, Serialize};

const FFT_SIZE: usize = 2 * BLOCK;
const BINS: usize = BLOCK + 1;
const _: () = assert!(FFT_SIZE.is_power_of_two());

/// Kernels are indexed [gate][output][feature][tap], with feature index
/// `power_index * cylinders + cylinder_index`. Labels describe raw outputs, not a decoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurrogateModel {
    pub rate: u32,
    pub cylinders: usize,
    pub flow_scale: f32,
    pub powers: Vec<u32>,
    pub centers: Vec<Vec<f32>>,
    pub widths: Vec<f32>,
    pub kernels: Vec<Vec<Vec<Vec<f32>>>>,
    #[serde(default)]
    pub biases: Vec<Vec<f32>>,
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Validated dimensions and labels, without copying the model's coefficient arrays.
#[derive(Debug, Clone, Copy)]
pub struct SurrogateMetadata<'a> {
    pub rate: u32,
    pub cylinders: usize,
    pub controls: usize,
    pub features: usize,
    pub gates: usize,
    pub outputs: usize,
    pub taps: usize,
    pub block_size: usize,
    pub labels: &'a [String],
}

fn checked_len<T>(dimensions: &[usize]) -> Result<usize, String> {
    let count = dimensions
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("surrogate dimensions overflow")?;
    if count > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err("surrogate allocation dimensions overflow".into());
    }
    Ok(count)
}

fn zeros<T: Copy + Default>(len: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| format!("cannot allocate surrogate buffers: {e}"))?;
    values.resize(len, T::default());
    Ok(values)
}

impl SurrogateModel {
    pub fn validate(&self) -> Result<(), String> {
        if self.rate == 0
            || self.cylinders == 0
            || !self.flow_scale.is_finite()
            || self.flow_scale <= 0.0
        {
            return Err("rate, cylinders and flowScale must be positive and finite".into());
        }
        if self.powers.is_empty() || self.powers.contains(&0) {
            return Err("powers must be nonempty positive integers".into());
        }
        let features = checked_len::<f32>(&[self.powers.len(), self.cylinders])?;
        if self.centers.is_empty() || self.centers.len() != self.kernels.len() {
            return Err("centers and kernels must have the same nonzero gate count".into());
        }
        if self.widths.iter().any(|&w| !w.is_finite() || w <= 0.0)
            || self
                .centers
                .iter()
                .any(|c| c.len() != self.widths.len() || c.iter().any(|v| !v.is_finite()))
        {
            return Err(
                "each center must match finite positive widths and contain finite values".into(),
            );
        }
        let outputs = self.outputs();
        if outputs == 0 {
            return Err("at least one output is required".into());
        }
        let taps = self.taps();
        if taps == 0 {
            return Err("kernels must have at least one tap".into());
        }
        for gate in &self.kernels {
            if gate.len() != outputs {
                return Err("inconsistent output count".into());
            }
            for output in gate {
                if output.len() != features {
                    return Err("inconsistent feature count".into());
                }
                for kernel in output {
                    if kernel.len() != taps || kernel.iter().any(|v| !v.is_finite()) {
                        return Err(
                            "kernels must have equal nonzero tap counts and finite coefficients"
                                .into(),
                        );
                    }
                }
            }
        }
        if !self.biases.is_empty()
            && (self.biases.len() != self.centers.len()
                || self
                    .biases
                    .iter()
                    .any(|b| b.len() != outputs || b.iter().any(|v| !v.is_finite())))
        {
            return Err("biases must have one finite value per gate and output".into());
        }
        if !self.labels.is_empty() && self.labels.len() != outputs {
            return Err("labels must match output count".into());
        }
        let parts = taps.div_ceil(BLOCK);
        checked_len::<f32>(&[self.cylinders, BLOCK])?;
        checked_len::<f32>(&[self.controls(), BLOCK])?;
        checked_len::<f32>(&[features, BLOCK])?;
        checked_len::<f32>(&[self.centers.len(), BLOCK])?;
        checked_len::<f64>(&[self.centers.len()])?;
        checked_len::<f64>(&[outputs, BLOCK])?;
        checked_len::<f32>(&[features, parts, BINS])?;
        checked_len::<f32>(&[self.centers.len(), outputs, features, parts, BINS])?;
        Ok(())
    }

    /// Returns zero for an empty/malformed model; `validate` checks the full shape.
    pub fn outputs(&self) -> usize {
        self.kernels.first().map_or(0, Vec::len)
    }
    /// Returns zero for an empty/malformed model; `validate` checks the full shape.
    pub fn taps(&self) -> usize {
        self.kernels
            .first()
            .and_then(|g| g.first())
            .and_then(|o| o.first())
            .map_or(0, Vec::len)
    }
    pub fn controls(&self) -> usize {
        self.widths.len()
    }
}

/// Fixed-size real-input FFT plan. Only the nonredundant bins are stored between blocks.
struct Fft {
    cos: [f32; BLOCK],
    sin: [f32; BLOCK],
    reverse: [usize; FFT_SIZE],
}

impl Fft {
    fn new() -> Self {
        let phase = -2.0 * std::f64::consts::PI / FFT_SIZE as f64;
        Self {
            cos: std::array::from_fn(|i| (phase * i as f64).cos() as f32),
            sin: std::array::from_fn(|i| (phase * i as f64).sin() as f32),
            reverse: std::array::from_fn(|i| {
                i.reverse_bits() >> (usize::BITS - FFT_SIZE.trailing_zeros())
            }),
        }
    }

    fn run(&self, re: &mut [f32; FFT_SIZE], im: &mut [f32; FFT_SIZE], inverse: bool) {
        for (i, &j) in self.reverse.iter().enumerate() {
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= FFT_SIZE {
            let half = len / 2;
            let step = FFT_SIZE / len;
            for start in (0..FFT_SIZE).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], sign * self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
            len *= 2;
        }
    }
}

/// Signed integer power, including the common linear/quadratic features without `powf`.
fn signed_power(q: f32, mut power: u32) -> f32 {
    if power == 1 {
        return q;
    }
    let mut base = q.abs();
    if power == 2 {
        return (base * base).copysign(q);
    }
    let mut value = 1.0;
    while power != 0 {
        if power & 1 != 0 {
            value *= base;
        }
        power >>= 1;
        if power != 0 {
            base *= base;
        }
    }
    value.copysign(q)
}

/// logit(g) - logit(reference), factored to cancel the common control-squared term.
/// Squaring absolute distances first can lose even decisive differences far from centers.
fn relative_logit(controls: &[f32], center: &[f32], reference: &[f32], widths: &[f32]) -> f64 {
    let mut difference = 0.0;
    for j in 0..widths.len() {
        let (c, r, u, w) = (
            center[j] as f64,
            reference[j] as f64,
            controls[j] as f64,
            widths[j] as f64,
        );
        difference += ((r - c) / w) * ((2.0 * u - (c + r)) / w);
    }
    -0.5 * difference
}

pub struct Surrogate {
    model: SurrogateModel,
    fft: Fft,
    features_count: usize,
    parts: usize,
    position: usize,
    features: Vec<f32>, // feature-major current block
    previous: Vec<f32>, // feature-major previous block
    h_re: Vec<f32>,     // [gate][output][feature][partition][bin]
    h_im: Vec<f32>,
    history_re: Vec<f32>, // [feature][partition ring slot][bin]
    history_im: Vec<f32>,
    pending_re: Vec<f32>, // [feature][bin], not committed until the block succeeds
    pending_im: Vec<f32>,
    weights: Vec<f32>, // gate-major blocks
    logits: Vec<f64>,
    mixed: Vec<f64>, // frame-interleaved; f64 avoids overflow while summing gated biases/tails
    re: [f32; FFT_SIZE],
    im: [f32; FFT_SIZE],
}

impl Surrogate {
    pub const BLOCK_SIZE: usize = BLOCK;

    pub fn from_json(json: &str) -> Result<Self, String> {
        let model =
            serde_json::from_str(json).map_err(|e| format!("invalid surrogate JSON: {e}"))?;
        Self::new(model)
    }

    pub fn new(model: SurrogateModel) -> Result<Self, String> {
        model.validate()?;
        let features_count = model.powers.len() * model.cylinders;
        let parts = model.taps().div_ceil(BLOCK);
        let spectra_len = checked_len::<f32>(&[
            model.centers.len(),
            model.outputs(),
            features_count,
            parts,
            BINS,
        ])?;
        let mut surrogate = Self {
            fft: Fft::new(),
            features_count,
            parts,
            position: 0,
            features: zeros(features_count * BLOCK)?,
            previous: zeros(features_count * BLOCK)?,
            h_re: zeros(spectra_len)?,
            h_im: zeros(spectra_len)?,
            history_re: zeros(features_count * parts * BINS)?,
            history_im: zeros(features_count * parts * BINS)?,
            pending_re: zeros(features_count * BINS)?,
            pending_im: zeros(features_count * BINS)?,
            weights: zeros(model.centers.len() * BLOCK)?,
            logits: zeros(model.centers.len())?,
            mixed: zeros(model.outputs() * BLOCK)?,
            re: [0.0; FFT_SIZE],
            im: [0.0; FFT_SIZE],
            model,
        };
        let mut offset = 0;
        for gate in &surrogate.model.kernels {
            for output in gate {
                for kernel in output {
                    for partition in kernel.chunks(BLOCK) {
                        surrogate.re.fill(0.0);
                        surrogate.im.fill(0.0);
                        surrogate.re[..partition.len()].copy_from_slice(partition);
                        surrogate
                            .fft
                            .run(&mut surrogate.re, &mut surrogate.im, false);
                        if surrogate.re[..BINS]
                            .iter()
                            .chain(&surrogate.im[..BINS])
                            .any(|v| !v.is_finite())
                        {
                            return Err("kernel FFT overflow: coefficients exceed the f32 decoder's numerical range".into());
                        }
                        surrogate.h_re[offset..offset + BINS]
                            .copy_from_slice(&surrogate.re[..BINS]);
                        surrogate.h_im[offset..offset + BINS]
                            .copy_from_slice(&surrogate.im[..BINS]);
                        offset += BINS;
                    }
                }
            }
        }
        Ok(surrogate)
    }

    pub fn model(&self) -> &SurrogateModel {
        &self.model
    }
    pub const fn block_size(&self) -> usize {
        BLOCK
    }
    pub fn metadata(&self) -> SurrogateMetadata<'_> {
        SurrogateMetadata {
            rate: self.model.rate,
            cylinders: self.model.cylinders,
            controls: self.model.controls(),
            features: self.features_count,
            gates: self.model.centers.len(),
            outputs: self.model.outputs(),
            taps: self.model.taps(),
            block_size: BLOCK,
            labels: &self.model.labels,
        }
    }

    /// Return to zero prehistory without allocating or changing coefficients/metadata.
    pub fn reset(&mut self) {
        self.previous.fill(0.0);
        self.history_re.fill(0.0);
        self.history_im.fill(0.0);
        self.position = 0;
    }

    /// `flow` is BLOCK frames interleaved by cylinder; `controls` is BLOCK frames
    /// interleaved by control dimension (empty for a model without controls).
    /// `output` is BLOCK frames interleaved by raw output channel.
    ///
    /// All gates use the entire feature history, including while their weight is zero.
    /// Successful processing and `reset` allocate nothing. Invalid dimensions, nonfinite
    /// inputs, transformed-feature/FFT overflow or unrepresentable output return an error,
    /// leaving both convolution history and the caller's output unchanged. No clipping or
    /// arbitrary fallback is used to hide numerical failure.
    pub fn process_block(
        &mut self,
        flow: &[f32],
        controls: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let m = &self.model;
        let outputs = m.outputs();
        if flow.len() != BLOCK * m.cylinders
            || controls.len() != BLOCK * m.controls()
            || output.len() != BLOCK * outputs
        {
            return Err("block dimensions do not match model".into());
        }
        if flow.iter().chain(controls).any(|v| !v.is_finite()) {
            return Err("nonfinite block input".into());
        }
        for (pi, &power) in m.powers.iter().enumerate() {
            for cyl in 0..m.cylinders {
                let feature = (pi * m.cylinders + cyl) * BLOCK;
                for t in 0..BLOCK {
                    let value = signed_power(flow[t * m.cylinders + cyl] / m.flow_scale, power);
                    if !value.is_finite() {
                        return Err("nonfinite transformed flow".into());
                    }
                    self.features[feature + t] = value;
                }
            }
        }
        if m.centers.len() == 1 {
            self.weights.fill(1.0);
        } else if m.controls() == 0 {
            self.weights.fill(1.0 / m.centers.len() as f32);
        } else {
            for t in 0..BLOCK {
                let control = &controls[t * m.controls()..(t + 1) * m.controls()];
                // A pairwise nearest-center search avoids losing softmax differences to a
                // huge common distance, even at the extremes of finite f32 controls/widths.
                let mut nearest = 0;
                for g in 1..m.centers.len() {
                    if relative_logit(control, &m.centers[g], &m.centers[nearest], &m.widths) > 0.0
                    {
                        nearest = g;
                    }
                }
                let mut max_logit = f64::NEG_INFINITY;
                for (g, center) in m.centers.iter().enumerate() {
                    let logit = relative_logit(control, center, &m.centers[nearest], &m.widths);
                    if !logit.is_finite() {
                        return Err("nonfinite gate distance".into());
                    }
                    self.logits[g] = logit;
                    max_logit = max_logit.max(logit);
                }
                let mut sum = 0.0;
                for logit in &mut self.logits {
                    *logit = (*logit - max_logit).exp();
                    sum += *logit;
                }
                for g in 0..m.centers.len() {
                    self.weights[g * BLOCK + t] = (self.logits[g] / sum) as f32;
                }
            }
        }
        for f in 0..self.features_count {
            self.re[..BLOCK].copy_from_slice(&self.previous[f * BLOCK..(f + 1) * BLOCK]);
            self.re[BLOCK..].copy_from_slice(&self.features[f * BLOCK..(f + 1) * BLOCK]);
            self.im.fill(0.0);
            self.fft.run(&mut self.re, &mut self.im, false);
            if self.re[..BINS]
                .iter()
                .chain(&self.im[..BINS])
                .any(|v| !v.is_finite())
            {
                return Err("feature FFT overflow".into());
            }
            self.pending_re[f * BINS..(f + 1) * BINS].copy_from_slice(&self.re[..BINS]);
            self.pending_im[f * BINS..(f + 1) * BINS].copy_from_slice(&self.im[..BINS]);
        }
        self.mixed.fill(0.0);
        for g in 0..m.centers.len() {
            // History belongs to features, not gates: zero-weight gates can be skipped
            // without losing their responses to old input when they become active again.
            if self.weights[g * BLOCK..(g + 1) * BLOCK]
                .iter()
                .all(|&w| w == 0.0)
            {
                continue;
            }
            for o in 0..outputs {
                self.re.fill(0.0);
                self.im.fill(0.0);
                for f in 0..self.features_count {
                    let path = ((g * outputs + o) * self.features_count + f) * self.parts * BINS;
                    for p in 0..self.parts {
                        let h = path + p * BINS;
                        let (xr, xi) = if p == 0 {
                            (
                                &self.pending_re[f * BINS..(f + 1) * BINS],
                                &self.pending_im[f * BINS..(f + 1) * BINS],
                            )
                        } else {
                            let slot = (self.position + self.parts - p) % self.parts;
                            let x = (f * self.parts + slot) * BINS;
                            (&self.history_re[x..x + BINS], &self.history_im[x..x + BINS])
                        };
                        let (hr, hi) = (&self.h_re[h..h + BINS], &self.h_im[h..h + BINS]);
                        for k in 0..BINS {
                            self.re[k] += xr[k] * hr[k] - xi[k] * hi[k];
                            self.im[k] += xr[k] * hi[k] + xi[k] * hr[k];
                        }
                    }
                }
                if self.re[..BINS]
                    .iter()
                    .chain(&self.im[..BINS])
                    .any(|v| !v.is_finite())
                {
                    return Err("convolution spectrum overflow".into());
                }
                // Normalize before the inverse transform, reducing intermediate overflow.
                for k in 0..BINS {
                    self.re[k] *= 1.0 / FFT_SIZE as f32;
                    self.im[k] *= 1.0 / FFT_SIZE as f32;
                }
                self.im[0] = 0.0;
                self.im[BLOCK] = 0.0;
                for k in 1..BLOCK {
                    self.re[FFT_SIZE - k] = self.re[k];
                    self.im[FFT_SIZE - k] = -self.im[k];
                }
                self.fft.run(&mut self.re, &mut self.im, true);
                let bias = if m.biases.is_empty() {
                    0.0
                } else {
                    m.biases[g][o] as f64
                };
                for t in 0..BLOCK {
                    let value = self.re[BLOCK + t] as f64 + bias;
                    if !value.is_finite() {
                        return Err("convolution overflow".into());
                    }
                    self.mixed[t * outputs + o] += self.weights[g * BLOCK + t] as f64 * value;
                }
            }
        }
        if self.mixed.iter().any(|&v| !(v as f32).is_finite()) {
            return Err("output exceeds finite f32 range".into());
        }
        // Commit only after every feature, gate and output has succeeded.
        for f in 0..self.features_count {
            let x = (f * self.parts + self.position) * BINS;
            self.history_re[x..x + BINS]
                .copy_from_slice(&self.pending_re[f * BINS..(f + 1) * BINS]);
            self.history_im[x..x + BINS]
                .copy_from_slice(&self.pending_im[f * BINS..(f + 1) * BINS]);
        }
        self.previous.copy_from_slice(&self.features);
        self.position = (self.position + 1) % self.parts;
        for (out, &value) in output.iter_mut().zip(&self.mixed) {
            *out = value as f32;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(taps: Vec<f32>) -> SurrogateModel {
        SurrogateModel {
            rate: 20_000,
            cylinders: 1,
            flow_scale: 1.0,
            powers: vec![1],
            centers: vec![vec![]],
            widths: vec![],
            kernels: vec![vec![vec![taps]]],
            biases: vec![],
            labels: vec![],
        }
    }

    fn close(actual: f32, expected: f64, tolerance: f64) {
        assert!(
            (actual as f64 - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn impulse_orientation_and_block_boundary() {
        let mut h = vec![0.0; BLOCK + 3];
        h[0] = 1.0;
        h[2] = 3.0;
        h[BLOCK + 2] = 5.0;
        let mut s = Surrogate::new(model(h)).unwrap();
        let mut x = vec![0.0; BLOCK];
        let mut y = vec![0.0; BLOCK];
        for b in 0..4 {
            x.fill(0.0);
            if b == 0 {
                x[BLOCK - 1] = 2.0;
            }
            s.process_block(&x, &[], &mut y).unwrap();
            for (t, &sample) in y.iter().enumerate() {
                let expected = match b * BLOCK + t {
                    n if n == BLOCK - 1 => 2.0,
                    n if n == BLOCK + 1 => 6.0,
                    n if n == 2 * BLOCK + 1 => 10.0,
                    _ => 0.0,
                };
                close(sample, expected, 1e-5);
            }
        }
    }

    #[test]
    fn matches_direct_convolution_across_partition_lengths_and_ring_wraps() {
        for taps in [
            1,
            BLOCK - 1,
            BLOCK,
            BLOCK + 1,
            2 * BLOCK,
            2 * BLOCK + 1,
            1024,
            1031,
        ] {
            let h: Vec<_> = (0..taps).map(|i| (i as f32 * 0.14).sin() * 0.01).collect();
            let mut s = Surrogate::new(model(h.clone())).unwrap();
            let blocks = 2 * taps.div_ceil(BLOCK) + 4;
            let input: Vec<_> = (0..BLOCK * blocks)
                .map(|i| {
                    if i < BLOCK * (blocks - taps.div_ceil(BLOCK) - 1) {
                        (i as f32 * 0.17).cos()
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut y = vec![0.0; BLOCK];
            for b in 0..blocks {
                s.process_block(&input[b * BLOCK..(b + 1) * BLOCK], &[], &mut y)
                    .unwrap();
                for (t, &sample) in y.iter().enumerate() {
                    let k = b * BLOCK + t;
                    let expected = (0..taps.min(k + 1))
                        .map(|j| h[j] as f64 * input[k - j] as f64)
                        .sum();
                    close(sample, expected, 2e-6);
                }
            }
        }
    }

    #[test]
    fn future_samples_and_controls_do_not_change_earlier_output() {
        let mut m = model(
            (0..BLOCK * 2 + 3)
                .map(|i| (i as f32 * 0.03).cos() * 0.01)
                .collect(),
        );
        m.widths = vec![0.3];
        m.centers = vec![vec![0.0], vec![1.0]];
        m.kernels.push(m.kernels[0].clone());
        for k in &mut m.kernels[1][0][0] {
            *k *= 2.0;
        }
        let mut a = Surrogate::new(m.clone()).unwrap();
        let mut b = Surrogate::new(m).unwrap();
        let mut x = vec![0.1; BLOCK];
        let mut control = vec![0.0; BLOCK];
        let mut ya = vec![0.0; BLOCK];
        let mut yb = vec![0.0; BLOCK];
        for _ in 0..3 {
            a.process_block(&x, &control, &mut ya).unwrap();
            b.process_block(&x, &control, &mut yb).unwrap();
        }
        a.process_block(&x, &control, &mut ya).unwrap();
        x[BLOCK / 2..].fill(-2.0);
        control[BLOCK / 2..].fill(1.0);
        b.process_block(&x, &control, &mut yb).unwrap();
        for t in 0..BLOCK / 2 {
            close(yb[t], ya[t] as f64, 2e-6);
        }
    }

    #[test]
    fn gate_switches_existing_tail_even_if_previously_zero_weight() {
        let mut m = model(vec![1.0; BLOCK + 20]);
        m.centers = vec![vec![0.0], vec![1.0]];
        m.widths = vec![0.01];
        m.kernels = vec![
            vec![vec![vec![1.0; BLOCK + 20]]],
            vec![vec![vec![3.0; BLOCK + 20]]],
        ];
        let mut s = Surrogate::new(m).unwrap();
        let mut x = vec![0.0; BLOCK];
        x[0] = 1.0;
        let mut y = vec![0.0; BLOCK];
        s.process_block(&x, &vec![0.0; BLOCK], &mut y).unwrap();
        x.fill(0.0);
        let mut controls = vec![0.0; BLOCK];
        controls[3..10].fill(1.0);
        s.process_block(&x, &controls, &mut y).unwrap();
        for (t, &sample) in y.iter().enumerate() {
            close(
                sample,
                if t >= 20 {
                    0.0
                } else if (3..10).contains(&t) {
                    3.0
                } else {
                    1.0
                },
                1e-5,
            );
        }
    }

    #[test]
    fn nonlinear_features_are_power_major_cylinder_minor_and_signed() {
        let mut m = model(vec![0.0]);
        m.cylinders = 2;
        m.flow_scale = 0.25;
        m.powers = vec![2, 1, 3];
        m.kernels = vec![(0..6)
            .map(|o| {
                (0..6)
                    .map(|f| vec![if o == f { 1.0 } else { 0.0 }])
                    .collect()
            })
            .collect()];
        let mut s = Surrogate::new(m).unwrap();
        let x: Vec<_> = (0..BLOCK).flat_map(|_| [-0.5, 0.75]).collect();
        let mut y = vec![0.0; BLOCK * 6];
        s.process_block(&x, &[], &mut y).unwrap();
        for frame in y.chunks_exact(6) {
            for (&actual, expected) in frame.iter().zip([-4.0, 9.0, -2.0, 3.0, -8.0, 27.0]) {
                close(actual, expected, 2e-6);
            }
        }
        assert_eq!(signed_power(-1.0, u32::MAX), -1.0);
    }

    #[test]
    fn mimo_gated_convolution_matches_direct_equation() {
        let (features, outputs, gates, taps, blocks) = (4, 2, 3, BLOCK + 7, 5);
        let mut m = model(vec![0.0]);
        m.cylinders = 2;
        m.powers = vec![1, 2];
        m.flow_scale = 0.3;
        m.centers = vec![vec![-0.5, 0.2], vec![0.4, -0.2], vec![0.9, 0.5]];
        m.widths = vec![0.7, 0.4];
        m.kernels = (0..gates)
            .map(|g| {
                (0..outputs)
                    .map(|o| {
                        (0..features)
                            .map(|f| {
                                (0..taps)
                                    .map(|k| {
                                        ((g * 3 + o * 7 + f * 13 + k) as f32 * 0.19).sin() * 0.02
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        m.biases = vec![vec![0.2, -0.4], vec![-0.1, 0.5], vec![0.6, -0.3]];
        let mut s = Surrogate::new(m.clone()).unwrap();
        let input: Vec<_> = (0..BLOCK * blocks * 2)
            .map(|i| (i as f32 * 0.17).cos() * 0.2)
            .collect();
        let controls: Vec<_> = (0..BLOCK * blocks * 2)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let mut y = vec![0.0; BLOCK * outputs];
        for b in 0..blocks {
            s.process_block(
                &input[b * BLOCK * 2..(b + 1) * BLOCK * 2],
                &controls[b * BLOCK * 2..(b + 1) * BLOCK * 2],
                &mut y,
            )
            .unwrap();
            for t in 0..BLOCK {
                let n = b * BLOCK + t;
                let logits: Vec<_> = m
                    .centers
                    .iter()
                    .map(|c| {
                        -0.5 * (0..2)
                            .map(|j| {
                                ((controls[n * 2 + j] as f64 - c[j] as f64) / m.widths[j] as f64)
                                    .powi(2)
                            })
                            .sum::<f64>()
                    })
                    .collect();
                let normalization: f64 = logits.iter().map(|&l| l.exp()).sum();
                for o in 0..outputs {
                    let expected = (0..gates)
                        .map(|g| {
                            let convolution: f64 = (0..features)
                                .map(|f| {
                                    (0..taps.min(n + 1))
                                        .map(|k| {
                                            let q = input[(n - k) * 2 + f % 2] / m.flow_scale;
                                            m.kernels[g][o][f][k] as f64
                                                * signed_power(q, m.powers[f / 2]) as f64
                                        })
                                        .sum::<f64>()
                                })
                                .sum();
                            logits[g].exp() / normalization * (convolution + m.biases[g][o] as f64)
                        })
                        .sum();
                    close(y[t * outputs + o], expected, 3e-6);
                }
            }
        }
    }

    #[test]
    fn zero_flow_biases_and_labels_are_raw_gated_outputs() {
        let mut m = model(vec![0.0]);
        m.centers = vec![vec![0.0], vec![1.0]];
        m.widths = vec![0.01];
        m.kernels = vec![vec![vec![vec![0.0]], vec![vec![0.0]]]; 2];
        m.biases = vec![vec![2.0, -5.0], vec![4.0, 7.0]];
        m.labels = vec!["pressure2".into(), "logTemperature".into()];
        let mut s = Surrogate::new(m).unwrap();
        let mut controls = vec![0.0; BLOCK];
        controls[1] = 1.0;
        controls[2] = 0.5;
        let mut y = vec![0.0; BLOCK * 2];
        s.process_block(&vec![0.0; BLOCK], &controls, &mut y)
            .unwrap();
        assert_eq!(&y[..6], &[2.0, -5.0, 4.0, 7.0, 3.0, 1.0]);
        assert_eq!(s.metadata().labels, &["pressure2", "logTemperature"]);
        let mut s = Surrogate::new(model(vec![0.0; BLOCK + 3])).unwrap();
        s.process_block(&vec![0.0; BLOCK], &[], &mut y[..BLOCK])
            .unwrap();
        assert!(y[..BLOCK].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn no_control_dimensions_give_uniform_gate_weights() {
        let mut m = model(vec![1.0]);
        m.centers = vec![vec![], vec![]];
        m.kernels = vec![vec![vec![vec![1.0]]], vec![vec![vec![3.0]]]];
        m.biases = vec![vec![2.0], vec![6.0]];
        let mut s = Surrogate::new(m).unwrap();
        let mut y = vec![0.0; BLOCK];
        s.process_block(&vec![1.0; BLOCK], &[], &mut y).unwrap();
        for value in y {
            close(value, 6.0, 1e-6);
        }
    }

    #[test]
    fn softmax_remains_finite_and_selective_for_extreme_finite_controls() {
        let mut m = model(vec![0.0]);
        m.centers = vec![vec![-f32::MAX], vec![0.0], vec![1.0]];
        m.widths = vec![f32::from_bits(1)];
        m.kernels = vec![vec![vec![vec![0.0]]]; 3];
        m.biases = vec![vec![1.0], vec![2.0], vec![3.0]];
        let mut s = Surrogate::new(m).unwrap();
        let mut y = vec![0.0; BLOCK];
        s.process_block(&vec![0.0; BLOCK], &vec![f32::MAX; BLOCK], &mut y)
            .unwrap();
        assert!(y.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn invalid_models_rejected_without_shape_panics() {
        let cases: &[(&str, fn(&mut SurrogateModel))] = &[
            ("rate", |m| m.rate = 0),
            ("cylinders", |m| m.cylinders = 0),
            ("overflow dimensions", |m| m.cylinders = usize::MAX),
            ("zero scale", |m| m.flow_scale = 0.0),
            ("negative scale", |m| m.flow_scale = -1.0),
            ("nan scale", |m| m.flow_scale = f32::NAN),
            ("inf scale", |m| m.flow_scale = f32::INFINITY),
            ("empty powers", |m| m.powers.clear()),
            ("zero power", |m| m.powers[0] = 0),
            ("empty gates", |m| m.centers.clear()),
            ("gate count", |m| m.centers.push(vec![])),
            ("control count", |m| m.centers[0].push(0.0)),
            ("nan center", |m| {
                m.widths.push(1.0);
                m.centers[0].push(f32::NAN);
            }),
            ("zero width", |m| {
                m.widths.push(0.0);
                m.centers[0].push(0.0);
            }),
            ("inf width", |m| {
                m.widths.push(f32::INFINITY);
                m.centers[0].push(0.0);
            }),
            ("empty kernels", |m| m.kernels.clear()),
            ("empty outputs", |m| m.kernels[0].clear()),
            ("empty features", |m| m.kernels[0][0].clear()),
            ("empty taps", |m| m.kernels[0][0][0].clear()),
            ("feature count", |m| m.kernels[0][0].push(vec![1.0])),
            ("nan kernel", |m| m.kernels[0][0][0][0] = f32::NAN),
            ("inconsistent outputs", |m| {
                m.centers.push(vec![]);
                m.kernels.push(vec![]);
            }),
            ("inconsistent taps", |m| {
                m.centers.push(vec![]);
                m.kernels.push(vec![vec![vec![1.0, 2.0]]]);
            }),
            ("bias gate count", |m| m.biases = vec![vec![0.0]; 2]),
            ("bias output count", |m| m.biases = vec![vec![]]),
            ("nan bias", |m| m.biases = vec![vec![f32::NAN]]),
            ("label count", |m| {
                m.labels = vec!["one".into(), "two".into()]
            }),
        ];
        for &(name, mutate) in cases {
            let mut m = model(vec![1.0]);
            mutate(&mut m);
            assert!(Surrogate::new(m).is_err(), "accepted {name}");
        }
        assert!(Surrogate::new(model(vec![f32::MAX; BLOCK])).is_err());
        assert!(Surrogate::from_json("{}").is_err());
        let json = r#"{"rate":20000,"cylinders":1,"flowScale":1,"powers":[1],"centers":[[]],"widths":[],"kernels":[[[[1]]]]}"#;
        assert!(Surrogate::from_json(json).is_ok());
    }

    fn rejection_preserves_state(m: SurrogateModel, valid: f32, bad: f32) {
        let mut a = Surrogate::new(m.clone()).unwrap();
        let mut b = Surrogate::new(m).unwrap();
        let x = vec![valid; BLOCK];
        let bad = vec![bad; BLOCK];
        let mut ya = vec![0.0; BLOCK];
        let mut yb = vec![0.0; BLOCK];
        a.process_block(&x, &[], &mut ya).unwrap();
        b.process_block(&x, &[], &mut yb).unwrap();
        ya.fill(123.0);
        assert!(a.process_block(&bad, &[], &mut ya).is_err());
        assert!(ya.iter().all(|&v| v == 123.0));
        a.process_block(&x, &[], &mut ya).unwrap();
        b.process_block(&x, &[], &mut yb).unwrap();
        assert_eq!(ya, yb);
    }

    #[test]
    fn bad_blocks_do_not_advance_history_or_overwrite_output() {
        rejection_preserves_state(model(vec![0.3; BLOCK + 3]), 0.1, f32::NAN);
        rejection_preserves_state(model(vec![0.3; BLOCK + 3]), 0.1, f32::MAX); // FFT overflow
        let mut quadratic = model(vec![0.3; BLOCK + 3]);
        quadratic.powers = vec![2];
        rejection_preserves_state(quadratic, 0.1, 1e30); // feature overflow
        rejection_preserves_state(model(vec![1e20; BLOCK + 3]), 1e-20, 1e20); // convolution overflow
        let mut m = model(vec![1e20]);
        m.biases = vec![vec![f32::MAX]];
        rejection_preserves_state(m, 0.0, 1e16); // finite convolution, unrepresentable biased output
        let mut s = Surrogate::new(model(vec![1.0])).unwrap();
        let mut y = vec![123.0; BLOCK];
        assert!(s.process_block(&[], &[], &mut y).is_err());
        assert!(s
            .process_block(&vec![0.0; BLOCK], &[], &mut y[..BLOCK - 1])
            .is_err());
        assert!(s
            .process_block(&vec![0.0; BLOCK], &[f32::NAN], &mut y)
            .is_err());
        assert!(y.iter().all(|&v| v == 123.0));
        let mut m = model(vec![1.0]);
        m.widths = vec![1.0];
        m.centers = vec![vec![0.0]];
        let mut s = Surrogate::new(m).unwrap();
        assert!(s
            .process_block(&vec![0.0; BLOCK], &vec![f32::INFINITY; BLOCK], &mut y)
            .is_err());
        assert!(y.iter().all(|&v| v == 123.0));
    }

    #[test]
    fn reset_reproduces_zero_history_and_metadata_matches_layout() {
        let mut m = model((0..BLOCK * 3 + 1).map(|i| (i as f32 * 0.3).sin()).collect());
        m.labels = vec!["raw".into()];
        let mut s = Surrogate::new(m).unwrap();
        let metadata = s.metadata();
        assert_eq!(
            (
                metadata.rate,
                metadata.cylinders,
                metadata.controls,
                metadata.features,
                metadata.gates,
                metadata.outputs,
                metadata.taps,
                metadata.block_size
            ),
            (20_000, 1, 0, 1, 1, 1, BLOCK * 3 + 1, BLOCK)
        );
        assert_eq!(s.block_size(), Surrogate::BLOCK_SIZE);
        let x: Vec<_> = (0..BLOCK * 7).map(|i| (i as f32 * 0.05).cos()).collect();
        let mut first = vec![0.0; x.len()];
        for (input, output) in x.chunks_exact(BLOCK).zip(first.chunks_exact_mut(BLOCK)) {
            s.process_block(input, &[], output).unwrap();
        }
        s.reset();
        let mut y = vec![0.0; BLOCK];
        for (input, expected) in x.chunks_exact(BLOCK).zip(first.chunks_exact(BLOCK)) {
            s.process_block(input, &[], &mut y).unwrap();
            assert_eq!(y, expected);
        }
    }
}
