//! Turns the engine's exhaust signal (one value per simulation step) into audio: resampling,
//! jitter, air noise, the exhaust's impulse response and levelling (engine-sim's
//! synthesizer.cpp and its filters). Single-threaded: write input, then render.

use crate::rng::{Rng, RAND_MAX};
use std::collections::VecDeque;


const PI_F32: f32 = crate::units::PI as f32;

/// A ring buffer with engine-sim's quirks (write index and start, no overflow check).
struct Ring<T: Copy + Default> {
    buffer: Vec<T>,
    write_index: usize,
    start: usize,
}

impl<T: Copy + Default> Ring<T> {
    fn new(capacity: usize) -> Self {
        Ring { buffer: vec![T::default(); capacity], write_index: 0, start: 0 }
    }
    fn capacity(&self) -> usize {
        self.buffer.len()
    }
    fn write(&mut self, v: T) {
        self.buffer[self.write_index] = v;
        self.write_index += 1;
        if self.write_index >= self.capacity() {
            self.write_index = 0;
        }
    }
    fn read(&self, index: usize) -> T {
        let i = self.start + index;
        if i >= self.capacity() {
            self.buffer[i - self.capacity()]
        } else {
            self.buffer[i]
        }
    }
    fn read_into(&self, target: &mut [T]) {
        for (i, t) in target.iter_mut().enumerate() {
            *t = self.read(i);
        }
    }
    fn remove_beginning(&mut self, n: usize) {
        self.start += n;
        if self.start >= self.capacity() {
            self.start -= self.capacity();
        }
    }
    fn size(&self) -> usize {
        if self.write_index < self.start {
            self.write_index + (self.capacity() - self.start)
        } else {
            self.write_index - self.start
        }
    }
}

/// Fourth-order Butterworth low-pass, in f32 or f64 as engine-sim instantiates it.
trait Real: Copy + Default + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Div<Output = Self> + std::ops::Neg<Output = Self> {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn tan(self) -> Self;
}
impl Real for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn tan(self) -> Self {
        f32::tan(self)
    }
}
impl Real for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn tan(self) -> Self {
        f64::tan(self)
    }
}

struct Butterworth<T: Real> {
    y: Ring<T>,
    x: Ring<T>,
    a: [T; 5],
    f_4: T,
}

impl<T: Real> Butterworth<T> {
    fn new() -> Self {
        let mut y = Ring::new(4);
        let mut x = Ring::new(4);
        for _ in 0..4 {
            y.write(T::default());
            x.write(T::default());
        }
        Butterworth { y, x, a: [T::default(); 5], f_4: T::default() }
    }

    fn f(&mut self, sample: T) -> T {
        let y_prev = [self.y.read(3), self.y.read(2), self.y.read(1), self.y.read(0)];
        let x_prev = [self.x.read(3), self.x.read(2), self.x.read(1), self.x.read(0)];
        let four = T::from_f64(4.0);
        let six = T::from_f64(6.0);
        let n = self.f_4 / self.a[0]
            * (sample + four * x_prev[0] + six * x_prev[1] + four * x_prev[2] + x_prev[3]);
        let d = -self.a[1] * y_prev[0] - self.a[2] * y_prev[1] - self.a[3] * y_prev[2] - self.a[4] * y_prev[3];
        let y = n + d;
        self.x.remove_beginning(1);
        self.x.write(sample);
        self.y.remove_beginning(1);
        self.y.write(y);
        y
    }

    /// The coefficients, with C++'s mix of T and double arithmetic.
    fn set_cutoff_frequency(&mut self, f_c: T, sample_rate: T) {
        let f = (T::from_f64(crate::units::PI) * f_c / sample_rate).tan();
        let f_2 = f * f;
        let f_3 = f_2 * f;
        let f_4 = f_2 * f_2;
        let m = T::from_f64(-2.0 * (5.0 * crate::units::PI / 8.0).cos());
        let n = T::from_f64(-2.0 * (7.0 * crate::units::PI / 8.0).cos());
        let d = |v: T| v.to_f64();
        let a0 = T::from_f64(
            1.0 + d((m + n) * f) + (2.0 + d(n * m)) * d(f_2) + d((m + n) * f_3) + d(f_4),
        );
        let a0d = d(a0);
        let a1 = (-4.0 - 2.0 * d(n + m) * d(f) + 2.0 * d(m + n) * d(f_3) + 4.0 * d(f_4)) / a0d;
        let a2 = (6.0 - 2.0 * (2.0 + d(m * n)) * d(f_2) + 6.0 * d(f_4)) / a0d;
        let a3 = (-4.0 + 2.0 * d(m + n) * d(f) - 2.0 * d(m + n) * d(f_3) + 4.0 * d(f_4)) / a0d;
        let a4 = (1.0 - d((n + m) * f) + (2.0 + d(m * n)) * d(f_2) - d((m + n) * f_3) + d(f_4)) / a0d;
        self.a = [a0, T::from_f64(a1), T::from_f64(a2), T::from_f64(a3), T::from_f64(a4)];
        self.f_4 = f_4;
    }
}

struct LowPass {
    y: f32,
    rc: f32,
    dt: f32,
}

impl LowPass {
    fn f(&mut self, sample: f32) -> f32 {
        let alpha = self.dt / (self.rc + self.dt);
        self.y = alpha * sample + (1.0 - alpha) * self.y;
        self.y
    }
}

struct Derivative {
    previous: f32,
    dt: f32,
}

impl Derivative {
    fn f(&mut self, sample: f32) -> f32 {
        let temp = self.previous;
        self.previous = sample;
        (sample - temp) / self.dt
    }
}

struct Jitter {
    history: Vec<f32>,
    offset: usize,
    noise: Butterworth<f32>,
    scale: f32,
    rng: Rng,
}

impl Jitter {
    fn new(max_jitter: usize, cutoff: f32, audio_frequency: f32) -> Self {
        let mut noise = Butterworth::new();
        noise.set_cutoff_frequency(cutoff, audio_frequency);
        Jitter { history: vec![0.0; max_jitter], offset: 0, noise, scale: 0.0, rng: Rng::new(7) }
    }

    fn f(&mut self, sample: f32) -> f32 {
        let max = self.history.len();
        self.history[self.offset] = sample;
        self.offset += 1;
        if self.offset >= max {
            self.offset = 0;
        }
        let u = (self.rng.unit() as f32) * (max - 1) as f32;
        let s = self.noise.f(u * self.scale * 1.0);
        let top = (max - 1) as f32;
        let s_i_0 = clamp_f32(s.floor(), 0.0, top);
        let s_i_1 = clamp_f32(s.ceil(), 0.0, top);
        let s_frac = s - s_i_0;
        let i_0 = s_i_0 as usize + self.offset;
        let i_1 = s_i_1 as usize + self.offset;
        let v0 = self.history[if i_0 >= max { i_0 - max } else { i_0 }];
        let v1 = self.history[if i_1 >= max { i_1 - max } else { i_1 }];
        v1 * s_frac + v0 * (1.0 - s_frac)
    }
}

fn clamp_f32(x: f32, x0: f32, x1: f32) -> f32 {
    if x <= x0 {
        x0
    } else if x >= x1 {
        x1
    } else {
        x
    }
}

struct Convolution {
    shift: Vec<f32>,
    impulse: Vec<f32>,
    offset: usize,
}

impl Convolution {
    fn f(&mut self, sample: f32) -> f32 {
        let n = self.impulse.len();
        if n == 0 {
            return 0.0;
        }
        self.shift[self.offset] = sample;
        let mut result = 0.0f32;
        let split = n - self.offset;
        for i in 0..split {
            result += self.impulse[i] * self.shift[i + self.offset];
        }
        for i in split..n {
            result += self.impulse[i] * self.shift[i - split];
        }
        self.offset = (self.offset + n - 1) % n;
        result
    }
}

/// Block size of the partitioned convolution (the AudioWorklet's render quantum).
pub const BLOCK: usize = 128;

/// A real FFT of 2 * BLOCK points, as the convolution needs it: f32, split re/im.
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        Fft {
            n,
            cos: (0..n / 2).map(|i| (-2.0 * std::f64::consts::PI * i as f64 / n as f64).cos() as f32).collect(),
            sin: (0..n / 2).map(|i| (-2.0 * std::f64::consts::PI * i as f64 / n as f64).sin() as f32).collect(),
            rev: (0..n).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect(),
        }
    }

    /// In place; `inverse` conjugates the twiddles (unscaled).
    fn run(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
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
            len <<= 1;
        }
    }
}

/// Uniformly partitioned overlap-save convolution: a long impulse response at a cost per
/// sample that grows with its length only through one multiply-add per partition and bin.
pub(crate) struct Partitioned {
    fft: Fft,
    bins: usize,
    /// Spectra of the impulse response's partitions, re then im.
    h_re: Vec<Vec<f32>>,
    h_im: Vec<Vec<f32>>,
    /// Responses to blend between (spectra as h_re/h_im; process_blended).
    sets: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    blend_tmp: Vec<f32>,
    /// Spectra of the last inputs (frequency-domain delay line).
    x_re: Vec<Vec<f32>>,
    x_im: Vec<Vec<f32>>,
    position: usize,
    previous: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    acc_re: Vec<f32>,
    acc_im: Vec<f32>,
}

impl Partitioned {
    pub(crate) fn new(impulse: &[f32]) -> Self {
        Self::blending(&[impulse.to_vec()])
    }

    /// Spectra of an impulse response's partitions.
    fn spectra(fft: &Fft, impulse: &[f32], parts: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n = 2 * BLOCK;
        let bins = BLOCK + 1;
        let (mut h_re, mut h_im) = (Vec::new(), Vec::new());
        for p in 0..parts {
            let mut re = vec![0.0f32; n];
            let mut im = vec![0.0f32; n];
            for (i, &v) in impulse.iter().skip(p * BLOCK).take(BLOCK).enumerate() {
                re[i] = v;
            }
            fft.run(&mut re, &mut im, false);
            h_re.push(re[..bins].to_vec());
            h_im.push(im[..bins].to_vec());
        }
        (h_re, h_im)
    }

    /// Convolution with a blend of `impulses` (see blend), starting at the first.
    pub(crate) fn blending(impulses: &[Vec<f32>]) -> Self {
        let n = 2 * BLOCK;
        let fft = Fft::new(n);
        let bins = BLOCK + 1;
        let parts = impulses.iter().map(|i| i.len()).max().unwrap_or(0).div_ceil(BLOCK).max(1);
        let sets: Vec<_> = impulses.iter().map(|i| Self::spectra(&fft, i, parts)).collect();
        let (h_re, h_im) = sets[0].clone();
        Partitioned {
            sets: if sets.len() > 1 { sets } else { Vec::new() },
            blend_tmp: vec![0.0; BLOCK],
            fft,
            bins,
            x_re: vec![vec![0.0; bins]; parts],
            x_im: vec![vec![0.0; bins]; parts],
            h_re,
            h_im,
            position: 0,
            previous: vec![0.0; BLOCK],
            re: vec![0.0; n],
            im: vec![0.0; n],
            acc_re: vec![0.0; bins],
            acc_im: vec![0.0; bins],
        }
    }

    /// Convolves one block of BLOCK samples (with the first response, or the only one).
    pub(crate) fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.push(input);
        let (h_re, h_im) = (std::mem::take(&mut self.h_re), std::mem::take(&mut self.h_im));
        self.convolve(&h_re, &h_im, output);
        (self.h_re, self.h_im) = (h_re, h_im);
        self.position = (self.position + 1) % self.h_re.len();
    }

    /// Convolves one block with the responses blended `x0` of the way through the sets at
    /// its start and `x1` at its end (0: the first, 1: the second, 1.5: half way to the
    /// third...): the outputs of the two neighbouring responses, crossfaded sample by
    /// sample. Changing a convolution's response under its history instead would click.
    /// Returns where the blend got to: `x1`, or the boundary of the pair `x0` is in if `x1` is
    /// beyond it (the next block should go on from there).
    pub(crate) fn process_blended(&mut self, input: &[f32], output: &mut [f32], x0: f64, x1: f64) -> f64 {
        if self.sets.is_empty() {
            self.process(input, output);
            return x1;
        }
        let top = (self.sets.len() - 1) as f64;
        let (x0, x1) = (x0.clamp(0.0, top), x1.clamp(0.0, top));
        self.push(input);
        // Between the pair x0 is in (going down from a boundary, the one below it); a block
        // that would cross into the next pair stops at their boundary (the next block goes
        // on from there).
        let i = if x1 >= x0 { x0.floor() } else { x0.ceil() - 1.0 }.clamp(0.0, top - 1.0) as usize;
        let f0 = (x0 - i as f64) as f32;
        let f1 = (x1 - i as f64).clamp(0.0, 1.0) as f32;
        let sets = std::mem::take(&mut self.sets);
        let mut b = std::mem::take(&mut self.blend_tmp);
        self.convolve(&sets[i].0, &sets[i].1, output);
        if f0 != 0.0 || f1 != 0.0 {
            self.convolve(&sets[i + 1].0, &sets[i + 1].1, &mut b);
            for (k, (o, &y)) in output.iter_mut().zip(&b).enumerate() {
                let f = f0 + (f1 - f0) * (k as f32 + 1.0) / BLOCK as f32;
                *o += f * (y - *o);
            }
        }
        self.blend_tmp = b;
        self.sets = sets;
        self.position = (self.position + 1) % self.x_re.len();
        i as f64 + f1 as f64
    }

    /// Adds a block of input to the frequency-domain delay line.
    fn push(&mut self, input: &[f32]) {
        self.re[..BLOCK].copy_from_slice(&self.previous);
        self.re[BLOCK..].copy_from_slice(input);
        self.previous.copy_from_slice(input);
        self.im.iter_mut().for_each(|v| *v = 0.0);
        self.fft.run(&mut self.re, &mut self.im, false);
        let slot = self.position;
        self.x_re[slot].copy_from_slice(&self.re[..self.bins]);
        self.x_im[slot].copy_from_slice(&self.im[..self.bins]);
    }

    /// The delay line convolved with the response `h` (partition spectra): this block.
    fn convolve(&mut self, h_re: &[Vec<f32>], h_im: &[Vec<f32>], output: &mut [f32]) {
        let n = 2 * BLOCK;
        let parts = self.x_re.len();
        let slot = self.position;
        self.acc_re.iter_mut().for_each(|v| *v = 0.0);
        self.acc_im.iter_mut().for_each(|v| *v = 0.0);
        for p in 0..parts {
            let x = (slot + parts - p) % parts;
            let (xr, xi) = (&self.x_re[x], &self.x_im[x]);
            let (hr, hi) = (&h_re[p], &h_im[p]);
            for k in 0..self.bins {
                self.acc_re[k] += xr[k] * hr[k] - xi[k] * hi[k];
                self.acc_im[k] += xr[k] * hi[k] + xi[k] * hr[k];
            }
        }
        // Back to the time domain: the Hermitian spectrum, the second half is this block.
        for k in 0..self.bins {
            self.re[k] = self.acc_re[k];
            self.im[k] = self.acc_im[k];
        }
        for k in 1..BLOCK {
            self.re[n - k] = self.acc_re[k];
            self.im[n - k] = -self.acc_im[k];
        }
        self.fft.run(&mut self.re, &mut self.im, true);
        let scale = 1.0 / n as f32;
        for (o, v) in output.iter_mut().zip(&self.re[BLOCK..]) {
            *o = v * scale;
        }
    }
}

/// Second-order sections of a Butterworth high-pass (bilinear transform).
pub(crate) struct HighPass {
    sections: Vec<([f64; 5], [f64; 4])>,
}

impl HighPass {
    /// Of `order` (even) at `cutoff` Hz.
    pub(crate) fn new(order: usize, cutoff: f64, sample_rate: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let sections = (0..order / 2)
            .map(|k| {
                let q = 1.0 / (2.0 * (std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64).sin());
                let alpha = w.sin() / (2.0 * q);
                let a0 = 1.0 + alpha;
                let c = w.cos();
                let b = [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0, -2.0 * c / a0, (1.0 - alpha) / a0];
                (b, [0.0; 4])
            })
            .collect();
        HighPass { sections }
    }

    pub(crate) fn f(&mut self, x: f32) -> f32 {
        let mut v = x as f64;
        for (c, s) in &mut self.sections {
            // s: x[n-1], x[n-2], y[n-1], y[n-2]
            let mut y = c[0] * v + c[1] * s[0] + c[2] * s[1] - c[3] * s[2] - c[4] * s[3];
            // A decayed state would linger as subnormals (slow on some hardware).
            if y.abs() < 1e-30 {
                y = 0.0;
            }
            *s = [v, s[0], y, s[2]];
            v = y;
        }
        v as f32
    }
}

struct Leveling {
    peak: f32,
    attenuation: f32,
    max_level: f32,
    min_level: f32,
    target: f32,
}

impl Leveling {
    fn f(&mut self, sample: f32) -> f32 {
        self.peak *= 0.999f32;
        if sample.abs() > self.peak {
            self.peak = sample.abs();
        }
        if self.peak == 0.0 {
            return 0.0;
        }
        let raw = self.target / self.peak;
        let mut attenuation = raw;
        if attenuation < self.min_level {
            attenuation = self.min_level;
        } else if attenuation > self.max_level {
            attenuation = self.max_level;
        }
        self.attenuation = (0.9 * self.attenuation as f64 + 0.1 * attenuation as f64) as f32;
        sample * self.attenuation
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AudioParameters {
    pub volume: f32,
    pub convolution: f32,
    pub df_f_mix: f32,
    pub input_sample_noise: f32,
    pub input_sample_noise_frequency_cutoff: f32,
    pub air_noise: f32,
    pub air_noise_frequency_cutoff: f32,
    pub leveler_target: f32,
    pub leveler_max_gain: f32,
    pub leveler_min_gain: f32,
}

impl Default for AudioParameters {
    fn default() -> Self {
        AudioParameters {
            volume: 1.0,
            convolution: 1.0,
            df_f_mix: 0.01,
            input_sample_noise: 0.5,
            input_sample_noise_frequency_cutoff: 10000.0,
            air_noise: 1.0,
            air_noise_frequency_cutoff: 2000.0,
            leveler_target: 30000.0,
            leveler_max_gain: 1.9,
            leveler_min_gain: 0.00001,
        }
    }
}

struct Channel {
    /// When set, the channel's impulse response is applied block by block (see render).
    partitioned: Option<Partitioned>,
    block_in: Vec<f32>,
    data: Ring<f32>,
    transfer: Vec<f32>,
    last_input_sample: f64,
    convolution: Convolution,
    derivative: Derivative,
    jitter: Jitter,
    air_noise_low_pass: Butterworth<f32>,
    input_dc: LowPass,
    antialiasing: Butterworth<f64>,
}

pub struct Synthesizer {
    channels: Vec<Channel>,
    /// With responses by flow (set_flow_responses): the flows (kg/s) they are for, and the
    /// last channel then carries the flow rather than sound.
    flow_levels: Vec<f64>,
    flow_sum: f64,
    /// Where the flow was among flow_levels at the end of the last block.
    flow_position: f64,
    /// Below hearing (and a car's speakers): the exhaust's flow changes radiate infrasound
    /// that would only take the leveler's headroom.
    hearing: HighPass,
    antialiasing: Butterworth<f32>,
    leveling: Leveling,
    pub params: AudioParameters,
    input_buffer_size: usize,
    input_write_offset: f64,
    last_input_sample_offset: f64,
    audio: Ring<i16>,
    /// Block mode: rendered samples waiting to be read (starts a block of silence ahead).
    block_out: VecDeque<f32>,
    block_mix: Vec<f32>,
    block_tmp: Vec<f32>,
    input_sample_rate: f32,
    audio_sample_rate: f32,
    rng: Rng,
}

impl Synthesizer {
    pub fn new(channels: usize, input_sample_rate: f32, audio_sample_rate: f32, params: AudioParameters) -> Self {
        let input_buffer_size = 44100;
        let audio_buffer_size = 44100;
        let channels = (0..channels)
            .map(|_| {
                let mut air_noise_low_pass = Butterworth::new();
                air_noise_low_pass.set_cutoff_frequency(params.air_noise_frequency_cutoff, audio_sample_rate);
                let mut input_dc = LowPass { y: 0.0, rc: 0.0, dt: 1.0 / 44100.0f32 };
                input_dc.rc = 1.0f32 / (10.0f32 * 2.0f32 * PI_F32);
                input_dc.dt = 1.0 / audio_sample_rate;
                let mut antialiasing = Butterworth::new();
                antialiasing.set_cutoff_frequency(1900.0f32 as f64, audio_sample_rate as f64);
                Channel {
                    partitioned: None,
                    block_in: Vec::with_capacity(BLOCK),
                    data: Ring::new(input_buffer_size),
                    transfer: vec![0.0; input_buffer_size],
                    last_input_sample: 0.0,
                    convolution: Convolution { shift: Vec::new(), impulse: Vec::new(), offset: 0 },
                    derivative: Derivative { previous: 0.0, dt: 1.0 / audio_sample_rate },
                    jitter: Jitter::new(10, params.input_sample_noise_frequency_cutoff, audio_sample_rate),
                    air_noise_low_pass,
                    input_dc,
                    antialiasing,
                }
            })
            .collect();
        let mut antialiasing = Butterworth::new();
        antialiasing.set_cutoff_frequency(audio_sample_rate * 0.45f32, audio_sample_rate);
        Synthesizer {
            channels,
            flow_levels: Vec::new(),
            flow_sum: 0.0,
            flow_position: 0.0,
            hearing: HighPass::new(4, 20.0, audio_sample_rate as f64),
            antialiasing,
            leveling: Leveling {
                peak: 30000.0,
                attenuation: 1.0,
                max_level: params.leveler_max_gain,
                min_level: params.leveler_min_gain,
                target: params.leveler_target,
            },
            params,
            input_buffer_size,
            input_write_offset: 0.0,
            last_input_sample_offset: 0.0,
            audio: Ring::new(audio_buffer_size),
            block_out: std::iter::repeat_n(0.0, BLOCK).collect(),
            block_mix: vec![0.0; BLOCK],
            block_tmp: vec![0.0; BLOCK],
            input_sample_rate,
            audio_sample_rate,
            rng: Rng::new(1),
        }
    }

    /// The exhaust's impulse response for channel `index` (16-bit samples).
    pub fn set_impulse_response(&mut self, index: usize, samples: &[i16], volume: f32) {
        let mut clipped = 0;
        for (i, s) in samples.iter().enumerate() {
            if (*s as i32).abs() > 100 {
                clipped = i + 1;
            }
        }
        let count = clipped.min(10000);
        let c = &mut self.channels[index].convolution;
        c.shift = vec![0.0; count];
        c.impulse = samples[..count].iter().map(|&s| volume * s as f32 / i16::MAX as f32).collect();
        c.offset = 0;
    }

    /// An impulse response at the audio rate for channel `index`, applied by partitioned
    /// convolution (for long ones, as acoustics.rs computes). The whole synthesizer then
    /// renders in blocks of BLOCK samples, BLOCK samples late.
    pub fn set_impulse_response_f32(&mut self, index: usize, impulse: &[f32]) {
        self.channels[index].partitioned = Some(Partitioned::new(impulse));
    }

    /// Impulse responses at the audio rate for channel `index` at each of `levels` of the
    /// exhaust's mass flow (kg/s, ascending), blended by the flow the last channel carries.
    pub fn set_flow_responses(&mut self, index: usize, levels: &[f64], impulses: &[Vec<f32>]) {
        assert_eq!(levels.len(), impulses.len());
        self.flow_levels = levels.to_vec();
        self.channels[index].partitioned = Some(Partitioned::blending(impulses));
    }

    /// Channels that carry sound (all but the flow's, if there is one).
    fn sound_channels(&self) -> usize {
        self.channels.len() - usize::from(!self.flow_levels.is_empty())
    }

    /// Where `flow` falls among flow_levels, as Partitioned::blend takes it.
    fn flow_position(&self, flow: f64) -> f64 {
        let l = &self.flow_levels;
        if flow <= l[0] {
            return 0.0;
        }
        for i in 1..l.len() {
            if flow <= l[i] {
                return (i - 1) as f64 + (flow - l[i - 1]) / (l[i] - l[i - 1]);
            }
        }
        (l.len() - 1) as f64
    }

    pub fn set_input_sample_rate(&mut self, rate: f32) {
        self.input_sample_rate = rate;
    }

    /// One simulation step's value per channel, resampled to the audio rate.
    pub fn write_input(&mut self, data: &[f64]) {
        let size = self.input_buffer_size as f64;
        self.input_write_offset += self.audio_sample_rate as f64 / self.input_sample_rate as f64;
        if self.input_write_offset >= size {
            self.input_write_offset -= size;
        }
        let distance_fn = |s1: f64, s0: f64| if s1 < s0 { size - s0 + s1 } else { s1 - s0 };
        for (ch, &value) in self.channels.iter_mut().zip(data) {
            let last = ch.last_input_sample;
            let base_index = ch.data.write_index as f64;
            let distance = distance_fn(self.input_write_offset, self.last_input_sample_offset);
            let mut s = distance_fn(base_index, self.last_input_sample_offset);
            while s <= distance {
                if s >= size {
                    s -= size;
                }
                let f = s / distance;
                let sample = last * (1.0 - f) + value * f;
                ch.data.write(ch.antialiasing.f(sample as f32 as f64) as f32);
                s += 1.0;
            }
            ch.last_input_sample = value;
        }
        self.last_input_sample_offset = self.input_write_offset;
    }

    /// Input samples (at the audio rate) not yet rendered.
    pub fn input_available(&self) -> usize {
        self.channels.first().map_or(0, |c| c.data.size())
    }

    /// Renders buffered input into the output buffer (engine-sim's renderAudio, without the
    /// thread); returns how many samples it rendered.
    pub fn render_pending(&mut self) -> usize {
        if self.channels.is_empty() {
            return 0;
        }
        let space = self.audio.capacity().saturating_sub(self.audio.size());
        let n = space.min(self.channels[0].data.size().min(self.input_buffer_size));
        if n == 0 {
            return 0;
        }
        self.leveling.min_level = self.params.leveler_min_gain;
        self.leveling.max_level = self.params.leveler_max_gain;
        for ch in &mut self.channels {
            let (data, transfer) = (&ch.data, &mut ch.transfer[..n]);
            data.read_into(transfer);
            ch.data.remove_beginning(n);
            ch.air_noise_low_pass
                .set_cutoff_frequency(self.params.air_noise_frequency_cutoff, self.audio_sample_rate);
            ch.jitter.scale = self.params.input_sample_noise;
        }
        for i in 0..n {
            let v = self.render_sample(i);
            self.audio.write(v);
        }
        n
    }

    /// Renders up to `out.len()` samples straight from buffered input, scaled to -1..1;
    /// returns how many there was input for (the rest is left untouched).
    pub fn render(&mut self, out: &mut [f32]) -> usize {
        if self.channels.is_empty() {
            return 0;
        }
        if self.channels.iter().any(|c| c.partitioned.is_some()) {
            return self.render_blocks(out);
        }
        let n = out.len().min(self.channels[0].data.size());
        if n == 0 {
            return 0;
        }
        self.leveling.min_level = self.params.leveler_min_gain;
        self.leveling.max_level = self.params.leveler_max_gain;
        for ch in &mut self.channels {
            let (data, transfer) = (&ch.data, &mut ch.transfer[..n]);
            data.read_into(transfer);
            ch.data.remove_beginning(n);
            ch.air_noise_low_pass
                .set_cutoff_frequency(self.params.air_noise_frequency_cutoff, self.audio_sample_rate);
            ch.jitter.scale = self.params.input_sample_noise;
        }
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o = self.render_sample(i) as f32 / 32768.0;
        }
        n
    }

    /// render, with the impulse responses applied a block at a time.
    fn render_blocks(&mut self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.channels[0].data.size());
        if n > 0 {
            self.leveling.min_level = self.params.leveler_min_gain;
            self.leveling.max_level = self.params.leveler_max_gain;
            for ch in &mut self.channels {
                let (data, transfer) = (&ch.data, &mut ch.transfer[..n]);
                data.read_into(transfer);
                ch.data.remove_beginning(n);
                ch.air_noise_low_pass
                    .set_cutoff_frequency(self.params.air_noise_frequency_cutoff, self.audio_sample_rate);
                ch.jitter.scale = self.params.input_sample_noise;
            }
        }
        let p = self.params;
        let sound = self.sound_channels();
        for i in 0..n {
            for c in 0..sound {
                let v_in = self.channel_input(c, i);
                self.channels[c].block_in.push(v_in);
            }
            if sound < self.channels.len() {
                self.flow_sum += self.channels[sound].transfer[i] as f64;
            }
            if self.channels[0].block_in.len() == BLOCK {
                let x0 = self.flow_position;
                if sound < self.channels.len() {
                    self.flow_position = self.flow_position(self.flow_sum / BLOCK as f64);
                    self.flow_sum = 0.0;
                }
                let x1 = self.flow_position;
                self.block_mix.iter_mut().for_each(|v| *v = 0.0);
                for ch in &mut self.channels[..sound] {
                    match &mut ch.partitioned {
                        Some(conv) => {
                            self.flow_position = conv.process_blended(&ch.block_in, &mut self.block_tmp, x0, x1);
                        }
                        None => self.block_tmp.copy_from_slice(&ch.block_in),
                    }
                    for (m, (&y, &x)) in self.block_mix.iter_mut().zip(self.block_tmp.iter().zip(&ch.block_in)) {
                        *m += p.convolution * y + (1.0 - p.convolution) * x;
                    }
                    ch.block_in.clear();
                }
                for k in 0..BLOCK {
                    let signal = self.hearing.f(self.antialiasing.f(self.block_mix[k]));
                    self.leveling.target = p.leveler_target;
                    let v = self.leveling.f(signal) * p.volume;
                    self.block_out.push_back(v.clamp(i16::MIN as f32, i16::MAX as f32) / 32768.0);
                }
            }
        }
        for o in out[..n].iter_mut() {
            *o = self.block_out.pop_front().unwrap_or(0.0);
        }
        n
    }

    /// One channel's signal ahead of its impulse response (render_sample's first half).
    fn channel_input(&mut self, c: usize, input: usize) -> f32 {
        let p = self.params;
        let _r_0 = self.rng.next();
        let ch = &mut self.channels[c];
        let f_in = ch.jitter.f(ch.transfer[input]);
        let f_dc = ch.input_dc.f(f_in);
        let f = f_in - f_dc;
        let f_p = ch.derivative.f(f_in);
        let noise = (2.0 * (self.rng.next() as f64 / RAND_MAX as f64) - 1.0) as f32;
        let r = self.channels[0].air_noise_low_pass.f(noise);
        let r_mixed = p.air_noise * r + (1.0 - p.air_noise);
        let v_in = f_p * p.df_f_mix + f * r_mixed * (1.0 - p.df_f_mix);
        if v_in.is_subnormal() {
            0.0
        } else {
            v_in
        }
    }

    /// Drops buffered input beyond `keep` samples (when the simulation ran ahead).
    pub fn drop_input(&mut self, keep: usize) {
        for ch in &mut self.channels {
            let size = ch.data.size();
            if size > keep {
                ch.data.remove_beginning(size - keep);
            }
        }
    }

    fn render_sample(&mut self, input: usize) -> i16 {
        let p = self.params;
        let mut signal = 0.0f32;
        for c in 0..self.sound_channels() {
            let _r_0 = (2.0 * (self.rng.next() as f64 / RAND_MAX as f64) - 1.0) as f32;
            let ch = &mut self.channels[c];
            let jittered = ch.jitter.f(ch.transfer[input]);
            let f_in = jittered;
            let f_dc = ch.input_dc.f(f_in);
            let f = f_in - f_dc;
            let f_p = ch.derivative.f(f_in);
            let noise = (2.0 * (self.rng.next() as f64 / RAND_MAX as f64) - 1.0) as f32;
            // engine-sim filters the noise of every channel with the first channel's filter.
            let r = self.channels[0].air_noise_low_pass.f(noise);
            let r_mixed = p.air_noise * r + (1.0 - p.air_noise);
            let mut v_in = f_p * p.df_f_mix + f * r_mixed * (1.0 - p.df_f_mix);
            if v_in.is_subnormal() {
                v_in = 0.0;
            }
            let ch = &mut self.channels[c];
            let v = p.convolution * ch.convolution.f(v_in) + (1.0 - p.convolution) * v_in;
            signal += v;
        }
        signal = self.antialiasing.f(signal);
        self.leveling.target = p.leveler_target;
        let v_leveled = self.leveling.f(signal) * p.volume;
        let r = v_leveled.round() as i64;
        r.clamp(i16::MIN as i64, i16::MAX as i64) as i16
    }

    /// Rendered samples waiting to be read.
    pub fn output_available(&self) -> usize {
        self.audio.size()
    }

    /// Reads up to `target.len()` rendered samples; pads with silence; returns how many it had.
    pub fn read_output(&mut self, target: &mut [i16]) -> usize {
        let available = self.audio.size();
        let n = available.min(target.len());
        self.audio.read_into(&mut target[..n]);
        self.audio.remove_beginning(n);
        for t in &mut target[n..] {
            *t = 0;
        }
        n
    }
}

/// Reads the first channel of a 16-bit PCM WAV file.
pub fn read_wav(data: &[u8]) -> Option<Vec<i16>> {
    if data.len() < 12 || &data[0..4] != b"RIFF" {
        return None;
    }
    let (mut channels, mut bits) = (1usize, 16u16);
    let mut p = 12;
    while p + 8 <= data.len() {
        let id = &data[p..p + 4];
        let size = u32::from_le_bytes(data[p + 4..p + 8].try_into().ok()?) as usize;
        let body = &data[p + 8..];
        if id == b"fmt " {
            channels = u16::from_le_bytes(body[2..4].try_into().ok()?) as usize;
            bits = u16::from_le_bytes(body[14..16].try_into().ok()?);
        } else if id == b"data" {
            if bits != 16 {
                return None;
            }
            let frames = size.min(body.len()) / (2 * channels);
            return Some(
                (0..frames)
                    .map(|i| i16::from_le_bytes([body[i * 2 * channels], body[i * 2 * channels + 1]]))
                    .collect(),
            );
        }
        p += 8 + size + (size & 1);
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioned_convolution_matches_direct() {
        let impulse: Vec<f32> = (0..1000).map(|i| ((i as f32 * 0.37).sin() * (-(i as f32) / 300.0).exp())).collect();
        let input: Vec<f32> = (0..BLOCK * 20).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
        let mut conv = Partitioned::new(&impulse);
        let mut fast = vec![0.0; input.len()];
        for (i, o) in input.chunks(BLOCK).zip(fast.chunks_mut(BLOCK)) {
            conv.process(i, o);
        }
        for n in 0..input.len() {
            let direct: f32 = (0..impulse.len().min(n + 1)).map(|k| impulse[k] * input[n - k]).sum();
            assert!((direct - fast[n]).abs() < 1e-3, "sample {n}: {direct} vs {}", fast[n]);
        }
    }

    #[test]
    fn blending_responses_is_convolving_with_their_blend() {
        // Responses are blended linearly: half way between two is convolution with their
        // mean.
        let a: Vec<f32> = (0..700).map(|i| ((i as f32 * 0.21).sin() * (-(i as f32) / 200.0).exp())).collect();
        let b: Vec<f32> = (0..900).map(|i| ((i as f32 * 0.05).cos() * (-(i as f32) / 400.0).exp())).collect();
        let mean: Vec<f32> = (0..900).map(|i| 0.5 * (a.get(i).unwrap_or(&0.0) + b[i])).collect();
        let input: Vec<f32> = (0..BLOCK * 20).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
        let (mut blended, mut direct) = (Partitioned::blending(&[a, b]), Partitioned::new(&mean));
        let (mut x, mut y) = (vec![0.0; BLOCK], vec![0.0; BLOCK]);
        for block in input.chunks(BLOCK) {
            blended.process_blended(block, &mut x, 0.5, 0.5);
            direct.process(block, &mut y);
            for (p, q) in x.iter().zip(&y) {
                assert!((p - q).abs() < 1e-3, "{p} vs {q}");
            }
        }
    }
}
