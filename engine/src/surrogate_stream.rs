//! Streaming adapter for auditioning a learned exhaust in the game. This is audio I/O,
//! not `voice.rs`: no acoustic geometry, steepener, noise or level fit. Inputs are the
//! game's raw valve mass flows plus five explicitly ordered controls. The engine's gas
//! and torque models do not change when this stream is selected.
//!
//! Input controls are linearly resampled to the model rate. PCM pressure is reconstructed
//! at the audio device rate with a 32-tap Blackman-windowed sinc, bounded look-ahead of
//! 16/model_rate seconds (<1 ms at 20 kHz), plus one input frame and one model block.
//! These are buffering delays, not a phase realignment learned from the target. There is
//! no extrapolation of missing input, per-clip normalization, or borrowed reference state.
use crate::surrogate::Surrogate;
use std::collections::VecDeque;
use std::f64::consts::PI;

pub const CONTROLS: usize = 5;
const LEFT: isize = 15;
const RIGHT: usize = 16;
const TAPS: usize = 32;
const PHASES: usize = 1024;

pub struct SurrogateStream {
    model: Surrogate,
    input: Vec<f32>,
    input_position: f64,
    input_step: f64,
    input_channels: usize,
    flow: Vec<f32>,
    controls: Vec<f32>,
    block_output: Vec<f32>,
    pcm: VecDeque<f32>,
    output_position: f64,
    output_step: f64,
    sinc: Vec<[f32; TAPS]>,
    full_scale: f64,
    failed: bool,
}

impl SurrogateStream {
    pub fn new(json: &str, input_rate: f64, output_rate: f64, full_scale: f64) -> Result<Self, String> {
        if !input_rate.is_finite() || !(100.0..=1_000_000.0).contains(&input_rate)
            || !output_rate.is_finite() || !(8000.0..=192000.0).contains(&output_rate)
            || !full_scale.is_finite() || full_scale <= 0.0
        {
            return Err("invalid surrogate stream rates/full scale".into());
        }
        let model = Surrogate::from_json(json)?;
        let m = model.metadata();
        if !(100..=192000).contains(&m.rate) || m.controls != CONTROLS || (!m.labels.is_empty() && m.labels.iter().any(|s| !s.starts_with("pressure"))) {
            return Err("game surrogate needs five controls and pressure outputs only".into());
        }
        let (n, no, block, rate) = (m.cylinders, m.outputs, m.block_size, m.rate as f64);
        let input_channels = n + CONTROLS;
        let cutoff = (output_rate / rate).min(1.0);
        let sinc = (0..PHASES).map(|phase| {
            let fraction = phase as f64 / PHASES as f64;
            let mut h = [0.0f32; TAPS];
            let mut sum = 0.0;
            for (k, v) in h.iter_mut().enumerate() {
                let d = k as f64 - LEFT as f64 - fraction;
                let window = if d.abs() < RIGHT as f64 {
                    0.42 + 0.5*(PI*d/RIGHT as f64).cos() + 0.08*(2.0*PI*d/RIGHT as f64).cos()
                } else { 0.0 };
                let x = PI*d*cutoff;
                let w = cutoff * if x.abs() < 1e-12 { 1.0 } else { x.sin()/x } * window;
                *v = w as f32;
                sum += w;
            }
            for v in &mut h { *v /= sum as f32; }
            h
        }).collect();
        Ok(Self {
            model,
            input: Vec::with_capacity(input_channels * 4096),
            input_position: 0.0,
            input_step: input_rate/rate,
            input_channels,
            flow: vec![0.0; block*n],
            controls: vec![0.0; block*CONTROLS],
            block_output: vec![0.0; block*no],
            pcm: VecDeque::with_capacity(block*4+TAPS),
            output_position: 0.0,
            output_step: rate/output_rate,
            sinc,
            full_scale,
            failed: false,
        })
    }

    /// A cached audio node reused for a newly created car must not retain the previous
    /// engine's pressure tails or interpolation phase.
    pub fn reset(&mut self) {
        self.model.reset();
        self.input.clear();
        self.pcm.clear();
        self.input_position = 0.0;
        self.output_position = 0.0;
        self.failed = false;
    }

    pub fn failed(&self) -> bool { self.failed }
    pub fn channels(&self) -> usize { self.input_channels }

    pub fn set_input_rate(&mut self, rate: f64) {
        if !rate.is_finite() || !(100.0..=1_000_000.0).contains(&rate) {
            self.failed = true;
            return;
        }
        self.input_step = rate/self.model.metadata().rate as f64;
    }

    /// Planar engine-step samples. Invalid dimensions/nonfinite input cannot partially
    /// append a packet. Errors latch until the host chooses its fallback backend.
    pub fn write_planar(&mut self, samples: &[f32], steps: usize, channels: usize) {
        if channels != self.input_channels || steps.checked_mul(channels) != Some(samples.len())
            || samples.iter().any(|v| !v.is_finite()) {
            self.failed = true;
            return;
        }
        if self.failed { return; }
        for t in 0..steps {
            for c in 0..channels { self.input.push(samples[c*steps+t]); }
        }
    }

    fn source_available(&self) -> usize {
        let last = (self.input.len()/self.input_channels) as f64 - 1.0;
        if last <= self.input_position { 0 } else { ((last-self.input_position)/self.input_step).ceil() as usize }
    }

    pub fn input_available(&self) -> usize {
        if self.failed { return 0; }
        let block = self.model.block_size();
        let count = self.pcm.len() + self.source_available()/block*block;
        let last = count as f64 - RIGHT as f64;
        if last <= self.output_position { 0 } else { ((last-self.output_position)/self.output_step).ceil() as usize }
    }

    fn decode_block(&mut self) -> bool {
        let (n, block, no) = (self.model.metadata().cylinders, self.model.block_size(), self.model.metadata().outputs);
        if self.source_available() < block { return false; }
        for t in 0..block {
            let i = self.input_position.floor() as usize;
            let fraction = (self.input_position-i as f64) as f32;
            for c in 0..self.input_channels {
                let a = self.input[i*self.input_channels+c];
                let b = self.input[(i+1)*self.input_channels+c];
                let value = a + fraction*(b-a);
                if c < n { self.flow[t*n+c] = value; }
                else { self.controls[t*CONTROLS+c-n] = value; }
            }
            self.input_position += self.input_step;
        }
        // Retain the next interpolation's left sample; no frame-boundary reset of histories.
        let behind = (self.input_position.floor() as usize).min(self.input.len()/self.input_channels-1);
        self.input.drain(..behind*self.input_channels);
        self.input_position -= behind as f64;
        if self.model.process_block(&self.flow, &self.controls, &mut self.block_output).is_err() {
            self.failed = true;
            return false;
        }
        for frame in self.block_output.chunks_exact(no) {
            let sample = frame.iter().map(|x| *x as f64).sum::<f64>()/self.full_scale;
            if !sample.is_finite() || !(sample as f32).is_finite() {
                self.failed = true;
                return false;
            }
            self.pcm.push_back(sample as f32);
        }
        true
    }

    /// Samples available for the device, zero initial history and continuous model state.
    /// The caller owns silence during underruns; no invented engine input is synthesized.
    pub fn render(&mut self, out: &mut [f32]) -> usize {
        let count = out.len().min(self.input_available());
        let mut written = 0;
        for sample in out.iter_mut().take(count) {
            let mut center = self.output_position.floor() as usize;
            // Nearest phase, including the next integer center. Flooring an almost-integer
            // phase would make packet-dependent floating-point roundoff choose opposite
            // ends of the phase table and introduce a one-phase-step discontinuity.
            let mut phase = ((self.output_position-center as f64)*PHASES as f64).round() as usize;
            if phase == PHASES { center += 1; phase = 0; }
            let right = if phase == 0 { RIGHT-1 } else { RIGHT };
            while self.pcm.len() <= center+right {
                if !self.decode_block() { return written; }
            }
            let mut value = 0.0;
            for (k, h) in self.sinc[phase].iter().enumerate() {
                let i = center as isize + k as isize - LEFT;
                if i >= 0 && *h != 0.0 { value += self.pcm[i as usize]*h; }
            }
            *sample = value;
            self.output_position += self.output_step;
            written += 1;
        }
        let behind = (self.output_position.floor() as usize).saturating_sub(LEFT as usize);
        self.pcm.drain(..behind.min(self.pcm.len()));
        self.output_position -= behind as f64;
        written
    }

    /// On a stall, advance the model through the queued history before discarding audio.
    /// Merely skipping source frames would leave resonant/filter state at the wrong time.
    pub fn drop_input(&mut self, keep: usize) {
        let mut scratch = [0.0; 256];
        while self.input_available() > keep {
            let n = (self.input_available()-keep).min(scratch.len());
            if self.render(&mut scratch[..n]) == 0 { break; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(h: Vec<f32>) -> String {
        json!({"rate":20000,"cylinders":1,"flowScale":1.,"powers":[1],"centers":[[0.,0.,0.,0.,0.]],
               "widths":[1.,1.,1.,1.,1.],"kernels":[[[h]]],"labels":["pressure0"]}).to_string()
    }
    fn stream() -> SurrogateStream { SurrogateStream::new(&model(vec![1.]),20000.,48000.,2.).unwrap() }
    fn write(s: &mut SurrogateStream, q: &[f32]) {
        let mut p = vec![0.;q.len()*6]; p[..q.len()].copy_from_slice(q);
        s.write_planar(&p,q.len(),6);
    }
    fn take(s: &mut SurrogateStream, chunk: usize) -> Vec<f32> {
        let mut out=Vec::new(); let mut scratch=vec![0.;chunk];
        while s.input_available()>0 {
            let n=s.render(&mut scratch); assert!(n>0); out.extend_from_slice(&scratch[..n]);
        }
        out
    }

    #[test]
    fn no_input_no_invented_audio_and_valid_calibration() {
        let mut s=stream(); assert_eq!(s.input_available(),0);
        assert_eq!(s.render(&mut [0.;128]),0);
        write(&mut s,&vec![0.4;1024]);
        let y=take(&mut s,128);
        assert!(!s.failed()); assert!(y.len()>2000);
        for v in &y[200..] { assert!((v-0.2).abs()<2e-6); }
    }

    #[test]
    fn packet_boundaries_preserve_state_and_resampling_phase() {
        let q:Vec<_>=(0..5000).map(|i| (i as f32*0.13).sin()).collect();
        let mut a=stream(); write(&mut a,&q); let x=take(&mut a,73);
        let mut b=stream(); let mut y=Vec::new();
        for packet in q.chunks(137) { write(&mut b,packet); y.extend(take(&mut b,128)); }
        assert_eq!(x.len(),y.len());
        assert!(x.iter().zip(y).all(|(a,b)| (a-b).abs()<2e-6));
    }

    #[test]
    fn source_resampling_and_anti_imaging() {
        let mut s=SurrogateStream::new(&model(vec![1.]),10000.,44100.,1.).unwrap();
        let q:Vec<_>=(0..2000).map(|i| (2.*PI*1000.*i as f64/10000.).sin() as f32).collect();
        write(&mut s,&q); let y=take(&mut s,128);
        assert!(y.len()>8500 && !s.failed());
        // Linear source interpolation has its known sinc^2 response; device reconstruction
        // must not add a phase alignment or an arbitrary load-dependent gain.
        let e:f64=y[500..].iter().enumerate().map(|(i,v)| {
            let target=(2.*PI*1000.*(i+500) as f64/44100.).sin(); (*v as f64-target).powi(2)
        }).sum::<f64>()/(y.len()-500) as f64;
        assert!(e.sqrt()<0.04,"resampling error {}",e.sqrt());
    }

    #[test]
    fn dropping_advances_history_not_just_time() {
        let q:Vec<_>=(0..3000).map(|i| (i as f32*0.21).sin()).collect();
        let mut a=stream(); write(&mut a,&q);
        let full=take(&mut a,128);
        let mut b=stream(); write(&mut b,&q); b.drop_input(500); let end=take(&mut b,128);
        assert_eq!(end.len(),500);
        assert!(end.iter().zip(&full[full.len()-500..]).all(|(a,b)| (a-b).abs()<2e-6));
    }

    #[test]
    fn reset_removes_previous_cars_tail_and_phase() {
        let q:Vec<_>=(0..1000).map(|i| (i as f32*0.23).sin()).collect();
        let mut a=stream(); write(&mut a,&q); let _=take(&mut a,128);
        a.reset(); write(&mut a,&q); let reset=take(&mut a,128);
        let mut b=stream(); write(&mut b,&q); let fresh=take(&mut b,128);
        assert_eq!(reset,fresh);
    }

    #[test]
    fn bad_input_latches_failure_without_panicking() {
        let mut s=stream(); s.write_planar(&[f32::NAN;6],1,6);
        assert!(s.failed()); assert_eq!(s.input_available(),0);
        assert_eq!(s.render(&mut [0.;128]),0);
        assert!(SurrogateStream::new(&model(vec![1.]),0.,48000.,1.).is_err());
        assert!(SurrogateStream::new(&model(vec![1.]),10000.,48000.,0.).is_err());
    }
}
