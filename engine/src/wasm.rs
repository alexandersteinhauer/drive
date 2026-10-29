//! WebAssembly entry points (C ABI, no bindings generator): the engine for the game's
//! simulation, the synthesizer for the audio thread. Each instance of the module holds
//! whatever it creates; pointers are opaque handles to JavaScript.

use crate::engine::Engine;
use crate::spec::EngineSpec;
use crate::synth::{read_wav, AudioParameters, Synthesizer};

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` and `len` must come from `alloc`.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    std::slice::from_raw_parts(ptr, len)
}

/// Builds an engine from its JSON spec; null if the spec doesn't parse.
///
/// # Safety
/// `ptr`/`len` must be a valid byte range.
#[no_mangle]
pub unsafe extern "C" fn engine_new(ptr: *const u8, len: usize) -> *mut Engine {
    match serde_json::from_slice::<EngineSpec>(bytes(ptr, len)) {
        Ok(spec) => Box::into_raw(Box::new(Engine::new(&spec))),
        Err(_) => std::ptr::null_mut(),
    }
}

macro_rules! engine_fns {
    ($($(#[$m:meta])* fn $name:ident($e:ident $(, $a:ident: $t:ty)*) $(-> $r:ty)? $body:block)*) => {
        $(
            $(#[$m])*
            /// # Safety
            /// `e` must come from `engine_new` and not be freed.
            #[no_mangle]
            pub unsafe extern "C" fn $name(e: *mut Engine $(, $a: $t)*) $(-> $r)? {
                let $e = &mut *e;
                $body
            }
        )*
    };
}

engine_fns! {
    fn engine_free(e) {
        drop(Box::from_raw(e as *mut Engine));
    }
    /// Runs `steps` simulation steps.
    fn engine_step(e, steps: u32) {
        for _ in 0..steps {
            e.step();
        }
    }
    fn engine_frequency(e) -> f64 { 1.0 / e.timestep() }
    /// Pedal 0..1.
    fn engine_set_throttle(e, pedal: f64) { e.set_speed_control(pedal) }
    fn engine_set_ignition(e, on: u32) { e.set_ignition(on != 0) }
    fn engine_set_starter(e, on: u32) { e.set_starter(on != 0) }
    /// Holds (or, without `hold`, only drives) the crank at `speed` rad/s.
    fn engine_set_dyno(e, enabled: u32, hold: u32, speed: f64) { e.set_dyno(enabled != 0, hold != 0, speed) }
    fn engine_dyno_torque(e) -> f64 { e.dyno_torque() }
    fn engine_filtered_dyno_torque(e) -> f64 { e.filtered_dyno_torque() }
    /// Crank speed in rad/s, positive forwards.
    fn engine_speed(e) -> f64 { -e.crank_speed() }
    fn engine_set_speed(e, speed: f64) { e.set_crank_speed(-speed) }
    /// Exhaust signal of the steps since the last clear, per exhaust system.
    fn engine_audio_len(e) -> usize { e.audio.first().map_or(0, |a| a.len()) }
    fn engine_audio_channels(e) -> usize { e.audio.len() }
    fn engine_audio_ptr(e, channel: usize) -> *const f32 { e.audio[channel].as_ptr() }
    fn engine_audio_clear(e) {
        for a in &mut e.audio {
            a.clear();
        }
    }
}

/// A synthesizer for `channels` exhaust systems fed at `input_rate` steps per second.
#[no_mangle]
pub extern "C" fn synth_new(channels: usize, input_rate: f32, hf_gain: f32, noise: f32, jitter: f32) -> *mut Synthesizer {
    let params = AudioParameters {
        df_f_mix: hf_gain,
        air_noise: noise,
        input_sample_noise: jitter,
        ..AudioParameters::default()
    };
    Box::into_raw(Box::new(Synthesizer::new(channels, input_rate, 44100.0, params)))
}

macro_rules! synth_fns {
    ($($(#[$m:meta])* fn $name:ident($s:ident $(, $a:ident: $t:ty)*) $(-> $r:ty)? $body:block)*) => {
        $(
            $(#[$m])*
            /// # Safety
            /// `s` must come from `synth_new` and not be freed.
            #[no_mangle]
            pub unsafe extern "C" fn $name(s: *mut Synthesizer $(, $a: $t)*) $(-> $r)? {
                let $s = &mut *s;
                $body
            }
        )*
    };
}

thread_local! {
    static OUT: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

synth_fns! {
    fn synth_free(s) {
        drop(Box::from_raw(s as *mut Synthesizer));
    }
    /// Loads the impulse response for `channel` from a 16-bit WAV file; 0 if it can't.
    fn synth_set_impulse_wav(s, channel: usize, ptr: *const u8, len: usize, volume: f32) -> u32 {
        match read_wav(bytes(ptr, len)) {
            Some(samples) => {
                s.set_impulse_response(channel, &samples, volume);
                1
            }
            None => 0,
        }
    }
    /// A fixed output gain (engine-sim's leveller held at one level).
    fn synth_set_gain(s, gain: f32) {
        s.params.leveler_min_gain = gain;
        s.params.leveler_max_gain = gain;
    }
    fn synth_set_input_rate(s, rate: f32) { s.set_input_sample_rate(rate) }
    /// Writes `steps` values per channel, channel after channel.
    fn synth_write(s, ptr: *const f32, steps: usize, channels: usize) {
        let data = std::slice::from_raw_parts(ptr, steps * channels);
        let mut frame = vec![0.0f64; channels];
        for i in 0..steps {
            for (c, f) in frame.iter_mut().enumerate() {
                *f = data[c * steps + i] as f64;
            }
            s.write_input(&frame);
        }
    }
    fn synth_input_available(s) -> usize { s.input_available() }
    fn synth_drop_input(s, keep: usize) { s.drop_input(keep) }
    /// Renders up to `frames` samples; returns a pointer to them (valid until the next call)
    /// and leaves their count in `synth_rendered`.
    fn synth_render(s, frames: usize) -> *const f32 {
        OUT.with(|out| {
            let mut out = out.borrow_mut();
            out.clear();
            out.resize(frames, 0.0);
            let n = s.render(&mut out);
            RENDERED.with(|r| r.set(n));
            out.as_ptr()
        })
    }
}

thread_local! {
    static RENDERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[no_mangle]
pub extern "C" fn synth_rendered() -> usize {
    RENDERED.with(|r| r.get())
}
