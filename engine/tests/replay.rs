//! The offline identification harness branches from a deep engine snapshot; a unit burn
//! factor must be exactly an unchanged replay, including the random sequences and mechanics.
use engine_sim::{engine::Engine, spec::EngineSpec, units::rpm};

fn engine() -> Engine {
    let mut spec: EngineSpec = serde_json::from_str(include_str!("vw1200.json")).unwrap();
    spec.physics.cycle_variation = Some(0.1);
    let mut e = Engine::new(&spec);
    e.collect_audio = false;
    e.seed_burns(17);
    e.set_ignition(true);
    e.set_crank_speed(-rpm(3000.0));
    e.set_dyno(true, true, rpm(3000.0));
    e.set_speed_control(0.8);
    e
}

#[test]
fn unit_burn_is_bit_exact_replay() {
    let mut a = engine();
    for _ in 0..1000 { a.step(); }
    let mut b = a.clone();
    for i in 0..b.cylinder_count() {
        b.burn_scales.push((i, b.burns(i), 1.0));
    }
    for _ in 0..2000 {
        a.step(); b.step();
        assert_eq!(a.crank_theta().to_bits(), b.crank_theta().to_bits());
        assert_eq!(a.dyno_torque().to_bits(), b.dyno_torque().to_bits());
        for i in 0..a.cylinder_count() {
            assert_eq!(a.burns(i), b.burns(i));
            assert_eq!(a.cylinder_pressure(i).to_bits(), b.cylinder_pressure(i).to_bits());
            assert_eq!(a.cylinder_temperature(i).to_bits(), b.cylinder_temperature(i).to_bits());
            assert_eq!(a.cylinder_exhaust_flow(i).to_bits(), b.cylinder_exhaust_flow(i).to_bits());
        }
    }
}

#[test]
fn independent_seed_is_reproducible_not_shared_state() {
    let mut a = engine();
    let mut b = a.clone();
    b.seed_burns(239);
    let mut differs = false;
    for _ in 0..3000 {
        a.step(); b.step();
        differs |= a.cylinder_pressure(0).to_bits() != b.cylinder_pressure(0).to_bits();
        assert!(a.cylinder_pressure(0).is_finite());
        assert!(b.cylinder_pressure(0).is_finite());
    }
    assert!(differs, "independent burn seeds must excite different burns");
}

#[test]
fn learned_audio_selector_does_not_change_engine_physics() {
    use engine_sim::spec::SurrogateAudioSpec;
    let mut spec: EngineSpec = serde_json::from_str(include_str!("vw1200.json")).unwrap();
    let mut original = Engine::new(&spec);
    spec.audio.surrogate = Some(SurrogateAudioSpec {
        model: "audio/surrogate/test.json".into(), full_scale: 16.,
        flap_speed: [200.,400.], flap_throttle: [0.3,0.6],
    });
    let mut learned = Engine::new(&spec);
    let cylinders = learned.cylinder_count();
    assert_eq!(learned.audio.len(), cylinders+5);
    for e in [&mut original, &mut learned] {
        e.set_ignition(true); e.set_speed_control(0.8);
        e.set_crank_speed(-rpm(3000.)); e.set_dyno(true,true,rpm(3000.));
    }
    for step in 0..1200 {
        if step == 400 {
            learned.set_surrogate_audio(false);
            assert_eq!(learned.audio.len(), original.audio.len());
            assert!(learned.audio.iter().all(Vec::is_empty));
        }
        if step == 800 {
            learned.set_surrogate_audio(true);
            assert_eq!(learned.audio.len(), cylinders+5);
            assert!(learned.audio.iter().all(Vec::is_empty));
        }
        original.step(); learned.step();
        assert_eq!(original.crank_theta().to_bits(), learned.crank_theta().to_bits());
        assert_eq!(original.dyno_torque().to_bits(), learned.dyno_torque().to_bits());
        for i in 0..cylinders {
            assert_eq!(original.cylinder_pressure(i).to_bits(), learned.cylinder_pressure(i).to_bits());
            if step >= 800 {
                assert_eq!(*learned.audio[i].last().unwrap(), learned.cylinder_exhaust_flow(i) as f32);
            }
        }
        if step >= 800 {
            assert_eq!(*learned.audio[cylinders].last().unwrap(), (learned.rpm()/1000.) as f32);
            assert_eq!(*learned.audio[cylinders+1].last().unwrap(), (learned.manifold_pressure()/1e5) as f32);
            assert_eq!(*learned.audio[cylinders+3].last().unwrap(), 0.8);
        }
    }
}
