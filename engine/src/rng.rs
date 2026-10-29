//! A small deterministic generator, in place of C's rand() (combustion efficiency scatter,
//! air noise), so runs repeat exactly.

pub struct Rng {
    state: u32,
}

pub const RAND_MAX: u32 = 0x7fff_ffff;

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng { state: seed }
    }

    /// 0..=RAND_MAX
    pub fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1_103_515_245).wrapping_add(12_345) & RAND_MAX;
        self.state
    }

    /// 0..=1, as rand() / RAND_MAX.
    pub fn unit(&mut self) -> f64 {
        self.next() as f64 / RAND_MAX as f64
    }
}
