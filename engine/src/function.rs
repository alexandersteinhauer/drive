//! A sampled function, read back with a triangle filter (engine-sim's function.h/.cpp).

use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(from = "RawFunction")]
pub struct Function {
    pub filter_radius: f64,
    /// (x, y), sorted by x; `from_samples` sorts them as engine-sim's insertion does.
    pub samples: Vec<(f64, f64)>,
}

/// As written: samples in any order.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawFunction {
    filter_radius: f64,
    samples: Vec<(f64, f64)>,
}

impl From<RawFunction> for Function {
    fn from(raw: RawFunction) -> Self {
        Function::from_samples(raw.filter_radius, &raw.samples)
    }
}

impl Function {
    pub fn new(filter_radius: f64) -> Self {
        Function { filter_radius, samples: Vec::new() }
    }

    /// Inserts samples in order, as engine-sim's addSample does (ties go after).
    pub fn from_samples(filter_radius: f64, samples: &[(f64, f64)]) -> Self {
        let mut f = Function::new(filter_radius);
        for &(x, y) in samples {
            f.add_sample(x, y);
        }
        f
    }

    pub fn add_sample(&mut self, x: f64, y: f64) {
        let closest = self.closest_sample(x);
        if closest < 0 {
            self.samples.clear();
            self.samples.push((x, y));
            return;
        }
        let closest = closest as usize;
        let index = if x < self.samples[closest].0 { closest } else { closest + 1 };
        self.samples.insert(index, (x, y));
    }

    pub fn sample_triangle(&self, x: f64) -> f64 {
        let s = &self.samples;
        let size = s.len();
        let closest = self.closest_sample(x);
        if size == 0 {
            return 0.0;
        } else if x >= s[size - 1].0 {
            return s[size - 1].1;
        } else if x <= s[0].0 {
            return s[0].1;
        }

        let closest = closest as usize;
        let mut sum = 0.0;
        let mut total_weight = 0.0;
        let mut i = closest as isize;
        while i >= 0 {
            let (xi, yi) = s[i as usize];
            i -= 1;
            if xi > x {
                continue;
            }
            if (x - xi).abs() > self.filter_radius {
                break;
            }
            let w = self.triangle(xi - x);
            sum += w * yi;
            total_weight += w;
        }
        for &(xi, yi) in &s[closest..] {
            if xi <= x {
                continue;
            }
            if (xi - x).abs() > self.filter_radius {
                break;
            }
            let w = self.triangle(xi - x);
            sum += w * yi;
            total_weight += w;
        }
        if total_weight != 0.0 {
            sum / total_weight
        } else {
            0.0
        }
    }

    #[inline]
    fn triangle(&self, x: f64) -> f64 {
        (self.filter_radius - x.abs()) / self.filter_radius
    }

    fn closest_sample(&self, x: f64) -> isize {
        if x.is_nan() {
            return 0;
        }
        let s = &self.samples;
        if s.is_empty() {
            return -1;
        }
        let mut l = 0usize;
        let mut r = s.len() - 1;
        if x <= s[l].0 {
            return l as isize;
        } else if x >= s[r].0 {
            return r as isize;
        }
        while l + 1 < r {
            let m = (l + r) / 2;
            if x > s[m].0 {
                l = m;
            } else if x < s[m].0 {
                r = m;
            } else {
                return m as isize;
            }
        }
        if x - s[l].0 < s[r].0 - x {
            l as isize
        } else {
            r as isize
        }
    }
}
