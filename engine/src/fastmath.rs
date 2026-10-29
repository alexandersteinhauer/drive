//! ln and exp for the physics mode's hot paths (gas.rs, the cylinders' heat transfer), where
//! the math library's (in WebAssembly compiled in, in software) took over half the time.
//! Range reduction and short series: within ~1e-14 relative of std, plenty for the gases.

const LN2: f64 = std::f64::consts::LN_2;

/// Natural logarithm of a positive, normal `x`.
#[inline]
pub fn ln(x: f64) -> f64 {
    // x = m 2^e with m in [sqrt(1/2), sqrt(2)).
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    // ln m = 2 atanh(s), s = (m - 1) / (m + 1), |s| < 0.172.
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let series = 1.0
        + s2 * (1.0 / 3.0
            + s2 * (1.0 / 5.0
                + s2 * (1.0 / 7.0
                    + s2 * (1.0 / 9.0 + s2 * (1.0 / 11.0 + s2 * (1.0 / 13.0 + s2 * (1.0 / 15.0 + s2 * (1.0 / 17.0))))))));
    e as f64 * LN2 + 2.0 * s * series
}

/// e^x for |x| < 700.
#[inline]
pub fn exp(x: f64) -> f64 {
    // x = k ln2 + r, |r| <= ln2 / 2; e^r by its series.
    let k = (x / LN2).round();
    let r = x - k * LN2;
    let p = 1.0
        + r * (1.0
            + r * (1.0 / 2.0
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0
                            + r * (1.0 / 720.0
                                + r * (1.0 / 5040.0
                                    + r * (1.0 / 40320.0
                                        + r * (1.0 / 362880.0
                                            + r * (1.0 / 3628800.0 + r * (1.0 / 39916800.0 + r * (1.0 / 479001600.0))))))))))));
    p * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

/// x^y for positive x.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    exp(y * ln(x))
}

#[cfg(test)]
mod tests {
    #[test]
    fn match_std() {
        let mut worst = (0.0f64, 0.0f64);
        let mut x = 1e-6;
        while x < 1e9 {
            worst.0 = worst.0.max(((super::ln(x) - x.ln()) / x.ln().abs().max(1.0)).abs());
            x *= 1.000731;
        }
        let mut x = -60.0;
        while x < 60.0 {
            worst.1 = worst.1.max(((super::exp(x) - x.exp()) / x.exp()).abs());
            x += 0.000917;
        }
        assert!(worst.0 < 1e-14 && worst.1 < 1e-14, "{worst:?}");
    }
}
