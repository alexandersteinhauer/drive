//! SI units and physical constants (engine-sim's units.h and constants.h).

pub const PI: f64 = 3.14159265359;
pub const R: f64 = 8.31446261815324;
pub const ROOT_2: f64 = 1.41421356237309504880168872420969807856967187537694807317667973799;
pub const E: f64 = 2.71828182845904523536028747135266249775724709369995;

pub const G: f64 = 1.0 / 1000.0;
pub const LBF: f64 = 4.44822;
pub const CM: f64 = 1.0 / 100.0;
pub const MM: f64 = 1.0 / 1000.0;
pub const INCH: f64 = CM * 2.54;
pub const THOU: f64 = INCH / 1000.0;
pub const CC: f64 = CM * CM * CM;
pub const LITRE: f64 = CC * 1000.0;
pub const DEG: f64 = PI / 180.0;
pub const FOOT: f64 = INCH * 12.0;
pub const FT_LB: f64 = FOOT * LBF;
pub const LBMOL: f64 = 453.59237;
pub const MINUTE: f64 = 60.0;
pub const SCFM: f64 = 0.002641 * LBMOL / MINUTE;
pub const KPA: f64 = 1000.0;
pub const ATM: f64 = 101.325 * KPA;
pub const PSI: f64 = LBF / (INCH * INCH);
pub const IN_HG: f64 = 3386.3886666666713;
pub const IN_H2O: f64 = IN_HG * 0.0734824;
pub const K0: f64 = 273.15;
pub const AIR_MOLECULAR_MASS: f64 = 28.97 * G;

pub const fn celcius(t: f64) -> f64 {
    t + K0
}

pub const fn rpm(rpm: f64) -> f64 {
    rpm * 0.104719755
}

pub const fn to_rpm(rad_s: f64) -> f64 {
    rad_s / 0.104719755
}
