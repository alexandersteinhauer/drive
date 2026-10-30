# /// script
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
# The reference simulation's sound at 48 kHz: the pressure its open ends radiate (the
# simulated waves leaving the tailpipes), and the tailpipe jets' turbulent mixing noise from
# the simulated exit flow (Stone, Groesbeck & Zola, NASA TM-82712, 1981: eq. A1 for the
# overall level of a static round jet, hot or cold, and Table I for its 1/3-octave spectrum
# by corrected angle and temperature-corrected Strouhal number), quasi-steady: each 1/4 ms,
# each tailpipe's bands at the level its exit velocity, density and temperature give then.
#   uv run engine/scripts/jetnoise.py <prefix> [listener angle from the inlet axis, deg (160)]
# Reads <prefix>.wav (examples/reference.rs) or <prefix>.f32 and <prefix>.json (a drive), and
# <prefix>.exits.f32 (and <prefix>.exits.txt, or the tailpipe diameter in DIAMETER and the
# listener's distance in DISTANCE). Writes <prefix>-tonal/-jet/-mix.wav (Pa, 48 kHz floats).
import sys, os, json, numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt
from fractions import Fraction
pre = sys.argv[1]
theta = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
here = os.path.dirname(os.path.abspath(sys.argv[0]))
# Stone, Groesbeck & Zola's Table I (1/3-octave SPL - OASPL_u by log S and corrected angle),
# transcribed; its columns sum to the printed overall levels within 0.05 dB.
table_path = os.path.join(here, "stone_table1.csv")
if os.path.exists(pre + ".wav"):
    p, rate = sf.read(pre + ".wav", dtype="float64")
else:
    rate = json.load(open(pre + ".json"))["rate"]
    p = np.fromfile(pre + ".f32", dtype=np.float32).astype(np.float64)
meta = {}
if os.path.exists(pre + ".exits.txt"):
    for line in open(pre + ".exits.txt"):
        k, *v = line.split()
        meta[k] = v
D = float(os.environ.get("DIAMETER", meta.get("diameters", ["0.085"])[0]))
R = float(os.environ.get("DISTANCE", meta.get("distance", ["5"])[0]))
ex = np.fromfile(pre + ".exits.f32", dtype=np.float32).astype(np.float64)
outlets = 2
ex = ex[: len(ex) // (3 * outlets) * 3 * outlets].reshape(-1, outlets, 3)
n = min(len(ex), len(p))
ex, p = ex[:n], p[:n]
FS = 48000
f = Fraction(FS, int(rate)).limit_denominator(2000)
tonal = resample_poly(p, f.numerator, f.denominator)
N = len(tonal)

# Control rate: 4 kHz, the exit state averaged over each 1/4 ms.
CR = 4000
k = int(round(rate / CR))
m = n // k
blk = ex[: m * k].reshape(m, k, outlets, 3).mean(axis=1)

# Ambient air (the duct model's): 298 K; ISA reference of Stone's eq. A1.
T_inf, rho_inf, c_inf = 298.15, 1.184, 346.1
rho_isa, c_isa = 1.225, 340.3
tab = np.genfromtxt(table_path, delimiter=",", names=True)
log_s = tab["logS"]
cols = [c for c in tab.dtype.names if c.startswith("th")]
col_angle = np.array([110.0 if c == "th0_110" else float(c[2:]) for c in cols])
grid = np.stack([tab[c] for c in cols], axis=1)  # rows: logS, columns: corrected angle
def table(thp, ls):
    # Interpolate across angle, then log S (clamped to the table).
    row = np.array([np.interp(thp, col_angle, grid[i]) for i in range(len(log_s))])
    return np.interp(ls, log_s, row)

centers = 1000.0 * 2.0 ** (np.arange(-16, 14) / 3.0)  # 25 Hz .. 20 kHz
centers = centers[centers < 0.45 * FS]
A = np.pi * D * D / 4
th = np.radians(theta)
def bands(V, rho_j, T_j):
    """1/3-octave SPL (dB) per band for one tailpipe's jet, or None (no outflow)."""
    if V < 1.0:
        return None
    Va = V / c_inf
    om = 3 * Va**3.5 / (0.6 + Va**3.5) - 1
    Mc = 0.62 * Va
    oaspl_u = (141 + 10 * np.log10((rho_inf / rho_isa) ** 2 * (c_inf / c_isa) ** 4) + 10 * np.log10(A / R**2)
               + 10 * om * np.log10(rho_j / rho_inf) + 75 * np.log10(Va)
               - 15 * np.log10((1 + Mc * np.cos(th)) ** 2 + 0.04 * Mc**2))
    thp = theta * Va**0.1
    S = centers * D / V * (T_j / T_inf) ** (0.4 * (1 + np.cos(np.radians(thp))))
    return oaspl_u + table(thp, np.log10(S))

# Band amplitudes (rms Pa) at the control rate, per tailpipe.
amp = np.zeros((outlets, m, len(centers)))
for o in range(outlets):
    for i in range(m):
        V, rho_j, T_j = blk[i, o]
        b = bands(V, rho_j, T_j)
        if b is not None:
            amp[o, i] = 2e-5 * 10 ** (b / 20)
tc = (np.arange(m) + 0.5) / CR
t48 = np.arange(N) / FS
rng = np.random.default_rng(1)
jet = np.zeros(N)
for o in range(outlets):
    for j, fc in enumerate(centers):
        lo, hi = fc / 2 ** (1 / 6), min(fc * 2 ** (1 / 6), 0.49 * FS)
        sos = butter(3, [lo, hi], btype="band", fs=FS, output="sos")
        x = sosfilt(sos, rng.standard_normal(N + 4800))[4800:]
        x /= np.sqrt(np.mean(x**2))
        jet += x * np.interp(t48, tc, amp[o, :, j])
mix = tonal + jet
def level(x):
    r = np.sqrt(np.mean(x[int(0.3 * FS):] ** 2))
    return f"{20 * np.log10(r / 2e-5 + 1e-30):.1f} dB"
print(f"{pre}: tonal {level(tonal)}, jet {level(jet)}, mix {level(mix)}; jet {D*1000:.0f} mm at {R} m, {theta:.0f} deg from the inlet axis")
for name, x in [("tonal", tonal), ("jet", jet), ("mix", mix)]:
    sf.write(f"{pre}-{name}.wav", x.astype(np.float32), FS, subtype="FLOAT")
