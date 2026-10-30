# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""Check burn-response superposition, from examples/superpose.rs, without audio alignment.

uv run engine/scripts/superpose_analyse.py <prefix>
Reports nonlinear second difference relative both to the summed first differences and to
base sound. A small perturbation is a local-linearization test, not full-amplitude linearity.
Outlets are evaluated jointly (no cancellation between outlets in the error measure).
"""
import json
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

prefix = sys.argv[1]
meta = json.loads(Path(prefix + ".json").read_text())
rate = 48000
ratio = Fraction(rate, int(meta["fluidRate"]))

def load(branch):
    x = np.fromfile(f"{prefix}.{branch}.ends.f32", dtype="<f4").astype(float)
    x = x.reshape(-1, meta["ends"])
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=0)

def norm(x, lo=20, hi=10000):
    f = np.fft.rfftfreq(len(x), 1/rate)
    spectrum = np.fft.rfft(x, axis=0)
    return np.linalg.norm(spectrum[(f >= lo) & (f < hi)])

def db(a, b):
    return float(20*np.log10(max(a, 1e-30)/max(b, 1e-30)))

base = load("base")
start = int(meta["burns"]["A"][0]/meta["rate"]*rate)
bands = [(20, 200), (200, 800), (800, 3200), (3200, 6400), (6400, 10000)]
print(f"{meta['rpm']:g} rpm, throttle {meta['throttle']:g}, {meta['record']:g} s; before directivity")
print("Error = y(a+b)-y(a)-y(b)+y(base). No sample or phase realignment.")
for factor in meta["factors"]:
    d = {key: load(f"{key}@{factor:g}")-base for key in ["A", "B", "C", "A2", "A+B", "A+C", "A+A2"]}
    print(f"\nfuel factor {factor:g}")
    for pair, (a, b) in {"A+B": ("A", "B"), "A+C": ("A", "C"), "A+A2": ("A", "A2")}.items():
        lin, residual = d[a]+d[b], d[pair]-d[a]-d[b]
        r, z = residual[start:], lin[start:]
        if not z.any():
            print(f"  {pair}: unchanged replay, exact={not r.any()}")
            continue
        print(f"  {pair}: {db(norm(r), norm(z)):+.1f} dB re linear response, "
              f"{db(norm(r), norm(base[start:])):+.1f} dB re base; bands re response: " +
              " ".join(f"{lo}-{hi}Hz {db(norm(r, lo, hi), norm(z, lo, hi)):+.1f}dB" for lo, hi in bands))
        # Differentiate immediate acoustics from slow engine/turbo feedback.
        for begin, end in [(0, .05), (.05, .15), (.15, meta["record"])]:
            first, last = start+round(begin*rate), min(start+round(end*rate), len(base))
            if last-first >= 240:
                print(f"    {begin*1e3:.0f}..{(last-start)/rate*1e3:.0f}ms: "
                      f"{db(norm(residual[first:last]), norm(lin[first:last])):+.1f} dB re response")
