# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""Check burn-response superposition without audio alignment.

uv run engine/scripts/superpose_analyse.py <examples/superpose.rs output prefix>
Uses f64 recordings when EVERY selected branch has one, otherwise consistently f32.
Reports pair second differences and the strength-linearization limit independently.
The latter is NOT established merely by a bounded, deterministic replay.
"""
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

RATE = 48000
BANDS = [(20, 200), (200, 800), (800, 3200), (3200, 6400), (6400, 10000)]


def norm(x, lo=20, hi=10000):
    f = np.fft.rfftfreq(len(x), 1/RATE)
    spectrum = np.fft.rfft(x, axis=0)
    return np.linalg.norm(spectrum[(f >= lo) & (f < hi)])


def db(a, b):
    return float(20*np.log10(max(a, 1e-30)/max(b, 1e-30)))


def analyse(prefix):
    meta = json.loads(Path(prefix + ".json").read_text())
    names = ["base", *meta["branches"]]
    precision = "f64" if all(Path(f"{prefix}.{name}.ends.f64").exists() for name in names) else "f32"
    ratio = Fraction(RATE, int(meta["fluidRate"]))

    def load(branch):
        x = np.fromfile(f"{prefix}.{branch}.ends.{precision}", dtype="<f8" if precision == "f64" else "<f4").astype(float)
        x = x.reshape(-1, meta["ends"])
        if not len(x) or not np.isfinite(x).all():
            raise ValueError(f"empty/nonfinite recording: {branch}")
        return resample_poly(x, ratio.numerator, ratio.denominator, axis=0)

    base = load("base")
    start = max(0, round(meta["burns"]["A"][0]/meta["rate"]*RATE)-round(.002*RATE))
    print(f"{meta['rpm']:g} rpm, throttle {meta['throttle']:g}, {meta['record']:g} s; {precision}, before directivity")
    print("Joint-outlet spectral norms, 20 Hz..10 kHz, from 2 ms before A; no phase realignment.")
    print("Pair error = y(a+b)-y(a)-y(b)+y(base). Strength error = D(eps)-eps/EPS*D(EPS).")
    strengths = []
    for factor in meta["factors"]:
        # Use recorded branch names, not :g formatting: .9999999 would round to '1'.
        branches = {name.rsplit("@", 1)[0]: name for name in meta["branches"]
                    if float(name.rsplit("@", 1)[1]) == factor}
        d = {}
        for key, name in branches.items():
            y = load(name)
            if y.shape != base.shape:
                raise ValueError(f"recording dimensions differ: {name}")
            d[key] = y-base
        if "A" in d:
            if factor == 1:
                print(f"\nunchanged A replay exact: {not d['A'].any()}")
            else:
                strengths.append((1-factor, d["A"][start:]))
        print(f"\nfuel factor {factor:.12g}")
        for pair, (a, b) in {"A+B": ("A", "B"), "A+C": ("A", "C"), "A+A2": ("A", "A2")}.items():
            if not all(key in d for key in (pair, a, b)):
                continue
            lin, residual = d[a]+d[b], d[pair]-d[a]-d[b]
            r, z = residual[start:], lin[start:]
            if not z.any():
                print(f"  {pair}: unchanged replay, exact={not r.any()}")
                continue
            print(f"  {pair}: {db(norm(r), norm(z)):+.1f} dB re linear response, "
                  f"{db(norm(r), norm(base[start:])):+.1f} dB re base; bands re response: " +
                  " ".join(f"{lo}-{hi}Hz {db(norm(r, lo, hi), norm(z, lo, hi)):+.1f}dB" for lo, hi in BANDS))
    if strengths:
        print("\nA strength limit (only this table tests amplitude linearization):")
        strengths.sort(key=lambda row: row[0])
        for j, (eps, delta) in enumerate(strengths):
            text = f"  eps {eps:.8g}: D {db(norm(delta), norm(base[start:])):+.2f} dB re base"
            if j+1 < len(strengths):
                larger, large_delta = strengths[j+1]
                residual = delta-(eps/larger)*large_delta
                actual_ratio = norm(large_delta)/max(norm(delta), 1e-30)
                text += (f", next/this norm {actual_ratio:.4g} (linear {larger/eps:.4g}), "
                         f"strength error {db(norm(residual), norm(delta)):+.2f} dB re D")
            print(text)


if __name__ == "__main__":
    analyse(sys.argv[1])
