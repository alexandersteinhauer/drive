# /// script
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""Experimental causal exhaust surrogate identified from paired abtrace runs.

Fits a bank of causal FIRs, with optional signed-power source features and Gaussian
operating-point gates. Gates act on the CURRENT output, including waves already in flight.
The fit is linear least squares, not a neural network; it is the simplest learned baseline
against which to justify a more complicated model. There is no recording fit or audio gain.

uv run engine/scripts/surrogate_fit.py <train-reference> <train-game> <test-reference>
    <test-game> <reference-config.json> <out-prefix> [--gates 6] [--taps 1024]

Splits by whole independently rendered trajectories and burn seeds. No phase realignment,
per-clip normalization, future inputs, noise waveform target or post-directivity training.
Pressure targets are anti-aliased to the model rate (20 kHz for the pilot, bandwidth <10 kHz).
A common resampling phase is part of this measurement convention, not listener directivity.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import lfilter, resample_poly, welch
from scipy.sparse.linalg import LinearOperator, lsqr


def load(prefix):
    meta = json.loads(Path(prefix + ".steps.json").read_text())
    ng, nc, n = len(meta["global"]), len(meta["cylinder"]), meta["cylinders"]
    raw = np.fromfile(prefix + ".steps.f32", dtype="<f4").reshape(-1, ng + nc * n)
    if not np.isfinite(raw).all():
        raise ValueError(f"nonfinite trace: {prefix}")
    glob = {k: raw[:, j].astype(float) for j, k in enumerate(meta["global"])}
    cyl = {k: raw[:, ng + j::nc].astype(float) for j, k in enumerate(meta["cylinder"])}
    return meta, glob, cyl


def controls(meta, glob, reference):
    """Only game-available state, and the flap actuator driven by those same controls."""
    flap_spec = reference.get("ducts", reference).get("flaps", [])
    opening = np.zeros_like(glob["rpm"])
    if flap_spec:
        f = flap_spec[0]
        def ramp(x, bounds):
            z = np.clip((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
            return z * z * (3 - 2 * z)
        target = np.maximum(ramp(glob["rpm"] * np.pi / 30, f["speed"]), ramp(glob["throttle"], f["throttle"]))
        # Match the reference's fluid-step actuator integration, not a look-ahead smooth.
        fluid_steps = int(round(meta.get("fluidRate", meta["rate"] * 12) / meta["rate"]))
        alpha = 1 - (1 - 1 / (meta["rate"] * fluid_steps * 0.15)) ** fluid_steps
        opening = lfilter([alpha], [1, -(1-alpha)], target)
    return np.stack([glob["rpm"] / 1000, glob["manifold"] / 1e5,
                     glob["wastegate"], glob["throttle"], opening], axis=1)


def gates(control, centers, widths):
    d = ((control[:, None, :] - centers[None, :, :]) / widths) ** 2
    logits = -0.5 * d.sum(axis=2)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def features(flow, powers, scale):
    return np.concatenate([np.sign(flow) * abs(flow / scale) ** p for p in powers], axis=1)


def targets(prefix, meta, count, exits=False):
    ne, fr, fs = meta["ends"], meta["fluidRate"], meta["rate"]
    if fr % fs != 0:
        raise ValueError("pilot requires integral fluid/model rate ratio")
    p = np.fromfile(prefix + ".ends.f32", dtype="<f4").reshape(-1, ne).astype(float)
    p = resample_poly(p, 1, int(fr / fs), axis=0)[:count]
    labels = [f"pressure{i}" for i in range(ne)]
    if exits:
        z = np.fromfile(prefix + ".exits.f32", dtype="<f4").reshape(-1, ne, 3).astype(float)
        # Step averages, like the simulation's control stream. Log state AFTER averaging.
        z = z[:count * int(fr/fs)].reshape(count, int(fr/fs), ne, 3).mean(axis=1)
        if (z[:, :, 1:] <= 0).any():
            raise ValueError("nonpositive exit density/temperature")
        p = np.concatenate([p, z[:, :, 0], np.log(z[:, :, 1]), np.log(z[:, :, 2])], axis=1)
        labels += [f"velocity{i}" for i in range(ne)] + [f"logDensity{i}" for i in range(ne)] + [f"logTemperature{i}" for i in range(ne)]
    if len(p) != count or not np.isfinite(p).all():
        raise ValueError("invalid target dimensions/state")
    return p, labels


def fit(x, weights, y, taps, iterations, damping, preemphasis, mask, zero_dc=None, penalty_scale=None):
    """Matrix-free causal MIMO regression; operator/adjoint are exact transposes."""
    n, nf = x.shape
    ng, no = weights.shape[1], y.shape[1]
    fftn = next_fast_len(n + taps - 1)
    X = rfft(x, fftn, axis=0)
    use_bias = True
    constrain_dc = False
    zero_dc = [False]*no if zero_dc is None else zero_dc
    size = ng * nf * taps
    # Include per-gate DC biases (exit mean state; pressure bias should stay near zero).
    def forward(h):
        H = h[:size].reshape(ng, nf, taps).transpose(2, 0, 1)
        if constrain_dc:
            H = H - H.mean(axis=0, keepdims=True)
        z = irfft((rfft(H, fftn, axis=0) * X[:, None, :]).sum(axis=2), fftn, axis=0)[:n]
        if use_bias:
            z += h[size:][None, :]
        v = (z * weights).sum(axis=1)
        return lfilter([1, -preemphasis], [1], v) * mask
    def adjoint(v):
        v = v * mask
        # Adjoint of y[t] - preemphasis*y[t-1], without circular wrapping.
        v = v.copy()
        v[:-1] -= preemphasis * v[1:].copy()
        z = v[:, None] * weights
        Z = rfft(z, fftn, axis=0)
        h = irfft(Z[:, :, None] * X.conj()[:, None, :], fftn, axis=0)[:taps]
        if constrain_dc:
            h -= h.mean(axis=0, keepdims=True)
        return np.r_[h.transpose(1, 2, 0).ravel(), z.sum(axis=0) if use_bias else np.zeros(ng)]
    # Explicit augmented ridge keeps the ORIGINAL coefficient prior under diagonal
    # preconditioning. Scalar LSQR damping after rescaling would change that prior.
    diagonal = damping * (np.ones(size+ng) if penalty_scale is None else penalty_scale)
    if diagonal.shape != (size+ng,):
        raise ValueError("regularization diagonal dimensions")
    def augmented_forward(h):
        return np.r_[forward(h), diagonal*h]
    def augmented_adjoint(v):
        return adjoint(v[:n]) + diagonal*v[n:]
    op = LinearOperator((n+size+ng, size+ng), matvec=augmented_forward, rmatvec=augmented_adjoint)
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=size+ng), rng.normal(size=n+size+ng)
    lhs, rhs = np.dot(augmented_forward(u), v), np.dot(u, augmented_adjoint(v))
    if not np.isclose(lhs, rhs, atol=1e-7, rtol=1e-9):
        raise AssertionError(f"operator adjoint mismatch: {lhs} {rhs}")
    kernels, biases, log = [], [], []
    for j in range(no):
        # Far-field monopole pressure has no steady-flow response. Orthogonally project
        # every FIR onto sum(h)=0 in BOTH operator/adjoint; no post-hoc output filtering.
        constrain_dc = zero_dc[j]
        use_bias = not constrain_dc
        rhs = lfilter([1, -preemphasis], [1], y[:, j]) * mask
        result = lsqr(op, np.r_[rhs, np.zeros(size+ng)], damp=0, atol=1e-7, btol=1e-7, iter_lim=iterations)
        h = result[0]
        kernel = h[:size].reshape(ng, nf, taps)
        if constrain_dc:
            kernel = kernel - kernel.mean(axis=-1, keepdims=True)
        kernels.append(kernel)
        biases.append(h[size:] if use_bias else np.zeros(ng))
        log.append({"iterations": result[2], "stop": result[1], "residual": result[3], "conditionEstimate": result[6]})
        print(f"output {j}: {log[-1]}", flush=True)
    return np.stack(kernels, axis=1), np.stack(biases, axis=1), log


def predict(x, weights, kernels, biases):
    n, taps = len(x), kernels.shape[-1]
    fftn = next_fast_len(n + taps - 1)
    X = rfft(x, fftn, axis=0)
    y = np.zeros((n, kernels.shape[1]))
    for g in range(kernels.shape[0]):
        for o in range(kernels.shape[1]):
            h = kernels[g, o].T
            z = irfft((rfft(h, fftn, axis=0) * X).sum(axis=1), fftn)[:n]
            y[:, o] += weights[:, g] * (z + biases[g, o])
    return y


def metrics(target, predicted, rate, start=0):
    t, p = target[start:].sum(axis=1), predicted[start:].sum(axis=1)
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    corr = float(np.corrcoef(t, p)[0, 1]) if t.std() and p.std() else None
    power = np.dot(t, t)
    result = {"rmsReferencePa": rms(t), "rmsPredictionPa": rms(p), "correlation": corr,
              "waveErrorDb": float(10*np.log10(np.dot(t-p, t-p) / max(power, 1e-30)))}
    freq, a = welch(t, rate, nperseg=min(8192, len(t)))
    _, b = welch(p, rate, nperseg=min(8192, len(p)))
    result["bands"] = {}
    for lo, hi in [(20, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400), (6400, 9500)]:
        mask = (freq >= lo) & (freq < hi)
        result["bands"][f"{lo}-{hi}"] = float(10*np.log10(max(b[mask].sum(), 1e-30) / max(a[mask].sum(), 1e-30)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["train_reference", "train_game", "test_reference", "test_game", "config", "out"]:
        ap.add_argument(key)
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--taps", type=int, default=1024)
    ap.add_argument("--powers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iterations", type=int, default=180)
    ap.add_argument("--damping", type=float, default=0.1)
    ap.add_argument("--preemphasis", type=float, default=0.9)
    ap.add_argument("--exits", action="store_true")
    ap.add_argument("--allow-pressure-dc", action="store_true", help="diagnostic unconstrained baseline; pressure normally has exactly zero DC")
    ap.add_argument("--oracle", action="store_true", help="reference inputs instead of game inputs, to measure deployment mismatch")
    args = ap.parse_args()
    started = time.monotonic()
    reference = json.loads(Path(args.config).read_text())
    rm, _, _ = load(args.train_reference)
    tm, _, _ = load(args.test_reference)
    sm, sg, sc = load(args.train_reference if args.oracle else args.train_game)
    vm, vg, vc = load(args.test_reference if args.oracle else args.test_game)
    if sm["rate"] != vm["rate"] or sm["cylinders"] != vm["cylinders"]:
        raise ValueError("incompatible run layouts")
    c, v = controls(sm, sg, reference), controls(vm, vg, reference)
    widths = np.maximum(c.std(axis=0), [0.4, 0.2, 0.1, 0.2, 0.2])
    if args.gates == 1:
        centers, widths = np.empty((1, 0)), np.empty(0)
        c, v = c[:, :0], v[:, :0]
    else:
        center, _ = kmeans2(c[::max(1, len(c)//20000)] / widths, args.gates, minit="++", seed=1, iter=40)
        centers = center * widths
    w, vw = gates(c, centers, widths), gates(v, centers, widths)
    scale = 0.1  # kg/s; feature units (also define the coefficient prior under ridge).
    x, vx = features(sc["exhaustFlow"], args.powers, scale), features(vc["exhaustFlow"], args.powers, scale)
    y, labels = targets(args.train_reference, rm, len(x), args.exits)
    vy, _ = targets(args.test_reference, tm, len(vx), args.exits)
    mask = np.ones(len(x))
    mask[:args.taps] = 0  # zero-start convolution history is not a target for warmed filters.
    print(f"fit {len(x)/sm['rate']:.2f}s; test {len(vx)/sm['rate']:.2f}s; {args.gates} gates; {x.shape[1]} features; {args.taps} taps", flush=True)
    # Diagonal source/gate preconditioning. Undo it in exported coefficients AND in
    # the augmented ridge penalty, preserving the original optimization objective.
    feature_scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-8)
    gate_scale = np.maximum(np.sqrt(np.mean(w*w, axis=0)), 1e-8)
    inverse_scale = 1/(gate_scale[:, None] * feature_scale[None, :])
    penalty_scale = np.r_[np.repeat(inverse_scale.ravel(), args.taps), 1/gate_scale]
    h, b, fitlog = fit(x/feature_scale, w/gate_scale, y, args.taps, args.iterations,
                       args.damping, args.preemphasis, mask,
                       [label.startswith("pressure") and not args.allow_pressure_dc for label in labels], penalty_scale)
    h /= gate_scale[:, None, None, None] * feature_scale[None, None, :, None]
    b /= gate_scale[:, None]
    model = {"rate": int(sm["rate"]), "cylinders": sm["cylinders"], "flowScale": scale,
             "powers": args.powers, "centers": centers.tolist(), "widths": widths.tolist(),
             "kernels": h.tolist(), "biases": b.tolist(), "labels": labels,
             "controlLabels": ["rpm/1000", "manifold/1e5", "wastegate", "throttle", "flap"] if c.shape[1] else []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".model.json").write_text(json.dumps(model, separators=(",", ":"), allow_nan=False))
    prediction = predict(vx, vw, h, b)
    training = predict(x, w, h, b)
    ne = rm["ends"]
    report = {"training": metrics(y[:, :ne], training[:, :ne], sm["rate"], args.taps),
              "test": metrics(vy[:, :ne], prediction[:, :ne], vm["rate"], args.taps),
              "fit": fitlog, "oracle": args.oracle, "modelBytes": Path(str(out)+".model.json").stat().st_size,
              "seconds": time.monotonic()-started}
    segments = []
    t = 0
    for duration, r0, r1, throttle in vm["program"]:
        a, end = round(t*vm["rate"]), round((t+duration)*vm["rate"])
        a = max(a, args.taps)
        segments.append({"start": t, "duration": duration, "rpm": [r0, r1], "throttle": throttle,
                         **metrics(vy[a:end, :ne], prediction[a:end, :ne], vm["rate"])})
        t += duration
    report["segments"] = segments
    if args.exits:
        report["exitRmse"] = {labels[j]: float(np.sqrt(np.mean((vy[:, j]-prediction[:, j])**2))) for j in range(ne, len(labels))}
    Path(str(out)+".report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez(str(out)+".test.npz", reference=vy, prediction=prediction, controls=v, flow=vc["exhaustFlow"], rate=vm["rate"])
    vc["exhaustFlow"].astype("<f4").tofile(str(out)+".flow.f32")
    v.astype("<f4").tofile(str(out)+".controls.f32")
    prediction.astype("<f4").tofile(str(out)+".prediction.f32")
    import soundfile as sf
    for name, value in [("reference", vy), ("prediction", prediction)]:
        from fractions import Fraction
        ratio = Fraction(48000, int(vm["rate"]))
        audio = resample_poly(value[:, :ne].sum(axis=1), ratio.numerator, ratio.denominator)
        sf.write(str(out)+f".{name}.wav", audio, 48000, subtype="FLOAT")
    print(json.dumps({k: report[k] for k in ["training", "test", "modelBytes", "seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
