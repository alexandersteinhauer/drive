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
Engine-step flows/controls are tagged at STEP END (first t=1/nativeRate). Raw
flows and all five controls are linearly interpolated exactly as SurrogateStream does,
then signed powers are evaluated. The teacher is anti-aliased onto that SAME physical
time grid, without independently aligning any run or window. The model rate is independent
of the mechanical simulation rate. All exported pressures are unnormalized Pa. --evaluate-model PATH reuses exactly the
same paired preparation/exports without optimizing. It reads rate, gates, powers, scale
and coefficients from that frozen model; the test positional pair may explicitly be a
validation trajectory during candidate selection (all four prefixes are in the report).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import lfilter, resample_poly, welch
from scipy.sparse.linalg import LinearOperator, lsqr


def integer_rate(value, name):
    if not np.isfinite(value) or value <= 0 or int(value) != value:
        raise ValueError(f"{name} must be a positive integer rate")
    return int(value)


def metadata(prefix):
    meta = json.loads(Path(str(prefix) + ".steps.json").read_text())
    meta["rate"] = integer_rate(meta["rate"], "native rate")
    if meta["cylinders"] <= 0 or int(meta["cylinders"]) != meta["cylinders"]:
        raise ValueError("invalid cylinder count")
    meta["cylinders"] = int(meta["cylinders"])
    return meta


def trace_frames(prefix, meta):
    stride = 4 * (len(meta["global"]) + len(meta["cylinder"]) * meta["cylinders"])
    size = Path(str(prefix) + ".steps.f32").stat().st_size
    if not stride or not size or size % stride:
        raise ValueError(f"invalid trace dimensions: {prefix}")
    return size // stride


def load(prefix):
    meta = metadata(prefix)
    ng, nc, n = len(meta["global"]), len(meta["cylinder"]), meta["cylinders"]
    count = trace_frames(prefix, meta)
    raw = np.fromfile(str(prefix) + ".steps.f32", dtype="<f4").reshape(count, ng + nc * n)
    if not np.isfinite(raw).all():
        raise ValueError(f"nonfinite trace: {prefix}")
    glob = {k: raw[:, j].astype(float) for j, k in enumerate(meta["global"])}
    cyl = {k: raw[:, ng + j::nc].astype(float) for j, k in enumerate(meta["cylinder"])}
    return meta, glob, cyl


def fluid_steps(meta):
    """Explicit substeps take precedence; old traces only recorded the fluid rate."""
    rate = integer_rate(meta["rate"], "native rate")
    if "fluidSteps" in meta:
        steps = integer_rate(meta["fluidSteps"], "fluidSteps")
    elif "fluidRate" in meta:
        fr = integer_rate(meta["fluidRate"], "fluid rate")
        if fr % rate:
            raise ValueError("fluid/native rate ratio must be integral")
        steps = fr // rate
    else:
        steps = 12  # Legacy abtrace default, never used for new marked actuator state.
    if "fluidRate" in meta and integer_rate(meta["fluidRate"], "fluid rate") != rate * steps:
        raise ValueError("fluidRate disagrees with fluidSteps and native rate")
    return steps


def program_array(meta):
    p = np.asarray(meta["program"], dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or not len(p) or not np.isfinite(p).all() or (p[:, 0] <= 0).any():
        raise ValueError("invalid prescribed program")
    return p


def validate_pair(reference, source, reference_count, source_count, name):
    """Only compare paired programs/seeds, not the intentionally independent test run."""
    rate = integer_rate(source["rate"], "source native rate")
    if integer_rate(reference["rate"], "teacher native rate") != rate:
        raise ValueError(f"{name}: teacher/source native rates differ")
    if reference["cylinders"] != source["cylinders"]:
        raise ValueError(f"{name}: teacher/source cylinder counts differ")
    rp, sp = program_array(reference), program_array(source)
    if rp.shape != sp.shape or not np.allclose(rp, sp, rtol=0, atol=1e-12):
        raise ValueError(f"{name}: teacher/source prescribed programs differ")
    if reference["seed"] != source["seed"]:
        raise ValueError(f"{name}: teacher/source burn seeds differ")
    expected = sum(int(np.floor(float(duration)*rate + 0.5)) for duration in sp[:, 0])
    if reference_count != source_count or source_count != expected:
        raise ValueError(f"{name}: teacher/source/program frame counts differ ({reference_count}, {source_count}, {expected})")
    fluid_steps(reference)
    fluid_steps(source)
    if "fluidRate" not in reference or reference.get("ends", 0) <= 0:
        raise ValueError(f"{name}: teacher has no pressure outputs/fluid rate")


def validate_runs(reference, source, test_reference, test_source, counts):
    validate_pair(reference, source, counts[0], counts[1], "training")
    validate_pair(test_reference, test_source, counts[2], counts[3], "test")
    if (source["rate"], source["cylinders"], reference["ends"]) != (test_source["rate"], test_source["cylinders"], test_reference["ends"]):
        raise ValueError("incompatible training/test layouts or native rates")


def controls(meta, glob, reference):
    """Five game-available controls, computed at the native rate BEFORE interpolation."""
    if meta.get("surrogateAudio", False):
        # This is the game's actual actuator, not a reconstructed reference filter.
        opening = glob["flap"]
    else:
        flap_spec = reference.get("ducts", reference).get("flaps", [])
        opening = np.zeros_like(glob["rpm"])
        if flap_spec:
            f = flap_spec[0]
            def ramp(x, bounds):
                z = np.clip((x - bounds[0]) / max(bounds[1] - bounds[0], 1e-9), 0, 1)
                return z * z * (3 - 2 * z)
            target = np.maximum(ramp(glob["rpm"] * np.pi / 30, f["speed"]), ramp(glob["throttle"], f["throttle"]))
            steps = fluid_steps(meta)
            alpha = 1 - (1 - min(1 / (meta["rate"] * steps * 0.15), 1)) ** steps
            opening = lfilter([alpha], [1, -(1-alpha)], target)
    return np.stack([glob["rpm"] / 1000, glob["manifold"] / 1e5,
                     glob["wastegate"], glob["throttle"], opening], axis=1)


def linear_resample(values, native_rate, model_rate):
    """SurrogateStream's f32 a + fraction*(b-a), with one future native frame.

    Positions are u=k*native_rate/model_rate, strictly u < native_frames-1. This
    never duplicates/extrapolates the last frame and does not reset at packet edges.
    """
    native_rate = integer_rate(native_rate, "native rate")
    model_rate = integer_rate(model_rate, "model rate")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("linear resampling needs at least two finite native frames")
    count = ((len(values)-1)*model_rate + native_rate-1) // native_rate
    position = np.arange(count, dtype=np.float64) * (native_rate / model_rate)
    left = np.floor(position).astype(np.int64)
    fraction = (position-left).astype(np.float32)[:, None]
    return values[left] + fraction * (values[left+1]-values[left])


def model_inputs(meta, glob, cyl, reference, model_rate):
    native_flow = np.asarray(cyl["exhaustFlow"], dtype=np.float32)
    native_controls = controls(meta, glob, reference).astype(np.float32)
    if native_flow.ndim != 2 or native_flow.shape != (len(native_controls), meta["cylinders"]):
        raise ValueError("native input dimensions disagree")
    sampled = linear_resample(np.concatenate([native_flow, native_controls], axis=1), meta["rate"], model_rate)
    return sampled[:, :meta["cylinders"]], sampled[:, meta["cylinders"]:], native_flow, native_controls


def gates(control, centers, widths):
    d = ((control[:, None, :] - centers[None, :, :]) / widths) ** 2
    logits = -0.5 * d.sum(axis=2)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def features(flow, powers, scale):
    # Use f64 for the regression, AFTER the runtime-equivalent f32 interpolation.
    flow = np.asarray(flow, dtype=float)
    return np.concatenate([np.sign(flow) * abs(flow / scale) ** p for p in powers], axis=1)


def target_grid(meta, source_rate, model_rate):
    """Raw teacher frame zero is at 1/fluidRate; source zero is at 1/sourceRate."""
    fr = integer_rate(meta["fluidRate"], "teacher fluid rate")
    source_rate = integer_rate(source_rate, "source native rate")
    model_rate = integer_rate(model_rate, "model rate")
    if integer_rate(meta["rate"], "teacher native rate") != source_rate:
        raise ValueError("teacher/source native rates differ")
    fluid_steps(meta)
    if fr % source_rate or fr % model_rate:
        raise ValueError("target sampling requires integral fluid/native and fluid/model rate ratios")
    return fr // source_rate - 1, fr // model_rate


def targets(prefix, meta, count, exits=False, *, source_rate=None, model_rate=None, native_count=None):
    source_rate = meta["rate"] if source_rate is None else source_rate
    model_rate = source_rate if model_rate is None else model_rate
    first, stride = target_grid(meta, source_rate, model_rate)
    if exits and model_rate != source_rate:
        raise ValueError("--exits requires model rate == native rate; interpolated exit state averages are not defined")
    ne = meta["ends"]
    p = np.fromfile(str(prefix) + ".ends.f32", dtype="<f4").reshape(-1, ne).astype(float)
    if native_count is not None and len(p) != native_count * fluid_steps(meta):
        raise ValueError("teacher pressure/native frame counts disagree")
    if count <= 0 or first + (count-1)*stride >= len(p) or not np.isfinite(p).all():
        raise ValueError("target grid exceeds measured teacher pressure or contains nonfinite values")
    # Shift the decimator's phase WITHOUT throwing away earlier measured pressure.
    # resample_poly is zero-phase anti-aliasing of the teacher only, with zero padding
    # outside the recording. Input flow/control interpolation never uses a sinc.
    pad = (-first) % stride
    if pad:
        p = np.pad(p, ((pad, 0), (0, 0)))
    start = (first+pad) // stride
    p = resample_poly(p, 1, stride, axis=0)[start:start+count]
    labels = [f"pressure{i}" for i in range(ne)]
    if exits:
        z = np.fromfile(str(prefix) + ".exits.f32", dtype="<f4").reshape(-1, ne, 3).astype(float)
        steps = fluid_steps(meta)
        if len(z) % steps or (native_count is not None and len(z) != native_count*steps):
            raise ValueError("teacher exit/native frame counts disagree")
        # Native-step averages carry step-end timestamps, just like the source controls.
        # Log state AFTER averaging. Different-rate exit-state interpolation is rejected.
        z = z.reshape(-1, steps, ne, 3).mean(axis=1)[:count]
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
        stop_reasons = {0: "zero_solution", 1: "compatible_residual_tolerance", 2: "least_squares_tolerance",
                        3: "condition_limit", 4: "compatible_machine_precision", 5: "least_squares_machine_precision",
                        6: "condition_machine_precision", 7: "iteration_limit"}
        log.append({"iterations": result[2], "iterationLimit": iterations, "stop": result[1],
                    "stopReason": stop_reasons[result[1]], "converged": result[1] in (0, 1, 2, 4, 5),
                    "iterationLimitReached": result[2] >= iterations, "residual": result[3],
                    "normalResidual": float(result[7]), "conditionEstimate": result[6]})
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


BANDS = [(20, 100), (100, 200), (200, 400), (400, 800), (800, 1600),
         (1600, 3200), (3200, 6400), (6400, 9500)]


def db_ratio(numerator, denominator):
    return float(10*np.log10(max(float(numerator), 1e-30) / max(float(denominator), 1e-30)))


def metrics(target, predicted, rate, start=0, *, rpm=None, cylinders=None):
    """Unaligned, unnormalized summed pressure, including per-band waveform residuals."""
    full_t, full_p = target.sum(axis=1), predicted.sum(axis=1)
    t, p = full_t[start:], full_p[start:]
    result = {"samples": len(t), "durationSeconds": len(t)/rate}
    if not len(t):
        return {**result, **dict.fromkeys(["rmsReferencePa", "rmsPredictionPa", "correlation", "rmsRatio", "levelRatioDb", "waveErrorDb", "normalizedWaveMse", "preemphasizedNormalizedWaveMse", "waveScore"]),
                "bands": {}, "bandWaveErrorDb": {}}
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    corr = float(np.corrcoef(t, p)[0, 1]) if t.std() and p.std() else None
    power, predicted_power = np.dot(t, t), np.dot(p, p)
    result.update({"rmsReferencePa": rms(t), "rmsPredictionPa": rms(p), "correlation": corr,
                   "rmsRatio": float(np.sqrt(predicted_power/power)) if power else None,
                   "levelRatioDb": db_ratio(predicted_power, power),
                   "waveErrorDb": db_ratio(np.dot(t-p, t-p), power)})
    # Fixed candidate-selection score. Filter the continuous run BEFORE slicing, so
    # a segment boundary never resets the causal preemphasis state or changes alignment.
    emphasized_t = lfilter([1, -0.97], [1], full_t)[start:]
    emphasized_p = lfilter([1, -0.97], [1], full_p)[start:]
    raw_mse = float(np.dot(t-p, t-p) / max(power, 1e-30))
    emphasized_mse = float(np.dot(emphasized_t-emphasized_p, emphasized_t-emphasized_p) /
                           max(np.dot(emphasized_t, emphasized_t), 1e-30))
    result.update({"normalizedWaveMse": raw_mse, "preemphasizedNormalizedWaveMse": emphasized_mse,
                   "waveScore": (raw_mse+emphasized_mse)/2})
    freq, a = welch(t, rate, nperseg=min(8192, len(t)))
    _, b = welch(p, rate, nperseg=min(8192, len(p)))
    _, e = welch(t-p, rate, nperseg=min(8192, len(p)))
    result["bands"], result["bandWaveErrorDb"] = {}, {}
    for lo, hi in BANDS:
        band = (freq >= lo) & (freq < hi)
        result["bands"][f"{lo}-{hi}"] = db_ratio(b[band].sum(), a[band].sum())
        result["bandWaveErrorDb"][f"{lo}-{hi}"] = db_ratio(e[band].sum(), a[band].sum())
    if rpm is not None and rpm > 0 and len(freq) > 1:
        rotation_hz, df = rpm/60, freq[1]-freq[0]
        half_width = max(df, rotation_hz*0.1)
        details = {"rpm": rpm, "rotationHz": rotation_hz, "firingOrder": cylinders/2,
                   "halfBandwidthHz": half_width, "orders": {}}
        # Half crank orders allow four-stroke periodicity. Skip unresolved short segments.
        if half_width < rotation_hz*0.25:
            for order in np.arange(0.5, 24.5, 0.5):
                f = order*rotation_hz
                if f + half_width >= rate/2:
                    break
                band = abs(freq-f) <= half_width
                details["orders"][f"{order:g}"] = {
                    "frequencyHz": f, "rmsReferencePa": float(np.sqrt(a[band].sum()*df)),
                    "rmsPredictionPa": float(np.sqrt(b[band].sum()*df)),
                    "levelRatioDb": db_ratio(b[band].sum(), a[band].sum()),
                    "waveErrorDb": db_ratio(e[band].sum(), a[band].sum())}
        result["harmonics"] = details
    return result


def segment_ranges(program, native_rate, model_rate, count):
    """Index the canonical step-END axis: segment j covers (start_j, end_j]."""
    native_rate = integer_rate(native_rate, "native rate")
    model_rate = integer_rate(model_rate, "model rate")
    before = 0
    for duration, r0, r1, throttle in program:
        after = before + int(np.floor(duration*native_rate + 0.5))
        # floor((boundary - first_timestamp)*model_rate)+1, using exact integer rates.
        a = max(0, min(count, ((before-1)*model_rate)//native_rate + 1))
        end = max(0, min(count, ((after-1)*model_rate)//native_rate + 1))
        yield {"start": before/native_rate, "duration": (after-before)/native_rate,
               "rpm": [r0, r1], "throttle": throttle}, a, end
        before = after


def time_axis(count, native_rate, model_rate):
    return {"rate": model_rate, "nativeRate": native_rate, "frames": count,
            "firstTime": 1/native_rate, "lastTime": 1/native_rate + (count-1)/model_rate,
            "endExclusive": 1/native_rate + count/model_rate, "durationSeconds": count/model_rate}


def balance_segments(target, program, native_rate, model_rate, skip=0):
    """Training-only joint-pressure RMS weights, not a physical output gain."""
    records = []
    for segment, a, end in segment_ranges(program, native_rate, model_rate, len(target)):
        a = max(a, skip)
        joint_rms = float(np.sqrt(np.mean(np.sum(target[a:end]**2, axis=1)))) if a < end else 0.0
        records.append({**segment, "startIndex": min(a, end), "endIndex": end, "jointRmsPa": joint_rms})
    positive = [r["jointRmsPa"] for r in records if r["jointRmsPa"] > 0]
    median = float(np.median(positive)) if positive else 0.0
    weights = np.ones(len(target))
    for record in records:
        weight = median/record["jointRmsPa"] if record["jointRmsPa"] > 0 else 1.0
        weights[record["startIndex"]:record["endIndex"]] = weight
        record["residualWeight"] = weight
    return weights, {"medianPositiveJointRmsPa": median, "segments": records,
                     "priorImpact": "Residual weights are squared in the least-squares objective; the original-coefficient ridge damping is unchanged. No deployment gain or held-out statistics."}


def frozen_parameters(path):
    """Read immutable runtime parameters. No clustering, feature changes or fitting."""
    model = json.loads(Path(path).read_text())
    rate = integer_rate(model["rate"], "frozen model rate")
    powers = model["powers"]
    cylinders = model["cylinders"]
    scale = model["flowScale"]
    h = np.asarray(model["kernels"], dtype=float)
    centers, widths = np.asarray(model["centers"], dtype=float), np.asarray(model["widths"], dtype=float)
    if cylinders <= 0 or int(cylinders) != cylinders:
        raise ValueError("invalid frozen cylinder count")
    if h.ndim != 4 or not all(h.shape) or h.shape[2] != cylinders*len(powers):
        raise ValueError("invalid frozen kernel dimensions")
    if centers.shape != (h.shape[0], 5) or widths.shape != (5,) or (widths <= 0).any():
        raise ValueError("frozen evaluation requires five controls and positive matching gate widths")
    b = np.asarray(model.get("biases") or np.zeros(h.shape[:2]).tolist(), dtype=float)
    if b.shape != h.shape[:2] or not np.isfinite(h).all() or not np.isfinite(b).all() or not np.isfinite(centers).all() or not np.isfinite(widths).all():
        raise ValueError("invalid frozen model state")
    if not powers or any(p <= 0 or int(p) != p for p in powers) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid frozen feature definition")
    return model, rate, powers, scale, centers, widths, h, b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["train_reference", "train_game", "test_reference", "test_game", "config", "out"]:
        ap.add_argument(key)
    ap.add_argument("--evaluate-model", metavar="PATH", help="no optimizer: use the frozen model's rate, gates, widths, powers and coefficients; fit options have no effect")
    ap.add_argument("--model-rate", type=int, default=20000, help="FIR rate, independent of native mechanical rate (default 20000)")
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--gate-width", type=float, default=1, help="scalar multiplier on Gaussian control widths")
    ap.add_argument("--taps", type=int, default=1024)
    ap.add_argument("--powers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iterations", type=int, default=180)
    ap.add_argument("--damping", type=float, default=0.1)
    ap.add_argument("--preemphasis", type=float, default=0.9)
    ap.add_argument("--balance-segments", action="store_true", help="training-only median/RMS joint-pressure residual weights by prescribed segment")
    ap.add_argument("--exits", action="store_true", help="also fit native-step-averaged exit state (requires model rate == native rate)")
    ap.add_argument("--allow-pressure-dc", action="store_true", help="diagnostic unconstrained baseline; pressure normally has exactly zero DC")
    ap.add_argument("--oracle", action="store_true", help="reference inputs instead of game inputs, to measure deployment mismatch")
    args = ap.parse_args()
    frozen = frozen_parameters(args.evaluate_model) if args.evaluate_model else None
    if frozen:
        model, args.model_rate, args.powers, scale, centers, widths, h, b = frozen
        args.gates, args.taps = h.shape[0], h.shape[-1]
    elif min(args.gates, args.taps, args.iterations, *args.powers) <= 0:
        ap.error("nonpositive gates/taps/iterations/powers")
    if not 100 <= args.model_rate <= 192000:
        ap.error("model rate must be in [100, 192000]")
    if not frozen and (not np.isfinite([args.gate_width, args.damping, args.preemphasis]).all() or args.gate_width <= 0 or args.damping < 0):
        ap.error("gate width must be positive; damping nonnegative; all fit settings finite")
    started = time.monotonic()
    reference = json.loads(Path(args.config).read_text())
    prefixes = [args.train_reference, args.train_game, args.test_reference, args.test_game]
    layouts = [metadata(prefix) for prefix in prefixes]
    counts = [trace_frames(prefix, meta) for prefix, meta in zip(prefixes, layouts)]
    rm, gm, tm, tgm = layouts
    validate_runs(rm, gm, tm, tgm, counts)
    sm, sg, sc = load(args.train_reference if args.oracle else args.train_game)
    vm, vg, vc = load(args.test_reference if args.oracle else args.test_game)
    flow, c, native_flow, native_c = model_inputs(sm, sg, sc, reference, args.model_rate)
    vflow, v, vnative_flow, vnative_c = model_inputs(vm, vg, vc, reference, args.model_rate)
    if len(flow) <= args.taps or len(vflow) <= args.taps:
        raise ValueError("training and test traces must exceed the FIR warm-up length")
    if args.gates > len(flow):
        raise ValueError("gate count exceeds training sample count")
    if frozen:
        if model["cylinders"] != sm["cylinders"]:
            raise ValueError("frozen model/source cylinder counts differ")
    else:
        base_widths = np.maximum(c.std(axis=0), [0.4, 0.2, 0.1, 0.2, 0.2])
        if args.gates == 1:
            # Still export the stream's five-control layout; one Gaussian has weight one.
            centers = c.mean(axis=0, keepdims=True)
        else:
            center, _ = kmeans2(c[::max(1, len(c)//20000)] / base_widths, args.gates, minit="++", seed=1, iter=40)
            centers = center * base_widths
        widths = base_widths * args.gate_width
        scale = 0.1  # kg/s; feature units also define the coefficient prior under ridge.
    w, vw = gates(c, centers, widths), gates(v, centers, widths)
    x, vx = features(flow, args.powers, scale), features(vflow, args.powers, scale)
    if not np.isfinite(x).all() or not np.isfinite(vx).all():
        raise ValueError("nonfinite signed-power input features")
    y, labels = targets(args.train_reference, rm, len(x), args.exits, source_rate=sm["rate"],
                        model_rate=args.model_rate, native_count=len(native_flow))
    vy, test_labels = targets(args.test_reference, tm, len(vx), args.exits, source_rate=vm["rate"],
                              model_rate=args.model_rate, native_count=len(vnative_flow))
    if labels != test_labels:
        raise ValueError("training/test output layouts differ")
    control_labels = ["rpm/1000", "manifold/1e5", "wastegate", "throttle", "flap"]
    print(f"{'evaluate' if frozen else 'fit'} {len(x)/args.model_rate:.2f}s; evaluation pair {len(vx)/args.model_rate:.2f}s; native {sm['rate']:g} Hz; model {args.model_rate} Hz; {args.gates} gates; {x.shape[1]} features; {args.taps} taps", flush=True)
    if frozen:
        if h.shape[1] != y.shape[1] or model.get("labels", labels) != labels:
            raise ValueError("frozen model/teacher output layouts differ (exit models require --exits)")
        fitlog = []
        optimizer = {"method": "none: frozen model evaluation", "evaluateModel": str(Path(args.evaluate_model).resolve()),
                     "gates": args.gates, "taps": args.taps, "powers": args.powers, "flowScale": scale}
    else:
        mask = np.ones(len(x))
        segment_balance = {"enabled": args.balance_segments}
        if args.balance_segments:
            mask, balance_report = balance_segments(y[:, :rm["ends"]], sm["program"], sm["rate"], args.model_rate, args.taps)
            segment_balance.update(balance_report)
        mask[:args.taps] = 0  # zero-start history is not a target for warmed filters.
        # Undo source/gate preconditioning in BOTH coefficients and augmented ridge.
        # This does not whiten lag correlations or promise convergence at an iteration cap.
        feature_scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-8)
        gate_scale = np.maximum(np.sqrt(np.mean(w*w, axis=0)), 1e-8)
        inverse_scale = 1/(gate_scale[:, None] * feature_scale[None, :])
        penalty_scale = np.r_[np.repeat(inverse_scale.ravel(), args.taps), 1/gate_scale]
        h, b, fitlog = fit(x/feature_scale, w/gate_scale, y, args.taps, args.iterations,
                           args.damping, args.preemphasis, mask,
                           [label.startswith("pressure") and not args.allow_pressure_dc for label in labels], penalty_scale)
        h /= gate_scale[:, None, None, None] * feature_scale[None, None, :, None]
        b /= gate_scale[:, None]
        model = {"rate": args.model_rate, "nativeRate": int(sm["rate"]), "cylinders": sm["cylinders"], "flowScale": scale,
                 "powers": args.powers, "centers": centers.tolist(), "widths": widths.tolist(),
                 "kernels": h.tolist(), "biases": b.tolist(), "labels": labels, "controlLabels": control_labels}
        optimizer = {"method": "matrix-free FFT LSQR with augmented original-coefficient ridge",
                     "iterations": args.iterations, "damping": args.damping, "preemphasis": args.preemphasis,
                     "gateWidth": args.gate_width, "segmentBalance": segment_balance}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".model.json").write_text(json.dumps(model, separators=(",", ":"), allow_nan=False))
    prediction = predict(vx, vw, h, b)
    training = predict(x, w, h, b)
    ne = rm["ends"]
    report = {"training": metrics(y[:, :ne], training[:, :ne], args.model_rate, args.taps),
              "test": metrics(vy[:, :ne], prediction[:, :ne], args.model_rate, args.taps),
              "trainingAxis": time_axis(len(x), sm["rate"], args.model_rate),
              "testAxis": time_axis(len(vx), vm["rate"], args.model_rate),
              "trainingNativeFrames": len(native_flow), "testNativeFrames": len(vnative_flow),
              "alignment": {"convention": "step-end; t=1/nativeRate+k/modelRate; no fitted shift, gain, or window alignment",
                            "trainingTeacherFirstIndex": target_grid(rm, sm["rate"], args.model_rate)[0],
                            "testTeacherFirstIndex": target_grid(tm, vm["rate"], args.model_rate)[0],
                            "trainingTeacherStride": target_grid(rm, sm["rate"], args.model_rate)[1],
                            "testTeacherStride": target_grid(tm, vm["rate"], args.model_rate)[1],
                            "targetFilter": "zero-phase scipy resample_poly anti-aliasing, zero padding outside the recording"},
              "heldOut": {"trainingSeed": sm["seed"], "testSeed": vm["seed"],
                          "independentSeed": sm["seed"] != vm["seed"], "independentProgram": sm["program"] != vm["program"]},
              "fit": fitlog, "optimizer": optimizer,
              "prefixes": {key: str(Path(getattr(args, key)).resolve()) for key in
                           ["train_reference", "train_game", "test_reference", "test_game"]},
              "oracle": args.oracle, "modelBytes": Path(str(out)+".model.json").stat().st_size,
              "seconds": time.monotonic()-started}
    segments = []
    for segment, a, end in segment_ranges(vm["program"], int(vm["rate"]), args.model_rate, len(vx)):
        a = min(end, max(a, args.taps))
        r0, r1 = segment["rpm"]
        segments.append({**segment, "startIndex": a, "endIndex": end,
                         "firstTime": 1/vm["rate"] + a/args.model_rate if a < end else None,
                         **metrics(vy[:end, :ne], prediction[:end, :ne], args.model_rate, a,
                                   rpm=r0 if r0 == r1 else None, cylinders=vm["cylinders"])})
    report["segments"] = segments
    scores = [segment["waveScore"] for segment in segments if segment["samples"]]
    report["segmentScore"] = {"mean": float(np.mean(scores)) if scores else None, "segments": len(scores),
                              "preemphasis": 0.97, "definition": "Equal-segment mean of 0.5*(raw normalized waveform MSE + continuously preemphasized normalized waveform MSE). Physical levels, no alignment."}
    if args.exits:
        report["exitRmse"] = {labels[j]: float(np.sqrt(np.mean((vy[:, j]-prediction[:, j])**2))) for j in range(ne, len(labels))}
    Path(str(out)+".report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    input_meta = {**time_axis(len(vflow), vm["rate"], args.model_rate), "nativeFrames": len(vnative_flow),
                  "cylinders": vm["cylinders"], "controls": len(control_labels), "controlLabels": control_labels,
                  "layout": "little-endian float32, frame-major interleaved channels in each file",
                  "modelFlow": out.name+".flow.f32", "modelControls": out.name+".controls.f32",
                  "nativeFlow": out.name+".native-flow.f32", "nativeControls": out.name+".native-controls.f32"}
    Path(str(out)+".inputs.json").write_text(json.dumps(input_meta, indent=2, allow_nan=False))
    np.savez(str(out)+".test.npz", reference=vy, prediction=prediction, controls=v, flow=vflow,
             rate=args.model_rate, nativeRate=vm["rate"], firstTime=1/vm["rate"],
             nativeFlow=vnative_flow, nativeControls=vnative_c)
    vflow.astype("<f4").tofile(str(out)+".flow.f32")
    v.astype("<f4").tofile(str(out)+".controls.f32")
    vnative_flow.astype("<f4").tofile(str(out)+".native-flow.f32")
    vnative_c.astype("<f4").tofile(str(out)+".native-controls.f32")
    prediction.astype("<f4").tofile(str(out)+".prediction.f32")
    import soundfile as sf
    from fractions import Fraction
    ratio = Fraction(48000, args.model_rate)
    for name, value in [("reference", vy), ("prediction", prediction)]:
        audio = resample_poly(value[:, :ne].sum(axis=1), ratio.numerator, ratio.denominator)
        sf.write(str(out)+f".{name}.wav", audio, 48000, subtype="FLOAT")
    print(json.dumps({k: report[k] for k in ["training", "test", "modelBytes", "seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
