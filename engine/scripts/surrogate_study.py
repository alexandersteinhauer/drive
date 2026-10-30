# /// script
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""Independent-trajectory learning curves for the existing causal FIR surrogate.

uv run engine/scripts/surrogate_study.py fit manifest.json out --stage baseline \
    --gates-from public/audio/surrogate/pista-retrained.json --iterations 8000
uv run engine/scripts/surrogate_study.py freeze manifest.json out.model.json frozen.json
uv run engine/scripts/surrogate_study.py evaluate manifest.json frozen.json final --split test

Manifest v1: {config, modelRate:20000, runs:[{id, split, reference, game,
seed?, program?}], stages?:[{id, trainRuns:[id,...]}]}. Capture paths are prefixes,
relative to the manifest or absolute. Splits are train/validation/test/diagnostic.
Only train runs enter fitting/statistics; fit evaluates train and validation, never
opens test/diagnostic captures. Test evaluation requires an explicitly frozen model.

Each run has its OWN zero-start convolution, preemphasis history and taps-sample
warmup. Segments do not reset history. One shared model minimizes
    (N_baseline / sum_run(N_run-taps)) * sum_run ||mask * E(prediction-target)||^2
    + damping^2 * ||original coefficients||^2.
N_baseline defaults to the effective samples of the 'baseline' stage, then the first
stage, then the first train run. --objective-samples fixes it explicitly. Thus one
baseline run matches surrogate_fit.py exactly, and duplicating data changes neither
objective nor prior. Feature/gate diagonal preconditioning and pressure zero-DC
projection are the existing fitter's, including the augmented original-unit ridge.
Completed outputs are saved to .checkpoint.npz, not mid-iteration solver state.
"Converged" means an LSQR residual/least-squares stopping tolerance, not exact
optimal coefficients. Reports include independently recomputed gradient norms.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import LinearOperator, lsqr

_spec = importlib.util.spec_from_file_location("surrogate_fit", Path(__file__).with_name("surrogate_fit.py"))
fitter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fitter)

SPLITS = ("train", "validation", "test", "diagnostic")
STOP_REASONS = {0: "zero_solution", 1: "compatible_residual_tolerance", 2: "least_squares_tolerance",
                3: "condition_limit", 4: "compatible_machine_precision", 5: "least_squares_machine_precision",
                6: "condition_machine_precision", 7: "iteration_limit"}


def json_text(value, *, indent=None):
    return json.dumps(value, sort_keys=True, separators=(",", ":") if indent is None else None,
                      indent=indent, allow_nan=False)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json_text(value, indent=2)+"\n")
    temporary.replace(path)


def load_manifest(path):
    """Validate declarations without opening ANY capture, including held-out files."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("manifest version must be 1")
    manifest["modelRate"] = fitter.integer_rate(manifest.get("modelRate", 20000), "model rate")
    if not 100 <= manifest["modelRate"] <= 192000:
        raise ValueError("model rate must be in [100, 192000]")
    def resolve(value):
        return str((path.parent / value).resolve())
    manifest["config"] = resolve(manifest["config"])
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError("manifest needs runs")
    ids, prefixes, seeds = set(), {}, {}
    for run in runs:
        rid = run.get("id")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("run IDs must be unique nonempty strings")
        ids.add(rid)
        split = run.get("split")
        if split not in SPLITS:
            raise ValueError(f"{rid}: invalid split")
        for key in ("reference", "game"):
            run[key] = resolve(run[key])
            prefix = run[key]
            if prefix in prefixes:
                raise ValueError(f"capture prefix reused: {rid}/{key} and {prefixes[prefix]}")
            prefixes[prefix] = f"{rid}/{key}"
        if "seed" in run and split != "diagnostic":
            seed = run["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"{rid}: seed must be an integer")
            if seed in seeds and seeds[seed] != split:
                raise ValueError(f"burn seed reused across protected splits: {seed}")
            seeds[seed] = split
        if "program" in run:
            fitter.program_array(run)
    stages = manifest.get("stages", [])
    stage_ids = set()
    by_id = {run["id"]: run for run in runs}
    for stage in stages:
        sid, selected = stage.get("id"), stage.get("trainRuns", [])
        if not isinstance(sid, str) or not sid or sid in stage_ids:
            raise ValueError("stage IDs must be unique nonempty strings")
        stage_ids.add(sid)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(f"{sid}: trainRuns must be nonempty and unique")
        for rid in selected:
            if rid not in by_id or by_id[rid]["split"] != "train":
                raise ValueError(f"{sid}: trainRuns may contain only declared train runs: {rid}")
    manifest["_path"], manifest["_sha256"] = str(path), file_digest(path)
    return manifest


def stage_runs(manifest, stage=None):
    if stage is None:
        selected = [run for run in manifest["runs"] if run["split"] == "train"]
    else:
        found = [entry for entry in manifest.get("stages", []) if entry["id"] == stage]
        if not found:
            raise ValueError(f"unknown stage: {stage}")
        by_id = {run["id"]: run for run in manifest["runs"]}
        selected = [by_id[rid] for rid in found[0]["trainRuns"]]
    if not selected:
        raise ValueError("no training runs selected")
    return selected


def split_runs(manifest, split, ids=None):
    selected = [run for run in manifest["runs"] if run["split"] == split]
    if ids is not None:
        by_id = {run["id"]: run for run in selected}
        if len(ids) != len(set(ids)) or any(rid not in by_id for rid in ids):
            raise ValueError(f"selected IDs must be unique declared {split} runs")
        selected = [by_id[rid] for rid in ids]
    return selected


def objective_samples(manifest, taps, explicit=None):
    if explicit is not None:
        if not np.isfinite(explicit) or explicit <= 0:
            raise ValueError("objective samples must be finite and positive")
        return float(explicit), {"explicit": True}
    stages = manifest.get("stages", [])
    baseline = next((stage for stage in stages if stage["id"] == "baseline"), stages[0] if stages else None)
    runs = stage_runs(manifest, baseline["id"]) if baseline else stage_runs(manifest)[:1]
    count = 0
    for run in runs:
        meta = fitter.metadata(run["game"])
        native_count = fitter.trace_frames(run["game"], meta)
        frames = ((native_count-1)*manifest["modelRate"]+meta["rate"]-1)//meta["rate"]
        if frames <= taps:
            raise ValueError(f"{run['id']}: baseline run must exceed warmup")
        count += frames-taps
    return float(count), {"baselineStage": baseline["id"] if baseline else None,
                          "baselineRuns": [run["id"] for run in runs]}


def load_run(run, reference, rate, taps):
    rm, gm = fitter.metadata(run["reference"]), fitter.metadata(run["game"])
    rc, gc = fitter.trace_frames(run["reference"], rm), fitter.trace_frames(run["game"], gm)
    fitter.validate_pair(rm, gm, rc, gc, run["id"])
    if "seed" in run and run["seed"] != gm["seed"]:
        raise ValueError(f"{run['id']}: manifest/capture seed mismatch")
    if "program" in run:
        declared, actual = fitter.program_array(run), fitter.program_array(gm)
        if declared.shape != actual.shape or not np.allclose(declared, actual, rtol=0, atol=1e-12):
            raise ValueError(f"{run['id']}: manifest/capture program mismatch")
    sm, glob, cyl = fitter.load(run["game"])
    flow, control, native_flow, _ = fitter.model_inputs(sm, glob, cyl, reference, rate)
    if len(flow) <= taps:
        raise ValueError(f"{run['id']}: run must exceed warmup")
    y, labels = fitter.targets(run["reference"], rm, len(flow), source_rate=sm["rate"],
                               model_rate=rate, native_count=len(native_flow))
    hashes = {str(Path(run[kind]+suffix)): file_digest(run[kind]+suffix)
              for kind, suffix in (("reference", ".steps.json"), ("reference", ".steps.f32"),
                                   ("reference", ".ends.f32"), ("game", ".steps.json"), ("game", ".steps.f32"))}
    return {"run": run, "meta": gm, "referenceMeta": rm, "flow": flow, "controls": control,
            "y": y, "labels": labels, "nativeFrames": len(native_flow), "captureSha256": hashes}


def assert_independent_seeds(runs, previous=None):
    seen = dict(previous or {})
    for run in runs:
        split, seed = run["run"]["split"], run["meta"]["seed"]
        if split == "diagnostic":
            continue
        if seed in seen and seen[seed] != split:
            raise ValueError(f"actual burn seed reused across protected splits: {seed}")
        seen[seed] = split
    return seen


def assert_layouts(runs, model=None):
    if not runs:
        return
    first = runs[0]
    expected = (first["meta"]["rate"], first["meta"]["cylinders"], first["labels"])
    for run in runs:
        if (run["meta"]["rate"], run["meta"]["cylinders"], run["labels"]) != expected:
            raise ValueError("incompatible run layouts or native rates")
    if model is not None and (model["cylinders"] != expected[1] or model.get("labels", expected[2]) != expected[2]
                              or len(model["kernels"][0]) != len(expected[2])):
        raise ValueError("model/capture output layouts or cylinder counts differ")


def preconditioning(xs, weights, taps):
    """Exactly the old RMS diagonal, extended by pooled training-only moments."""
    n = sum(len(x) for x in xs)
    feature_scale = np.maximum(np.sqrt(sum(np.sum(x*x, axis=0) for x in xs)/n), 1e-8)
    gate_scale = np.maximum(np.sqrt(sum(np.sum(w*w, axis=0) for w in weights)/n), 1e-8)
    inverse = 1/(gate_scale[:, None]*feature_scale[None, :])
    penalty = np.r_[np.repeat(inverse.ravel(), taps), 1/gate_scale]
    return feature_scale, gate_scale, penalty


class IndependentRunsOperator(LinearOperator):
    """Stack independent existing data operators, but augment ridge only ONCE."""
    def __init__(self, xs, weights, masks, taps, preemphasis, diagonal, baseline_samples,
                 *, backend="overlap-add", block_size=None):
        if not xs or not (len(xs) == len(weights) == len(masks)):
            raise ValueError("independent-run array counts differ or are empty")
        nf, ng = xs[0].shape[1], weights[0].shape[1]
        self.size = ng*nf*taps
        self.dimension = self.size+ng
        self.diagonal = np.asarray(diagonal, dtype=float)
        if self.diagonal.shape != (self.dimension,) or not np.isfinite(self.diagonal).all():
            raise ValueError("regularization diagonal dimensions/state")
        self.lengths = [len(x) for x in xs]
        self.effective_samples = sum(np.count_nonzero(mask) for mask in masks)
        if self.effective_samples <= 0 or not np.isfinite(baseline_samples) or baseline_samples <= 0:
            raise ValueError("normalization needs positive baseline/effective samples")
        self.data_scale = float(np.sqrt(baseline_samples/self.effective_samples))
        self.parts = []
        for x, w, mask in zip(xs, weights, masks):
            if x.shape != (len(x), nf) or w.shape != (len(x), ng) or mask.shape != (len(x),):
                raise ValueError("independent-run array dimensions differ")
            if not all(np.isfinite(a).all() for a in (x, w, mask)):
                raise ValueError("nonfinite independent-run data")
            convolution = fitter._CausalConvolution(x, taps, backend, block_size)
            self.parts.append(fitter._RegressionOperator(convolution, w, preemphasis, mask*self.data_scale,
                                                         np.zeros(self.dimension)))
        self._zero_dc = False
        super().__init__(dtype=np.dtype(float), shape=(sum(self.lengths)+self.dimension, self.dimension))

    @property
    def zero_dc(self):
        return self._zero_dc

    @zero_dc.setter
    def zero_dc(self, value):
        self._zero_dc = bool(value)
        for part in self.parts:
            part.zero_dc = self._zero_dc

    def _matvec(self, value):
        value = np.asarray(value).reshape(-1)
        return np.concatenate([part.matvec(value)[:n] for part, n in zip(self.parts, self.lengths)]
                              + [self.diagonal*value])

    def _rmatvec(self, value):
        value = np.asarray(value).reshape(-1)
        gradient = np.zeros(self.dimension)
        offset = 0
        for part, n in zip(self.parts, self.lengths):
            gradient += part.rmatvec(np.r_[value[offset:offset+n], np.zeros(self.dimension)])
            offset += n
        return gradient+self.diagonal*value[offset:]

    def target(self, ys, output):
        return np.concatenate([fitter.lfilter([1, -part.preemphasis], [1], y[:, output])*part.mask
                               for part, y in zip(self.parts, ys)] + [np.zeros(self.dimension)])


class ProgressOperator(LinearOperator):
    def __init__(self, operator, output, every, emit):
        self.operator, self.output, self.every, self.emit = operator, output, every, emit
        self.iterations = 0
        super().__init__(dtype=operator.dtype, shape=operator.shape)

    def _matvec(self, value):
        result = self.operator.matvec(value)
        self.iterations += 1
        if self.every and self.iterations % self.every == 0:
            self.emit({"event": "lsqr_progress", "output": self.output, "iteration": self.iterations})
        return result

    def _rmatvec(self, value):
        return self.operator.rmatvec(value)


def fit_runs(xs, weights, ys, masks, taps, damping, preemphasis, penalty, baseline_samples,
             *, zero_dc=None, iterations=8000, atol=1e-7, btol=1e-7, conlim=1e8,
             backend="overlap-add", block_size=None, progress_every=100, emit=lambda event: None,
             checkpoint=None):
    op = IndependentRunsOperator(xs, weights, masks, taps, preemphasis, damping*penalty,
                                 baseline_samples, backend=backend, block_size=block_size)
    penalty = np.asarray(penalty)
    if penalty.shape != (op.dimension,) or not np.isfinite(penalty).all() or (penalty <= 0).any():
        raise ValueError("positive finite original-coefficient preconditioning diagonal required")
    no, nf, ng = ys[0].shape[1], xs[0].shape[1], weights[0].shape[1]
    if len(ys) != len(xs) or any(y.shape != (n, no) or not np.isfinite(y).all() for y, n in zip(ys, op.lengths)):
        raise ValueError("independent-run target dimensions/state")
    zero_dc = [True]*no if zero_dc is None else zero_dc
    if len(zero_dc) != no:
        raise ValueError("zero-DC flags/output count differ")
    kernels, biases, logs, solutions = [], [], [], []
    for j in range(no):
        op.zero_dc = zero_dc[j]
        rhs = op.target(ys, j)
        emit({"event": "lsqr_start", "output": j, "iterations": iterations, "atol": atol, "btol": btol,
              "effectiveSamples": int(op.effective_samples), "dataScale": op.data_scale})
        progress = ProgressOperator(op, j, progress_every, emit)
        result = lsqr(progress, rhs, atol=atol, btol=btol, conlim=conlim, iter_lim=iterations)
        solution = result[0]
        residual = op.matvec(solution)-rhs
        gradient = op.rmatvec(residual)
        normal_norm, residual_norm = float(np.linalg.norm(gradient)), float(np.linalg.norm(residual))
        rhs_gradient = op.rmatvec(rhs)
        original_normal_norm = float(np.linalg.norm(gradient/penalty))
        data_sse = float(np.dot(residual[:-op.dimension], residual[:-op.dimension]))
        ridge_sse = float(np.dot(residual[-op.dimension:], residual[-op.dimension:]))
        log = {"output": j, "iterations": int(result[2]), "iterationLimit": iterations,
               "stop": int(result[1]), "stopReason": STOP_REASONS[result[1]],
               "converged": result[1] in (0, 1, 2, 4, 5), "iterationLimitReached": result[1] == 7,
               "atol": atol, "btol": btol, "conditionLimit": conlim, "conditionEstimate": float(result[6]),
               "residual": float(result[3]), "normalResidual": float(result[7]),
               "exactResidual": residual_norm, "exactNormalResidual": normal_norm,
               "relativeNormalResidual": normal_norm/max(float(np.linalg.norm(rhs_gradient)), 1e-300),
               "originalCoefficientNormalResidual": original_normal_norm,
               "originalCoefficientRelativeNormalResidual": original_normal_norm/max(float(np.linalg.norm(rhs_gradient/penalty)), 1e-300),
               "operatorNormEstimate": float(result[5]), "preconditionedSolutionNorm": float(result[8]),
               "leastSquaresToleranceRatio": normal_norm/max(float(result[5])*residual_norm, 1e-300),
               "scaledDataSse": data_sse, "originalCoefficientRidgeSse": ridge_sse,
               "objective": data_sse+ridge_sse, "meanWeightedDataSse": data_sse/baseline_samples,
               "zeroDc": bool(zero_dc[j]), "convolution": op.parts[0].convolution.metadata}
        h = solution[:op.size].reshape(ng, nf, taps)
        if zero_dc[j]:
            h = h-h.mean(axis=-1, keepdims=True)
        kernels.append(h)
        biases.append(np.zeros(ng) if zero_dc[j] else solution[op.size:])
        logs.append(log)
        solutions.append(solution)
        if checkpoint:
            checkpoint(np.stack(solutions), logs)
        emit({"event": "lsqr_done", **log})
    return np.stack(kernels, axis=1), np.stack(biases, axis=1), logs


def metric_statistics(target, predicted, rate, start):
    """Sufficient statistics pool runs WITHOUT inventing a boundary in the signal."""
    full_t, full_p = target.sum(axis=1), predicted.sum(axis=1)
    t, p = full_t[start:], full_p[start:]
    et = fitter.lfilter([1, -0.97], [1], full_t)[start:]
    ep = fitter.lfilter([1, -0.97], [1], full_p)[start:]
    values = {"samples": len(t), "sumT": float(t.sum()), "sumP": float(p.sum()),
              "tt": float(t@t), "pp": float(p@p), "tp": float(t@p), "ee": float((t-p)@(t-p)),
              "ett": float(et@et), "eee": float((et-ep)@(et-ep)), "spectra": {}}
    if len(t):
        freq, a = fitter.welch(t, rate, nperseg=min(8192, len(t)))
        _, b = fitter.welch(p, rate, nperseg=min(8192, len(t)))
        _, e = fitter.welch(t-p, rate, nperseg=min(8192, len(t)))
        for lo, hi in fitter.BANDS:
            band = (freq >= lo) & (freq < hi)
            # Duration-weighted per-run PSD integrals; never Welch a concatenation.
            df = freq[1]-freq[0] if len(freq) > 1 else rate
            values["spectra"][f"{lo}-{hi}"] = (np.array([a[band].sum(), b[band].sum(), e[band].sum()])
                                                  *len(t)*df).tolist()
    return values


def pooled_metrics(statistics, rate):
    n = sum(value["samples"] for value in statistics)
    if not n:
        return {"samples": 0, "durationSeconds": 0}
    total = {key: sum(value[key] for value in statistics) for key in ("sumT", "sumP", "tt", "pp", "tp", "ee", "ett", "eee")}
    tt, pp = total["tt"], total["pp"]
    denominator = np.sqrt(max(tt-total["sumT"]**2/n, 0)*max(pp-total["sumP"]**2/n, 0))
    raw = total["ee"]/max(tt, 1e-30)
    emphasized = total["eee"]/max(total["ett"], 1e-30)
    result = {"samples": n, "durationSeconds": n/rate, "rmsReferencePa": float(np.sqrt(tt/n)),
              "rmsPredictionPa": float(np.sqrt(pp/n)), "correlation": float((total["tp"]-total["sumT"]*total["sumP"]/n)/denominator) if denominator else None,
              "rmsRatio": float(np.sqrt(pp/tt)) if tt else None, "levelRatioDb": fitter.db_ratio(pp, tt),
              "waveErrorDb": fitter.db_ratio(total["ee"], tt), "normalizedWaveMse": raw,
              "preemphasizedNormalizedWaveMse": emphasized, "waveScore": (raw+emphasized)/2,
              "bands": {}, "bandWaveErrorDb": {}}
    for lo, hi in fitter.BANDS:
        key = f"{lo}-{hi}"
        a, b, e = np.sum([value["spectra"].get(key, [0, 0, 0]) for value in statistics], axis=0)
        result["bands"][key], result["bandWaveErrorDb"][key] = fitter.db_ratio(b, a), fitter.db_ratio(e, a)
    return result


def equal_metrics(records):
    records = [record for record in records if record["samples"]]
    result = {"count": len(records)}
    for key in ("normalizedWaveMse", "preemphasizedNormalizedWaveMse", "waveScore", "correlation", "levelRatioDb", "waveErrorDb"):
        values = [record[key] for record in records if record.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    for field in ("bands", "bandWaveErrorDb"):
        result[field] = {f"{lo}-{hi}": float(np.mean([record[field][f"{lo}-{hi}"] for record in records]))
                         for lo, hi in fitter.BANDS} if records else {}
    return result


def evaluate_runs(runs, model, rate, powers, scale, centers, widths, h, b, taps, out=None):
    rows, stats = [], {}
    assert_layouts(runs, model)
    for index, run in enumerate(runs):
        x = fitter.features(run["flow"], powers, scale)
        w = fitter.gates(run["controls"], centers, widths)
        prediction = fitter.predict(x, w, h, b)
        y, meta = run["y"], run["meta"]
        row = {key: run["run"][key] for key in ("id", "split", "reference", "game")}
        row.update({"seed": meta["seed"], "program": meta["program"], "nativeFrames": run["nativeFrames"],
                    "axis": fitter.time_axis(len(y), meta["rate"], rate), "captureSha256": run["captureSha256"],
                    "metrics": fitter.metrics(y, prediction, rate, taps), "segments": []})
        for segment, a, end in fitter.segment_ranges(meta["program"], meta["rate"], rate, len(y)):
            a = min(end, max(a, taps))
            r0, r1 = segment["rpm"]
            row["segments"].append({**segment, "startIndex": a, "endIndex": end,
                                     **fitter.metrics(y[:end], prediction[:end], rate, a,
                                                      rpm=r0 if r0 == r1 else None, cylinders=meta["cylinders"])})
        row["segmentScore"] = {"mean": equal_metrics(row["segments"])["waveScore"],
                               "segments": sum(bool(segment["samples"]) for segment in row["segments"])}
        stats[row["id"]] = metric_statistics(y, prediction, rate, taps)
        if out is not None:
            np.savez(str(out)+f".run-{index}.npz", reference=y, prediction=prediction,
                     flow=run["flow"], controls=run["controls"], rate=rate, nativeRate=meta["rate"],
                     firstTime=1/meta["rate"], runId=row["id"])
        rows.append(row)
    aggregate = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        if selected:
            aggregate[split] = {"runs": len(selected),
                                "pooled": pooled_metrics([stats[row["id"]] for row in selected], rate),
                                "equalRun": equal_metrics([row["metrics"] for row in selected]),
                                "equalSegment": equal_metrics([segment for row in selected for segment in row["segments"]])}
    return rows, aggregate


def parameter_digest(model):
    return digest({key: value for key, value in model.items() if key != "study"})


def check_study_model(manifest, model, *, frozen=False):
    study = model.get("study", {})
    if study.get("manifestSha256") != manifest["_sha256"]:
        raise ValueError("study model was not fitted/frozen with this exact manifest")
    if study.get("configSha256") != file_digest(manifest["config"]):
        raise ValueError("reference config changed since fitting")
    if study.get("parameterSha256") and study["parameterSha256"] != parameter_digest(model):
        raise ValueError("study parameters changed since fitting")
    by_id = {run["id"]: run for run in manifest["runs"]}
    for key, split in (("trainingRuns", "train"), ("validationRuns", "validation")):
        if key not in study or (key == "trainingRuns" and not study[key]):
            raise ValueError("missing study training/validation provenance")
        if any(rid not in by_id or by_id[rid]["split"] != split for rid in study[key]):
            raise ValueError("study provenance contains held-out or mislabeled runs")
    if frozen:
        stamp = study.get("frozen", {})
        if not stamp or stamp.get("parameterSha256") != parameter_digest(model):
            raise ValueError("test evaluation requires an unchanged explicitly frozen model")
    return study


def gate_parameters(runs, args, rate):
    if args.gates_from:
        source, source_rate, powers, scale, centers, widths, _, _ = fitter.frozen_parameters(args.gates_from)
        if source_rate != rate or source["cylinders"] != runs[0]["meta"]["cylinders"]:
            raise ValueError("gate source model/capture rates or cylinders differ")
        return powers, scale, centers, widths, {"path": str(Path(args.gates_from).resolve()),
                                                "sha256": file_digest(args.gates_from),
                                                "definition": "fixed centers, widths, powers and flowScale; no reclustering"}
    control = np.concatenate([run["controls"] for run in runs])
    if args.gates > len(control):
        raise ValueError("gate count exceeds training sample count")
    base = np.maximum(control.std(axis=0), [0.4, 0.2, 0.1, 0.2, 0.2])
    if args.gates == 1:
        centers = control.mean(axis=0, keepdims=True)
    else:
        center, _ = kmeans2(control[::max(1, len(control)//20000)]/base, args.gates, minit="++", seed=1, iter=40)
        centers = center*base
    return args.powers, 0.1, centers, base*args.gate_width, {"definition": "training-only kmeans2, seed=1, iter=40"}


def fit_command(args, manifest):
    started = time.monotonic()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(manifest["config"]).read_text())
    rate = manifest["modelRate"]
    validation_declared = split_runs(manifest, "validation", args.validation_runs)
    warmup = max(args.taps, 1024) if args.warmup_samples is None else args.warmup_samples
    if warmup < args.taps:
        raise ValueError("report warmup cannot be shorter than model taps")
    selected = stage_runs(manifest, args.stage)
    nbaseline, normalization_source = objective_samples(manifest, args.taps, args.objective_samples)
    training = [load_run(run, reference, rate, args.taps) for run in selected]
    validation = [load_run(run, reference, rate, args.taps) for run in validation_declared]
    assert_layouts(training+validation)
    seeds = assert_independent_seeds(training+validation)
    powers, scale, centers, widths, gate_source = gate_parameters(training, args, rate)
    xs = [fitter.features(run["flow"], powers, scale) for run in training]
    weights = [fitter.gates(run["controls"], centers, widths) for run in training]
    ys = [run["y"] for run in training]
    masks = [np.r_[np.zeros(args.taps), np.ones(len(x)-args.taps)] for x in xs]
    fs, gs, penalty = preconditioning(xs, weights, args.taps)
    effective = sum(len(x)-args.taps for x in xs)
    optimizer = {"method": "independent-run matrix-free LSQR; one augmented original-coefficient ridge",
                 "iterations": args.iterations, "atol": args.atol, "btol": args.btol, "conditionLimit": args.conlim,
                 "damping": args.damping, "preemphasis": args.preemphasis, "warmupPerRun": args.taps,
                 "normalization": {"baselineSamples": nbaseline, "effectiveSamples": effective,
                                   "dataResidualScale": float(np.sqrt(nbaseline/effective)), **normalization_source,
                                   "definition": "N_baseline/N_effective * summed squared residuals + damping^2 * original-coefficient squared norm"},
                 "featureScale": fs.tolist(), "gateScale": gs.tolist(), "gateSource": gate_source}
    with Path(str(out)+".progress.jsonl").open("w") as progress:
        def emit(event):
            event = {**event, "elapsedSeconds": time.monotonic()-started}
            text = json_text(event)
            progress.write(text+"\n")
            progress.flush()
            print(text, flush=True)
        def checkpoint(solutions, logs):
            path = Path(str(out)+".checkpoint.npz")
            with path.with_name(path.name+".tmp").open("wb") as stream:
                np.savez(stream, solutions=solutions, featureScale=fs, gateScale=gs,
                         metadata=json_text({"manifestSha256": manifest["_sha256"], "stage": args.stage,
                                             "completedOutputs": len(logs), "fit": logs, "optimizer": optimizer,
                                             "note": "Completed output solutions in preconditioned coordinates, not mid-LSQR state."}))
            path.with_name(path.name+".tmp").replace(path)
        h, b, fitlog = fit_runs([x/fs for x in xs], [w/gs for w in weights], ys, masks, args.taps,
                                args.damping, args.preemphasis, penalty, nbaseline,
                                iterations=args.iterations, atol=args.atol, btol=args.btol, conlim=args.conlim,
                                backend=args.backend, block_size=args.block_size,
                                progress_every=args.progress_every, emit=emit, checkpoint=checkpoint)
    h /= gs[:, None, None, None]*fs[None, None, :, None]
    b /= gs[:, None]
    model = {"rate": rate, "nativeRate": training[0]["meta"]["rate"], "cylinders": training[0]["meta"]["cylinders"],
             "flowScale": scale, "powers": powers, "centers": centers.tolist(), "widths": widths.tolist(),
             "kernels": h.tolist(), "biases": b.tolist(), "labels": training[0]["labels"],
             "controlLabels": ["rpm/1000", "manifold/1e5", "wastegate", "throttle", "flap"],
             "study": {"version": 1, "manifest": manifest["_path"], "manifestSha256": manifest["_sha256"],
                       "configSha256": file_digest(manifest["config"]), "stage": args.stage,
                       "trainingRuns": [run["id"] for run in selected],
                       "validationRuns": [run["id"] for run in validation_declared],
                       "fitConverged": all(log["converged"] for log in fitlog), "fit": fitlog, "optimizer": optimizer,
                       "status": "converged" if all(log["converged"] for log in fitlog) else "diagnostic_unconverged",
                       "seedsBySplit": {split: [seed for seed, actual in seeds.items() if actual == split]
                                        for split in ("train", "validation")},
                       "captureSha256": {run["run"]["id"]: run["captureSha256"] for run in training+validation},
                       "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                                    "studySha256": file_digest(__file__), "fitterSha256": file_digest(_spec.origin)}}}
    model["study"]["parameterSha256"] = parameter_digest(model)
    write_json(str(out)+".model.json", model)
    rows, aggregate = evaluate_runs(training+validation, model, rate, powers, scale, centers, widths, h, b,
                                    warmup, out if args.save_predictions else None)
    report = {"version": 1, "role": "training and validation only; test/diagnostic captures unopened",
              "manifest": manifest["_path"], "manifestSha256": manifest["_sha256"], "stage": args.stage,
              "parameterSha256": parameter_digest(model), "fit": fitlog, "optimizer": optimizer, "warmupPerRun": warmup,
              "runs": rows, "aggregate": aggregate, "status": model["study"]["status"], "software": model["study"]["software"], "seconds": time.monotonic()-started,
              "metricsDefinition": "Summed pressure Pa, no gain/alignment. Pooled waveform sufficient statistics; duration-weighted independent Welch spectra. Equal-run/segment are arithmetic metric means, including dB. Score preemphasis=.97, with per-run history only."}
    write_json(str(out)+".report.json", report)
    print(json_text({"report": str(out)+".report.json", "converged": model["study"]["fitConverged"],
                     "validationEqualSegment": aggregate.get("validation", {}).get("equalSegment"),
                     "seconds": report["seconds"]}), flush=True)


def freeze_command(args, manifest):
    if Path(args.out).exists():
        raise ValueError("refusing to overwrite an existing frozen selection")
    model, *_ = fitter.frozen_parameters(args.model)
    study = check_study_model(manifest, model)
    if not study.get("fitConverged") and not args.allow_unconverged:
        raise ValueError("refusing to freeze an unconverged fit (explicit --allow-unconverged required)")
    if study.get("frozen"):
        raise ValueError("model already frozen; do not reselect on final holdout")
    study["frozen"] = {"atUtc": datetime.now(timezone.utc).isoformat(),
                       "parameterSha256": parameter_digest(model), "sourceModelSha256": file_digest(args.model),
                       "sourceModel": str(Path(args.model).resolve()),
                       "selectionSplit": "validation", "allowUnconverged": args.allow_unconverged}
    write_json(args.out, model)
    print(json_text({"frozen": str(Path(args.out).resolve()), "parameterSha256": study["frozen"]["parameterSha256"]}))


def evaluate_command(args, manifest):
    model, rate, powers, scale, centers, widths, h, b = fitter.frozen_parameters(args.model)
    if rate != manifest["modelRate"]:
        raise ValueError("manifest/model rates differ")
    if args.split == "test":
        check_study_model(manifest, model, frozen=True)
    elif "study" in model:
        check_study_model(manifest, model)
    selected = split_runs(manifest, args.split, args.run_ids)
    if not selected:
        raise ValueError(f"no {args.split} runs")
    reference = json.loads(Path(manifest["config"]).read_text())
    taps = max(h.shape[-1], 1024) if args.warmup_samples is None else args.warmup_samples
    if taps < h.shape[-1]:
        raise ValueError("evaluation warmup cannot be shorter than the model's taps")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = [load_run(run, reference, rate, taps) for run in selected]
    seeds = {seed: split for split, values in model.get("study", {}).get("seedsBySplit", {}).items() for seed in values}
    assert_independent_seeds(runs, seeds)
    rows, aggregate = evaluate_runs(runs, model, rate, powers, scale, centers, widths, h, b,
                                    taps, out if args.save_predictions else None)
    report = {"version": 1, "role": "final frozen holdout evaluation; not for selection" if args.split == "test" else f"{args.split} evaluation",
              "manifest": manifest["_path"], "manifestSha256": manifest["_sha256"],
              "model": str(Path(args.model).resolve()), "parameterSha256": parameter_digest(model),
              "frozen": model.get("study", {}).get("frozen"), "warmupPerRun": taps,
              "runs": rows, "aggregate": aggregate}
    write_json(str(out)+".report.json", report)
    print(json_text({"report": str(out)+".report.json", "aggregate": aggregate}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit a train-only stage and score validation; never read final holdout")
    fit.add_argument("manifest")
    fit.add_argument("out")
    fit.add_argument("--stage")
    fit.add_argument("--validation-runs", "--validation-run", nargs="+", help="score only these validation IDs (default all)")
    fit.add_argument("--warmup-samples", type=int, help="report-only warmup, default max(taps,1024); training still excludes taps")
    fit.add_argument("--gates-from", help="reuse exact gates, widths, powers and flow scale from this model")
    fit.add_argument("--gates", type=int, default=6)
    fit.add_argument("--gate-width", type=float, default=1)
    fit.add_argument("--taps", type=int, default=512)
    fit.add_argument("--powers", type=int, nargs="+", default=[1, 2])
    fit.add_argument("--iterations", type=int, default=8000)
    fit.add_argument("--damping", type=float, default=1)
    fit.add_argument("--preemphasis", type=float, default=0.9)
    fit.add_argument("--objective-samples", type=float)
    fit.add_argument("--atol", type=float, default=1e-7)
    fit.add_argument("--btol", type=float, default=1e-7)
    fit.add_argument("--conlim", type=float, default=1e8)
    fit.add_argument("--backend", choices=["overlap-add", "full-fft"], default="overlap-add")
    fit.add_argument("--block-size", type=int)
    fit.add_argument("--progress-every", type=int, default=100, help="LSQR iterations per JSON progress event; 0 disables")
    fit.add_argument("--save-predictions", action="store_true")
    freeze = commands.add_parser("freeze", help="commit the selected model before any final holdout evaluation")
    freeze.add_argument("manifest")
    freeze.add_argument("model")
    freeze.add_argument("out")
    freeze.add_argument("--allow-unconverged", action="store_true")
    evaluate = commands.add_parser("evaluate", help="no fitting or clustering; test requires explicit freeze")
    evaluate.add_argument("manifest")
    evaluate.add_argument("model")
    evaluate.add_argument("out")
    evaluate.add_argument("--split", choices=SPLITS, default="validation")
    evaluate.add_argument("--run-ids", nargs="+", help="evaluate only these IDs from the selected split")
    evaluate.add_argument("--warmup-samples", "--warmup", type=int, help="common evaluation exclusion, default max(taps,1024)")
    evaluate.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "fit":
            if min(args.taps, args.gates, args.iterations, *args.powers) <= 0 or args.progress_every < 0:
                raise ValueError("positive taps/gates/iterations/powers and nonnegative progress interval required")
            if not np.isfinite([args.damping, args.preemphasis, args.atol, args.btol, args.conlim, args.gate_width]).all():
                raise ValueError("fit settings must be finite")
            if min(args.damping, args.atol, args.btol, args.conlim) < 0 or args.gate_width <= 0:
                raise ValueError("nonnegative damping/tolerances/condition limit and positive gate width required")
            fit_command(args, manifest)
        elif args.command == "freeze":
            freeze_command(args, manifest)
        else:
            evaluate_command(args, manifest)
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
