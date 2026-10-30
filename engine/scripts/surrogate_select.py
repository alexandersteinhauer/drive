# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""Select a bounded surrogate set on a separate validation trajectory only.

uv run engine/scripts/surrogate_select.py <native-game.steps.json> <selection.json>
    <candidate-prefix> ... [--criterion wave|spectrum]
Each prefix has .test.npz/.model.json/.report.json; "test" means the validation
trajectory at this stage. Never supply the final confirmation labels to select weights.

wave: equal-segment raw/.97-preemphasized normalized waveform error.
spectrum: worst then mean absolute global band-level error, with a waveform-correlation
floor (default .7). All candidates share the longest FIR's warm-up exclusion and the
fitter's canonical (start,end] step-end intervals. Neither criterion changes playback gain.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

module_spec = importlib.util.spec_from_file_location("surrogate_fit", Path(__file__).with_name("surrogate_fit.py"))
fitter = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(fitter)


def score(prefix, meta, warmup=None):
    report = json.loads(Path(prefix+".report.json").read_text())
    recorded_segments = report["segments"]
    if len(recorded_segments) != len(meta["program"]):
        raise ValueError(f"validation segment count mismatch: {prefix}")
    for actual, expected in zip(recorded_segments, meta["program"]):
        duration, r0, r1, throttle = expected
        if not np.allclose([actual["duration"], *actual["rpm"], actual["throttle"]],
                           [duration, r0, r1, throttle], rtol=0, atol=1e-10):
            raise ValueError(f"candidate evaluated on another trajectory: {prefix}")
    if warmup is None:
        value = report["segmentScore"]["mean"]
        if value is None or not np.isfinite(value):
            raise ValueError(f"no finite validation score: {prefix}")
        return {"prefix":prefix, "score":value, "segments":recorded_segments}
    model = json.loads(Path(prefix+".model.json").read_text())
    rate = model["rate"]
    labels = model.get("labels", [])
    outputs = sum(s.startswith("pressure") for s in labels) if labels else len(model["kernels"][0])
    if not outputs:
        raise ValueError("model has no pressure outputs")
    with np.load(prefix+".test.npz") as data:
        y, p = data["reference"][:, :outputs], data["prediction"][:, :outputs]
        if "firstTime" in data and not np.isclose(float(data["firstTime"]), 1/meta["rate"], rtol=0, atol=1e-15):
            raise ValueError("different input timestamp convention")
    full = fitter.metrics(y, p, rate, warmup)
    segments = []
    for segment, a, end in fitter.segment_ranges(meta["program"], meta["rate"], rate, len(y)):
        a = min(end, max(a, warmup))
        segments.append({**segment, **fitter.metrics(y[:end], p[:end], rate, a)})
    values = [s["waveScore"] for s in segments if s["samples"]]
    errors = np.abs(list(full["bands"].values()))
    return {"prefix":prefix, "score":float(np.mean(values)), "segments":segments,
            "global":full, "maxAbsBandErrorDb":float(errors.max()),
            "meanAbsBandErrorDb":float(errors.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metadata")
    ap.add_argument("out")
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--criterion", choices=["wave", "spectrum"], default="wave")
    ap.add_argument("--min-correlation", type=float)
    args = ap.parse_args()
    meta = json.loads(Path(args.metadata).read_text())
    warmup = max(len(json.loads(Path(p+".model.json").read_text())["kernels"][0][0][0]) for p in args.prefixes)
    rows = [score(p, meta, warmup) for p in args.prefixes]
    floor = args.min_correlation if args.min_correlation is not None else (0.7 if args.criterion == "spectrum" else -1.)
    for row in rows:
        corr = row["global"]["correlation"]
        row["eligible"] = corr is not None and corr >= floor
        row["rank"] = [row["maxAbsBandErrorDb"], row["meanAbsBandErrorDb"], row["score"]] if args.criterion == "spectrum" else [row["score"]]
    rows.sort(key=lambda r:(not r["eligible"], r["rank"]))
    if not any(r["eligible"] for r in rows):
        raise ValueError("no validation candidate meets the waveform-correlation floor")
    result = {"role":"validation only", "criterion":args.criterion, "minCorrelation":floor,
              "commonWarmupSamples":warmup, "winner":rows[0]["prefix"], "candidates":rows}
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in rows:
        print(f"{row['prefix']}: eligible={row['eligible']} rank={row['rank']}")
    print(f"selected: {result['winner']}")


if __name__ == "__main__":
    main()
