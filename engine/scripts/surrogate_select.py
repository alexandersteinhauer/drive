# /// script
# dependencies = ["numpy"]
# ///
"""Select a bounded surrogate candidate set on a separate validation trajectory only.

uv run engine/scripts/surrogate_select.py <native-game.steps.json> <selection.json>
    <candidate-prefix> ...
Each prefix has the fitter's .test.npz and .model.json, where "test" is the validation
trajectory for this stage. Final test data must not be supplied here to choose parameters.

The preregistered score averages raw and preemphasized normalized waveform errors,
then weights each prescribed segment equally. This balances quiet/loud states and
high-frequency detail without normalizing or equalizing the deployed waveform.
"""
import json
from pathlib import Path
import sys

import numpy as np


def score(prefix, meta):
    # Consume the fitter's canonical (start,end] step-end intervals and fixed score,
    # rather than reimplementing its boundary indexing here.
    report = json.loads(Path(prefix+".report.json").read_text())
    value = report["segmentScore"]["mean"]
    if value is None or not np.isfinite(value):
        raise ValueError(f"no finite validation score: {prefix}")
    segments = report["segments"]
    if len(segments) != len(meta["program"]):
        raise ValueError(f"validation segment count mismatch: {prefix}")
    for actual, expected in zip(segments, meta["program"]):
        duration, r0, r1, throttle = expected
        if not np.allclose([actual["duration"], *actual["rpm"], actual["throttle"]],
                           [duration, r0, r1, throttle], rtol=0, atol=1e-10):
            raise ValueError(f"candidate evaluated on another trajectory: {prefix}")
    return {"prefix":prefix, "score":value, "segments":segments}


def main():
    meta = json.loads(Path(sys.argv[1]).read_text())
    candidates = [score(p, meta) for p in sys.argv[3:]]
    candidates.sort(key=lambda x:x["score"])
    result = {"role":"validation only", "criterion":"mean segment 0.5*(raw NMSE + .97 preemphasized NMSE)",
              "winner":candidates[0]["prefix"], "candidates":candidates}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in candidates:
        print(f"{row['prefix']}: {row['score']:.6g}")
    print(f"selected: {result['winner']}")


if __name__ == "__main__":
    main()
