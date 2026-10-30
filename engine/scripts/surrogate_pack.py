# /// script
# dependencies = []
# ///
"""Pack model JSON decimals without changing any runtime f32 coefficient.

uv run engine/scripts/surrogate_pack.py <input.json> <output.json>
Nine significant decimal digits round-trip IEEE binary32. This is NOT quantization of
runtime weights, and it does not alter labels, control conventions or model dimensions.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(value):
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k)+":"+encode(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(map(encode, value)) + "]"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite model coefficient")
        return format(f32(value), ".9g")
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def verify(before, after):
    if isinstance(before, dict):
        assert before.keys() == after.keys()
        for k in before:
            verify(before[k], after[k])
    elif isinstance(before, list):
        assert len(before) == len(after)
        for a, b in zip(before, after):
            verify(a, b)
    elif isinstance(before, float):
        assert struct.pack("<f", before) == struct.pack("<f", after)
    else:
        assert before == after


def main():
    before = json.loads(Path(sys.argv[1]).read_text())
    result = encode(before)
    verify(before, json.loads(result))
    Path(sys.argv[2]).write_text(result)
    print(f"{len(result)} bytes, runtime f32 weights bit-identical; sha256 {hashlib.sha256(result.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
