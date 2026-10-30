#!/usr/bin/env bash
# Experimental exhaust-only surrogate pipeline. Writes data/weights ONLY to the supplied
# directory, never opts a game car into an unvalidated model. Python always runs with uv.
#   bash engine/scripts/surrogate-pilot.sh /absolute/output/directory
# Native+WASM model output is pre-directivity pressure (Pa), not normalized game samples.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?supply an output directory (renders are large and take minutes)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
bun -e 'import {ferrari488Pista} from "./src/sim/config.ts"; await Bun.write(process.argv[1],JSON.stringify(ferrari488Pista.powertrain.engine.spec));' "$OUT/spec.json"
(cd engine && cargo build --release --features parallel --example abtrace --example surrogate)
B="${CARGO_TARGET_DIR:-$ROOT/engine/target}/release/examples/abtrace"
R="$ROOT/engine/scripts/pista-reference.json"
TRAIN='2:1000:1000:0,1:1000:3000:1,2:3000:3000:1,0.5:3000:4500:1,2:4500:4500:1,0.5:4500:6000:1,2:6000:6000:1,0.5:6000:7500:1,2:7500:7500:1,1:7500:4500:0,1:4500:4500:0.3,0.8:4500:4500:0.65,1:4500:4500:1,1:4500:1000:0,1:1000:1000:0'
TEST='1.5:1000:1000:0,1.5:1000:3800:0.8,1:3800:3800:1,1.5:3800:7000:1,0.3:7000:5000:0,0.8:5000:5000:1,0.5:5000:5000:0,1:5000:2500:0,1:2500:2500:0.25,1:2500:5000:0.6,1:5000:1000:0,1:1000:1000:0'
# Independent trajectories AND independent cycle-indexed random draws in the held-out run.
for mode in lumped ducted; do
    tag=game; [[ "$mode" == ducted ]] && tag=ref
    SEED=17 DUCT_THREADS="${DUCT_THREADS:-8}" "$B" "$OUT/spec.json" "$R" "$OUT/train-$tag" "$mode" "$TRAIN" > "$OUT/train-$tag.log"
    SEED=239 DUCT_THREADS="${DUCT_THREADS:-8}" "$B" "$OUT/spec.json" "$R" "$OUT/test-$tag" "$mode" "$TEST" > "$OUT/test-$tag.log"
done
uv run engine/scripts/surrogate_fit.py "$OUT/train-ref" "$OUT/train-game" "$OUT/test-ref" "$OUT/test-game" "$R" "$OUT/model" \
    --gates "${GATES:-6}" --taps "${TAPS:-1024}" --iterations "${ITERATIONS:-350}" --powers 1 2 > "$OUT/fit.log"
"${CARGO_TARGET_DIR:-$ROOT/engine/target}/release/examples/surrogate" "$OUT/model.model.json" "$OUT/model.flow.f32" "$OUT/model.controls.f32" "$OUT/native.f32" --bench > "$OUT/native.log"
bash engine/build.sh > "$OUT/wasm-build.log" 2>&1
bun engine/scripts/surrogate-wasm.ts "$OUT/model.model.json" "$OUT/model.flow.f32" "$OUT/model.controls.f32" "$OUT/wasm.f32" --bench > "$OUT/wasm.log"
echo "Model, metrics, A/B pressure WAVs, native and WASM outputs: $OUT"
cat "$OUT/native.log" "$OUT/wasm.log"
