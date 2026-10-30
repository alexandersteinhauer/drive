#!/usr/bin/env bash
# Experimental, deployment-matched exhaust-only training pipeline. Large files stay outside
# the repo; nothing automatically replaces the in-game asset. Python always runs with uv.
#   bash engine/scripts/surrogate-pilot.sh /absolute/output/directory
# Optional fit settings: GATES,TAPS,ITERATIONS,GATE_WIDTH,PREEMPHASIS,BALANCE_SEGMENTS=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?supply an output directory (renders are large and take minutes)}"
mkdir -p "$OUT/data"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
bun -e 'import {withPistaSurrogateExhaust} from "./src/sim/config.ts"; const c=withPistaSurrogateExhaust(); if(c.powertrain.type!=="combustion"||c.powertrain.engine.type!=="simulated") throw Error("Pista"); await Bun.write(process.argv[1],JSON.stringify(c.powertrain.engine.spec));' "$OUT/spec.json"
(cd engine && cargo build --release --features parallel --example abtrace --example surrogate)
B="${CARGO_TARGET_DIR:-$ROOT/engine/target}/release/examples/abtrace"
R="$ROOT/engine/scripts/pista-reference.json"
TRAIN='1.5:1000:1000:0,1:1000:1800:0.2,1:1800:1800:0.2,0.8:1800:2500:0.45,1.2:2500:2500:0.45,0.5:2500:3300:0.8,1.5:3300:3300:1,0.4:3300:4200:1,1.5:4200:4200:1,0.4:4200:5200:1,1.5:5200:5200:1,0.4:5200:6500:1,1.5:6500:6500:1,0.4:6500:7500:1,1.5:7500:7500:1,0.25:7500:5800:0,0.6:5800:5800:1,0.4:5800:5800:0,0.8:5800:3200:0,1:3200:3200:0.2,1:3200:3200:0.55,0.7:3200:3200:1,0.7:3200:1800:0,1:1800:1800:0.1,0.7:1800:2800:0.7,1.3:2800:2800:1,0.6:2800:1000:0,1:1000:1000:0'
VAL='1:1000:1000:0,1:1000:2800:0.65,0.7:2800:2800:0.35,1.4:2800:4000:1,0.8:4000:6200:1,0.25:6200:4400:0,0.6:4400:4400:1,0.8:4400:2400:0,0.6:2400:2400:0.25,0.8:2400:1000:0,0.5:1000:1000:0'
TEST='1.5:1000:1000:0,1.5:1000:3800:0.8,1:3800:3800:1,1.5:3800:7000:1,0.3:7000:5000:0,0.8:5000:5000:1,0.5:5000:5000:0,1:5000:2500:0,1:2500:2500:0.25,1:2500:5000:0.6,1:5000:1000:0,1:1000:1000:0'
REV=$(git rev-parse HEAD)
# Student uses stock game physics and actual 10 kHz/four-substep sampling. Teacher adds only the
# corrected exhaust, with 24 substeps for 240 kHz fluid output. No boost/C2C parameter overrides.
for mode in lumped ducted; do
    tag=game; fluid=4
    if [[ "$mode" == ducted ]]; then tag=ref; fluid=24; fi
    for split in train validation test; do
        seed=17; program="$TRAIN"
        if [[ "$split" == validation ]]; then seed=73; program="$VAL"; fi
        if [[ "$split" == test ]]; then seed=239; program="$TEST"; fi
        SOURCE_REVISION="$REV" EXHAUST_ONLY=1 FREQ=10000 FLUID="$fluid" SEED="$seed" DUCT_THREADS="${DUCT_THREADS:-8}" \
            "$B" "$OUT/spec.json" "$R" "$OUT/data/$split-$tag" "$mode" "$program" > "$OUT/data/$split-$tag.log"
    done
done
FIT=(--model-rate 20000 --gates "${GATES:-6}" --taps "${TAPS:-1024}" --iterations "${ITERATIONS:-350}" \
    --gate-width "${GATE_WIDTH:-1}" --preemphasis "${PREEMPHASIS:-0.9}" --powers 1 2)
if [[ "${BALANCE_SEGMENTS:-0}" == 1 ]]; then FIT+=(--balance-segments); fi
# Validation is the positional "test" dataset during fitting; final test is untouched.
uv run engine/scripts/surrogate_fit.py "$OUT/data/train-ref" "$OUT/data/train-game" "$OUT/data/validation-ref" "$OUT/data/validation-game" "$R" "$OUT/model" "${FIT[@]}" > "$OUT/fit.log"
uv run engine/scripts/surrogate_fit.py "$OUT/data/train-ref" "$OUT/data/train-game" "$OUT/data/test-ref" "$OUT/data/test-game" "$R" "$OUT/model-test" \
    --evaluate-model "$OUT/model.model.json" > "$OUT/test.log"
"${CARGO_TARGET_DIR:-$ROOT/engine/target}/release/examples/surrogate" "$OUT/model.model.json" "$OUT/model-test.flow.f32" "$OUT/model-test.controls.f32" "$OUT/native.f32" --bench > "$OUT/native.log"
bash engine/build.sh > "$OUT/wasm-build.log" 2>&1
bun engine/scripts/surrogate-wasm.ts "$OUT/model.model.json" "$OUT/model-test.flow.f32" "$OUT/model-test.controls.f32" "$OUT/wasm.f32" --bench > "$OUT/wasm.log"
bun engine/scripts/surrogate-stream-wasm.ts "$OUT/model.model.json" "$OUT/model-test.native-flow.f32" "$OUT/model-test.native-controls.f32" \
    10000 44100 "$OUT/stream.f32" --bench > "$OUT/stream.log"
echo "Physical-pressure A/B WAVs, held-out reports, model and native/WASM checks: $OUT"
cat "$OUT/native.log" "$OUT/wasm.log" "$OUT/stream.log"
