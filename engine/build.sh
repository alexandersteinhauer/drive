#!/usr/bin/env bash
# Builds the engine to WebAssembly: src/sim/engine.wasm (used by the game and its tests).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
RUSTFLAGS="-C target-feature=+simd128,+bulk-memory,+nontrapping-fptoint,+sign-ext" \
  cargo build --release --target wasm32-unknown-unknown --lib
WASM=target/wasm32-unknown-unknown/release/engine_sim.wasm
OUT="$HERE/../src/sim/engine.wasm"
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext "$WASM" -o "$OUT"
else
  cp "$WASM" "$OUT"
fi
ls -l "$OUT" | awk '{print $5, $9}'
