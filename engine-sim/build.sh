#!/usr/bin/env bash
# Builds engine-sim (https://github.com/ange-yaghi/engine-sim, MIT) to WebAssembly for the
# engine sound: the simulation core, its script compiler (piranha) and bridge.cpp, with the
# engine-sim script library and our engine scripts embedded.
#
# Needs: emscripten, bison >= 3.2 and flex (brew install emscripten bison flex).
# Output: src/audio/engine-sim.mjs (glue) and public/audio/engine-sim.wasm.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
CACHE="$HERE/.cache"
ES="$CACHE/engine-sim"
OUT="$ROOT/src/audio"
COMMIT=85f7c3b959a908ed5232ede4f1a4ac7eafe6b630
PIRANHA_COMMIT=432f0b122bb1663b686c553c7e7269300afac3bc
SCS_COMMIT=e009f4ff1c9c4c5874e865e893cdb62e208fb2b3

BISON="${BISON:-$(brew --prefix bison 2>/dev/null)/bin/bison}"
FLEX="${FLEX:-$(brew --prefix flex 2>/dev/null)/bin/flex}"
FLEX_INCLUDE="$(dirname "$(dirname "$FLEX")")/include"

fetch() { # dir url commit
  if [ ! -d "$1/.git" ]; then
    git init -q "$1"
    git -C "$1" remote add origin "$2"
  fi
  if [ "$(git -C "$1" rev-parse -q --verify HEAD || true)" != "$3" ]; then
    git -C "$1" fetch -q --depth 1 origin "$3"
    git -C "$1" checkout -q --force "$3"
  fi
}

mkdir -p "$CACHE"
fetch "$ES" https://github.com/ange-yaghi/engine-sim.git "$COMMIT"
PIRANHA="$ES/dependencies/submodules/piranha"
SCS="$ES/dependencies/submodules/simple-2d-constraint-solver"
fetch "$PIRANHA" https://github.com/ange-yaghi/piranha.git "$PIRANHA_COMMIT"
fetch "$SCS" https://github.com/ange-yaghi/simple-2d-constraint-solver.git "$SCS_COMMIT"

git -C "$ES" checkout -q -- include src scripting
git -C "$ES" apply "$HERE/engine-sim.patch"

# The parser includes "../include/...", so it is generated next to piranha's include/.
GEN="$PIRANHA/autogen"
mkdir -p "$GEN"
"$BISON" -o "$GEN/parser.auto.cpp" --defines="$GEN/parser.auto.h" "$PIRANHA/flex-bison/specification.y"
"$FLEX" -o "$GEN/scanner.auto.cpp" --header-file="$GEN/scanner.auto.h" "$PIRANHA/flex-bison/scanner.l"

CORE=(audio_buffer camshaft crankshaft combustion_chamber connecting_rod convolution_filter
  cylinder_bank cylinder_head delay_filter derivative_filter direct_throttle_linkage dynamometer
  engine exhaust_system feedback_comb_filter filter fuel function gas_system gaussian_filter
  governor ignition_module impulse_response intake jitter_filter leveling_filter low_pass_filter
  part piston piston_engine_simulator simulator standard_valvetrain starter_motor synthesizer
  throttle transmission utilities valvetrain vehicle vehicle_drag_constraint vtec_valvetrain)
SOURCES=("$HERE/bridge.cpp" "$GEN/parser.auto.cpp" "$GEN/scanner.auto.cpp")
for f in "${CORE[@]}"; do SOURCES+=("$ES/src/$f.cpp"); done
SOURCES+=("$ES"/scripting/src/*.cpp "$PIRANHA"/src/*.cpp "$SCS"/src/*.cpp)

OBJ="$CACHE/obj"
mkdir -p "$OBJ" "$OUT" "$CACHE/node"
FLAGS=(-O3 -std=c++17 -msimd128 -DNDEBUG
  -I"$HERE/shim" -I"$ES" -I"$GEN" -I"$PIRANHA/include" -I"$FLEX_INCLUDE" -I"$ES/dependencies/submodules"
  "-D__forceinline=inline __attribute__((always_inline))" "-D__int64=long long"
  -fdelayed-template-parsing -Wno-everything)

# Compile in parallel; each object depends only on its source and the flags.
obj() { echo "$OBJ/$(basename "$1" .cpp)-$(echo "$1" | shasum | cut -c1-8).o"; }
{
  echo '#!/bin/sh'
  # Skip objects newer than their source, this script and the patch (headers only change with it).
  echo '[ "$2" -nt "$1" ] && [ "$2" -nt "'"$HERE/build.sh"'" ] && [ "$2" -nt "'"$HERE/engine-sim.patch"'" ] && exit 0'
  printf 'exec em++'; printf ' %q' "${FLAGS[@]}"; echo ' -c "$1" -o "$2"'
} > "$CACHE/cc.sh"
chmod +x "$CACHE/cc.sh"
for s in "${SOURCES[@]}"; do printf '%s\n%s\n' "$s" "$(obj "$s")"; done |
  xargs -P "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" -n 2 "$CACHE/cc.sh"

OBJECTS=()
for s in "${SOURCES[@]}"; do OBJECTS+=("$(obj "$s")"); done

link() { # output environment
  em++ "${OBJECTS[@]}" -O3 -msimd128 -o "$1" \
    -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT="$2" \
    -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=64MB \
    -sEXPORTED_FUNCTIONS=_es_init,_es_set_controls,_es_set_volume,_es_set_gain,_es_render,_es_engine_rpm,_es_overrun,_es_dyno_torque,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPF32,stringToNewUTF8 \
    --embed-file "$ES/es@/es" \
    --embed-file "$HERE/engines@/engines" \
    --exclude-file '*.md'
}

# For the AudioWorklet ("shell": no DOM, no Node); the page fetches the wasm and hands it over.
link "$OUT/engine-sim.mjs" shell
mkdir -p "$ROOT/public/audio"
mv "$OUT/engine-sim.wasm" "$ROOT/public/audio/engine-sim.wasm"
# For render.mjs, the offline check under Node.
link "$CACHE/node/engine-sim.mjs" node

ls -la "$OUT/engine-sim.mjs" "$ROOT/public/audio/engine-sim.wasm"
