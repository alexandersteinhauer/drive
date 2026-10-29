#!/usr/bin/env bash
# Builds the original engine-sim (pinned, with engine-sim.patch) natively as a reference for
# the Rust port, compiled so its arithmetic matches (no fused multiply-add) and with rand()
# replaced by the port's generator, then runs it on vw1200.mr: .build/{beetle.json,trace.txt}.
# Needs bison >= 3.2 and flex (brew install bison flex). Then: cargo test --release.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/.build"
ES="$OUT/engine-sim"
PIRANHA="$ES/dependencies/submodules/piranha"
SCS="$ES/dependencies/submodules/simple-2d-constraint-solver"
GEN="$PIRANHA/autogen"
BISON="${BISON:-$(brew --prefix bison)/bin/bison}"
FLEX="${FLEX:-$(brew --prefix flex)/bin/flex}"
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
mkdir -p "$OUT"
fetch "$ES" https://github.com/ange-yaghi/engine-sim.git 85f7c3b959a908ed5232ede4f1a4ac7eafe6b630
fetch "$PIRANHA" https://github.com/ange-yaghi/piranha.git 432f0b122bb1663b686c553c7e7269300afac3bc
fetch "$SCS" https://github.com/ange-yaghi/simple-2d-constraint-solver.git e009f4ff1c9c4c5874e865e893cdb62e208fb2b3
# Fixes for clang, and single-threaded synthesizer entry points.
git -C "$ES" checkout -q -- include src scripting
git -C "$ES" apply "$HERE/engine-sim.patch"
mkdir -p "$GEN"
"$BISON" -o "$GEN/parser.auto.cpp" --defines="$GEN/parser.auto.h" "$PIRANHA/flex-bison/specification.y" 2>/dev/null
"$FLEX" -o "$GEN/scanner.auto.cpp" --header-file="$GEN/scanner.auto.h" "$PIRANHA/flex-bison/scanner.l"

OUT="$HERE/.build"
mkdir -p "$OUT/obj"

FLAGS=(-O2 -std=c++17 -ffp-contract=off -fdelayed-template-parsing -DNDEBUG -w
  -I"$HERE/shim" -I"$ES" -I"$GEN" -I"$PIRANHA/include" -I"$FLEX_INCLUDE"
  -I"$ES/dependencies/submodules" -I"$HERE"
  "-D__forceinline=inline __attribute__((always_inline))" "-D__int64=long long")

CORE=(audio_buffer camshaft crankshaft combustion_chamber connecting_rod convolution_filter
  cylinder_bank cylinder_head delay_filter derivative_filter direct_throttle_linkage dynamometer
  engine exhaust_system feedback_comb_filter filter fuel function gas_system gaussian_filter
  governor ignition_module impulse_response intake jitter_filter leveling_filter low_pass_filter
  part piston piston_engine_simulator simulator standard_valvetrain starter_motor synthesizer
  throttle transmission utilities valvetrain vehicle vehicle_drag_constraint vtec_valvetrain)

jobs=()
compile() { # source object extra-flags...
  local src="$1" obj="$2"; shift 2
  if [ ! "$obj" -nt "$src" ] || [ "$HERE/build.sh" -nt "$obj" ] || [ "$HERE/engine-sim.patch" -nt "$obj" ]; then
    clang++ "${FLAGS[@]}" "$@" -c "$src" -o "$obj" &
    jobs+=($!)
  fi
}
OBJS=()
for f in "${CORE[@]}"; do
  extra=(-include "$HERE/rand_shim.h")
  [ "$f" = synthesizer ] && extra+=(-DES_RAND_SYNTH)
  compile "$ES/src/$f.cpp" "$OUT/obj/es_$f.o" "${extra[@]}"
  OBJS+=("$OUT/obj/es_$f.o")
done
for src in "$ES"/scripting/src/*.cpp "$PIRANHA"/src/*.cpp "$SCS"/src/*.cpp "$GEN"/parser.auto.cpp "$GEN"/scanner.auto.cpp; do
  obj="$OUT/obj/$(basename "$src" .cpp)-$(echo "$src" | shasum | cut -c1-8).o"
  compile "$src" "$obj"
  OBJS+=("$obj")
done
compile "$HERE/reference.cpp" "$OUT/obj/reference.o"
OBJS+=("$OUT/obj/reference.o")
for j in ${jobs[@]+"${jobs[@]}"}; do wait "$j"; done
clang++ "${OBJS[@]}" -o "$OUT/reference"
# engine-sim finds its library (es/) relative to the working directory.
(cd "$ES" && "$OUT/reference" "$HERE/vw1200.mr" "$OUT/beetle.json" "$OUT/trace.txt")
echo "wrote $OUT/beetle.json, $OUT/trace.txt"
