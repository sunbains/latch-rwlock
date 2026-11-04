#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Measures each technique's contribution by disabling it, one at a time.
# Usage: scripts/ablation.sh [output.csv]
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-ablation.csv}
work=$(mktemp -d)
bench=$root/build/rw_bench
args="ALength=64 NCSLength=50 WCSLength=4 RCSLength=10 _Duration=2000"
declare -A variants=(
  [default]="" [no_bias]="-DLATCH_BIAS=0" [fenced_bias]="-DLATCH_ASYMMETRIC=0"
  [no_parking]="-DLATCH_PARKING=0" [no_eager_reopen]="-DLATCH_EAGER_REOPEN=0"
  [flat_ticket]="-DLATCH_PROPORTIONAL_TICKET=0" [no_prefetchw]="-DLATCH_PREFETCHW=0"
  [split_indicator]="-DLATCH_SPLIT_INDICATOR=1" [no_pause_calibration]="-DLATCH_CALIBRATE_PAUSE=0"
  [fetch_tdma]="-DLATCH_FETCH_POLICY=1" [fetch_numa]="-DLATCH_FETCH_POLICY=2" [fetch_xadd]="-DLATCH_FETCH_POLICY=3"
  [cpu_rdtscp]="-DLATCH_CPU_SOURCE=2"
)
for name in "${!variants[@]}"; do
  cmake -S "$root" -B "$work/$name" -DLATCH_BUILD_BENCH=OFF -DLATCH_BUILD_TESTS=OFF \
        -DCMAKE_CXX_FLAGS="${variants[$name]}" >/dev/null
  cmake --build "$work/$name" -j16 --target latch_interpose >/dev/null 2>&1 || echo "build failed: $name"
done
run() {  # label preload env-assignment...
  local label=$1 lib=$2; shift 2
  # A missing library is ignored by ld.so, which would record glibc under this label.
  if [[ ! -f "$lib" ]]; then echo "skipping $label: no $lib" >&2; return; fi
  for load in "96 0" "96 1" "96 5" "96 50" "384 5" "384 50" "8 50" "1 100"; do
    set -- "$@"; read -r threads pct <<<"$load"
    for rep in 1 2; do
      ops=$(env "$@" LD_PRELOAD="$lib" timeout 40 "$bench" $args "+$threads@$pct" 2>/dev/null |
            sed -nE 's/^SYNOPSIS.*: ([0-9]+) Ops.*/\1/p')
      echo "$label,$threads,$pct,$rep,${ops:-0}" >>"$out"
    done
  done
}
echo "variant,threads,write_pct,rep,ops" >"$out"
for name in "${!variants[@]}"; do run "$name" "$work/$name/liblatch_interpose.so" X=0; done
run no_admission "$work/default/liblatch_interpose.so" AdmissionLimit=0
run no_node_placement "$work/default/liblatch_interpose.so" NodePlacement=0
rm -rf "$work"
