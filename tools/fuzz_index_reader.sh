#!/usr/bin/env bash
# Runs the libFuzzer index reader target for a fixed time, seeded with the golden files.
#   tools/fuzz_index_reader.sh <build-dir of preset linux-clang-fuzz> <seconds> [corpus-dir]
# Crash, leak and timeout inputs are written to ./fuzz-artifacts/. Exit code is non-zero on findings.
set -euo pipefail

build_dir=${1:?build directory}
seconds=${2:?seconds}
corpus=${3:-fuzz-corpus}
root=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$corpus" fuzz-artifacts
cp -n "$root"/tests/data/golden/*.vfidx "$corpus"/ 2>/dev/null || true

export ASAN_OPTIONS="detect_leaks=1:allocator_may_return_null=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
"$build_dir/tests/vf_fuzz_index_reader_libfuzzer" "$corpus" \
  -max_total_time="$seconds" -max_len=131072 -timeout=20 -rss_limit_mb=4096 \
  -artifact_prefix=fuzz-artifacts/ -print_final_stats=1
