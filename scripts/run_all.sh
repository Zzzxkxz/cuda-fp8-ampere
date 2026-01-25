#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmake -S "$root_dir" -B "$root_dir/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root_dir/build" -j

"$root_dir/build/gpu_bench" --list
"$root_dir/build/gpu_bench" --bench tex
"$root_dir/build/gpu_bench" --bench lop3
"$root_dir/build/gpu_bench" --bench transpose
