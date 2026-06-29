#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build"

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"
cmake --build "${build_dir}" --parallel

ctest --test-dir "${build_dir}" --output-on-failure

"${build_dir}/operators/transpose/transpose_benchmark" \
    --csv --check > "${repo_root}/docs/transpose_benchmark_results.csv"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    python3 "${repo_root}/operators/transpose/benchmark/plot.py" \
        "${repo_root}/docs/transpose_benchmark_results.csv" \
        --output "${repo_root}/docs/images/transpose_bandwidth.png"
else
    echo "matplotlib is not installed; skipping plot generation"
fi

echo "Results: ${repo_root}/docs/transpose_benchmark_results.csv"
