#!/usr/bin/env bash
#
# Reproduce every figure and table of the paper from the existing results.
# Run the experiments first (see README); this script only reads results/full
# and writes into figures/ and results/full/tables/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"

echo "== tables =="
for script in scripts/tables/make_table_*.py; do
    echo "-- ${script}"
    "${PYTHON}" "${script}"
done

echo "== diagnostics (model-free controls used by the text) =="
for script in \
    scripts/diagnostics/channel_label_observability.py \
    scripts/diagnostics/observability_table.py \
    scripts/diagnostics/scene_report.py \
    scripts/diagnostics/lpi_detectability.py
do
    echo "-- ${script}"
    "${PYTHON}" "${script}"
done

echo "== figures =="
for script in scripts/figures/make_fig_*.py; do
    echo "-- ${script}"
    "${PYTHON}" "${script}"
done

echo "== done: figures/ and results/full/tables/ are up to date =="
