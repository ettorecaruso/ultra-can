#!/usr/bin/env bash

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
for script in \
    scripts/figures/make_fig_architecture.py \
    scripts/figures/make_fig_ber_full_linear.py \
    scripts/figures/make_fig_peer_ranging.py \
    scripts/figures/make_fig_retraining_control.py \
    scripts/figures/make_fig_velocity_track.py
do
    echo "-- ${script}"
    "${PYTHON}" "${script}"
done

echo "== done: figures/ and results/full/tables/ are up to date =="
