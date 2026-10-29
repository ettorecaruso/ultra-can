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
    scripts/diagnostics/lpi_detectability.py \
    scripts/diagnostics/ber_gap_ci.py
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

echo "== microcontroller deployment table (needs a captured firmware log) =="
if compgen -G "results/full/mcu_bench/*.log" > /dev/null; then
    "${PYTHON}" src/experiments/mcu_footprint.py --logs results/full/mcu_bench/*.log
else
    echo "-- skipped: no results/full/mcu_bench/*.log (flash firmware/esp32 first)"
fi

echo "== done: figures/ and results/full/tables/ are up to date =="
