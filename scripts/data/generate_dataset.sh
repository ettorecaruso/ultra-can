#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${1:-configs/experiments.yaml}"
case "${CONFIG_PATH}" in
    /*) CONFIG_ABS="${CONFIG_PATH}" ;;
    *)  CONFIG_ABS="${REPO_ROOT}/${CONFIG_PATH}" ;;
esac

EXTRA_ARGS=()
if [ "$#" -gt 1 ]; then
    EXTRA_ARGS=("${@:2}")
fi

if [ ! -f "${CONFIG_ABS}" ]; then
    echo "ERROR: config not found: '${CONFIG_PATH}' (resolved to '${CONFIG_ABS}')" >&2
    exit 1
fi

GENERATOR_SCRIPT="src/data/dataset_generator.py"
if [ ! -f "${REPO_ROOT}/${GENERATOR_SCRIPT}" ]; then
    echo "ERROR: generator not found: '${REPO_ROOT}/${GENERATOR_SCRIPT}'" >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "INFO: starting dataset generation"
echo "INFO: config     = ${CONFIG_ABS}"
echo "INFO: generator  = ${GENERATOR_SCRIPT}"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "INFO: extra args = ${EXTRA_ARGS[*]}"
fi

python "${GENERATOR_SCRIPT}" --config "${CONFIG_ABS}" "${EXTRA_ARGS[@]}"

echo "INFO: dataset generation completed"
exit 0
