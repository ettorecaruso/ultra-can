#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${1:-configs/experiments.yaml}"
case "${CONFIG_PATH}" in
    /*) CONFIG_ABS="${CONFIG_PATH}" ;;
    *)  CONFIG_ABS="${REPO_ROOT}/${CONFIG_PATH}" ;;
esac

SPLIT_ARG=""
if [ "$#" -ge 2 ]; then
    for arg in "${@:2}"; do
        if [[ "$arg" == "--split" ]] && [ "$#" -ge 3 ]; then
            SPLIT_VALUE="${3}"
            if [[ "$SPLIT_VALUE" =~ ^(train|val|test)$ ]]; then
                SPLIT_ARG="--split $SPLIT_VALUE"
                echo "INFO: generating split '$SPLIT_VALUE' only"
                break
            else
                echo "ERROR: --split must be train, val or test" >&2
                exit 1
            fi
        fi
    done
fi

if [ ! -f "${CONFIG_ABS}" ]; then
    echo "ERROR: config not found: '${CONFIG_PATH}' (resolved to '${CONFIG_ABS}')" >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "INFO: starting dataset generation"
echo "INFO: config = ${CONFIG_ABS}"
if [ -n "$SPLIT_ARG" ]; then
    echo "INFO: split limited to ${SPLIT_ARG#--split }"
fi

python3 -c "
from pathlib import Path
from src.experiments.pipeline import prepare_all_datasets
import sys
config_path = Path('${CONFIG_ABS}')
split = None
if '${SPLIT_ARG}' != '':
    split = '${SPLIT_ARG#--split }'
prepare_all_datasets(config_path, split=split)
"

echo "INFO: dataset generation completed"
exit 0
