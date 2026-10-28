#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=''
export DGLBACKEND=pytorch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"${PYTHON:-python}" -B -m unittest discover -s "$ROOT/tests" -v
