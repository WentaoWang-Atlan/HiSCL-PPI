#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/build_tm.sh STRUCTURE_DIR DICTIONARY_CSV OUTPUT_DIR" >&2
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STRUCTURES=$(cd "$1" && pwd)
DICTIONARY=$2
OUTPUT=$3
FOLDSEEK=${FOLDSEEK:-foldseek}
THREADS=${THREADS:-8}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/pdb"
shopt -s nullglob
for pdb in "$STRUCTURES"/*.pdb; do
  ln -s "$pdb" "$WORK/pdb/$(basename "$pdb")"
done
if [[ -z "$(ls -A "$WORK/pdb")" ]]; then
  echo 'No .pdb structures found' >&2
  exit 2
fi
"$FOLDSEEK" easy-search "$WORK/pdb" "$WORK/pdb" "$WORK/aln.m8" "$WORK/tmp" \
  --format-output 'query,target,qtmscore' -e 0.1 --threads "$THREADS"
"${PYTHON:-python}" -B "$ROOT/src/tm_matrix/build_tm_matrix.py" \
  --alignment "$WORK/aln.m8" --structures "$STRUCTURES" \
  --dictionary "$DICTIONARY" --output "$OUTPUT"
