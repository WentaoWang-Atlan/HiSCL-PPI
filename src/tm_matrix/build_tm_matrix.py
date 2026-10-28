"""Build a directed sparse matrix from Foldseek query,target,qtmscore hits."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def corpus_names(names_file):
    """Read identifiers in the exact processed protein dictionary order."""
    import csv
    names = []
    with open(names_file, newline='', encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if len(row) >= 2 and row[0].startswith('9606.'):
                names.append(row[0])
    if not names:
        raise ValueError('No 9606.* names in ' + str(names_file))
    return names, 'names_file'


def build(alignment, structures, output, names_file, t_min=0.5):
    if not 0 <= t_min <= 1:
        raise ValueError('t_min must be in [0,1]')
    names, order_source = corpus_names(names_file)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate names in ' + str(names_file))
    present = {p.name.removesuffix('.pdb') for p in Path(structures).iterdir()
               if p.is_file()}
    missing = [name for name in names if name not in present]
    if len(missing) == len(names):
        raise ValueError('No dictionary identifier has a structure in ' + str(structures))
    index = {name: i for i, name in enumerate(names)}
    lookup = dict(index)
    for name, i in index.items():
        lookup[name + '.pdb'] = i

    hits = {}
    with open(alignment, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 3:
                if not fields:
                    continue
                raise ValueError('Expected 3 columns at line {}'.format(line_no))
            i = lookup.get(fields[0], -1)
            j = lookup.get(fields[1], -1)
            if i < 0 or j < 0:
                raise ValueError('Foldseek ID absent from name mapping: '
                                 + fields[0] + ' / ' + fields[1])
            t = float(fields[2])
            if not np.isfinite(t) or t < 0:
                raise ValueError('Invalid TM-score at line {}'.format(line_no))
            if t > 1.0:
                t = 1.0
            if i != j and t >= t_min:
                hits[i, j] = max(t, hits.get((i, j), 0.))
    rows, cols, values = [], [], []
    for (i, j), t in hits.items():
        rows.append(i); cols.append(j); values.append(t)
    matrix = sparse.csr_matrix((np.asarray(values, dtype=np.float32), (rows, cols)),
                               shape=(len(names), len(names)))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('tm_matrix.npz', 'mapping.json'):
        if (out/name).exists():
            raise FileExistsError('Use a new output directory: ' + str(out/name))
    sparse.save_npz(out/'tm_matrix.npz', matrix)
    mapping = dict(row_to_id=names, id_to_row=index, t_min=t_min,
                   count=len(names), directed=True, duplicate_policy='max',
                   missing_value=0.0, clamped_over_unity=True,
                   order_source=order_source, missing_ids=missing,
                   numpy_version=np.__version__)
    (out/'mapping.json').write_text(json.dumps(mapping, indent=2)+'\n', encoding='utf-8')
    print('Stored {} directed hits for {} proteins ({} without structure)'
          .format(matrix.nnz, len(names), len(missing)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', required=True)
    parser.add_argument('--structures', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.alignment, args.structures, args.output, args.dictionary)
