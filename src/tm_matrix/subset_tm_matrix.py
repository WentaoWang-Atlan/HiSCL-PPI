"""Subset a corpus TM matrix (e.g. STRING) to a smaller nested corpus.

Row order of the output equals the TARGET dictionary file order (= target pkl
order), independent of the source order. Target names must be a subset of the
source rows; otherwise the source matrix cannot serve this corpus.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def read_names(dictionary):
    names = []
    with open(dictionary, newline='', encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if len(row) >= 2 and row[0].startswith('9606.'):
                names.append(row[0])
    if not names:
        raise ValueError('No 9606.* names in ' + str(dictionary))
    if len(set(names)) != len(names):
        raise ValueError('Duplicate names in ' + str(dictionary))
    return names


def subset(source, dictionary, output):
    root = Path(source)
    mapping = json.loads((root/'mapping.json').read_text(encoding='utf-8'))
    names = read_names(dictionary)
    src_index = mapping['id_to_row']
    absent = [n for n in names if n not in src_index]
    if absent:
        raise ValueError('{} target names absent from source rows (first 5: {})'
                         .format(len(absent), absent[:5]))
    rows = [src_index[n] for n in names]
    with np.load(root/'tm_matrix.npz', allow_pickle=False) as archive:
        if archive['format'].item() != b'csr':
            raise ValueError('Expected scipy CSR NPZ')
        matrix = sparse.csr_matrix((archive['data'], archive['indices'],
                                    archive['indptr']),
                                   shape=tuple(archive['shape']))
    sub = matrix[np.ix_(rows, rows)].tocsr()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('tm_matrix.npz', 'mapping.json'):
        if (out/name).exists():
            raise FileExistsError('Use a new output directory: ' + str(out/name))
    sparse.save_npz(out/'tm_matrix.npz', sub)
    missing = [n for n in mapping.get('missing_ids', []) if n in set(names)]
    new_mapping = dict(mapping)
    new_mapping.update(row_to_id=names,
                       id_to_row={n: i for i, n in enumerate(names)},
                       count=len(names), missing_ids=missing,
                       order_source='subset_of:{}'.format(root.name))
    (out/'mapping.json').write_text(json.dumps(new_mapping, indent=2)+'\n',
                                    encoding='utf-8')
    print('Subset {} -> {}: {} rows, {} hits ({} without structure)'
          .format(root, out, len(names), sub.nnz, len(missing)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    subset(args.source, args.dictionary, args.output)
