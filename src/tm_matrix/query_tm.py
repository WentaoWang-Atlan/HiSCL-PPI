"""CSR batch lookup without scipy in the training process (numpy/torch only)."""
import json
from pathlib import Path

import numpy as np
import torch


class TMMatrix:
    def __init__(self, directory, expected_count, t_min=0.5):
        root = Path(directory)
        if not (root/'mapping.json').is_file():
            raise FileNotFoundError('Missing TM row mapping: ' + str(root/'mapping.json'))
        self.mapping = json.loads((root/'mapping.json').read_text(encoding='utf-8'))
        names = self.mapping['row_to_id']
        # Order is free (target-corpus rows follow dictionary/pkl order), but
        # it must be unique and consistent with the reverse mapping.
        if len(names) != expected_count or len(set(names)) != len(names):
            raise ValueError('Invalid mapping count/uniqueness')
        if self.mapping['id_to_row'] != {s: i for i, s in enumerate(names)}:
            raise ValueError('TM mapping directions disagree')
        stored_min = self.mapping['t_min']
        if (0 if t_min is None else t_min) < stored_min:
            raise ValueError('Requested threshold below stored threshold; rebuild TM matrix')
        self.t_min = t_min
        with np.load(root/'tm_matrix.npz', allow_pickle=False) as archive:
            if archive['format'].item() != b'csr':
                raise ValueError('Expected scipy CSR NPZ')
            self.data = archive['data'].copy()
            self.indices = archive['indices'].copy()
            self.indptr = archive['indptr'].copy()
            shape = tuple(archive['shape'])
        n = len(names)
        if shape != (n, n) or self.indptr.shape != (n+1,):
            raise ValueError('TM matrix and mapping dimensions disagree')
        if (self.indptr[0] != 0 or self.indptr[-1] != len(self.data)
                or len(self.indices) != len(self.data) or (np.diff(self.indptr) < 0).any()
                or (self.indices < 0).any() or (self.indices >= n).any()
                or not np.isfinite(self.data).all() or (self.data < 0).any()
                or (self.data > 1).any()):
            raise ValueError('Malformed CSR data')
        self.count = n

    def query(self, batch_ids, device=None):
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.detach().cpu().tolist()
        ids = list(batch_ids)
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= self.count for i in ids):
            raise IndexError('Batch protein ID outside TM mapping')
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate protein IDs would create self-positive pairs')
        result = np.zeros((len(ids), len(ids)), dtype=np.float32)
        column = {value: j for j, value in enumerate(ids)}
        for row, source in enumerate(ids):
            start, end = self.indptr[source:source+2]
            for target, value in zip(self.indices[start:end], self.data[start:end]):
                j = column.get(int(target))
                if j is not None and (self.t_min is None or value >= self.t_min):
                    result[row, j] = value
        np.fill_diagonal(result, -np.inf)
        return torch.as_tensor(result, device=device)
