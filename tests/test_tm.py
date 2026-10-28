"""Synthetic tests for directed target matrices and dictionary-order subsets."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from tm_matrix.build_tm_matrix import build
from tm_matrix.query_tm import TMMatrix
from tm_matrix.subset_tm_matrix import subset


class TMTests(unittest.TestCase):
    def test_order_duplicates_missing_structures_and_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structures = root / 'structures'
            structures.mkdir()
            for name in ('9606.b', '9606.a'):
                (structures / (name + '.pdb')).write_text('fixture')
            dictionary = root / 'dictionary.csv'
            with dictionary.open('w', newline='') as stream:
                csv.writer(stream).writerows((name, 'AA') for name in ('9606.b', '9606.a', '9606.c'))
            alignment = root / 'aln.m8'
            alignment.write_text('9606.a.pdb\t9606.b.pdb\t0.6\n9606.a\t9606.b\t0.8\n'
                                 '9606.b\t9606.a\t0.7\n9606.a\t9606.a\t1\n')
            build(alignment, structures, root / 'matrix', dictionary)
            tm = TMMatrix(root / 'matrix', expected_count=3)
            torch.testing.assert_close(tm.query([1, 0, 2]), torch.tensor([
                [-torch.inf, .8, 0.], [.7, -torch.inf, 0.], [0., 0., -torch.inf]]))
            self.assertEqual(tm.mapping['missing_ids'], ['9606.c'])
            with self.assertRaises(ValueError):
                tm.query([0, 0])
            with self.assertRaises(IndexError):
                tm.query([3])
            target = root / 'target.csv'
            target.write_text('9606.a,AA\n9606.b,AA\n')
            subset(root / 'matrix', target, root / 'subset')
            selected = TMMatrix(root / 'subset', expected_count=2)
            torch.testing.assert_close(selected.query([0, 1]), tm.query([1, 0]))
            self.assertEqual(json.loads((root / 'subset/mapping.json').read_text())['count'], 2)

    def test_unknown_identifier_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '9606.a.pdb').write_text('fixture')
            (root / 'names.csv').write_text('9606.a,AA\n')
            (root / 'aln').write_text('9606.a\t9606.unknown\t0.9\n')
            with self.assertRaises(ValueError):
                build(root / 'aln', root, root / 'out', root / 'names.csv')


if __name__ == '__main__':
    unittest.main()
