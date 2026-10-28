"""Exercise all implemented variants through the actual command-line runner on CPU."""
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paths import ROOT
from settings import VARIANTS, baseline_batch, load_settings
import dgl
import torch
from tm_matrix.build_tm_matrix import build


class PipelineTests(unittest.TestCase):
    def test_final_config_and_ablation_switches(self):
        expected = {'baseline': (0., 0.), 'full': (1., .3),
                    'without_hcl': (0., .3), 'without_tcl': (1., 0.)}
        for variant in VARIANTS:
            backbone, method = load_settings('STRING', 'bfs', 4, variant)
            self.assertEqual(backbone['seed'], 4)
            self.assertEqual((method['lambda_hcl'], method['lambda_tm']), expected[variant])
            self.assertEqual(method['batch_size_proteins'], 128)
            self.assertEqual(method['r_cap'], 12000)
        self.assertEqual(baseline_batch('STRING', 'random', 0), 512)
        self.assertEqual(baseline_batch('STRING', 'random', 3), 128)

    def test_all_variants_and_result_aggregation(self):
        with tempfile.TemporaryDirectory(prefix='hiscl_fixture_') as directory:
            root = Path(directory)
            processed = root / 'data/processed_data'
            processed.mkdir(parents=True)
            torch.manual_seed(17)
            graphs = []
            for n in (8, 9, 10, 11):
                edges = [(j, j + 1) for j in range(n - 1)] + [(j + 1, j) for j in range(n - 1)]
                graph = dgl.heterograph({('amino_acid', e, 'amino_acid'): edges
                                        for e in ('SEQ', 'STR_KNN', 'STR_DIS')})
                graph.ndata['x'] = torch.randn(n, 7)
                graphs.append(graph)
            for name, value in {
                'SHS27k_protein_graphs.pkl': graphs,
                'SHS27k_ppi.pkl': [[0, 1], [1, 2], [2, 3], [0, 3]],
                'SHS27k_ppi_label.pkl': [[1, 0, 0, 0, 0, 0, 0]] * 4,
            }.items():
                with (processed / name).open('wb') as stream:
                    pickle.dump(value, stream)
            (processed / 'SHS27k_random.json').write_text(json.dumps(
                dict(train_index=[0, 1], val_index=[2], test_index=[3])))
            dictionary = processed / 'protein.SHS27k.sequences.dictionary.csv'
            dictionary.write_text(''.join('9606.P{},AAAA\n'.format(i) for i in range(4)))
            structures = root / 'structures'
            structures.mkdir()
            for i in range(4):
                (structures / ('9606.P{}.pdb'.format(i))).write_text('fixture')
            alignment = root / 'aln.m8'
            alignment.write_text('9606.P0 9606.P1 .8\n9606.P1 9606.P0 .8\n'
                                 '9606.P2 9606.P3 .9\n9606.P3 9606.P2 .9\n')
            build(alignment, structures, root / 'data/tm_matrix/shs27k_af2_qtm', dictionary)
            config = json.loads((ROOT / 'configs/backbone.json').read_text())
            config['SHS27k']['random'].update(prot_hidden_dim=8, num_embeddings=16,
                ppi_hidden_dim=8, prot_num_layers=2, pre_epoch=2, max_epoch=2, log_num=1)
            config_path = root / 'backbone.json'
            config_path.write_text(json.dumps(config))
            # An explicit test override keeps the fixture fast; shipped settings are unchanged.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            commands = []
            for variant in VARIANTS:
                cmd = [sys.executable, '-B', str(ROOT / 'src/run.py'),
                       '--dataset', 'SHS27k', '--split', 'random', '--seed', '0',
                       '--variant', variant, '--gpu', 'cpu', '--data-root', str(root / 'data'),
                       '--output-root', str(root / 'runs'), '--backbone-config', str(config_path)]
                commands.append(cmd)
                result = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True)
                if result.returncode:
                    logs = list((root / 'runs').glob('*/run.log'))
                    self.fail(result.stderr + '\n' + '\n'.join(path.read_text() for path in logs))
                run = root / 'runs' / (variant + '__SHS27k__random__s0')
                self.assertEqual((run / 'DONE').read_text().strip(), '0')
                with (run / 'PerformMetrics_Metrics.csv').open() as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(len(rows), 1)
                self.assertTrue(0 <= float(rows[0][-2]) <= 1)
                if variant != 'baseline':
                    effective = json.loads((run / 'pretrain/config.json').read_text())
                    self.assertEqual(effective['effective_batch_size'], 128)
            repeat = subprocess.run(commands[1], cwd=root, env=env, text=True, capture_output=True, check=True)
            self.assertIn('Already completed', repeat.stdout)
            summary = subprocess.run([sys.executable, '-B', str(ROOT / 'src/aggregate.py'),
                                      '--root', str(root / 'runs'), '--expected-seeds', '0'],
                                     cwd=root, env=env, text=True, capture_output=True, check=True)
            for variant in VARIANTS:
                self.assertIn('| ' + variant + ' |', summary.stdout)


if __name__ == '__main__':
    unittest.main()
