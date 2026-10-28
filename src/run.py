"""Run one implemented HiSCL-PPI experiment with isolated configuration, outputs, and seed."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from paths import ROOT, BACKBONE_SOURCE
from settings import DATASETS, SPLITS, VARIANTS, TM_DIRECTORIES, baseline_batch, load_settings


def validate_inputs(data_root, dataset, split, needs_tm, tm_dir):
    processed = data_root / 'processed_data'
    required = [dataset + '_' + split + '.json', dataset + '_ppi.pkl',
                dataset + '_ppi_label.pkl', dataset + '_protein_graphs.pkl',
                'protein.' + dataset + '.sequences.dictionary.csv']
    for name in required:
        if not (processed / name).is_file():
            raise FileNotFoundError('Required prepared input: ' + str(processed / name))
    partitions = json.loads((processed / (dataset + '_' + split + '.json')).read_text())
    groups = [set(partitions[key]) for key in ('train_index', 'val_index', 'test_index')]
    if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, validation, and test edge splits must be nonempty and disjoint')
    if needs_tm:
        for name in ('mapping.json', 'tm_matrix.npz'):
            if not (tm_dir / name).is_file():
                raise FileNotFoundError('Required target TM matrix: ' + str(tm_dir / name))


def run_experiment(dataset, split, seed, variant, gpu, data_root, output_root,
                   backbone_path=None, method_path=None, tm_dir=None):
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    tm_dir = Path(tm_dir).resolve() if tm_dir else data_root / 'tm_matrix' / TM_DIRECTORIES[dataset]
    backbone_path = Path(backbone_path or ROOT / 'configs/backbone.json').resolve()
    method_path = Path(method_path or ROOT / 'configs/method.json').resolve()
    backbone, method = load_settings(dataset, split, seed, variant, backbone_path, method_path)
    validate_inputs(data_root, dataset, split, bool(method['lambda_tm']), tm_dir)
    run_id = '{}__{}__{}__s{}'.format(variant, dataset, split, seed)
    directory = output_root / run_id
    signature_data = dict(backbone=backbone, method=method, variant=variant,
                          data_root=str(data_root), tm_dir=str(tm_dir),
                          baseline_batch=baseline_batch(dataset, split, seed))
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode())
    for folder in (ROOT / 'src', BACKBONE_SOURCE):
        for source in sorted(folder.rglob('*.py')):
            signature.update(source.relative_to(ROOT).as_posix().encode())
            signature.update(source.read_bytes())
    for input_path in (data_root / 'processed_data' / (dataset + '_' + split + '.json'),
                       data_root / 'processed_data' / ('protein.' + dataset + '.sequences.dictionary.csv')):
        signature.update(input_path.read_bytes())
    if method['lambda_tm']:
        signature.update((tm_dir / 'mapping.json').read_bytes())
    digest = signature.hexdigest()
    if directory.exists():
        if (directory / 'DONE').is_file() and (directory / 'signature').is_file():
            if (directory / 'DONE').read_text().strip() == '0' and (directory / 'signature').read_text() == digest:
                print('Already completed: ' + run_id, flush=True)
                return directory
        raise FileExistsError('Run exists with incomplete or different settings; use a new output root: ' + str(directory))
    directory.mkdir(parents=True)
    (directory / 'work').mkdir()
    (directory / 'signature').write_text(digest)
    config = json.loads(backbone_path.read_text())
    config[dataset][split] = backbone
    local_backbone = directory / 'backbone.json'
    local_backbone.write_text(json.dumps(config, indent=2) + '\n')
    local_method = directory / 'method.json'
    local_method.write_text(json.dumps(method, indent=2) + '\n')
    (directory / 'experiment.json').write_text(json.dumps(signature_data, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if str(gpu) == 'cpu' else str(gpu),
               PPI_DATA_ROOT=str(data_root), PYTHONDONTWRITEBYTECODE='1')
    # Keep dynamic-library lookup consistent with the activated interpreter.
    conda_lib = Path(sys.prefix) / 'lib'
    if conda_lib.is_dir():
        env['LD_LIBRARY_PATH'] = str(conda_lib) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    downstream = [sys.executable, '-B', str(BACKBONE_SOURCE / 'train.py'),
                  '--dataset', dataset, '--split_mode', split,
                  '--param-config', str(local_backbone), '--pretrain-batch',
                  str(baseline_batch(dataset, split, seed) if variant == 'baseline' else 128)]
    with (directory / 'run.log').open('w') as log:
        if variant != 'baseline':
            pretrain = [sys.executable, '-B', str(ROOT / 'src/pretrain.py'),
                        '--dataset', dataset, '--split', split, '--seed', str(seed),
                        '--variant', variant, '--data-root', str(data_root),
                        '--backbone-config', str(local_backbone), '--method-config', str(local_method),
                        '--tm-dir', str(tm_dir), '--output', str(directory / 'pretrain')]
            subprocess.run(pretrain, cwd=directory / 'work', env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            downstream += ['--ckpt_path', str(directory / 'pretrain/vae_model.ckpt')]
        subprocess.run(downstream, cwd=directory / 'work', env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    metrics_path = directory / 'PerformMetrics_Metrics.csv'
    with metrics_path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 1 or not math.isfinite(float(rows[0][-2])) or not 0 <= float(rows[0][-2]) <= 1:
        raise ValueError('Expected one finite test_val metric in ' + str(metrics_path))
    (directory / 'DONE').write_text('0\n')
    print('Completed: ' + run_id, flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--gpu', default='0', help='GPU index, or cpu')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--backbone-config', type=Path)
    parser.add_argument('--method-config', type=Path)
    parser.add_argument('--tm-dir', type=Path)
    args = parser.parse_args()
    run_experiment(args.dataset, args.split, args.seed, args.variant, args.gpu,
                   args.data_root, args.output_root, args.backbone_config, args.method_config, args.tm_dir)


if __name__ == '__main__':
    main()
