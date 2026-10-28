"""Execute the implemented baseline, full model, and two component ablations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path
from queue import Empty, Queue

from paths import ROOT
from settings import DATASETS, SPLITS, VARIANTS
from run import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated GPU indices, or cpu')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--backbone-config', type=Path)
    parser.add_argument('--method-config', type=Path)
    args = parser.parse_args()
    gpus = args.gpu_ids.split(',')
    if len(set(gpus)) != len(gpus) or any(g != 'cpu' and not g.isdigit() for g in gpus):
        raise ValueError('Use unique numeric GPU indices, or cpu')
    tasks = list(itertools.product(args.variants, args.datasets, args.splits, args.seeds))
    if len(set(tasks)) != len(tasks):
        raise ValueError('Duplicate experiment tasks')
    jobs = Queue()
    for task in tasks:
        jobs.put(task)

    def worker(gpu):
        failures = []
        while True:
            try:
                variant, dataset, split, seed = jobs.get_nowait()
            except Empty:
                return failures
            try:
                run_experiment(dataset, split, seed, variant, gpu, args.data_root,
                               args.output_root, args.backbone_config, args.method_config)
            except Exception as error:
                failure = '{} {} {} s{}: {}'.format(variant, dataset, split, seed, error)
                print(failure, flush=True)
                failures.append(failure)
            finally:
                jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        failures = [failure for batch in executor.map(worker, gpus) for failure in batch]
    if failures:
        raise SystemExit('{} experiments failed; inspect run-local logs'.format(len(failures)))


if __name__ == '__main__':
    main()
