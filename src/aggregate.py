"""Summarize completed HiSCL-PPI experiments using test-at-best-validation micro-F1."""
import argparse
import csv
import math
from pathlib import Path
import statistics

from paths import ROOT
from settings import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--expected-seeds', nargs='+', type=int, default=list(range(5)))
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.root.glob('*/PerformMetrics_Metrics.csv')):
        marker = path.parent / 'DONE'
        if not marker.is_file() or marker.read_text().strip() != '0':
            continue
        variant, dataset, split, seed = path.parent.name.split('__')
        if variant not in VARIANTS:
            continue
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        if len(rows) != 1:
            raise ValueError('Expected one metrics row: ' + str(path))
        value = float(rows[0][-2]) * 100
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Invalid test_val: ' + str(path))
        groups.setdefault((variant, dataset, split), {})[int(seed[1:])] = value
    if not groups:
        raise SystemExit('No completed experiments found')
    print('| Method | Dataset | Split | Micro-F1 mean +/- sample std (%) | Seeds | Missing |')
    print('|---|---|---|---:|---:|---|')
    means = {}
    for (variant, dataset, split), values in sorted(groups.items()):
        xs = list(values.values())
        average = statistics.mean(xs)
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        missing = sorted(set(args.expected_seeds) - set(values))
        print('| {} | {} | {} | {:.2f} +/- {:.2f} | {} | {} |'.format(
            variant, dataset, split, average, std, len(values), ','.join(map(str, missing))))
        means.setdefault(variant, []).append((average, not missing))
    print('\nNine-setting means:')
    for variant, cells in sorted(means.items()):
        if len(cells) == 9 and all(complete for _, complete in cells):
            print('{}: {:.2f}'.format(variant, statistics.mean(value for value, _ in cells)))
        else:
            print('{}: incomplete ({}/9 settings)'.format(variant, len(cells)))


if __name__ == '__main__':
    main()
