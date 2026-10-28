"""Build target-benchmark graph caches before running parallel experiments."""
import argparse
import os
from pathlib import Path

from paths import ROOT
from settings import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    processed = args.data_root.resolve() / 'processed_data'
    os.environ['PPI_DATA_ROOT'] = str(args.data_root.resolve())
    from dataloader import load_pretrain_data, load_data
    for dataset in args.datasets:
        cache = processed / (dataset + '_protein_graphs.pkl')
        if cache.is_file():
            print(dataset + ': existing graph cache retained')
        else:
            for kind, suffix in (('nodes', 'pt'), ('knn.edges', 'npy'), ('rball.edges', 'npy')):
                path = processed / ('protein.' + kind + '.' + dataset + '.' + suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
            graphs = load_pretrain_data(dataset)
            print(dataset + ': built ' + str(len(graphs)) + ' graphs')
        if not all((processed / (dataset + suffix)).is_file()
                   for suffix in ('_ppi.pkl', '_ppi_label.pkl')):
            for name in (dataset + '_random.json', 'protein.actions.' + dataset + '.txt',
                         'protein.' + dataset + '.sequences.dictionary.csv'):
                if not (processed / name).is_file():
                    raise FileNotFoundError('Required input for PPI cache preparation: ' + str(processed / name))
            load_data(dataset, 'random', 0)
            print(dataset + ': prepared PPI pair and label caches')


if __name__ == '__main__':
    main()
