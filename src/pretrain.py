"""Target-protein pretraining with HCL and soft TM-score supervision."""
import argparse
import csv
import json
from pathlib import Path

from paths import ROOT
from settings import DATASETS, SPLITS, TM_DIRECTORIES, load_settings
import dgl
import torch
from torch.utils.data import DataLoader, Dataset

from dataloader import load_pretrain_data, collate
from models import CodeBook, device
from utils import set_seed
from losses import build_two_view_batch, pyramid, protein_mean_by_node, hcl_loss, tcl_loss
from tm_matrix.query_tm import TMMatrix


class IndexedProteins(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return index, self.data[index]


def indexed_collate(samples):
    indices, graphs = zip(*samples)
    return list(indices), collate(list(graphs))


def baseline_forward(model, graph, capture_unmasked=False):
    """Clone VQ output before CodeBook's e[mask_index]=0 in-place operation.

    A temporary hook sees the original quantized tensor and preserves gradients;
    it neither replaces the VQ output nor runs the encoder/VQ a second time.
    With capture disabled, there is no hook, clone, or extra forward operation.
    """
    captured = []
    handle = None
    if capture_unmasked:
        def capture(_module, _inputs, output):
            captured.append(output[0].clone())
        handle = model.vq_layer.register_forward_hook(capture)
    try:
        z, e, e_q_loss, recon_loss, mask_loss = model(graph)
    finally:
        if handle is not None:
            handle.remove()
    return z, e, e_q_loss, recon_loss, mask_loss, (captured[0] if captured else None)


def training_loss(model, graph, hcl, tm_batch=None):
    use_hcl = hcl['lambda_hcl'] != 0
    use_tm = hcl['lambda_tm'] != 0
    z, _, e_q_loss, recon_loss, mask_loss, unmasked_e = baseline_forward(model, graph, use_tm)
    loss_vae = e_q_loss + recon_loss + mask_loss * model.param['mask_loss']
    # Exact baseline tensor, without even adding floating-point zeros.
    loss = loss_vae
    metrics = dict(vae=loss_vae, e_q=e_q_loss, recon=recon_loss, mask=mask_loss)
    zero = loss_vae.detach().new_zeros(())
    metrics.update(R=zero, S=zero, P=zero, hcl=zero, tm=zero, tm_valid=zero)
    # Codebook survival: fraction of codebook entries used by this forward
    # (diagnostic for the batch-size/codebook-collapse finding).
    surv = torch.unique(model.vq_layer.get_code_indices(z)).numel()
    metrics['codebook_surv'] = torch.tensor(
        surv / model.vq_layer.num_embeddings, device=z.device)
    n = graph.batch_num_nodes()
    if use_hcl:
        views = build_two_view_batch(dgl.unbatch(graph), **hcl['view'])
        h = model.Protein_Encoder.encoding(views)
        quantized, _, _ = model.vq_layer(h)
        u = torch.cat([h, quantized], dim=-1)
        m = int(n.sum())
        r1, r2 = u[:m], u[m:]
        s1, p1 = pyramid(r1, n, hcl['S'])
        s2, p2 = pyramid(r2, n, hcl['S'])
        lh, parts = hcl_loss(r1, r2, s1, s2, p1, p2,
                             taus=tuple(hcl['taus'][key] for key in ('R', 'S', 'P')),
                             lambdas=tuple(hcl['lambdas'][key] for key in ('R', 'S', 'P')),
                             r_max_anchors=hcl.get('r_cap'),
                             generator=hcl.get('generator'))
        metrics.update(parts)
        metrics['hcl'] = lh
        loss = loss + hcl['lambda_hcl'] * lh
    if use_tm:
        if tm_batch is None:
            raise ValueError('TCL enabled but TM batch is missing')
        residues = torch.cat([z, unmasked_e], dim=-1)
        protein = protein_mean_by_node(residues, n)
        lt, targets = tcl_loss(protein, tm_batch, hcl['tau_tm'], hcl['t_min'], hcl['tau_t'])
        metrics['tm'] = lt
        metrics['tm_valid'] = (targets >= 0).sum()
        loss = loss + hcl['lambda_tm'] * lt
    metrics['total'] = loss
    return loss, metrics


def assert_row_order(tm, data_root, dataset, count):
    dictionary = Path(data_root) / 'processed_data' / ('protein.' + dataset + '.sequences.dictionary.csv')
    with dictionary.open(newline='', encoding='utf-8') as stream:
        names = [row[0] for row in csv.reader(stream) if len(row) >= 2 and row[0].startswith('9606.')]
    if len(names) != count or tm.mapping['row_to_id'] != names:
        raise ValueError('TM rows must match the dictionary and cached graph order')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--variant', choices=('full', 'without_hcl', 'without_tcl'), required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--tm-dir', type=Path)
    parser.add_argument('--backbone-config', type=Path)
    parser.add_argument('--method-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    # The shared loader reads the root at import time; the runner sets this env
    # before starting either stage, while direct invocation sets the module value.
    import dataloader
    dataloader.PROCESSED_DATA = data_root / 'processed_data'
    param, hcl = load_settings(args.dataset, args.split, args.seed, args.variant,
                               args.backbone_config, args.method_config)
    if args.output.exists():
        raise FileExistsError('Use a new pretraining output directory')
    cache = data_root / 'processed_data' / (args.dataset + '_protein_graphs.pkl')
    if not cache.is_file():
        raise FileNotFoundError('Build the target protein graph cache first: ' + str(cache))
    set_seed(param['seed'])
    protein_data = load_pretrain_data(args.dataset)
    tm = None
    if hcl['lambda_tm']:
        tm_dir = args.tm_dir or data_root / 'tm_matrix' / TM_DIRECTORIES[args.dataset]
        tm = TMMatrix(tm_dir, expected_count=len(protein_data), t_min=hcl['t_min'])
        assert_row_order(tm, data_root, args.dataset, len(protein_data))
    loader = DataLoader(IndexedProteins(protein_data) if tm else protein_data,
                        batch_size=hcl['batch_size_proteins'], shuffle=True,
                        collate_fn=indexed_collate if tm else collate)
    model = CodeBook(param, DataLoader(protein_data, batch_size=128,
                                      shuffle=False, collate_fn=collate)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['learning_rate']),
                                 weight_decay=float(param['weight_decay']))
    args.output.mkdir(parents=True)
    effective = dict(param=param, method=hcl, variant=args.variant,
                     tm_directory=str(tm_dir) if tm else None,
                     effective_batch_size=hcl['batch_size_proteins'])
    (args.output / 'config.json').write_text(json.dumps(effective, indent=2) + '\n')
    fields = ['epoch', 'batch', 'total', 'vae', 'e_q', 'recon', 'mask',
              'R', 'S', 'P', 'hcl', 'tm', 'tm_valid', 'codebook_surv']
    with (args.output / 'train_log.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, param['pre_epoch'] + 1):
            for iteration, item in enumerate(loader):
                ids, graph = item if tm else (None, item)
                graph = graph.to(device)
                scores = tm.query(ids, device=device) if tm else None
                loss, metrics = training_loss(model, graph, hcl, scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                row = {key: value.detach().item() for key, value in metrics.items()}
                row.update(epoch=epoch, batch=iteration)
                writer.writerow(row)
                if (epoch - 1) % param['log_num'] == 0 and iteration == 0:
                    print(json.dumps(row), flush=True)
                    stream.flush()
                if iteration % 50 == 49 or iteration + 1 == len(loader):
                    torch.cuda.empty_cache()
    torch.save(model.state_dict(), args.output / 'vae_model.ckpt')


if __name__ == '__main__':
    main()
