"""Shared configuration and the four implemented HCL/soft-TCL experiment settings."""
import copy
import json
from pathlib import Path

from paths import ROOT

DATASETS = ('SHS27k', 'SHS148k', 'STRING')
SPLITS = ('random', 'dfs', 'bfs')
VARIANTS = ('baseline', 'full', 'without_hcl', 'without_tcl')
TM_DIRECTORIES = {'SHS27k': 'shs27k_af2_qtm',
                  'SHS148k': 'shs148k_af2_qtm', 'STRING': 'string_af2_qtm'}


def load_settings(dataset, split, seed, variant, backbone_path=None, method_path=None):
    if dataset not in DATASETS or split not in SPLITS or variant not in VARIANTS:
        raise ValueError('Unsupported dataset, split, or implemented variant')
    backbone_path = Path(backbone_path or ROOT / 'configs/backbone.json')
    method_path = Path(method_path or ROOT / 'configs/method.json')
    backbone = copy.deepcopy(json.loads(backbone_path.read_text())[dataset][split])
    backbone.update(dataset=dataset, split_mode=split, seed=seed)
    method = json.loads(method_path.read_text())
    if variant in ('baseline', 'without_hcl'):
        method['lambda_hcl'] = 0.0
    if variant in ('baseline', 'without_tcl'):
        method['lambda_tm'] = 0.0
    if method['batch_size_proteins'] <= 0 or backbone['pre_epoch'] <= 0 or method['r_cap'] <= 0:
        raise ValueError('Batch size, pretraining epochs, and contrastive cap must be positive')
    if any(method[key] < 0 for key in ('lambda_hcl', 'lambda_tm')):
        raise ValueError('Loss weights must be nonnegative')
    if any(method['taus'][key] <= 0 or method['lambdas'][key] < 0 for key in ('R', 'S', 'P')):
        raise ValueError('Invalid scale temperatures or weights')
    if method['tau_tm'] <= 0 or method['tau_t'] <= 0 or not 0 < method['t_min'] <= 1:
        raise ValueError('Invalid soft-TCL temperatures or threshold')
    return backbone, method


def baseline_batch(dataset, split, seed):
    """Retain the published local reproduction's historical batch exception."""
    return 512 if dataset == 'STRING' and split == 'random' and seed in (0, 1, 2) else 128
