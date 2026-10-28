"""Masked two-view HCL and soft TM-score supervision used by HiSCL-PPI."""
import math

import dgl
import torch
import torch.nn.functional as F


def build_two_view_batch(graph_list, mask_ratio=0.15, block_w=10,
                         strategy=('iid', 'block'), x_mask_value=0.0,
                         generator=None):
    """Never mutate source features/topology. RNG follows set_seed by default.

    Block windows: round-half-up(rho*N/w), at least one if rho>0;
    uniformly sampled starts without replacement, overlap allowed. Realized
    coverage is approximate. rho=0/1 explicitly masks none/all.
    An explicit generator must match the feature tensor's device.
    """
    if not graph_list or not 0 <= mask_ratio <= 1 or block_w < 1:
        raise ValueError('Need nonempty graphs, rho in [0,1], block_w >= 1')
    if len(strategy) != 2 or any(s not in ('iid', 'block') for s in strategy):
        raise ValueError('strategy must contain two iid/block entries')
    views = []
    for method in strategy:
        for graph in graph_list:
            g = graph.local_var()
            x = graph.ndata['x']
            n = x.shape[0]
            if n == 0:
                raise ValueError('Empty proteins are unsupported')
            if method == 'iid':
                mask = torch.rand(n, device=x.device, generator=generator) < mask_ratio
            else:
                mask = torch.zeros(n, dtype=torch.bool, device=x.device)
                if mask_ratio == 1:
                    mask[:] = True
                elif mask_ratio > 0:
                    width = min(block_w, n)
                    num = min(n - width + 1, max(1, math.floor(mask_ratio*n/width + 0.5)))
                    starts = torch.randperm(n-width+1, device=x.device,
                                           generator=generator)[:num]
                    positions = starts[:, None] + torch.arange(width, device=x.device)
                    mask[positions.flatten()] = True
            g.ndata['x'] = torch.where(mask[:, None],
                                      torch.as_tensor(x_mask_value, dtype=x.dtype, device=x.device), x)
            views.append(g)
    return dgl.batch(views)


def site_grouping(batch_num_nodes, S=50):
    """Return node slots, site sizes, and B+1 exclusive site offsets."""
    n = torch.as_tensor(batch_num_nodes, dtype=torch.long)
    if n.ndim != 1 or n.numel() == 0 or (n <= 0).any() or S < 1:
        raise ValueError('Need positive protein sizes and S')
    k = (n + S - 1) // S
    offsets = torch.cat([k.new_zeros(1), k.cumsum(0)])
    starts = n.cumsum(0) - n
    local = torch.arange(int(n.sum()), device=n.device) - torch.repeat_interleave(starts, n)
    slot = torch.repeat_interleave(offsets[:-1], n) + local // S
    counts = torch.bincount(slot, minlength=int(k.sum()))
    return slot, counts, offsets


def segment_mean(values, slot, counts):
    return values.new_zeros((len(counts), values.shape[1])).index_add(
        0, slot, values) / counts.to(values.dtype)[:, None]


def pyramid(residues, batch_num_nodes, S=50):
    slot, counts, offsets = site_grouping(batch_num_nodes.to(residues.device), S)
    if residues.shape[0] != slot.numel():
        raise ValueError('Residue rows do not match graph sizes')
    sites = segment_mean(residues, slot, counts)
    k = offsets[1:] - offsets[:-1]
    site2prot = torch.repeat_interleave(torch.arange(len(k), device=k.device), k)
    return sites, segment_mean(sites, site2prot, k)


def protein_mean_by_node(values, batch_num_nodes):
    """Plain per-protein residue mean (== dgl.mean_nodes over [h; z]) — the
    exact unmasked counterpart of the downstream prot_embed readout geometry
    (downstream readout). Used for the TCL anchor by default; 'site_mean' (Eq. 7)
    remains available as a config switch."""
    n = torch.as_tensor(batch_num_nodes, dtype=torch.long).to(values.device)
    gid = torch.repeat_interleave(torch.arange(len(n), device=values.device), n)
    return segment_mean(values, gid, n)


def info_nce_matrix(z1, z2, tau, max_anchors=None, generator=None):
    if tau <= 0 or z1.ndim != 2 or z1.shape != z2.shape or len(z1) == 0:
        raise ValueError('Need equal nonempty matrices and positive temperature')
    if max_anchors is not None and 0 < max_anchors < len(z1):
        # Select matching rows in both views, limiting anchors and candidates.
        perm = torch.randperm(len(z1), device=z1.device, generator=generator)
        sel = perm[:max_anchors]
        z1, z2 = z1[sel], z2[sel]
    a = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).t() / tau
    y = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(a, y) + F.cross_entropy(a.t(), y)) * 0.5


def hcl_loss(r1, r2, s1, s2, p1, p2, taus=(0.5, 0.3, 0.1), lambdas=(1., 1., 1.),
             r_max_anchors=None, generator=None):
    if len(taus) != 3 or len(lambdas) != 3:
        raise ValueError('taus/lambdas order is R,S,P')
    parts = {}
    for name, left, right, tau, weight in zip(
            ('R', 'S', 'P'), (r1, s1, p1), (r2, s2, p2), taus, lambdas):
        if weight != 0:
            l = info_nce_matrix(left, right, tau, max_anchors=r_max_anchors,
                                generator=generator)
        else:
            l = left.sum()*0
        parts[name] = l
    return sum(w*parts[n] for n, w in zip(('R', 'S', 'P'), lambdas)), parts


def tcl_loss(p_unmasked, tm_batch, tau_tm=0.1, t_min=0.5,
             tau_t=0.1):
    """Return mean over valid anchors; invalid j_star=-1, empty loss has gradient.

    Soft labels use only threshold-qualified partners, while the model
    denominator still contains ALL non-self batch partners. No 0 * -inf.
    """
    p = p_unmasked
    b = len(p)
    targets = torch.full((b,), -1, dtype=torch.long, device=p.device)
    zero = p.sum()*0
    if tau_tm <= 0 or tau_t <= 0:
        raise ValueError('Invalid TCL temperature or variant')
    if tm_batch is None or b < 2:
        return zero, targets
    if tuple(tm_batch.shape) != (b, b):
        raise ValueError('TM batch shape mismatch')
    eye = torch.eye(b, dtype=torch.bool, device=p.device)
    tm = tm_batch.detach().to(device=p.device, dtype=p.dtype).masked_fill(eye, -torch.inf)
    if torch.isnan(tm).any() or torch.isposinf(tm).any():
        raise ValueError('Invalid TM values')
    qualified = torch.isfinite(tm) & ~eye
    if t_min is not None:
        qualified &= tm >= t_min
    valid = qualified.any(dim=1)
    if not valid.any():
        return zero, targets
    targets[valid] = tm[valid].argmax(dim=1)
    norm = F.normalize(p, dim=-1)
    logits = (norm @ norm.t() / tau_tm).masked_fill(eye, -torch.inf)
    q = F.softmax((tm[valid]/tau_t).masked_fill(~qualified[valid], -torch.inf), dim=1)
    logp = F.log_softmax(logits[valid], dim=1).masked_fill(~qualified[valid], 0.)
    return -(q * logp).sum(dim=1).mean(), targets
