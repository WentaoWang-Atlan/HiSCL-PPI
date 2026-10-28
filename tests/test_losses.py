"""Synthetic tests for the HCL/TCL loss implementation."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: F401
import dgl
import torch
import torch.nn.functional as F

from losses import (info_nce_matrix, build_two_view_batch, site_grouping,
                        pyramid, protein_mean_by_node, tcl_loss)
from models import CodeBook, device
from pretrain import training_loss, baseline_forward
from utils import set_seed


def graph(n):
    edges = [(j, j+1) for j in range(n-1)] + [(j+1, j) for j in range(n-1)]
    g = dgl.heterograph({('amino_acid', e, 'amino_acid'): edges
                        for e in ('SEQ', 'STR_KNN', 'STR_DIS')},
                       num_nodes_dict={'amino_acid': n}).to(device)
    g.ndata['x'] = (torch.arange(n*7, device=device).reshape(n, 7).float()+1)/17
    return g


def parameter():
    return dict(prot_num_layers=2, prot_hidden_dim=8, input_dim=7,
                dropout_ratio=0., num_embeddings=16, commitment_cost=0.25,
                mask_ratio=0.5, sce_scale=1.5, mask_loss=1.)


class LossTests(unittest.TestCase):
    def setUp(self):
        set_seed(123)

    def test_infonce_double_loop(self):
        a, b = torch.randn(7, 5, dtype=torch.double), torch.randn(7, 5, dtype=torch.double)
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        losses = []
        for left, right in ((a, b), (b, a)):
            for i in range(7):
                denominator = sum(torch.exp(torch.dot(left[i], right[j])/0.3) for j in range(7))
                losses.append(-torch.log(torch.exp(torch.dot(left[i], right[i])/0.3)/denominator))
        torch.testing.assert_close(info_nce_matrix(a, b, 0.3), torch.stack(losses).mean())

    def test_transpose_direction(self):
        a, b = F.normalize(torch.randn(5, 4), dim=1), F.normalize(torch.randn(5, 4), dim=1)
        logits = a @ b.t()/0.2
        target = torch.arange(5)
        torch.testing.assert_close(F.cross_entropy(logits.t(), target),
                                   F.cross_entropy(b @ a.t()/0.2, target))

    def test_identical_views_not_zero_infonce(self):
        # Identical views do not imply zero InfoNCE when negatives remain.
        z = torch.ones(4, 3)
        torch.testing.assert_close(info_nce_matrix(z, z, 0.5), torch.tensor(4.).log())

    def test_small_protein_and_site_weighting(self):
        n = torch.tensor([3, 51])
        slot, counts, offsets = site_grouping(n)
        self.assertEqual(counts.tolist(), [3, 50, 1])
        self.assertEqual(offsets.tolist(), [0, 1, 3])
        self.assertEqual(slot[:3].tolist(), [0, 0, 0])
        r = torch.cat([torch.ones(3, 1), torch.zeros(50, 1), torch.tensor([[10.]])])
        _, proteins = pyramid(r, n)
        torch.testing.assert_close(proteins, torch.tensor([[1.], [5.]]))

    def test_two_views_alignment_topology_and_encoder(self):
        graphs = [graph(31), graph(53)]
        originals = [g.ndata['x'].clone() for g in graphs]
        bg = build_two_view_batch(graphs, mask_ratio=0.4)
        n = bg.batch_num_nodes()
        self.assertEqual(n.tolist(), [31, 53, 31, 53])
        pieces = dgl.unbatch(bg)
        for i in range(2):
            torch.testing.assert_close(graphs[i].ndata['x'], originals[i])
            for etype in graphs[i].etypes:
                for actual in (pieces[i], pieces[i+2]):
                    for u, v in zip(actual.edges(etype=etype), graphs[i].edges(etype=etype)):
                        torch.testing.assert_close(u, v)
        slot1 = site_grouping(n[:2])[0]
        slot2 = site_grouping(n[2:])[0]
        torch.testing.assert_close(slot1, slot2)
        model = CodeBook(parameter(), []).to(device).eval()
        h = model.Protein_Encoder.encoding(bg)  # exactly one call across 2B
        e, _, _ = model.vq_layer(h)
        r = torch.cat([h, e], dim=-1)
        self.assertGreater(torch.linalg.vector_norm(r[:84]-r[84:]).item(), 0.)

    def test_soft_filter_and_gradients(self):
        p = torch.randn(4, 5, requires_grad=True)
        tm = torch.tensor([[1., .8, .6, .1], [.8, 1., .1, .1],
                           [.6, .1, 1., .1], [.1, .1, .1, 1.]])
        loss, targets = tcl_loss(p, tm, tau_t=0.2)
        self.assertEqual(targets[-1].item(), -1)
        z = F.normalize(p, dim=1)
        expected = []
        for i in range(3):
            others = [j for j in range(4) if i != j]
            logits = torch.stack([torch.dot(z[i], z[j])/0.1 for j in others])
            qualified = [j for j in others if tm[i, j] >= .5]
            weights = F.softmax(torch.stack([tm[i, j]/.2 for j in qualified]), dim=0)
            expected.append(sum(w*(torch.logsumexp(logits, 0)-torch.dot(z[i], z[j])/.1)
                                for w, j in zip(weights, qualified)))
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_empty_valid_and_singleton(self):
        for b in (1, 3):
            p = torch.randn(b, 4, requires_grad=True)
            loss, target = tcl_loss(p, torch.zeros(b, b))
            self.assertEqual(loss.item(), 0.)
            self.assertTrue((target == -1).all())
            loss.backward()
            torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_zero_weights_exact_baseline_update(self):
        a = CodeBook(parameter(), []).to(device)
        b = copy.deepcopy(a)
        bg = dgl.batch([graph(8), graph(9)])
        oa = torch.optim.Adam(a.parameters(), lr=.001)
        ob = torch.optim.Adam(b.parameters(), lr=.001)
        set_seed(99)
        _, _, eq, recon, mask = a(bg)
        loss_a = eq + recon + mask*a.param['mask_loss']
        oa.zero_grad(); loss_a.backward(); oa.step()
        rng_a = torch.get_rng_state().clone()
        set_seed(99)
        loss_b, _ = training_loss(b, bg, {'lambda_hcl': 0., 'lambda_tm': 0.})
        ob.zero_grad(); loss_b.backward(); ob.step()
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        torch.testing.assert_close(rng_a, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_tcl_capture_before_mcm(self):
        param = parameter(); param['mask_ratio'] = 1.
        model = CodeBook(param, []).to(device)
        bg = dgl.batch([graph(8), graph(9)])
        _, mutated_e, _, _, _, original_e = baseline_forward(model, bg, True)
        self.assertEqual(mutated_e.count_nonzero().item(), 0)
        self.assertGreater(original_e.norm().item(), 0.)
        original_e.sum().backward()
        self.assertIsNotNone(model.Protein_Encoder.fc[0].weight.grad)

    def test_anchor_pooling_paths(self):
        # residue_mean == plain residue mean (downstream readout geometry);
        # site_mean == Eq. 7 equal-weight site mean. They must differ when the
        # last site is partial.
        n = torch.tensor([3, 51])
        r = torch.cat([torch.ones(3, 1), torch.zeros(50, 1), torch.tensor([[10.]])])
        torch.testing.assert_close(protein_mean_by_node(r, n),
                                   torch.tensor([[1.], [10. / 51]]))
        _, site_mean = pyramid(r, n)
        torch.testing.assert_close(site_mean, torch.tensor([[1.], [5.]]))

    def test_joint_loss_and_gradients(self):
        model = CodeBook(parameter(), []).to(device)
        bg = dgl.batch([graph(8), graph(120), graph(11)])
        cfg = {'lambda_hcl': 1., 'lambda_tm': .3, 'tau_tm': .1, 'tau_t': .1,
               't_min': .5, 'S': 50, 'taus': {'R': .5, 'S': .3, 'P': .1},
               'lambdas': {'R': 1., 'S': 1., 'P': 1.},
               'view': {'mask_ratio': .15, 'block_w': 10, 'strategy': ['iid', 'block']}}
        tm = torch.tensor([[0., .8, .7], [.8, 0., .6], [.7, .6, 0.]])
        loss, metrics = training_loss(model, bg, cfg, tm)
        torch.testing.assert_close(loss, metrics['vae'] + metrics['hcl'] + .3 * metrics['tm'])
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(int(metrics['tm_valid']), 3)


if __name__ == '__main__':
    unittest.main()
