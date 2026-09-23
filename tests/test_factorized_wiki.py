import unittest

import torch

from utils.factorized_wiki import FactorizedEvidenceAdapter, factorized_route
from utils.utility_wiki import utility_candidate_specialization_loss, utility_supervision_loss, utility_decision_loss


class FactorizedWikiTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_zero_initialization_physical_veto_and_trainable_semantics(self):
        module = FactorizedEvidenceAdapter(8, 3, 2, 6, 4)
        history = torch.randn(2, 1, 5, 8, requires_grad=True)
        trend = torch.tensor([[.2, .3, .5]] * 2)
        support = torch.tensor([[1., 0.], [0., 0.]])
        keys = torch.randn(2, 6, requires_grad=True)
        physical = torch.randn(2, 4)
        out, _ = module(history, trend, support, support, physical, keys)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        optimizer = torch.optim.Adam(module.parameters(), lr=.01)
        (out - .1).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        out, _ = module(history, trend, support, support, physical, keys)
        self.assertTrue(torch.all(out[..., 1] == 0))
        self.assertTrue(torch.all(out[1] == 0))
        self.assertLessEqual(out.abs().max().item(), .2)
        (out - .1).square().sum().backward()
        self.assertGreater(module.semantic_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in module.query.parameters()), 0)
        self.assertIsNone(history.grad)
        self.assertIsNone(keys.grad)

    def test_horizon_selection_all_candidates_and_exact_abstention(self):
        base = torch.tensor([[[10.], [10.], [10.]]])
        corrections = torch.tensor([[[[1., 2., 3.]], [[1., 2., 3.]], [[1., 2., 3.]]]])
        scores = torch.tensor([[[.1, .2, 999.], [.5, .1, 999.], [-1., -2., 999.]]], requires_grad=True)
        available = torch.tensor([[True, True, False]])
        pred, soft, hard, action = factorized_route(base, corrections, scores, available, 0., .05, False)
        torch.testing.assert_close(pred, torch.tensor([[[12.], [11.], [10.]]]))
        torch.testing.assert_close(action, torch.tensor([[2, 1, 0]]))
        soft.sum().backward()
        self.assertEqual(scores.grad[..., 2].abs().sum().item(), 0)
        self.assertGreater(scores.grad[..., :2].abs().sum().item(), 0)
        pred, soft, _, action = factorized_route(base, corrections, scores, torch.zeros_like(available), 0., .05, True)
        self.assertTrue(torch.equal(pred, base))
        self.assertTrue(torch.equal(soft, base))
        self.assertTrue(torch.all(action == 0))

    def test_each_expert_supervised_masked_and_base_detached(self):
        base = torch.zeros(2, 3, 1, requires_grad=True)
        correction = torch.zeros(2, 3, 1, 3, requires_grad=True)
        scores = torch.zeros(2, 3, 3, requires_grad=True)
        aux = {'factor_predictions': base.detach()[..., None] + correction,
               'base_prediction': base, 'factor_utilities': scores,
               'factor_availability': torch.tensor([[True, False, False], [False, True, False]])}
        loss, ranking, _ = utility_candidate_specialization_loss(aux, torch.ones_like(base))
        (loss + ranking).backward()
        self.assertIsNone(base.grad)
        self.assertGreater(correction.grad[0, ..., 0].abs().sum().item(), 0)
        self.assertGreater(correction.grad[1, ..., 1].abs().sum().item(), 0)
        self.assertEqual(correction.grad[..., 2].abs().sum().item(), 0)
        self.assertEqual(correction.grad[0, ..., 1].abs().sum().item(), 0)
        loss, targets = utility_supervision_loss(aux, torch.ones_like(base))
        self.assertEqual(targets.shape, (2, 3, 3))
        self.assertTrue(torch.isfinite(loss))
        regret, _ = utility_decision_loss(aux, torch.ones_like(base))
        self.assertTrue(torch.isfinite(regret))

    def test_absolute_gain_and_no_evidence_loss(self):
        scores = torch.zeros(1, 1, 3, requires_grad=True)
        aux = {'factor_predictions': torch.tensor([[[[9., 8., 20.]]]], requires_grad=True),
               'base_prediction': torch.tensor([[[10.]]]), 'factor_utilities': scores,
               'factor_availability': torch.tensor([[True, True, False]])}
        loss, gains = utility_supervision_loss(aux, torch.zeros(1, 1, 1))
        torch.testing.assert_close(gains, torch.tensor([[[1., 2., -10.]]]))
        loss.backward()
        self.assertTrue(torch.all(scores.grad[..., :2] < 0))
        self.assertEqual(scores.grad[..., 2].item(), 0)
        aux['factor_availability'].zero_()
        loss, rank, _ = utility_candidate_specialization_loss(aux, torch.zeros(1, 1, 1))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(rank.item(), 0)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
