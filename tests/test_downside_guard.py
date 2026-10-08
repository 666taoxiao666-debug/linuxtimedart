import unittest

import torch

from utils.factorized_wiki import FactorUtilityGate, factorized_loss


class DownsideGuardTests(unittest.TestCase):
    def test_head_starts_at_existing_min_gain_and_round_trips(self):
        gate = FactorUtilityGate(
            d_model=4,
            pred_len=3,
            physical_dim=2,
            num_modes=3,
            temperature=0.05,
            min_gain=0.003,
            num_candidates=2,
            downside_guard=True,
            downside_weight=1.0,
        )
        hidden = torch.randn(2, 5, 4)
        trend = torch.softmax(torch.randn(2, 3), dim=-1)
        physical = torch.randn(2, 2)
        descriptors = torch.randn(2, 2, 4)
        corrections = torch.randn(2, 3, 1, 2)
        support = torch.ones(2, 2)
        rules = torch.ones(2, 2)
        scores = gate(
            hidden, trend, physical, descriptors, corrections, support, rules
        )
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))
        downside = torch.nn.functional.softplus(gate._last_downside_raw)
        torch.testing.assert_close(
            downside, torch.full_like(downside, 0.003), rtol=1e-5, atol=1e-7
        )

        restored = FactorUtilityGate(
            d_model=4,
            pred_len=3,
            physical_dim=2,
            num_modes=3,
            temperature=0.05,
            min_gain=0.003,
            num_candidates=2,
            downside_guard=True,
            downside_weight=1.0,
        )
        restored.load_state_dict(gate.state_dict())
        restored(hidden, trend, physical, descriptors, corrections, support, rules)
        torch.testing.assert_close(
            torch.nn.functional.softplus(restored._last_downside_raw), downside
        )

    def test_severity_loss_uses_positive_excess_error_not_binary_label(self):
        predicted_downside = torch.tensor([[[0.1], [0.1]]], requires_grad=True)
        raw_gain = torch.zeros(1, 2, 1, requires_grad=True)
        base = torch.zeros(1, 2, 1)
        target = torch.ones(1, 2, 1)
        candidates = torch.tensor([[[[0.5]], [[3.0]]]])
        aux = {
            "base_prediction": base,
            "factor_predictions": candidates,
            "factor_availability": torch.ones(1, 2, 1, dtype=torch.bool),
            "factor_utilities": raw_gain,
            "factor_route_scores": raw_gain - predicted_downside,
            "factor_downside": predicted_downside,
        }
        loss, _ = factorized_loss(
            aux,
            target,
            "decision",
            min_gain=0.003,
            temperature=0.05,
            downside_loss_weight=1.0,
        )
        loss.backward()
        # Beneficial candidate target severity is zero, harmful candidate is 1.
        self.assertGreater(predicted_downside.grad[0, 0, 0].item(), 0.0)
        self.assertLess(predicted_downside.grad[0, 1, 0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
