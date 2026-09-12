import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from layers.TimeDART_EncDec import CompositionalEventWikiRouter
from utils.wind_regime_wiki import (
    compute_event_factor_rule_logits,
    load_wind_regime_wiki_spec,
)
from utils.wind_wiki_lifecycle import evolve_event_wiki


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "wind_event_factor_wiki.json"


def _base_spec():
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evidence(candidates, *, held_out=None):
    return {
        "schema_version": 1,
        "source_split": "train_oof",
        "as_of_step": 1000,
        "source_turbines": [1, 2, 3],
        "held_out_turbines": list(held_out or [4]),
        "candidates": candidates,
    }


def _candidate(identifier, factor_id, *, prompt="validated insight", rule=None, step=1000):
    candidate = {
        "id": identifier,
        "factor_id": factor_id,
        "prompt": prompt,
        "created_step": 0,
        "last_evidence_step": step,
        "turbine_evidence": [
            {
                "turbine_id": turbine_id,
                "n_windows": 120,
                "mean_utility": 0.08 + 0.002 * turbine_id,
                "std_utility": 0.01,
                "source_split": "train_oof",
            }
            for turbine_id in (1, 2, 3)
        ],
    }
    if rule is not None:
        candidate["rule"] = rule
    return candidate


class WindWikiLifecycleTests(unittest.TestCase):
    def test_unassessed_seed_knowledge_is_not_silently_deleted(self):
        evolved, audit = evolve_event_wiki(_base_spec(), _evidence([]))
        weights = [
            scene["lifecycle"]["deployment_weight"] for scene in evolved["scenes"]
        ]
        self.assertEqual(weights, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(audit["active_factor_count"], 4)

    def test_rejects_validation_evidence_and_held_out_turbine_leakage(self):
        forbidden = _evidence([])
        forbidden["source_split"] = "val"
        with self.assertRaisesRegex(ValueError, "train_oof"):
            evolve_event_wiki(_base_spec(), forbidden)

        leaked = _candidate("leak", "gust_or_turbulent")
        leaked["turbine_evidence"][0]["turbine_id"] = 4
        with self.assertRaisesRegex(ValueError, "held-out turbines"):
            evolve_event_wiki(_base_spec(), _evidence([leaked]))

    def test_merge_forget_and_cross_turbine_weight_are_deterministic(self):
        fresh_a = _candidate(
            "gust-a", "gust_or_turbulent", prompt="Respond conservatively to gust onset."
        )
        fresh_b = _candidate(
            "gust-b", "gust_or_turbulent", prompt="Avoid overshooting after abrupt wind jumps."
        )
        stale = _candidate("stale-idle", "low_wind_idle", step=0)
        evolved, audit = evolve_event_wiki(
            _base_spec(),
            _evidence([fresh_a, fresh_b, stale]),
            forget_half_life_steps=100,
        )
        scenes = {scene["id"]: scene for scene in evolved["scenes"]}
        gust = scenes["gust_or_turbulent"]
        idle = scenes["low_wind_idle"]
        self.assertEqual(
            gust["lifecycle"]["merged_candidate_ids"], ["gust-a", "gust-b"]
        )
        self.assertGreater(gust["lifecycle"]["deployment_weight"], 0.0)
        self.assertIn("Cross-turbine OOF validated refinements", gust["prompt"])
        self.assertEqual(idle["lifecycle"]["status"], "retired")
        self.assertEqual(idle["lifecycle"]["deployment_weight"], 0.0)
        self.assertEqual(audit["source_split"], "train_oof")
        self.assertEqual(audit["held_out_turbines"], [4])

    def test_new_observable_rule_can_be_added_and_executed(self):
        rule = {
            "combine": "all",
            "conditions": [
                {
                    "feature": "power_ratio",
                    "statistic": "trend_delta",
                    "operator": "above",
                    "threshold": 0.05,
                }
            ],
        }
        candidate = _candidate("ramp-up", "rapid_power_ramp", rule=rule)
        evolved, _ = evolve_event_wiki(_base_spec(), _evidence([candidate]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evolved.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(evolved, handle)
            loaded = load_wind_regime_wiki_spec(path)
        self.assertEqual(loaded["scene_ids"][-1], "rapid_power_ramp")
        history = torch.zeros(2, 12, 2)
        history[0, :, 0] = 7.0
        history[0, :, 1] = torch.linspace(100.0, 1000.0, 12)
        history[1, :, 0] = 7.0
        history[1, :, 1] = 500.0
        _, targets = compute_event_factor_rule_logits(
            history,
            ["Wspd", "power"],
            rated_power=1500.0,
            factor_ids=loaded["scene_ids"],
            factor_rules=loaded["factor_rules"],
            rule_defaults=loaded["rule_defaults"],
        )
        self.assertEqual(tuple(targets.shape), (2, 5))
        self.assertEqual(targets[:, -1].tolist(), [1.0, 0.0])

    def test_zero_lifecycle_reliability_forces_exact_abstention(self):
        router = CompositionalEventWikiRouter(
            np.random.default_rng(7).standard_normal((2, 8)).astype(np.float32),
            d_model=4,
            factor_reliability=[0.0, 1.0],
            activation_threshold=0.0,
            rule_weight=100.0,
            dropout=0.0,
        )
        _, _, _, activations = router(
            torch.randn(1, 4, 4), torch.tensor([[1.0, -1.0]])
        )
        self.assertEqual(float(activations[0, 0]), 0.0)
        self.assertEqual(float(activations[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
