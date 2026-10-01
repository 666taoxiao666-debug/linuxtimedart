import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from layers.TimeDART_EncDec import CompositionalEventWikiRouter
from utils.wind_regime_wiki import (
    compute_event_factor_rule_logits,
    load_wind_regime_wiki_bundle,
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

    def test_fold_specific_lifecycle_cannot_be_reused_across_folds(self):
        evidence = _evidence([])
        evidence["sdwpf_fold"] = 0
        evolved, audit = evolve_event_wiki(_base_spec(), evidence)
        self.assertEqual(audit["sdwpf_fold"], 0)
        later = _evidence([])
        later["sdwpf_fold"] = 1
        later["as_of_step"] = 1100
        with self.assertRaisesRegex(ValueError, "cannot cross SDWPF folds"):
            evolve_event_wiki(evolved, later)

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

    def test_horizon_utility_survives_only_stable_profitable_steps(self):
        evidence = _evidence([_candidate("gust-horizon", "gust_or_turbulent")])
        evidence["pred_len"] = 3
        for row in evidence["candidates"][0]["turbine_evidence"]:
            row["horizon_mean_utility"] = [0.08, -0.02, 0.06]
            row["horizon_std_utility"] = [0.001, 0.001, 0.001]
        evolved, audit = evolve_event_wiki(_base_spec(), evidence)
        lifecycle = evolved["scenes"][0]["lifecycle"]
        self.assertEqual(lifecycle["horizon_deployment_weight"], [1.0, 0.0, 1.0])
        self.assertEqual(audit["pred_len"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wiki.json"
            path.write_text(json.dumps(evolved), encoding="utf-8")
            loaded = load_wind_regime_wiki_spec(path)
            bundle_path = Path(temporary) / "wiki.npz"
            np.savez_compressed(
                bundle_path,
                embeddings=np.zeros((len(loaded["scene_ids"]), 8), dtype=np.float32),
                scene_ids=np.asarray(loaded["scene_ids"]),
                encoder_name=np.asarray("unit-test"),
                config_sha256=np.asarray(loaded["sha256"]),
                factor_reliability=np.asarray(loaded["factor_reliability"], dtype=np.float32),
                factor_horizon_reliability=np.asarray(
                    loaded["factor_horizon_reliability"], dtype=np.float32
                ),
            )
            bundle = load_wind_regime_wiki_bundle(
                bundle_path, expected_scene_ids=loaded["scene_ids"]
            )
            self.assertEqual(bundle["factor_horizon_reliability"].shape, (4, 3))
        self.assertEqual(loaded["factor_horizon_reliability"][0], [1.0, 0.0, 1.0])
        self.assertEqual(loaded["factor_horizon_reliability"][1], [1.0, 1.0, 1.0])

        invalid = copy.deepcopy(evidence)
        invalid["candidates"][0]["turbine_evidence"][0]["horizon_mean_utility"] = [0.08, 0.06]
        with self.assertRaisesRegex(ValueError, "horizon utilities"):
            evolve_event_wiki(_base_spec(), invalid)

    def test_horizon_only_gain_can_survive_negative_macro_and_decay(self):
        evidence = _evidence([_candidate("ramp", "gust_or_turbulent", step=1000)])
        evidence["pred_len"] = 2
        for row in evidence["candidates"][0]["turbine_evidence"]:
            row["mean_utility"] = -0.01
            row["horizon_mean_utility"] = [0.08, -0.10]
            row["horizon_std_utility"] = [0.001, 0.001]
        evolved, _ = evolve_event_wiki(_base_spec(), evidence)
        self.assertEqual(evolved["scenes"][0]["lifecycle"]["horizon_deployment_weight"], [1.0, 0.0])
        later = _evidence([])
        later["pred_len"] = 2
        later["as_of_step"] = 1100
        carried, _ = evolve_event_wiki(evolved, later, forget_half_life_steps=100)
        self.assertEqual(carried["scenes"][0]["lifecycle"]["horizon_deployment_weight"], [0.5, 0.0])

    @unittest.skipUnless(importlib.util.find_spec("reformer_pytorch"), "full model dependencies missing")
    def test_pretrain_horizon_can_differ_from_factorized_forecast_horizon(self):
        from models.TimeDART import PromptGuidedModel
        from run import build_parser, configure_args

        evidence = _evidence([_candidate("gust-horizon", "gust_or_turbulent")])
        evidence["pred_len"] = 3
        for row in evidence["candidates"][0]["turbine_evidence"]:
            row["horizon_mean_utility"] = [0.08, 0.08, 0.08]
            row["horizon_std_utility"] = [0.001, 0.001, 0.001]
        evolved, _ = evolve_event_wiki(_base_spec(), evidence)
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "wiki.json"
            config.write_text(json.dumps(evolved), encoding="utf-8")
            spec = load_wind_regime_wiki_spec(config)
            bundle = Path(temporary) / "wiki.npz"
            np.savez_compressed(
                bundle,
                embeddings=np.zeros((len(spec["scene_ids"]), 16), dtype=np.float32),
                scene_ids=np.asarray(spec["scene_ids"]),
                encoder_name=np.asarray("unit-test"),
                config_sha256=np.asarray(spec["sha256"]),
                factor_reliability=np.asarray(spec["factor_reliability"], dtype=np.float32),
                factor_horizon_reliability=np.asarray(
                    spec["factor_horizon_reliability"], dtype=np.float32
                ),
            )
            common = [
                "--model_id", "SDWPF", "--model", "PromptTimeDART", "--data", "SDWPF",
                "--prompt_router", "compositional_wiki", "--scene_wiki_config", str(config),
                "--scene_wiki_embeddings", str(bundle), "--input_len", "24",
                "--patch_len", "6", "--stride", "6", "--d_model", "16",
                "--d_ff", "32", "--n_heads", "4", "--e_layers", "1",
                "--d_layers", "1", "--no-use_gpu",
            ]
            pretrain = configure_args(build_parser().parse_args(
                ["--task_name", "pretrain", "--pred_len", "6", *common]
            ))
            pretrain.device = torch.device("cpu")
            PromptGuidedModel(pretrain)
            factorized = [
                "--task_name", "finetune", "--is_training", "0", "--utility_wiki",
                "--utility_factorized", "--utility_adapter_mode", "calibrated_evidence",
                "--utility_intervention_floor", "1",
                "--pred_len", "6", *common,
            ]
            with self.assertRaisesRegex(ValueError, "pred_len differs"):
                configure_args(build_parser().parse_args(factorized))


if __name__ == "__main__":
    unittest.main()
