import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.factorized_wiki import FactorUtilityGate
from utils.wiki_gain_calibration import calibration_blocks, fit_gain_policy


class GainCalibrationTests(unittest.TestCase):
    def fixture(self):
        n = 180
        base = np.zeros((n, 2))
        target = np.ones_like(base)
        candidates = np.stack([np.full_like(base, 2), np.full_like(base, -2)], -1)
        return dict(base=base, candidates=candidates, target=target,
                    available=np.ones((n, 2), bool), trend=np.tile(np.arange(3), n // 3),
                    block_ids=np.repeat(np.arange(3), n // 3), source_split='train', min_windows=3)

    def test_shrink_overshoot_and_reject_harmful_event(self):
        alpha, gain, audit = fit_gain_policy(**self.fixture())
        np.testing.assert_array_equal(alpha[..., 0], .5)
        np.testing.assert_array_equal(alpha[..., 1], 0)
        np.testing.assert_array_equal(gain[..., 0], 1)
        self.assertTrue(audit)

    def test_validation_test_and_nonfinite_are_rejected(self):
        for split in ('val', 'test'):
            data = self.fixture()
            data['source_split'] = split
            with self.assertRaises(ValueError):
                fit_gain_policy(**data)
        data = self.fixture()
        data['target'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_gain_policy(**data)

    def test_unavailable_and_rare_events_abstain(self):
        data = self.fixture()
        data['available'][:, 0] = False
        alpha, _, _ = fit_gain_policy(**data)
        self.assertTrue(np.all(alpha == 0))
        data = self.fixture()
        data['min_windows'] = 999
        alpha, _, _ = fit_gain_policy(**data)
        self.assertTrue(np.all(alpha == 0))

    def test_one_good_time_block_cannot_override_two_harmful_blocks(self):
        data = self.fixture()
        data['candidates'][data['block_ids'] != 0, :, 0] = -.1
        alpha, _, _ = fit_gain_policy(**data)
        self.assertTrue(np.all(alpha == 0))

    def test_gain_matches_clipped_power_metric(self):
        data = self.fixture()
        data['base'][:] = 2.
        data['candidates'][:] = 1.
        alpha, _, _ = fit_gain_policy(**data, clip_bounds=(0., 1.))
        self.assertTrue(np.all(alpha == 0))

    def test_pooled_fallback_and_history_trend_conditioning(self):
        data = self.fixture()
        data['trend'][:] = 0
        alpha, _, audit = fit_gain_policy(**data)
        self.assertTrue(any(row['pooled_fallback'] for row in audit))
        np.testing.assert_array_equal(alpha[..., 0], .5)
        data = self.fixture()
        data['target'][data['trend'] == 1] = -1
        alpha, _, _ = fit_gain_policy(**data)
        np.testing.assert_array_equal(alpha[1, :, 0], 0)
        np.testing.assert_array_equal(alpha[1, :, 1], .5)

    def test_window_order_does_not_define_temporal_blocks(self):
        dates = np.arange(60).astype('timedelta64[m]') + np.datetime64('2020-01-01')
        dataset = SimpleNamespace(flag='train', window_starts=np.tile(np.arange(35), 2),
                                  dates=dates, seq_len=3, pred_len=5)
        blocks, audit = calibration_blocks(dataset)
        np.testing.assert_array_equal(blocks[:35], blocks[35:])
        self.assertGreater(audit['purged_cross_block_windows'], 0)
        dataset.flag = 'val'
        with self.assertRaises(ValueError):
            calibration_blocks(dataset)

    def test_old_checkpoint_loading_and_policy_roundtrip(self):
        gate = FactorUtilityGate(4, 2, 3, num_candidates=2).eval()
        old = {k: v for k, v in gate.state_dict().items() if not k.startswith('policy_')}
        gate.load_state_dict(old, strict=True)
        self.assertFalse(gate.policy_enabled)
        alpha, gain, _ = fit_gain_policy(**self.fixture())
        gate.install_policy(alpha, gain)
        restored = copy.deepcopy(gate)
        restored.load_state_dict(gate.state_dict(), strict=True)
        self.assertTrue(restored.policy_enabled)
        actual = restored(torch.randn(1, 2, 4), torch.tensor([[0., 1., 0.]]), None, None, None, None, None)
        torch.testing.assert_close(actual, torch.tensor(gain[1:2], dtype=torch.float32))
        broken = dict(gate.state_dict())
        del broken['policy_gain']
        with self.assertRaises(RuntimeError):
            gate.load_state_dict(broken, strict=True)

    def test_source_requires_trained_last_checkpoint(self):
        from scripts.calibrate_factorized_wiki import resolve_source
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir = root / 'runs/f0_s2024'
            log_dir.mkdir(parents=True)
            selected = root / 'checkpoint.pth'
            selected.touch()
            (log_dir / 'finetune.log').write_text(f'[AUDIT] FINETUNE_CHECKPOINT={selected}\n', encoding='utf-8')
            with self.assertRaises(FileNotFoundError):
                resolve_source(root, 0, 2024)


if __name__ == '__main__':
    unittest.main()
