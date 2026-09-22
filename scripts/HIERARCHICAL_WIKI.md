# Hierarchical evidence Wiki: quick server experiment

For the calibrated soft-training/hard-inference follow-up, use
[CALIBRATED_WIKI.md](CALIBRATED_WIKI.md). This file documents the older mode.

Run from the server checkout with the `timedart` environment active:

```bash
git pull --ff-only
bash scripts/train/SDWPF_launch_hierarchical_wiki.sh
```

The launcher runs in the background. The default experiment is fold 0, seed
2024, horizon 12 (2 hours), 3 adapter epochs followed by up to 7 gate epochs,
patience 4, utility LR 1e-4. It reuses the September 13 compositional pretraining
and September 14 trend CV checkpoints already present on this server. Override
`SOURCE_CV_DIR` and `TREND_CV_DIR` for another checkout. No pretraining rerun or
model download is needed when those assets are present.

```bash
bash scripts/train/SDWPF_launch_hierarchical_wiki.sh --status
```

All logs use `outputs/logs/SDWPF/YYYYMMDD/NNN_hierarchical_wiki_<parameters>/`.
Long names retain a prefix plus hash; complete parameters are in `cv.env`,
`log_meta.json` and checkpoint `run_manifest.json`. `launch.log` records the
background console, `cv.log` the CV process, `runs/f*_s*/finetune.log` individual
runs, and `wiki_vs_trend.txt` the paired MAE gain against the validated trend
model. `--status` works in a fresh shell without setting `LAUNCH_LOG`.

## Changes to test

* Freeze the base model's Dropout behavior as well as its weights. The former
  training loop re-enabled Dropout with `model.train()` even with frozen weights.
* Replace flattened residual heads with history/evidence-conditioned down,
  stable and up experts, softly combined using predicted trend probabilities.
* Event correction is bounded by 0.5 times input-history power standard
  deviation; composition adds an increment bounded by 0.25 times that deviation.
  Those are scale units, not fixed kW limits. The composition loss cannot change
  the inherited event correction through its gradient.
* Gate input includes both proposed correction trajectories. Cost-sensitive
  expected regret supplements natural-frequency decision classification.
  Selected corrections are executed at full strength, so the supervised
  candidate and executed candidate coincide. Candidate bounds control amplitude.
* Candidate supervision uses 80% MAE and 20% smooth-L1 plus a small correction
  penalty. Training-history trend/event combinations receive tempered inverse
  frequency weights (square root, capped at 3 before normalization). This is
  training-loss reweighting, not removal, resampling, label editing or resplitting.

Validation and test populations, target labels and the sealed test protocol are
unchanged. The logged trend-conditioned groups use predicted classes rather
than oracle future labels. Gate targets are still fitted on training windows,
not out-of-fold residuals: this version does not claim cross-fitted utility
calibration or establish semantic causality from the presence of Wiki alone.

`legacy` remains the default in the general scripts for checkpoint compatibility.
The new launcher explicitly selects `UTILITY_ADAPTER_MODE=hierarchical_evidence`.
For standalone evaluation, pass that same mode and the recorded caps, with
`UTILITY_INTERVENTION_FLOOR=1`; runtime contract checks reject mismatched modes.

The first experiment measures whether the new module improves MAE over its
matched trend model. A positive gain on fold 0 is a pilot result, not a 3-fold
claim. To run the same recipe over all folds/seeds:

```bash
FOLDS='0 1 2' SEEDS='2024 2025 2026' bash scripts/train/SDWPF_launch_hierarchical_wiki.sh
```

These changes are hypotheses to test. Unit tests verify behavior, gradients,
checkpoint loading and reporting; they do not demonstrate SDWPF accuracy.
