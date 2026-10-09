# SDWPF accuracy optimization v5

## Objective and scope

Optimize MAE and RMSE together on the unchanged 10-minute SDWPF task: input
336, horizon 12, fold 1, seed 2024. R² and both persistence Skill scores are
reported, not independently tuned. This is numerical trend-backbone
optimization; existing Wiki methods and thresholds remain frozen.

The previous v4 tradeoff on this fold was MAE 129.350779 kW, RMSE
206.082559 kW, R² 0.679514. Matched trend errors were MAE 128.131697 kW,
RMSE 212.459186 kW, R² 0.659374. Lower RMSE did not imply lower MAE.

## Fixed execution plan

1. Fit one new inner pretrain on the first 80% of the outer training period.
   Its selection interval is the next 10%; both precede outer validation.
   The remaining 10% is unused by this parameter search.
2. Sequentially fit four frozen loss settings for eight epochs each, using
   the same inner pretrain, data split, learning rates, and seed:

   | Candidate | MSE fraction | Last-horizon weight | Power-weight alpha |
   | --- | ---: | ---: | ---: |
   | reference | 0.20 | 1.00 | 0.00 |
   | balanced | 0.50 | 1.00 | 0.00 |
   | moderate_tail | 0.35 | 1.25 | 0.25 |
   | tail_v4 | 0.50 | 1.50 | 0.50 |

3. Define the reference as its lowest-MAE inner epoch. Admit only candidate
   epochs whose inner MAE and RMSE are both no worse. Rank admitted epochs by
   the mean of the two error ratios to persistence, breaking ties by frozen
   candidate order then epoch. If no alternative qualifies, retain reference.
4. Refit only the selected configuration on the outer training fold. Use its
   inner-selected epoch, keeping the original eight-epoch learning-rate
   schedule. Outer validation never changes this epoch or selects a fallback.
5. Export original-scale MAE, RMSE, R², MAE/RMSE Skill, capacity-normalized
   errors, tolerance hit rates, and prediction/error plots.

The outer refit retains the original matched trend pretrain, which was selected
under the historical validation protocol. That checkpoint is frozen; the new
objective/finetuning epoch selection is train-only. The outer validation has
already been examined in earlier experiments: this is development evidence,
not an untouched confirmatory test. No sealed test is accessed, no power label
is altered, and no additional candidate is added after the outer report.

## Server commands

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
conda activate timedart
bash scripts/train/SDWPF_launch_accuracy_tuning.sh --status
```

Follow the combined stage/epoch log without choosing a dated directory:

```bash
tail -n 80 -f "$(cat outputs/logs/SDWPF/accuracy_tuning_latest.txt)/launch.log"
```

`Ctrl+C` stops log viewing, not background training. After completion:

```bash
cat "$(cat outputs/logs/SDWPF/accuracy_tuning_latest.txt)/result.json"
```

The first launch (performed by Codex after tests and a GPU-idle check) is:

```bash
bash scripts/train/SDWPF_launch_accuracy_tuning.sh
```

Do not run that command again during training. If a genuine failure occurs,
inspect `progress.json` and the current stage's `launcher.log`, repair the
identified code/environment error, then use `--resume`. Completed stages are
reused only after checkpoint hash validation; the launcher never starts two GPU
stages at once. `selected.json` and `protocol.json` cannot change on resume.

Canonical protocol: `configs/sdwpf_accuracy_tuning_protocol_v5.json`.
Results: `outputs/logs/SDWPF/<date>/<sequence>_accuracy_tuning_<parameters>/`.
Final plots: `validation/artifacts/forecast_accuracy_overview.png` and
`validation/artifacts/forecast_trace.png`.
