# SDWPF operating-angle semantics and matched pilot

The [official SDWPF schema (Table 2)](https://arxiv.org/html/2208.04360v2)
defines `Wdir` as the angle between wind and the nacelle, and `Ndir` as nacelle
direction. Subtracting `Ndir` from `Wdir` therefore does not produce yaw error.
For example, `Wdir=0, Ndir=180` means aligned wind, not a 180-degree yaw error.

The CLI and manifests now distinguish two conventions:

- `legacy_difference` (default): old eight-channel layout, including circular
  `Wdir` and circular `Wdir-Ndir`. Retained for reproduction of frozen results.
- `relative_wdir`: circular `Ndir` gives nacelle heading, circular `Wdir` gives
  relative yaw. Layout: `Wspd, Etmp, Ndir_sin, Ndir_cos, yaw_sin, yaw_cos,
  Pab_mean, power`. No assumed sign convention for reconstructed absolute wind.

The corrected frame keeps eight channels and the same per-slot initial mixer
scales. It also preserves the legacy joint-angle missingness mask so it cannot
improve metrics by admitting/removing different windows. Power labels and
causal filling rules are unchanged. Checkpoint transfer, forecast overlay,
utility warm-start and evaluation reject convention or feature-order mismatch.
Corrected pretraining has a distinct checkpoint namespace and records its
convention in the checkpoint and adjacent manifest. Old files are not rewritten.

## Frozen v14 pilot

`configs/sdwpf_angle_semantics_protocol_v14.json` changes only this coordinate
convention. It runs matched new pretraining and fine-tuning for fold1 seed2024,
input336, horizon12, plain trend (Wiki and ramp off). Numerical normalization,
loss (MIXED MSE weight0.2), LR1e-6/new5e-6, strides and raw data stay at the v5
reference values; the independent v6 normalization change is not included.

The source plan is v5's original train-period first80% fit and following10%
internal holdout. The launcher verifies the source protocol, reference
checkpoint, dataset and history hashes. Before training it hashes the selected
train/holdout windows, timestamps, turbine IDs, target labels, availability mask
and target scaler for both frames and requires exact equality. Completed
stages are reused on resume. No outer validation/test inference is run.

Fine-tuning runs eight epochs and evaluates fixed epoch8, without choosing an
earlier epoch. The comparator is the frozen v5 **internal** reference at epoch8
(MAE151.1694846kW, RMSE254.199838kW), not the outer MAE128.1317kW, because these
are different evaluation periods. The internal holdout has already been seen
during development; this pilot is exploratory, not an independent confirmation.

Start once in the server's `timedart` environment:

```bash
bash scripts/train/SDWPF_launch_angle_semantics.sh
bash scripts/train/SDWPF_launch_angle_semantics.sh --status
tail -f "$(cat outputs/logs/SDWPF/angle_semantics_latest.txt)/launch.log"
```

`--resume` is only for an exited, failed controller after the real cause is
fixed; it does not restart a live job or a completed stage. Root artifacts are
`protocol.json`, `provenance.json`, `window_audit.json`, `progress.json`,
`result.json`. The prediction report is `internal_validation/artifacts/`, with
MAE, MSE, RMSE, R², MAE/RMSE Skill, capacity-tolerance hit rates, per-horizon and
power-bin errors, saved predictions, and accuracy/trace/scatter plots.

The prespecified decision requires both fixed-epoch MAE and RMSE to improve.
There is no automatic outer refit or seed expansion. This is an implementation
semantics correction, not a new Wiki innovation or a guaranteed performance gain.
