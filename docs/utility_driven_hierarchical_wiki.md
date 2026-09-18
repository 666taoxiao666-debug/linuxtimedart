# Evidence-Conditioned Utility-Adaptive Wiki (ECUA-Wiki)

## Core mechanism

ECUA-Wiki turns knowledge granularity selection and intervention timing into one
horizon-wise utility decision:

1. **Physical evidence filter.** Observable SCADA rules and semantic confidence
   form a sparse event-factor candidate set. Unsupported factors are exactly zero.
2. **Adaptive knowledge granularity.** The model constructs a trend-only branch,
   a coarse single-event branch using the strongest supported factor, and a fine
   compositional branch using all supported Top-K factors.
3. **Per-horizon gain estimation.** A small utility estimator predicts the
   relative absolute-error reduction of both knowledge branches against the
   trend branch for every forecast step.
4. **Selective intervention.** The higher-utility available granularity is used
   only when its predicted gain exceeds the configured threshold. Otherwise the
   final output is exactly the trend forecast.

For horizon `h` and candidate granularity `g`, the train-only supervision target is

`u*(h,g) = clip((|y_base-y| - |y_g-y|) / max(|y_base-y|, eps), -1, 1)`.

The inference decision is

`g_h = argmax_g u_hat(h,g)` and `intervene_h = 1[u_hat(h,g_h) > tau]`,

subject to the hard physical-availability mask. Validation/test labels never
enter retrieval, granularity selection, or intervention decisions.

## Checkpoint and leakage contract

- The compositional Wiki and trend encoder retain the existing fold-matched
  pretraining checkpoint contract.
- The utility estimator is created only for supervised fine-tuning and is not a
  required pretraining tensor, so existing compositional-Wiki checkpoints remain
  reusable.
- Utility targets are computed only inside the training loop from the current
  training batch. Validation remains read-only and selects checkpoints using the
  configured forecast metric.
- Evaluation requires the same `utility_wiki`, temperature, and minimum-gain
  settings recorded in the fine-tuning manifest.

## Required paper comparisons

1. Trend prompt only.
2. Static compositional Wiki.
3. Physical filter + fixed single-event granularity.
4. Physical filter + fixed compositional granularity.
5. ECUA-Wiki without abstention (`utility_min_gain` forced very low).
6. Full ECUA-Wiki.

Report overall and event-conditioned MAE/RMSE skill, horizon-wise intervention
fractions, single-event/composition selection fractions, abstention fraction, and
within-fold seed variation. The full method is supported only if it improves the
static Wiki across folds while retaining non-trivial abstention.
