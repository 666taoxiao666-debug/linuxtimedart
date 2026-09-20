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
4. **Decision-aligned supervision.** A balanced train-only three-way objective
   directly teaches abstention, single-event retrieval, and compositional
   retrieval. Its abstention logit is the same minimum-gain boundary used at
   inference, so utility regression and the actual routing decision cannot
   silently optimize different tasks.
5. **Selective intervention.** The higher-utility available granularity is used
   only when its predicted gain exceeds the configured threshold. Otherwise the
   final output is exactly the trend forecast.
6. **Candidate specialization.** On training data only, every physically
   available event/composition branch is fitted to the target and ranked against
   a detached trend baseline. This prevents hard abstention from starving the
   knowledge branches before the utility estimator learns to select them.
7. **Isolated residual adaptation.** Single-event and composition knowledge use
   separate zero-initialized residual adapters over their prompt-induced feature
   contrasts. Candidate supervision updates these adapters without pulling the
   shared trend head away from its validated solution.

For horizon `h` and candidate granularity `g`, the train-only supervision target is

`u*(h,g) = clip((|y_base-y| - |y_g-y|) / max(|y_base-y|, eps), -1, 1)`.

The inference decision is

`g_h = argmax_g u_hat(h,g)` and `intervene_h = 1[u_hat(h,g_h) > tau]`,

subject to the hard physical-availability mask. Validation/test labels never
enter retrieval, granularity selection, or intervention decisions.

The decision loss uses the same rule to construct a training action:
`abstain` when neither available branch exceeds `tau`; otherwise it labels the
larger realized-gain branch. Losses are averaged per present action class before
being combined, preventing frequent abstention windows from suppressing rare
but useful compositional events.

Once a candidate passes the threshold, its blend strength starts at
`utility_intervention_floor` and grows with the predicted gain. Thus a reported
intervention is a material correction rather than a numerically negligible gate.
The shared forecast modules and isolated Wiki modules use separate learning
rates: the trend path remains conservative while the utility estimator and
residual adapters learn sparse selection/correction more quickly.

## Checkpoint and leakage contract

- The compositional Wiki and trend encoder retain the existing fold-matched
  pretraining checkpoint contract.
- The utility estimator is created only for supervised fine-tuning and is not a
  required pretraining tensor, so existing compositional-Wiki checkpoints remain
  reusable.
- Utility targets are computed only inside the training loop from the current
  training batch. Validation remains read-only and selects checkpoints using the
  configured forecast metric.
- Evaluation requires the same `utility_wiki`, temperature, minimum-gain, and
  intervention-floor settings recorded in the fine-tuning manifest.

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
