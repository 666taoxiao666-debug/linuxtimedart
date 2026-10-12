# Expert-bound physical event knowledge

This research change asks when an event correction, rather than merely a
similar event description, can be reused at a later time or on another turbine.
It replaces hidden-state-dependent event residuals in the new pilot with a
declared physical-coordinate residual bank. It is not a reimplementation or a
claimed improvement of the frozen v1 neural semantic-Wiki method.

## Mechanism

For event e and horizon h, define a capacity-normalized physical function
`delta[e,h](z)` and forecast `clip(base[h] + delta[e,h](z), 0, rated_power)`.
Its inputs are historical mean/last power, mean/variability/change of wind,
historical power change, and the base forecast relative to last power. There
is no backbone hidden state and no turbine identifier in these coordinates.
Each event uses a fitted regularized linear function; corrections are bounded
by 20% capacity and their deterministic composition by 5%. This is a specific
new expert class, not a placeholder approximation of the earlier neural bank.

The utility evaluated in calibration is the reduction in original-scale
absolute error of that exact candidate relative to that exact base. Its
contract includes the coefficients, physical normalization, implementation,
event rules, capacity, horizon, base checkpoint and base manifest. Changing
any of these invalidates the evidence, even if tensor shapes or event names
still match. Expert fitting completes before any utility calibration; neither
the expert nor its feature normalization is updated afterward.

## Why transport might work, and its limits

Transport requires overlap in observable physical working points and stable
conditional correction utility. The generic distinction between changing
input distributions and unchanged conditional outcomes is established in
[covariate-shift research](https://www.jmlr.org/papers/v8/sugiyama07a.html).
This implementation does not claim to prove conditional invariance. It checks
two non-overlapping training calibration blocks and balances utility across
source turbines; future hidden faults, control changes or weather can still
break the assumption.

Expert-specific history support is fitted from its training inputs alone using
a fixed 95th-percentile standardized physical-distance radius. Utility cells
use the existing fixed power/wind bins and require 64 windows and 8 source
turbines in each calibration block. Both turbine-balanced MAE and MSE gain
must be positive, with at least two thirds of source turbines showing positive
MAE gain. The routing score is the worse of the two temporal mean gains; it is
a descriptive conservative score, not a nominal confidence bound.

Inference uses the finest supported joint-power/wind cell, otherwise a power
cell, otherwise the event-level cell. A sufficiently supported fine cell with
negative utility vetoes the event: it cannot fall back to a positive pooled
average. History outside the expert's support, insufficient evidence or
nonpositive score yields the exact base forecast. Competing event candidates
are compared using these scores separately at each forecast horizon.

## Frozen pilot

Protocol: `configs/sdwpf_knowledge_transport_protocol_v12.json`.
The previously generated fold1/seed2024 trend OOF forecasts and audited
history are reused; this pilot fits actual new event residual functions.
The original-train OOF calendar is split into first50% expert fitting,
next25% utility calibration (two purged time blocks), and last25% checking.
Turbines with integer ID divisible by5 are excluded from both residual expert
fitting and utility calibration. The final block reports future source
turbines and future residual-held-out turbines separately. The existing trend
base has seen these turbines during earlier training, so this is residual
knowledge transfer, not a cold-start or entirely unseen-turbine result.

Comparisons use the same base and the same frozen experts: base only,
event-level utility only, and physical conditional transport. Report original
MAE/MSE/RMSE/R2, both Persistence Skills, physical/support coverage, intervention,
harm, abstention, and paired temporal statistics. Original labels, earlier
results and outer validation/test are unchanged. The OOF interval has been
inspected in previous development; this remains exploratory train-period
evidence, not independent publication confirmation. A positive pilot does
not automatically trigger an outer evaluation or additional parameter search.

```bash
conda activate timedart
bash scripts/train/SDWPF_launch_knowledge_transport.sh
bash scripts/train/SDWPF_launch_knowledge_transport.sh --status
tail -f "$(cat outputs/logs/SDWPF/knowledge_transport_latest.txt)/launch.log"
```

If a real code/environment failure interrupts the run, inspect the traceback
and use `--resume` only after repair. Saved expert and utility artifacts are
identity-checked and reused, rather than refitted. The run writes
`expert_model.json`, `utility_policy.json`, `provenance.json`, `result.json`,
and the two check prediction files. The runtime prediction API deliberately
accepts no future targets or turbine IDs.

For a compact metric summary and rejection audit of a completed run:

```bash
python scripts/summarize_sdwpf_knowledge_transport.py
cat "$(cat outputs/logs/SDWPF/knowledge_transport_latest.txt)/summary.txt"
```

The summarizer verifies the saved completion hashes and expert identity before
writing new `summary.txt` and `calibration_audit.json` artifacts. It does not
refit any component, change the policy, or rewrite original results.

## Executed pilot: 2026-10-12

The v12 protocol SHA-256 was
`b564c45ae61765e6f8bfc1bf86c06190bfb8a7c78ff0cdd1d978352284585732`.
Run: `outputs/logs/SDWPF/20261012/003_knowledge_transport_h12_f1_s2024_expertbound_time50-25-25_turbmod5_cpu2`.
Fitting used 5,458 source-turbine windows. The two calibration blocks used
1,020 and 991 windows. Expert bundle:
`ba0d7375bea539387fb9967082f0785316642b5684fbda5c94b4abaac22f371f`.

The run completed successfully but found **no supported/accepted utility
cells**. Both the event-only comparison and physical transport abstained on
all check windows and exactly preserved the same base predictions.

- Future source turbines: 2,008 windows, 94 turbines; MAE 118.216866 kW,
  MSE 33,233.140250 kW², RMSE 182.299589 kW, R² 0.468039.
  Persistence-relative MAE/RMSE Skill: 1.909498% / 1.179826%.
- Future residual-held-out turbines: 498 windows, 22 turbines; MAE
  116.089386 kW, MSE 32,124.959810 kW², RMSE 179.234371 kW, R² 0.469129.
  Persistence-relative MAE/RMSE Skill: 2.376139% / 2.811122%.
- Both checks: gain versus the same base 0 kW; intervention 0%, abstention
  100%. Harm among interventions is undefined, not a measured 0% harm rate.
  Zero-gain bootstrap intervals and DM p=1 only reflect identical forecasts.

The training-side rejection audit explains why this is not simply a strict
threshold problem. The gust expert had 886 fit windows. Across all 12 horizons
its in-support calibration MAE gains were negative: block1 -16.386442 to
-0.299115 kW; block2 -187.231268 to -31.340584 kW. Block2 also had only 38–42
in-support windows per horizon, below the fixed 64-window requirement.
The low-wind expert had 2,122 fit windows and mixed gains in block1, but no
in-support observations in block2. High-wind/low-power had only 60 fit windows
and was inactive; rated saturation had none. No check labels were used to
change these gates or experts.

This establishes an executable expert/evidence identity contract and exposes
insufficient or harmful transport evidence. It does **not** establish useful
knowledge transport or higher accuracy. In particular, the new linear physical
experts are not yet adequate replacements for the old semantic adapters. The
original frozen v1 results and model defaults remain unchanged. Any next
expert-capacity or training-coverage change requires a separate declared
protocol; lowering these gates cannot be presented as evidence of invariance.

Both check groups have generated `forecast_accuracy_overview.png` under their
own `_artifacts` directories. Their captions identify the original-train
scope and residual-only held-out status. These figures were numerically
checked against the saved predictions and visually inspected.

## Local empirical expert follow-up (v13)

A source-fit and calibration audit found a concrete v12 expert failure. The
gust expert's average correction changed from +13.852606 kW in the fit block
to -199.831798 kW in late calibration. 34.94% of its late-calibration physical
candidate corrections reached the 300 kW bound. Mean actual residual in that
block was +35.400153 kW. Correlated level features and the base-change feature
contributed large extrapolated offsets after the physical distribution shifted.
These diagnostics use original-train source turbines only.

The declared follow-up changes the expert to a local empirical function. For
each event and horizon it retrieves 32 source-fit cases in the same ten physical
coordinates, and predicts their turbine-balanced weighted median residual.
Each represented source turbine receives equal total mass. A local correction
cannot extrapolate beyond the residual values represented by its neighbors.
The same final capacity bounds apply. Fit-only leave-one-turbine-out neighbor
distances define a 95th-percentile support radius; at least eight neighboring
turbines must support a query. This is an empirical support test, not a guarantee
that target conditional errors are invariant.

All stored cases, their residuals, feature scaling, neighbor rules and support
radius are frozen before utility calibration and bound to the actual expert's
hash. Updating case memory creates a new expert identity and requires new
utility evidence. v12 code remains byte-identical so that old bundle identities
can still be checked. v13 is a new expert family, not a retroactive correction
of v12's published result artifacts.

Protocol: `configs/sdwpf_local_knowledge_transport_protocol_v13.json`.
The source fit/calibration/check dates, turbine split, utility acceptance rules,
capacity bounds and comparison methods remain those of v12. In addition to the
final checks, `train_candidate_audit.json` compares the actual v12 and v13
corrections on the same source fit and calibration windows. Fit measurements
are in-sample descriptions. The calibration comparisons do not change the
frozen neighbor count, support rule, thresholds or seed.

```bash
conda activate timedart
bash scripts/train/SDWPF_launch_local_knowledge_transport.sh
bash scripts/train/SDWPF_launch_local_knowledge_transport.sh --status
RESULT_DIR="$(cat outputs/logs/SDWPF/local_knowledge_transport_latest.txt)"
python scripts/summarize_sdwpf_knowledge_transport.py --result-dir "$RESULT_DIR"
cat "$RESULT_DIR/summary.txt"
```

### Executed v13 pilot: 2026-10-12

Protocol SHA-256:
`3817c553c45ac312740b64ff6a7ded7b0bec963271f98db943fda03814a10d36`.
Run: `outputs/logs/SDWPF/20261012/004_local_know_hed6097d92412_h12_f1_s2024_knn32_wind8_time50-25-25_cpu2`.
Expert bundle:
`d083b02dc5eb8c1ee6494eba04ec1fd3d657aecbdc880bbebb60b06b688715ea`.
Policy:
`ecb8af2cbbc62396494a3201dff94fda6f6edbe31ecb8db64ade430f0ff4d824`.

The local expert reduced the linear extrapolation failure, but did not establish
profitable transport. On the same 104 late-calibration gust windows, the mean
correction changed from -199.831798 to -81.939571 kW, and the fraction reaching
the correction bound fell from 34.935897% to 0.641026%. The all-physical-candidate
MAE gain improved from -89.573695 to -34.016187 kW; it remained negative.
The early-calibration gust candidate gain similarly improved from -91.634848
to -34.342608 kW. These compare identical physical windows before either
expert's support or utility filter, not selected final predictions.

Low-wind early-calibration candidate MAE gain improved from -19.133791 to
+4.032353 kW on the same 397 windows. Its RMSE gain was still -13.063776 kW,
so this was not a joint accuracy improvement. There were no low-wind physical
event windows in the later calibration block. For gust, local in-support
counts in that block were only 21–26 per horizon, and turbine-balanced MAE
gains were negative at every horizon (-118.074620 to -1.917185 kW).

Consequently no utility cell met the unchanged two-block requirements.
Event-only and physical-conditional policies both abstained on every check
window. The two final checks exactly reproduced v12's base-only metrics:
source turbines MAE 118.216866 / RMSE 182.299589 kW / R² 0.468039, and
residual-held-out turbines MAE 116.089386 / RMSE 179.234371 kW / R² 0.469129.
Intervention was 0%, abstention 100%, and net improvement 0 kW; intervention
harm is undefined. These are original-train development checks, not outer
validation/test scores or an accuracy improvement attributable to Wiki.

The evidence/executor identity contract is implemented and tested. The
remaining scientific gap is useful conditional error stability: similar
observed histories and bounded local residuals alone do not establish it.
The current data also lack temporally repeated calibration support for some
events. Lowering the gates or calling abstention a successful knowledge
transfer would not address either problem. This pilot is retained as a
negative result; no automatic outer refit, extra seed, or threshold search
follows it.
