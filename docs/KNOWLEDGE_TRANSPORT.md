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
