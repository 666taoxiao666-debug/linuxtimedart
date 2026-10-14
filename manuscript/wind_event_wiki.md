# Execution-Bound Event Knowledge for Wind Power Forecasting under Temporal Shift

Research manuscript draft, 14 October 2026. This is the canonical working text,
not a submission package. The completed experiments below evaluate the frozen
transition Wiki and forecast-relative extension. The episode-balanced extension
has a working implementation; its independent forecast evaluation is pending.
Venue and final manuscript format remain undecided.

## Abstract

Event knowledge can supplement a numerical wind-power forecaster, but physical
relevance does not establish that an event correction will improve a future
prediction. We study this distinction through execution-bound knowledge: every
benefit estimate is attached to the exact correction rule, memory, numerical
forecaster, normalization and forecast horizon that produced it. A structured
event memory supplies candidate power transitions, while chronological training
evidence determines the supported condition and lead time for intervention. A
study on six Kelmarsh turbines shows that correct execution identity and positive
benefit in two training periods are insufficient for temporal transfer. Across
nine fold–seed runs, the frozen Wiki increases mean absolute error from 141.668 to
141.973 kW relative to its numerical base. A forecast-relative readout reduces
this degradation in a three-fold pilot but does not outperform the base jointly
in MAE and RMSE. Training-only inspection further finds that 32 retrieved windows
typically represent three or four 72-hour time blocks, despite covering five
turbines. These findings motivate an episode-balanced, timestamped memory that
separates shared weather support from turbine diversity. The resulting research
question is when executable event knowledge remains useful after a change of
time or turbine, rather than whether retrieval alone increases model capacity.

## 1. Introduction

Short-term wind-power forecasting supports operational decisions under variable
weather and turbine control. Numerical sequence models can describe regular power
evolution, but unusual operating conditions may require information that is sparse
in the training distribution. An event memory offers an explicit representation
of such information: an observed condition identifies relevant historical cases,
and their outcomes suggest a correction to the numerical forecast.

Three different propositions are involved. The event must be compatible with
current measurements; its historical cases must cover the query conditions; and
the correction produced from those cases must improve the actual forecaster.
Conflating them makes it possible to retrieve physically plausible but harmful
knowledge. Updating the numerical base or correction rule introduces a second
problem: benefit measured for the previous execution path may no longer describe
the revised model.

Wind forecasting already includes regime-dependent calibration and temporal or
spatial adaptation [1,2]. Knowledge-augmented language systems likewise distinguish
retrieval, support and usefulness [5], and organize evolving memories through
structured notes and links [6,7]. These findings motivate a more specific question
for numerical prediction: what is the unit of reusable evidence for an executable
event correction, and what changes invalidate that evidence?

We investigate execution-bound knowledge through a reproducible event-memory
pipeline. Its contribution at the current stage is an explicit model–evidence
contract and an empirical separation between execution consistency and temporal
transport. We also implement an episode-balanced extension that prevents numerous
correlated donor windows from dominating a local outcome distribution solely
through their multiplicity. Its forecast contribution must be established by the
pending evaluation rather than inferred from its construction.

## 2. Related work

### 2.1 Wind forecasting and adaptation

Calibrated regime-switching forecasting models the correction as a function of
the current regime and its duration, and updates forecasts in chronological rolls
[1]. Meta-learning-based probabilistic forecasting separates offline adaptation
across location/lead-time tasks from online updates using recently observed
outcomes [2]. These methods show that adaptation has a temporal protocol, not
just a shared representation. Our current forecaster is static during evaluation;
adding case timestamps does not implement their online-learning algorithms.

Temporal collaborative attention combines long-range temporal representations,
variable interactions and short-range information [3]. Its evaluated temporal
aggregation and attention operations differ from our independently issued hourly
SCADA forecasts, so its published errors are not comparable to ours. Regional
conformalized forecasting separates point prediction, conditional uncertainty
and calibration [4]. We adopt this separation as an evidence-design principle,
while retaining raw power observations that are incompatible with a bounded
logit-power assumption.

### 2.2 Retrieval and structured memory

Self-RAG evaluates retrieval demand, passage relevance, support and output
usefulness separately [5]. HippoRAG organizes document relations for associative
retrieval [6], whereas A-Mem constructs timestamped notes and updates their
contextual relations [7]. A-RAG exposes keyword, semantic and chunk-level access
to a retrieval agent [8]. Accordingly, hierarchical access, memory linking and
selective retrieval are established ideas. Our numerical problem differs in its
evidence target: the signed error reduction of a particular correction against
a particular base at a particular lead time.

Nearest-neighbor forecasting already retrieves cached time-series outcomes [9].
The memory/readout component therefore is not presented as an original invention
of nearest-neighbor prediction. The distinction to investigate is whether
execution-bound, time-aware benefit evidence supports selective use of those
outcomes under temporal and turbine changes. A semantic relation between events
is not treated as an additive causal effect.

## 3. Problem setting

Let x(i,t) contain the observations available for turbine i at forecast origin t.
The numerical base f produces a two-step hourly forecast b(i,t,h), h=1,2, from
the preceding 72 hours. Inputs include wind speed, ambient temperature, nacelle
heading represented by sine/cosine, mean blade pitch, rotor speed and power.
Future weather and future control signals are unavailable.

The observed power y may be negative or slightly exceed rated capacity; such
measurements are retained. The nominal 2050 kW capacity C normalizes transitions
and bounds corrections, not target values. An event candidate is

    candidate(i,t,h,e) = anchor(i,t) + C * transition(i,t,h,e).

Its correction is candidate minus b, bounded in magnitude by 0.2C. The four
event families are hourly wind variability, high-wind/low-power operation,
near-rated operation and low-wind idle. These are observable-condition categories;
hourly variability is not a direct measurement of turbulence or a verified fault.

## 4. Execution-bound event memory

### 4.1 Observable conditions and executable cases

Each case records an event category, historical physical/control coordinates,
source turbine and observed future power transition. The new implementation
additionally preserves every target timestamp. A case can be queried only after
its complete stored outcome trajectory is available. The hourly forecast origin
is one hour before the first target timestamp.

The execution contract identifies the case bank, coordinate transformation,
readout, physical rules, scaler, base checkpoint and forecast horizon. Utility
evidence is attached to the resulting expert identity. Changing its readout or
base invalidates the evidence and requires reassessment. Editing a human-readable
event explanation cannot manufacture a numerical benefit observation.

### 4.2 Forecast-relative distributional readout

For a query, retrieve K=32 physically similar cases and their normalized power
transitions z. Let b denote the numerical forecast change relative to the last
observed power, expressed in the same units. A local distribution supplies its
weighted median m and weighted mean mu. Define

    d = clip(m - b, -0.2, 0.2),
    alpha = clip((mu - b) / d, 0, 1),
    q = b + alpha*d,

with alpha=0 for d=0. The squared-loss minimizer on the segment from b to b+d is
the projection of mu onto that segment. Since b belongs to the segment, its
empirical squared loss cannot increase. The segment also points toward a weighted
median, so convexity makes empirical absolute loss non-increasing along it.
This is a local property for the stored weighted outcomes, not a statement about
future query labels or distribution shift.

### 4.3 Episode-balanced probability mass

Balancing source turbines does not balance shared weather periods. We group
stored cases into fixed 24-hour UTC blocks. Within a retrieved neighborhood, each
represented block receives equal total mass, each turbine within that block
receives equal mass, and cases within a block–turbine pair share that mass:

    w(j) = 1 / (number_of_blocks * turbines_in_block(j)
                * cases_in_block_and_turbine(j)).

These weights replace the turbine-only weights in the same readout. Coordinates,
neighbor count, base, memory outcomes, physical rules and correction bound remain
fixed. A matched turbine-only control uses the same timestamp checks and expert
API. The block length is predeclared, not selected using outer accuracy.

Block balance is a probability-mass convention. Adjacent blocks and 72-hour
history windows remain dependent. We report block count and the Kish weight count
as descriptive concentration diagnostics, not independent sample sizes or a
transport confidence bound.

### 4.4 Benefit-driven granularity and intervention

Chronological donor-only utility periods evaluate the actual new candidate
outputs. For each lead time and event, evidence can be grouped at event level,
power-bin level or joint power/wind-bin level. A cell requires the frozen minimum
support and positive MAE/MSE gain in both periods, with at least two thirds of
represented turbines showing positive MAE gain. The finest supported cell is used;
a supported harmful fine cell blocks a coarse fallback.

The routing score is the smaller of the two-period realized mean MAE gain and
the query's local empirical gain. Among available candidates, the highest
positive score is selected; otherwise the forecast equals the base exactly.
The minimum of two means is an empirical stability rule, not a lower confidence
bound. Composition is disabled in this extension because adding multiple
conditional future-level forecasts would double-count the same base.

### 4.5 Knowledge evolution and its current boundary

A deployable lifecycle would create a new expert revision when cases are added,
merged or removed, reassess its actual outputs using available chronological
evidence, and retain the old revision for comparison. Automatic online revision,
learned forgetting and cross-farm transfer are planned studies. The current
implementation provides immutable timestamped cases and evidence invalidation;
it does not yet demonstrate a self-evolving agent or causal event intervention.

## 5. Experimental design

### 5.1 Data and chronological protocol

Kelmarsh SCADA observations are obtained from the public dataset [13]. Six
ten-minute records form one hourly bin, labeled by the bin end in UTC; directional
variables use circular aggregation. Missing input hours remain missing. A window
requires complete historical inputs and valid future power, but does not require
future weather validity. The study starts with the first full calendar month
after commercial operation, May 2016.

Three rolling folds end outer training at 60%, 70% and 80% of the declared
timeline, respectively; each uses the following 10% for outer validation. The
final 10% is reserved and has not been evaluated. Within each outer training
period, 60% fits the numerical model, 10% selects its epoch, 10% constructs the
case memory, and two subsequent 10% periods assess utility. Memory/utility labels
come from turbines 1–5. Turbine 6 is held out from knowledge fitting, but has
been observed by the numerical base; this is not unseen-turbine neural transfer.

### 5.2 Comparisons and measures

The numerical comparisons are Persistence, daily Persistence, lag Ridge, DLinear
[10], PatchTST [11], and the project's PromptTimeDART adaptation of TimeDART [12].
The latter uses historical regime prompts, not an LLM prediction of future trend.
Matched-label-budget Ridge residual controls use the same donor memory/utility
labels as the Wiki. Frozen runs use three seeds per fold. The forecast-relative
pilot uses seed 2024 on three folds and must be reported separately.

MAE, MSE, RMSE and R² are computed on original kW values. MAE Skill and RMSE Skill
are 100*(1-error_model/error_Persistence). Capacity-tolerance hit rates count
errors within 5% or 10% of C; they are not classification accuracy. Wiki reporting
includes intervention coverage, abstention, selected gain and harm. A harm window
has greater mean absolute error than its base after intervention. Undefined harm
when no action occurs remains undefined.

Paired statistical analysis resamples issue-time clusters while retaining the
associated turbine predictions and reports time-block bootstrap intervals and
Newey–West-adjusted Diebold–Mariano statistics. Macro R² means the mean of run
R² values, not a pooled R². Development validation has been inspected during
earlier research, so new confirmatory conclusions require a separately frozen
future evaluation.

## 6. Completed findings

### 6.1 The frozen Wiki does not improve overall precision

Across the nine frozen Kelmarsh runs, mean MAE/RMSE/R² are 141.798/209.250/0.745214
for Persistence, 141.668/206.013/0.752596 for PromptTimeDART, and
141.973/206.611/0.751334 for the full Wiki. The matched-label numerical residual
control achieves 141.503/206.449/0.751677. Thus the present event organization
does not establish an accuracy advantage over its numerical base or label-budget
control. These are outcomes of the specified implementations and budgets, not
general rankings of the named architecture families.

The Wiki intervenes in 9.004% of windows on average. Only two of nine runs have
nonzero intervention; both have negative net gain. The mean harm rate among
these active runs is 50.898%. The remaining runs fall back exactly to the base,
so their harm rates cannot be averaged as zeros. The active paired confidence
intervals include zero.

### 6.2 Execution consistency does not establish temporal transport

In the three-fold seed-2024 pilot, MAE/RMSE are 142.495/205.715 kW for the numerical
base, 143.294/207.279 for the legacy Wiki, and 142.682/206.607 for the
forecast-relative readout. The extension reduces the legacy error but does not
jointly outperform the base. Its active fold has positive utility-period gains
of approximately 2.358 and 2.278 kW, yet negative outer gain of 0.560 kW. This
provides a concrete counterexample to treating two positive training periods and
correct expert identity as sufficient transfer evidence.

### 6.3 Multiple donor turbines do not imply many weather episodes

Training-only inspection of the late utility queries finds median retrieved donor
diversity of five turbines in all folds. However, 32 cases correspond to median
72-hour block counts of 3, 4 and 4; their median daily counts are 6, 5 and 7.
These counts identify evidence concentration. They do not establish that this
concentration caused the observed outer degradation.

### 6.4 Episode-balanced extension

A paired training-only probe refits utility evidence for each readout, reusing
the same numerical checkpoints. In fold 0, episode weighting increases early/late
MAE from 106.829/137.036 to 108.256/138.977 kW relative to the matched
turbine-weight control. Both RMSE values also increase. In fold 1, it reduces
early/late MAE from 152.480/151.280 to 151.769/150.956 kW, with RMSE reductions
from 228.897/220.358 to 228.337/220.046 kW. Fold 2 has zero intervention in
both variants. The fitted utility data are reused for these diagnostics, so this
is descriptive training evidence, not an independent forecast comparison.

The mixed directions do not support a global replacement of the current readout
or selecting only the favorable fold. Independent outer MAE/RMSE/R², coverage,
harm and paired intervals for this extension are TBD. No numerical parameters
were retrained and no new outer evaluation was launched for the probe.

## 7. Discussion and conclusion

The experiments separate three questions that are often conflated in a
knowledge-augmented predictor: whether an event is observable, whether benefit
evidence describes the executing expert, and whether that benefit transfers to a
new period. Correct execution identity addresses the second question; it cannot
by itself answer the third. The current results favor a research program about
conditional knowledge validity rather than a claim of established Wiki-driven
accuracy improvement.

The scope is six turbines at one site, short history-only forecasts and limited
support for rare operating conditions. Correlated weather, missing control
signals and repeated development validation constrain interpretation. The next
discriminating evaluation compares episode-balanced and turbine-balanced experts
under a frozen future-period protocol, then separates temporal transport from
held-out-knowledge turbine performance. Automatic knowledge merging or forgetting
must preserve revision-specific benefit evidence before it can support a
self-evolution claim.

## References

1. Spatio-temporal short-term wind forecast: a calibrated regime-switching method. Annals of Applied Statistics, 2019. https://doi.org/10.1214/19-AOAS1243
2. Meng, Z., Guo, Y., Sun, H. An Adaptive Approach for Probabilistic Wind Power Forecasting Based on Meta-Learning. IEEE TSTE, 2024. https://doi.org/10.1109/TSTE.2024.3379835
3. Hu, Y., Liu, H., Wu, S., Zhao, Y., Wang, Z., Liu, X. Temporal collaborative attention for wind power forecasting. Applied Energy, 2024. https://doi.org/10.1016/j.apenergy.2023.122502
4. Jonkers, J., Nieves Avendano, D., Van Wallendael, G., Van Hoecke, S. A novel day-ahead regional and probabilistic wind power forecasting framework using deep CNNs and conformalized regression forests. Applied Energy, 2024. https://doi.org/10.1016/j.apenergy.2024.122900
5. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR, 2024. https://arxiv.org/abs/2310.11511
6. Jiménez Gutiérrez, B., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS, 2024. https://proceedings.nips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html
7. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. A-Mem: Agentic Memory for LLM Agents. NeurIPS, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html
8. Du, M., Xu, B., Zhu, C., Wang, S., Wang, P., Wang, X., Mao, Z. A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces. arXiv:2602.03442, 2026.
9. Zhang, H., Nie, P., Sun, L., Boulet, B. Nearest Neighbor Multivariate Time Series Forecasting. IEEE TNNLS, 2025. https://doi.org/10.1109/TNNLS.2024.3490603
10. Zeng, A., Chen, M., Zhang, L., Xu, Q. Are Transformers Effective for Time Series Forecasting? AAAI, 2023. https://doi.org/10.1609/aaai.v37i9.26317
11. Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers. ICLR, 2023. https://openreview.net/forum?id=Jbdc0vTOcol
12. Wang, D., Cheng, M., Liu, Z., Liu, Q. TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation. ICML, 2025. https://proceedings.mlr.press/v267/wang25r.html
13. Kelmarsh wind farm SCADA data. Public dataset record, CC BY 4.0. https://zenodo.org/records/16807551

## Reproduction and remaining evidence

Completed result sources: `output/kelmarsh_hourly/review_5629269/` and
`output/kelmarsh_hourly/model_review/`; protocol definitions:
`configs/kelmarsh_hourly_protocol_v1.json` and
`configs/kelmarsh_pareto_protocol_v2.json`.
New executable path: `utils/kelmarsh_episode_wiki.py`; training-only probe:
`scripts/audit_kelmarsh_episode_wiki.py`.

Remaining empirical work: episode-balanced future-period comparison, exact
paired temporal statistics, source/target turbine separation, a close retrieval
forecasting baseline, and independent multi-site evaluation. The current
manuscript does not report these as completed results.
