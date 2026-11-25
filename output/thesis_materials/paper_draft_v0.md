# Which exogenous information matters for turbine-level ultra-short-term wind power forecasting? A pre-registered three-farm study of numerical weather prediction, upstream observations and alternative predictors

Working draft opened 17 November 2026. Literature and interpretation updated 21 November 2026; Section 5.6 (upstream value under NWP latency) and the X4 adaptation results added 22 November 2026. The matched-coverage analysis (Section 5.6), the comparison with three published wind-power models (Section 5.7) and the evidence status of each claim added 25 November 2026. The primary sealed comparison and later exploratory analyses are identified in the text.

---

## Abstract

For a wind farm with access to a numerical weather forecast, how much do upstream observations still improve turbine-level power predictions? We evaluated hourly forecasts at 1–4 h for 41 turbines on three UK farms. A factorial comparison gave the same local SCADA predictor upstream station observations, GFS forecasts, both sources, or neither, under a six-hour forecast-cycle availability rule. On first-opened test data, adding GFS to a predictor already using upstream observations reduced mean absolute error (MAE) by 8.7% at Kelmarsh and 8.6% at Penmanshiel, and by 6.9% and 7.2% in a single pre-registered confirmation spanning two held-out years at Hill of Towie; all four 95% intervals excluded zero. Conversely, upstream observations reduced TSMixer MAE by 5.7–11.0% without GFS but by only 0.4–1.5% with GFS, an increment whose interval excluded zero in two of the four test sets. On development data with single-seed LightGBM, the upstream increment was 1.82–2.28 kW when GFS cycles were selected 6 h after issue and 15.50–24.38 kW at 18 h, when the extracted forecasts no longer covered any target hour. A post-hoc analysis at 15 h partly separated the two changes: truncating a recent cycle to the lead coverage of the older cycle raised the upstream increment by 3.75–12.80 kW, whereas switching to the older cycle at matched coverage had no detectable additional effect. Three published wind-power models, adapted with a single seed to the same development windows, were less accurate than LightGBM given the same information in 29 of 30 paired comparisons, with one difference not detected; for the two that use GFS, the upstream increment also rose from 6 h to 18 h at all three farms. An upstream timing correction, fine-tuning and ensembling of Chronos-2, and cross-farm adaptation missed their pre-registered thresholds. For these sites, a forecaster with station access gained consistently from an additional GFS feed. On these development sets, the station feed yielded larger MAE reductions when the extracted GFS forecasts covered fewer target hours.

**Keywords:** wind power forecasting; numerical weather prediction; upstream observations; information decomposition; pre-registration; negative results

---

## 1 Introduction

Hourly turbine-level forecasts support decisions made before the next four hours of generation are observed. A surface station upwind of a farm can observe a weather change before it reaches the turbines. Spatial forecasting models exploit this lead and the dependence between nearby sites (Gneiting et al., 2006, Hering and Genton, 2010, Tastu et al., 2011). A numerical weather prediction (NWP) model also represents regional weather evolution. This creates a practical question for an operator who already receives one feed. How much does acquiring the other reduce forecast error?

The unit of prediction changes the comparison. A six-hour forecast of total farm output averages over turbines, whereas an hourly turbine-level forecast retains local variation and turbine operating states. The timing of the weather input also matters. A forecast issued before a weather change and a measurement taken during it answer different operational questions. Here we vary the weather inputs on the same forecast windows and keep the issue-time rules fixed.

This paper asks four questions under a protocol designed to avoid these pitfalls:

1. How much does each exogenous source (upstream observations, NWP) contribute, and how much of the upstream contribution remains once NWP is available?
2. Does the incremental NWP gain persist on two reserved years at a third farm, and does the ordering of source contributions persist with a second predictor?
3. How does the residual value of upstream observations change when the available NWP forecast is older, and does the change follow forecast age or the loss of forecasts for the target hours?
4. Do commonly proposed mechanisms for using NWP better (displacement correction, anchored increments, bias correction), or alternative predictors, including recently published wind-power models, add anything beyond a well-specified full-information baseline?

**Contributions.**

1. A leakage-free hourly turbine-level benchmark on three public UK wind farms, combining SCADA, MIDAS upstream stations and GFS forecasts with a causal forecast-cycle rule (Section 3–4).
2. A 2 × 2 comparison of the two weather sources, with paired inference by forecast issue time. The pre-registered third-farm test covers the increment from adding NWP to the upstream model (Sections 5.1–5.2).
3. A second decomposition using LightGBM on development data, with the same ordering of source contributions as TSMixer (Section 5.3).
4. A latency curve for the upstream feed: the same predictors re-trained with GFS cycles selected under delays of 6–18 h, and a post-hoc matched-coverage analysis that separates the loss of target-hour forecasts from forecast age at a 15 h delay (Section 5.6).
5. Matched comparisons that quantify how much selected NWP corrections, foundation-model adaptations and three published wind-power models change error relative to LightGBM given the same information (Sections 5–6).

---

## 2 Related work

**Spatial information and forecast timing.** Tastu et al. (2011) traced the spatial dependence of short-term wind power forecast errors across sites in western Denmark. Liu et al. (2023) compared measured and forecast wind-speed sequences, scored candidate time lags and combined the corresponding NWP forecasts for power prediction. Peduto et al. (2026) used sparse, irregular marine observations to update GFS winds at other locations. These studies motivate distinct controls for our upstream correction. Offsite measurements can carry information about a forecast error even when they do not identify its timing. We therefore compare a timing model with predictors that receive the station information without being required to estimate a lag. A useful correction to marine wind speed also has to survive the conversion from wind to turbine power to yield a gain on our endpoint.

**Information sources and comparison across farms.** Spiliotis and Theodorou (2025) compared 21 wind-speed bias corrections using hourly measurements from 75 turbines at ten Greek farms and forecasts supplied a day ahead. The accuracy after correction depended on the quality of the original wind forecast. Deignan et al. (2025) studied transfer of model settings for six-hour farm-level forecasts in Ireland and the UK, including Kelmarsh and Penmanshiel. Xu et al. (2025) compared neural forecasters on eight turbine- and farm-scale datasets. WPBench extends comparisons to 26 public datasets and 19 models and evaluates forecast-curve fidelity as well as point error (Zhu et al., 2026, preprint). Its Kelmarsh case uses ten-minute records from the same six turbines studied here. Our comparison holds the turbine, forecast origin and target hours fixed within each farm. Crossing upstream observations with an issued GFS forecast lets us estimate both the value of an upstream feed on its own and the value it retains once the GFS feed is available.

**Foundation models and transfer.** Chronos-2 can use known future covariates in multivariate forecasting (Ansari et al., 2025). DeepWind learns from a wind-specific pretraining corpus and evaluates zero- and few-shot probabilistic forecasting across eight datasets (Wang et al., 2026). Wind-specific pretraining and adapting a general model have different data requirements. In our comparison, Chronos-2 receives GFS as a future covariate. For target farms with little labelled history, a LightGBM model trained with data from the other farms provides a reference for what ordinary supervised transfer can achieve under the same calendar cutoff.

**Recent wind-power models.** Bouche et al. (2023) combined ECMWF forecasts with local observations for 10-min to 4-h predictions at three French farms, including direct power regression with Nyström kernel ridge regression. Wu et al. (2025) proposed Cabin, which fuses power history with ambient variables through ambient representation modules, temporal mixing and Kolmogorov–Arnold networks. Stratigakos and Andrianesis (2026) learned partitions of the set of possibly missing features, with linear decision rules that adapt a forecaster to the features available at prediction time. We adapt each model to our windows and compare it with LightGBM given the same information (Section 5.7).

---

## 3 Data

### 3.1 Wind farms

| Farm | Turbines | Rated (kW) | Data used | Fit / validation / test |
|---|---|---|---|---|
| Kelmarsh (England) | 6 × Senvion MM92 | 2050 | 2016-05 … 2022 | 2016-05–2020 / 2021 / 2022 |
| Penmanshiel (Scotland) | 14 × Senvion MM82 | 2050 | 2016-08 … 2021 | 2016-08–2019 / 2020 / 2021 |
| Hill of Towie (Scotland) | 21 × Siemens SWT-2.3-VS-82 | 2300 | 2016 … 2025 | 2016–2023-06 / 2023-H2 / **2024, 2025 sealed** |

<!-- sources: configs/g1_protocol.json path_T; configs/g2b_protocol.json case_A; output/x1/A0/results.json windows -->

SCADA records (10 min) are aggregated to hour-end means; a label is kept only when all six 10-min values are present, and labels are never imputed, clipped or filtered. Hill of Towie (HoT) underwent an aerodynamic upgrade (AeroUp) between 2021-09 and 2023-05; windows overlapping a turbine's works period are excluded, and a retrofit flag is an input. A pre-registered diagnostic (same calendar months before/after) found post-retrofit power at 5–11 m/s nacelle wind 6–23 % lower for 18 of 21 turbines; we report this as an association, not as an effect of the retrofit.

### 3.2 Upstream observations

The registered offsite rosters contained 36 MIDAS stations for Kelmarsh, 11 for Penmanshiel and 12 for Hill of Towie. At each issue time the feature generator summarised the available stations, including those in the upwind sector. Eleven handcrafted features describe wind speed, its three-hour change and the surface pressure gradient. Station observations enter with a one-hour delay.

<!-- Development metadata: output/g1/preparation_78/prepared/{Kelmarsh,Penmanshiel}/calibration.json (exclude the farm entry); output/hill_of_towie_e1/l0b_r2/qualified_stations.json. -->

### 3.3 NWP

We extracted GFS 0.25° 10 m wind at the farm centre and station locations. At forecast origin t0, cycle C is eligible when C + 6 h ≤ t0. This assigns cycle ages of 6–11 h using a fixed publication allowance. Forecast leads from 3 to 18 h cover the four power targets. If an eligible cycle failed integrity checks, we marked it unavailable for every model. We used the 10 m product because the 100 m level could not be verified throughout the study period.

---

## 4 Evaluation protocol

- **Windows.** 24 h of hourly input, targets at t0 + 1 … 4 h, stride 3 h per turbine. Every model in a comparison is evaluated on exactly the same windows.
- **Splits.** Chronological by year. All model and hyper-parameter choices use the validation year only.
- **Sealed confirmation.** HoT 2024 and 2025 were hashed at download and opened once, after the protocol, the 14 fitted models and the evaluation code were frozen and verified; a rehearsal on the 2023 validation half reproduced all frozen validation errors to within 3 × 10⁻⁵ kW before the sealed years were opened.
- **Metrics.** MAE (primary), RMSE, R² and the accuracy used by Chinese grid codes, 1 − RMSE / rated power. Because HoT test years are entirely post-retrofit while the fit period is not, absolute errors at HoT are reported "under post-retrofit conditions".
- **Inference.** Paired per-window loss differences, clustered by issue time; 95 % moving-block bootstrap intervals (block 8 issue times, 5000 replicates) and Newey–West tests (lag 7); block 56 as sensitivity. Multi-seed models are compared on seed-averaged losses. For the sealed confirmation, the p-value of each question is the larger of the two years, with Holm correction across questions.

---

## 5 Results

### 5.1 Information decomposition (TSMixer)

The neural predictor N is a TSMixer (Chen et al., 2023) trained with a 0.8 MAE + 0.2 MSE loss, early stopping on the validation year and three seeds. Four information sets are compared: local SCADA only (B0), + upstream features (B), + NWP (N0), and both (N).

**Table 1.** Test MAE (kW, seed mean) and paired gains.

| | Kelmarsh 2022 | Penmanshiel 2021 | HoT 2024 (sealed) | HoT 2025 (sealed) |
|---|---|---|---|---|
| B0 local only | 180.15 | 195.87 | 228.88 | 225.25 |
| B + upstream | 160.37 | 184.79 | 215.67 | 208.11 |
| N0 + NWP | 147.05 | 170.73 | 201.67 | 196.05 |
| N + both | 146.46 | 168.98 | 200.56 | 193.03 |
| Upstream without NWP (B vs B0) | 19.78 [16.94, 22.39] | 11.08 [8.48, 13.83] | 13.22 | 17.13 |
| NWP without upstream (N0 vs B0) | 33.09 [28.94, 36.51] | 25.14 [19.96, 30.40] | 27.21 | 29.19 |
| NWP given upstream (N vs B) | 13.91 [11.42, 16.26] | 15.80 [11.60, 19.91] | 14.91 [12.19, 17.60] | 15.05 [12.40, 17.73] |
| Upstream given NWP (N vs N0) | 0.59 [−0.14, 1.62] | 1.75 [0.03, 3.61] | 0.91 [−0.57, 2.29] | 2.99 [1.46, 4.52] |

<!-- K/P: output/g2a/results81/results.json, output/g1/results_78; HoT: output/g2b/sealed_eval/server/results.json (N = 5-seed mean MAE; paired N vs B uses the first 3 seeds). HoT B vs B0, N0 vs B0 descriptive (CIs in results.json). -->

For TSMixer, adding NWP to local SCADA reduced MAE by 12–18%. With upstream observations already available, it reduced MAE by 7–9%. The upstream increment was 6–11% without NWP and 0.4–1.5% with NWP. For the latter comparison, the interval excluded zero at Hill of Towie 2025 and, narrowly, at Penmanshiel, where the Newey–West test gave p = 0.0512 and the longer-block interval crossed zero; it included zero at Kelmarsh and Hill of Towie 2024. A small residual upstream gain is therefore supported on some test sets but not confirmed across all four.

The evidence status differs between the cells of Table 1. N versus B was the pre-registered primary comparison on first-opened test data at all three farms. At Kelmarsh and Penmanshiel, B0 and N0 were fitted afterwards and evaluated on the already-opened test years, so their contrasts are exploratory. At Hill of Towie all four arms were evaluated in the single sealed run, with N versus N0 as a pre-registered secondary comparison and the other contrasts descriptive. On the two development farms, the reduction in the upstream increment after adding GFS was 19.19 kW at Kelmarsh and 9.33 kW at Penmanshiel. These descriptive differences use the four fitted models and the same forecast windows.

### 5.2 Confirmation on a third farm

Hill of Towie took no part in any design choice; its development years were used only to fit and select the frozen models. The pre-registered primary family on the sealed HoT years was (Q1h) N vs B and (Qe) a five-seed ensemble vs a single model. Both passed: Q1h gains 14.91 and 15.05 kW (Holm p < 10⁻²⁰), Qe gains 3.28 and 2.93 kW. The NWP gain is positive for all 21 turbines in both years (7–19 kW). Because both years are evaluated with the same fitted models, they constitute one out-of-sample confirmation on a third farm, not two independent replications.

### 5.3 Is the decomposition predictor-dependent? (LightGBM)

A gradient-boosted tree model (LightGBM, one model per lead, L1 objective, 4-cell grid selected on validation) was given exactly the same inputs as N (last-step channels, 24-h power and wind history, turbine id). On the test years it reaches MAE 145.25 / 161.01 / 194.40 / 186.72 kW, i.e. 1.2 (n.s.), 8.0, 6.4 and 6.3 kW below N, while its accuracy (1 − nRMSE) differs from N by at most 0.2 percentage points; the advantage is specific to MAE and consistent with its L1 objective.

Repeating the decomposition with LightGBM on the second half of the validation year (development data; seed 2024) gives: upstream alone −8.5 to −11.1 %, NWP alone −17.6 to −23.1 %, NWP given upstream −9.1 to −14.3 %, upstream given NWP −1.0 to −1.7 %. The ordering is the same as for TSMixer. On the (already opened) test years the upstream-given-NWP increment of LightGBM is positive in all four sets (2.5–6.7 kW, 1.7–3.4 %); we report this as exploratory.

<!-- output/x2/x2_91d4d20/report_a/{dev_decisions.json, results.json}; verified by independent recomputation (DECISIONS stage X2) -->

### 5.4 External baselines

**Table 2.** Test MAE (kW) and gain of N over each baseline (95 % CI).

| Baseline | Kelmarsh 2022 | Penmanshiel 2021 | HoT 2024 | HoT 2025 |
|---|---|---|---|---|
| Persistence | 193.87; +47.4 [42.5, 51.7] | 206.07; +37.1 [29.3, 44.4] | 245.97; +45.2 [40.7, 49.8] | 241.98; +48.9 [44.2, 53.8] |
| GFS + per-turbine power curve | 219.50; +73.0 [66.1, 80.6] | 242.00; +73.0 [62.2, 84.0] | 314.95; +114.2 [103.0, 126.3] | 291.14; +98.1 [88.5, 108.3] |
| LightGBM, same inputs | 145.25; −1.2 [−2.5, 0.2] | 161.01; −8.0 [−10.9, −5.6] | 194.40; −6.4 [−8.2, −4.6] | 186.72; −6.3 [−7.9, −4.8] |
| Chronos-2 zero-shot, 24-h context | 190.1; +43.6 [39.1, 48.1] | 205.9; +37.0 [30.0, 43.8] | 248.1; +47.4 [42.8, 52.2] | 241.8; +48.7 [44.4, 53.3] |
| Chronos-2 zero-shot, 168-h context | 163.5; +17.1 [14.5, 20.1] | 185.4; +16.4 [10.9, 21.4] | 223.9; +23.2 [19.9, 26.6] | 217.7; +24.7 [21.4, 28.0] |

<!-- output/x1/A0, A2, A1 results.json; Chronos-2: output/x1/x1_e4819b0_gpu_r1/A3/results.json (model revision 29ec3766) -->

With a 24-hour context, Chronos-2 approached persistence on these forecasts. With 168 hours it improved on all four datasets. Adding its forecasts as inputs to TSMixer yielded 0.26–0.55% at best on validation, below the pre-registered 1% threshold. A subsequent development study selected a 1,024-hour context at each farm. Under its 20,000-window fine-tuning budget, the pooled relative MAE gain over a three-seed LightGBM reference was −2.56%, with a multiplicity-adjusted interval of [−4.37%, −0.75%]. The lower bound missed the −1% non-inferiority margin. Combining the two predictors reduced LightGBM MAE by 2.88%, 1.44% and 0.27% at Kelmarsh, Penmanshiel and Hill of Towie, respectively. The registered criterion required at least 1% at every farm. All three development sets had been used in earlier experiments.

A further development round tested two adaptations. Raising the fine-tuning learning rate (3 × 10⁻⁶ selected at every farm) gave a pooled gain over LightGBM of −2.19% [−3.85%, −0.53%], again below the non-inferiority margin. The ensemble gains were 3.64%, 1.80% and 0.54%, so the every-farm 1% criterion failed at Hill of Towie once more. For a target farm with one month of labels, a two-stage pipeline first fine-tuned Chronos-2 on the other two farms and then adapted it to the target month. Its pooled gain over a LightGBM model trained on the same source and target windows was −2.54% [−4.45%, −0.64%], with farm values between −2.61% and −2.49%. Relative to adapting the base model directly to the target month, the pipeline reduced MAE by 5.13, 9.66 and 2.51 kW at Kelmarsh, Penmanshiel and Hill of Towie, and relative to zero-shot use by 8.05, 12.77 and 3.83 kW. These descriptive gains do not identify the source stage as the cause, and the transferred LightGBM remained more accurate. A LightGBM proxy diagnostic of whether event groups of source windows have a stable transfer value did not reach its pre-registered continuation threshold.

<!-- X3: output/x3/x3_f8af1b6/report/dev_decisions.json (Q2 and Q3, V2); X4: output/x4/x4_1c0a125/report/dev_decisions.json (A, B1, B2, D0; Bonferroni z = 2.394 over three questions); previously opened test years are exploratory. -->

### 5.4b Base screening

Six predictor families were trained on identical inputs (the 50-channel information set, HoT retrofit flag), identical splits and the same training recipe, with early stopping on the first half of the validation year (V1) and selection on the second half (V2). The neural candidates include the author's two earlier architectures: a TCN-GRU, with and without a domain-adversarial branch (standard gradient-reversal layer, time-segment domains, λ selected on V1), and CNN and BiLSTM networks with attention, also combined with LightGBM by ridge stacking fitted on V1 only.

**Table 2b.** V2 MAE (kW) and mean relative difference to LightGBM.

| Base | Kelmarsh | Penmanshiel | HoT | vs LightGBM |
|---|---|---|---|---|
| LightGBM | 132.05 | 176.28 | 186.39 | — |
| TSMixer | 133.40 | 182.98 | 198.44 | +3.8 % |
| Stacking (CNN + BiLSTM + LightGBM) | 132.31 | 184.35 | 199.25 | +3.9 % |
| BiLSTM | 137.54 | 195.32 | 199.94 | +7.4 % |
| CNN | 139.37 | 192.22 | 201.01 | +7.5 % |
| DA-TCN-GRU | 139.97 | 193.10 | 215.31 | +10.3 % |
| TCN-GRU | 140.39 | 193.82 | 218.08 | +11.1 % |

LightGBM is selected by the pre-registered rule. The domain-adversarial branch did not improve its TCN-GRU twin (gains 0.4–2.8 kW, all intervals include zero). The stacking ensemble had a worse MAE than LightGBM alone on two farms but the highest accuracy (1 − nRMSE) on all four test sets (89.4 / 87.0 / 86.0 / 86.4 %), consistent with its squared-error meta-learner; the ranking of predictors therefore depends on the metric, and we keep MAE as the pre-registered primary metric.

<!-- output/x2/x2_0f24875/report/{dev_decisions.json, results.json}; verified by independent recomputation (DECISIONS stage X2-B2) -->

### 5.5 Accuracy under different reporting conventions

Turbine-level, 1–4 h averaged accuracy (1 − nRMSE) of N is 89.2 % (Kelmarsh), 86.6 % (Penmanshiel), 85.6 % and 86.1 % (HoT). Two common conventions raise these values:

- Farm-level aggregation (sum over turbines at the same issue time, issue times with ≥ 80 % of turbines): 90.9 % (Kelmarsh; lead 1 → 4: 93.3 → 89.6 %), 87.9 % (Penmanshiel; 91.9 → 85.4 %), 87.6 % and 88.1 % (HoT 2024, 2025; 91.6 → 85.4 % and 91.8 → 85.9 %). Persistence reaches 84.3 % and 84.6 % at HoT on the same basis.
- Excluding stoppage and curtailment cells (pre-registered rule, 1.7–6.3 % of cells): +0.1 to +1.1 percentage points.

<!-- output/x1/E1/{Kelmarsh_2022,Penmanshiel_2021}.json; output/x1/E1_hot_31ce5d3_r2/HoT_202{4,5}.json; output/x1/E2/results.json -->

### 5.6 Upstream value under NWP latency

Section 5.1 measured the upstream increment with the six-hour availability rule. To see how this increment depends on forecast age, we rebuilt the GFS inputs of the LightGBM information sets under delays d ∈ {6, 9, 12, 15, 18} h. At origin t0 the latest cycle C with C + d ≤ t0 is used; d = 6 reproduces the frozen inputs window by window. For each delay, LG_d (local + GFS) and LUG_d (local + upstream + GFS) were trained and evaluated with the same delay, and LU (local + upstream) is unaffected. U(d) = MAE(LG_d) − MAE(LUG_d) is the upstream increment at delay d, and G(d) = MAE(LU) − MAE(LUG_d) is the GFS increment. All models use seed 2024. Development statistics are on the second half of the validation year (V2), and intervals are paired moving-block bootstrap intervals clustered by issue time (block 8, 2000 replicates).

**Table 4.** Upstream increment U(d) on V2 (kW) and its rise from 6 h to 18 h (95 % CI).

| Farm | U(6) | U(9) | U(12) | U(15) | U(18) | U(18) − U(6) |
|---|---|---|---|---|---|---|
| Kelmarsh | 2.28 | 2.27 | 3.02 | 6.75 | 15.50 | 13.21 [9.95, 16.86] |
| Penmanshiel | 2.18 | 2.65 | 3.63 | 10.91 | 15.99 | 13.81 [10.20, 18.13] |
| Hill of Towie | 1.82 | 2.57 | 3.49 | 13.46 | 24.38 | 22.56 [17.38, 27.56] |

<!-- output/x5/x5_e01cc6b/run/X5/results.json (commit e01cc6b); independent recomputation by Claude and Codex (DECISIONS stage X5-result); block-1 references from output/x5/local_blocks12/results.json -->

From 6 h to 12 h, U increased by only 0.7–1.7 kW. From 12 h to 15 h it increased by 3.7–10.0 kW. The proportion of target hours whose required lead exceeded the 18 h limit of the extracted forecasts was 12.50%, 12.50% and 25.11% at 12 h, 62.51%, 62.50% and 75.11% at 15 h, and 100% at 18 h (Kelmarsh, Penmanshiel, Hill of Towie). Some past-hour forecasts remained available at 18 h. The rise in U therefore coincides with the loss of forecast coverage. Because the delay design changes forecast age and coverage together, it cannot separate their contributions; the matched-coverage analysis below does so at 15 h. The GFS increment fell from G(6) = 13.21, 28.69 and 30.98 kW to G(18) = −0.33, 1.67 and 0.95 kW. For reference, the upstream increment without any GFS input (Section 5.3, same split and seed) was 17.68, 19.06 and 27.22 kW.

On the four previously opened test years the rise from 6 h to 18 h ranged from 8.65 to 13.81 kW, with paired 95 % intervals above zero; these are exploratory. A single model trained on a mixture of delays, which sees the age of the selected cycle as an input but not the delay setting, differed from the matched-delay models by −0.9 to +3.1 kW MAE over 35 comparisons, most within ±1.5 kW.

**Matched coverage at 15 h (post-hoc).** After seeing the latency curve, we compared the recent cycle used at a 6 h delay (state A) with the older cycle used at 15 h (state C) on the same windows, adding two states: A restricted to the lead coverage of C (D), and C restricted to the lead coverage of A (Cc). A lead counts as covered when cycle age plus offset lies within 3–18 h; removed values and their validity flags were set to missing together, and each state kept its own cycle age and missing-cycle flag. The rise U(C) − U(A) then splits into a truncation term U(D) − U(A), a cycle term U(Cc) − U(D) at common coverage, and a remainder U(C) − U(Cc). LightGBM was trained separately for each state with seed 2024. U(A) and U(C) reproduced U(6) and U(15) of Table 4 exactly, and the actual availability of the D and Cc inputs agreed in every V2 window, so the cycle term is not affected by different gaps in the forecast archive.

**Table 4b.** Decomposition of the upstream increment between 6 h and 15 h on V2 (kW, 95 % CI).

| Farm | U(C) − U(A) | Truncation U(D) − U(A) | Cycle term U(Cc) − U(D) | Remainder U(C) − U(Cc) |
|---|---|---|---|---|
| Kelmarsh | 4.47 [2.78, 6.34] | 3.75 [2.36, 5.39] | 0.40 [−0.82, 1.59] | 0.32 [−0.65, 1.28] |
| Penmanshiel | 8.73 [5.74, 11.73] | 5.76 [3.08, 8.37] | 1.45 [−0.47, 3.50] | 1.53 [0.34, 2.92] |
| Hill of Towie | 11.64 [7.88, 15.33] | 12.80 [9.20, 16.32] | −0.36 [−1.98, 1.26] | −0.80 [−1.79, 0.17] |

<!-- output/x5/x5_b7018c0/ (commit b7018c0, X5b); reviewed in DECISIONS stage X5b-result (Claude) and docs/review/X5b-result-R1-codex.md -->

Truncating the recent cycle to the older cycle's coverage raised the upstream increment at all three farms, with intervals above zero; the effect was concentrated at the third and fourth target hours, whose leads exceed 18 h at a 15 h delay. At common coverage, changing from the recent to the older cycle altered the increment by 0.40, 1.45 and −0.36 kW, and all three intervals included zero. At Kelmarsh and Hill of Towie the whole interval lay within ±50 % of the truncation point estimate; at Penmanshiel it did not, and a cycle effect of about 3.5 kW cannot be excluded. The remainder, which reflects leads covered by the older cycle but not by the recent one, was between −0.80 and 1.53 kW. On the previously opened test years (exploratory) the truncation term was 2.94–6.77 kW with intervals above zero, and the cycle term −0.35 to +0.55 kW with intervals including zero. These results apply to 6 h versus 15 h, leads f003–f018 and this predictor; they do not show that forecast age has no effect.

With the six-hour rule, the upstream increment was also concentrated in a small part of the data. Ranking windows by the disagreement between the local + upstream and local + GFS forecasts, with the 80th-percentile threshold taken from the first half of the validation year, the top fifth of V2 windows carried 54%, 85% and 65% of the net upstream gain. The high-versus-low disagreement contrast excluded zero at Kelmarsh and Penmanshiel but not at Hill of Towie.

In summary, on the V2 development sets of the three farms, with LightGBM trained and evaluated at the same delay, adding upstream observations reduced 1–4 h MAE by 1.82–2.28 kW when GFS cycles were selected 6 h after issue, by 6.75–13.46 kW at 15 h and by 15.50–24.38 kW at 18 h. The exploratory test years also showed a larger gain at 18 h than at 6 h. In the post-hoc matched-coverage analysis, the rise at 15 h went together with the loss of target-hour coverage, and no additional effect of the older cycle was detected at common coverage. These descriptive results are limited to the forecast-lead coverage of our GFS extraction, the discrete delay settings and the predictor used.

### 5.7 Published wind-power models under matched information (development)

We adapted three published models to the V2 development windows and compared each with a LightGBM reference given the same information. These comparisons describe the adaptations on our data and do not reproduce the original studies. They have no pass threshold, use seed 2024 throughout, and use V2 data that had been used in earlier rounds. Each model selected among four pre-specified candidates by mean V1 MAE over the four target hours, and all its V1 selections were frozen before any V2 prediction was made. Intervals are computed as in Section 5.6.

- **Nyström kernel ridge regression** of direct power (Bouche et al., 2023): Gaussian kernel with 300 centres and inputs identical to the LightGBM information sets LG and LUG, with the turbine one-hot encoded. It was trained and evaluated at GFS delays of 6 h and 18 h and compared with LightGBM trained at the same delay (Section 5.6).
- **Cabin** (Wu et al., 2025): the authors' code at a fixed commit with one change. Its softmax across the samples of a batch lets later-issued windows change the predictions for earlier ones, so we switched it off with the authors' own option. For every trained candidate, a window's prediction changed by at most 1.6 × 10⁻⁷ (normalised power) when later windows were added to its batch, whereas with the option on the change was 7.7 × 10⁻³. Cabin receives 24 h of power history and of local (L0) or local plus upstream (LU) observations, without GFS. Its reference is a LightGBM that receives the identical scaled 24 h inputs, flattened.
- **Adaptive robust regression with learned partitions** (Stratigakos and Andrianesis, 2026): the authors' code at a fixed commit, trained on the mixture of GFS delays of Section 5.6 and evaluated at 6 h and 18 h. GFS validity flags enter only through the method's missing-feature mask. The authors' trees have a single output, so one tree was fitted per target hour, each on 50,000 training windows per farm. The reference is the mixed-delay LightGBM of Section 5.6, which used all training windows and receives the validity flags as features.

**Table 5.** Excess V2 MAE of each published model over its LightGBM reference (kW; positive means LightGBM was more accurate). For kernel ridge regression and adaptive robust regression the range covers {LG, LUG} × {6 h, 18 h}; each of these intervals excluded zero.

| Farm | Kernel ridge regression | Cabin, L0 | Cabin, LU | Adaptive robust regression |
|---|---|---|---|---|
| Kelmarsh | 11.49–13.83 | 7.29 [4.88, 10.21] | 2.42 [−0.02, 4.87] | 29.79–36.17 |
| Penmanshiel | 12.46–18.39 | 7.27 [3.43, 11.49] | 8.21 [4.23, 12.87] | 16.58–22.75 |
| Hill of Towie | 36.80–44.82 | 25.65 [19.65, 31.92] | 29.84 [22.42, 37.34] | 26.74–49.92 |

<!-- output/wjb/wjb_1fe7ab6/report/results.json (run commit 1fe7ab6); full cell table output/thesis_materials/wjb_development_comparisons.md; independent recomputation in output/wjb/wjb_1fe7ab6/claude_review/ (DECISIONS stage wind-journal-baselines-result) -->

LightGBM was more accurate in 29 of the 30 pairs. In the remaining pair, Cabin with upstream observations at Kelmarsh, the excess of 2.42 kW had an interval that just included zero, so no difference was detected under the pre-registered reading. This cell lies at the boundary: the point estimate favoured LightGBM at all four target hours, and about half of 50 alternative bootstrap seeds gave an interval excluding zero (post-hoc check). At Kelmarsh, Cabin also gained 4.87 kW [2.42, 7.70] more from upstream observations than its reference; this post-hoc difference was not seen at the other two farms.

Selection chose the neural-network base model of the adaptive robust regression at all three farms. Its linear base model is initialised by an unregularised least-squares fit, and our inputs contain collinear columns, such as GFS values interpolated from the same leads and the latest power and wind also present in the 24 h history. At Kelmarsh and Penmanshiel this linear variant failed numerically: robust training losses exceeded 10⁵ (normalised power) from the first epoch, and its V1 MAE was 836–1,120 kW. We report this as an interaction between our inputs and the implementation, not as a property of the method.

**Table 6.** Upstream increment U = MAE(LG) − MAE(LUG) of each model at 6 h and 18 h, and its change (V2, kW, 95 % CI).

| Model (training) | Farm | U(6) | U(18) | U(18) − U(6) |
|---|---|---|---|---|
| Kernel ridge regression (same delay) | Kelmarsh | 2.84 | 13.64 | 10.80 [8.00, 14.02] |
| | Penmanshiel | 1.84 | 10.55 | 8.71 [5.92, 11.66] |
| | Hill of Towie | −0.17 | 18.55 | 18.72 [14.33, 22.73] |
| LightGBM (same delay) | Kelmarsh | 2.28 | 15.50 | 13.21 [9.95, 16.86] |
| | Penmanshiel | 2.18 | 15.99 | 13.81 [10.20, 18.13] |
| | Hill of Towie | 1.82 | 24.38 | 22.56 [17.38, 27.56] |
| Adaptive robust regression (mixed delays) | Kelmarsh | 4.56 | 17.02 | 12.46 [9.95, 15.22] |
| | Penmanshiel | 1.33 | 8.14 | 6.81 [2.45, 10.89] |
| | Hill of Towie | −18.13 | 5.42 | 23.55 [19.59, 27.22] |
| LightGBM (mixed delays) | Kelmarsh | 2.74 | 14.63 | 11.89 [8.99, 15.54] |
| | Penmanshiel | 1.61 | 14.31 | 12.70 [9.69, 15.98] |
| | Hill of Towie | 2.99 | 26.12 | 23.12 [18.43, 27.63] |

For both models that use GFS, the upstream increment rose from 6 h to 18 h at all three farms, with intervals above zero, in the same direction as their LightGBM references. At 18 h all four target hours lie beyond the extracted forecast leads, so the rise describes missing forecast coverage; the effect of forecast age is addressed by the matched-coverage analysis of Section 5.6. At Hill of Towie the adaptive robust regression had a negative increment at 6 h. At 6 h, adding upstream observations increased MAE at each of the 21 turbines. Grouped by GFS cycle age, the mean LUG prediction exceeded the mean LG prediction by 34.2 kW at ages 8–11 h and by 50.1 kW at ages 6–7 h. The LUG model already had higher V1 MAE (221.08 vs 208.50 kW). Its absolute increment therefore does not measure the value of the stations. The pre-registered contrast measures how the increment changes between delays. From 6 h to 18 h the LG and LUG models lost 51.17 and 27.62 kW, a difference of 23.55 kW, close to the 23.12 kW of its LightGBM reference.

---

## 6 Pre-registered negative results

Every mechanism below was registered with its gate before it was run; selection used development data only.

**Table 3.** Summary.

| Mechanism | Idea | Gate | Outcome |
|---|---|---|---|
| NWP displacement correction (G1) | match upstream observations to local NWP sequences and transfer the estimated shift to the farm | ≥ 1.5% gain on each development farm and significance against the validation-selected information control | MAE was 0.60% higher at Kelmarsh and 0.96% higher at Penmanshiel. Q3 Holm-adjusted p = 0.071. The offline effective shift is a diagnostic proxy. |
| Station-network operator (N0/N1) | learn a station-identity-free upstream operator on the MIDAS network, transfer to farms | beats handcrafted features on held-out stations | better than handcrafted features on 0 of 8 held-out stations; transfer failed |
| Regime-weighted conformal transfer (3 rounds) | transfer residual calibration across farms by upstream-regime density ratios | coverage 88–92 % and interval score ≥ 3 % better than matched power-similarity control | no round passed; best variant ≤ 0.3 % better than its matched control, and significantly worse in the reverse transfer direction |
| LLM-generated feature programs (3 rounds) | a local LLM writes feature-encoding programs, selected on validation | beats handcrafted features by ≥ 1.5 % on both farms | no round passed |
| Anchored NWP increment (ANI) on LightGBM | convert the NWP change into a power change and anchor it at current power | ≥ 1 % on every farm and pooled CI > 0 over the full-information model | pooled −0.04 % [−0.34, 0.25] |
| Wind-anchored / ratio / bias-corrected NWP (3 variants, Bonferroni) | anchor the NWP change at observed nacelle wind; ratio form; 6-h bias correction | same gate | pooled +0.14 %, −0.25 %, +0.06 %, none significant |
| Chronos-2 fine-tuning (two rounds) | fine-tune on 20,000 windows per farm (lr 10⁻⁶; then {3 × 10⁻⁶, 10⁻⁵}) | pooled lower bound > −1 % vs LightGBM, no farm entirely below −1 % | −2.56 % [−4.37, −0.75]; −2.19 % [−3.85, −0.53] |
| LightGBM + Chronos-2 ensemble (two rounds) | weight selected on V1 | ≥ 1 % at every farm, pooled CI > 0, and better than a LightGBM + TSMixer ensemble | Hill of Towie 0.27 % and 0.54 %; not passed |
| Cross-farm two-stage adaptation, one-month cold start | fine-tune on the other farms, then adapt to the target month | every farm gain > 0 and pooled CI > 0 vs transferred LightGBM | −2.54 % [−4.45, −0.64] |
| Event-group transfer value (LightGBM proxy) | rank source event groups by their transfer value to a pseudo-target | Spearman ≥ 0.5 and top groups beat random on ≥ 2 farms | threshold not reached under this proxy and partition |

<!-- DECISIONS stages G1, N0, M2–M4 (C line), L1/M5–M7, X2, X2-A2, X3-result, X5 (X4 results); fine-tuning, ensemble and cold-start rows use Bonferroni z = 2.394 within their round -->

The registered transformations added little accuracy over predictors with access to the same weather inputs. Three hourly observations also give limited leverage for estimating timing and amplitude separately. For a locally linear forecast with slope v, a time shift δ changes the forecast by vδ, which an additive bias can reproduce exactly. This limits the physical interpretation of the estimated shift on constant-slope segments. Jin et al. (2024) attributed about 70% of hub-height wind forecast error to the sequence component in a study of 262 Chinese wind farms. Their decomposition sorts the observed and forecast values independently. A single bounded shift can correct only a subset of the discrepancies removed by that reordering.

---

## 7 Discussion

### 7.1 How do these errors compare with the literature?

At Hill of Towie, aggregating turbines raised forecast accuracy on the same years (Section 5.5). This difference matters when comparing our turbine-level errors with the farm-level forecasts in Deignan et al. (2025). Forecast lead matters for the NWP results as well. Spiliotis and Theodorou (2025) evaluated corrections to wind forecasts supplied a day ahead. Our correction must improve a 1–4 h power forecast whose baseline already sees recent power, upstream observations and GFS. The available information and the baseline determine which part of a weather forecast error a correction can remove.

### 7.2 Limitations

- We evaluated one global NWP product, using 10 m winds and a six-hour availability rule. A regional product or a verified hub-height forecast would provide a direct test of how the source increments depend on weather forecast quality.
- The latency curve (Section 5.6) uses discrete delays and forecasts extracted only to an 18 h lead, so longer delays also remove forecast coverage. The matched-coverage analysis separates age and coverage only for 6 h versus 15 h, was designed after the latency results were seen, and cannot exclude a cycle effect of about 3.5 kW at Penmanshiel. Both use one LightGBM seed, and their development split had been used in earlier rounds.
- The published-model comparisons (Section 5.7) are adaptations to our windows and inputs, not reproductions: one seed, four candidates each, reused development data, and for the adaptive robust regression 50,000 training windows per farm, one tree per target hour and a neural base model selected on V1 at each farm. The linear candidates failed numerically at Kelmarsh and Penmanshiel.
- The validation halves and the earlier test years were used repeatedly across rounds. The multiplicity corrections apply within each round, not to the whole sequence of analyses, so later development results are reported as development-stage evidence.
- The model screen used a four-configuration LightGBM grid and one training configuration for each neural candidate. Its ranking applies to those budgets.
- Three farms in one climate region (UK).
- HoT test years are entirely post-retrofit, while most fit data are pre-retrofit.
- The two HoT test years share the same fitted models.
- The 15-min resolution of GB/T 40607 is not available from hourly aggregation.

---

## 8 Conclusion

On three UK wind farms, an issued GFS forecast reduced 1–4 h turbine-level MAE even when upstream observations were already available. The two reserved Hill of Towie years confirmed this increment with the same fitted models. The converse gain was smaller and was not confirmed on every test set. With a recent GFS cycle, upstream observations added about 2 kW on the development sets. At an 18 h delay, when the extracted forecasts covered none of the target hours, their value rose to 15.50–24.38 kW and approached the value they had without any GFS input. In a post-hoc comparison at 15 h, the rise went together with the loss of target-hour coverage, and no additional effect of the older cycle was detected at matched coverage. For this predictor at these farms, the station feed yielded larger MAE reductions in the conditions with incomplete target-hour coverage. Three published wind-power models, adapted to the same information, were less accurate than LightGBM in 29 of 30 development comparisons; the two that use GFS showed the same rise of the upstream increment. Fine-tuning and cross-farm adaptation of a time-series foundation model narrowed, but did not close, its gap to LightGBM under the registered budgets. Whether the same patterns hold with regional or hub-height NWP, or with forecasts extracted beyond 18 h, remains open.

---

## Data and code availability

Kelmarsh and Penmanshiel SCADA (Zenodo, CC-BY-4.0); Hill of Towie (Zenodo 22662930); MIDAS Open (OGL v3, attribution required); GFS (NOAA, public). Code, protocols, hashes and all result files: [repository, PENDING].

## References (to verify)

- Gneiting T. et al. (2006) Calibrated probabilistic forecasting at the Stateline wind energy center. JASA.
- Hering A.S., Genton M.G. (2010) Powering up with space-time wind forecasting. JASA.
- Tastu J. et al. (2011). Spatio-temporal analysis and modeling of short-term wind power forecast errors. *Wind Energy* 14, 43–60. [Article](https://doi.org/10.1002/we.401)
- Messner J.W., Pinson P. (2019) Online adaptive lasso estimation in vector autoregressive models for high dimensional wind power forecasting. IJF.
- Hong T., Pinson P., Fan S. (2014) Global Energy Forecasting Competition 2012. IJF.
- Ansari A. F. et al. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. [Technical report](https://arxiv.org/abs/2510.15821)
- Chen S.-A. et al. (2023). TSMixer: An all-MLP architecture for time series forecasting. *Transactions on Machine Learning Research*. [Article](https://openreview.net/forum?id=wbpxTuXgm0)
- Ke G. et al. (2017) LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
- Ganin Y. et al. (2016) Domain-adversarial training of neural networks. JMLR.
- Deignan C. et al. (2025). Portability of short term wind power forecasting: investigating model calibration using wind power data from Ireland and UK. *Scientific Reports* 15, 33891. [Article](https://doi.org/10.1038/s41598-025-08751-3)
- Liu C. et al. (2023). Series-wise attention network for wind power forecasting considering temporal lag of numerical weather prediction. *Applied Energy* 336, 120815. [Article](https://doi.org/10.1016/j.apenergy.2023.120815)
- Jin C. et al. (2024). Evaluation of forecasted wind speed at turbine hub height and wind ramps by five NWP models with observations from 262 wind farms over China. *Meteorological Applications* 31, e70007. [Article](https://doi.org/10.1002/met.70007)
- Spiliotis E. and Theodorou E. (2025). Improving wind power forecasting accuracy through bias correction of wind speed predictions. *Sustainable Energy Technologies and Assessments* 83, 104599. [Article](https://doi.org/10.1016/j.seta.2025.104599)
- Xu X. et al. (2025). A cross-dataset benchmark for neural network-based wind power forecasting. *Renewable Energy*, 123463. [Article](https://doi.org/10.1016/j.renene.2025.123463)
- Peduto M. et al. (2026). Observation-driven correction of numerical weather prediction for marine winds. *Journal of Geophysical Research: Machine Learning and Computation* 3, e2025JH001174. [Article](https://doi.org/10.1029/2025JH001174)
- Wang H. et al. (2026). DeepWind: A foundation model for zero-shot wind power forecasting. *Energy* 360, 141793. [Article](https://doi.org/10.1016/j.energy.2026.141793)
- Zhu Y. et al. (2026). WPBench: A comprehensive benchmark for wind power forecasting. Preprint, arXiv:2609.24444. [Preprint](https://arxiv.org/abs/2609.24444)
- GB/T 40607-2021. Technical requirements for wind/PV power forecasting systems on the dispatch side.
- Bouche D., Flamary R., d'Alché-Buc F., Plougonven R., Clausel M., Badosa J., Drobinski P. (2023). Wind power predictions from nowcasts to 4-hour forecasts: A learning approach with variable selection. *Renewable Energy* 211, 938–947. [Article](https://doi.org/10.1016/j.renene.2023.05.005)
- Wu S., Chen Y., He X., Wang Z., Liu X., Fu Y. (2025). Cabin: A collaborative and adaptive framework for wind power forecasting integrating ambient variables. *Energy* 335, 137753. [Article](https://doi.org/10.1016/j.energy.2025.137753) · code: github.com/xiufengliu/cabin-wind-forecasting (commit 392f103)
- Stratigakos A., Andrianesis P. (2026). Learning data-driven uncertainty set partitions for robust and adaptive energy forecasting with missing data. *IEEE Transactions on Smart Grid* 17(1), 806–817. doi:10.1109/TSG.2025.3620957 (suffix to verify) · code: github.com/akylasstrat/wind-forecast-missing-data (commit 409f20f)
