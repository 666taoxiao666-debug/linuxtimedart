# Which exogenous information matters for turbine-level ultra-short-term wind power forecasting? A pre-registered three-farm study of numerical weather prediction, upstream observations and alternative predictors

Draft v0 (Claude, 2026-11-17). Status: everything that does not depend on the running B line (base screening) or the Chronos-2 queue is filled in from verified result files; open items are marked **[PENDING]**. Numbers carry their source in `<!-- -->` comments so that every value can be traced. Target journal: Advanced Engineering Informatics (fallback: IJEPES / SETA). Wording rules from DECISIONS (G2c, X2) apply: no "independent replication" for the two HoT years (shared models), no "complementarity confirmed", no causal claim about the AeroUp retrofit.

---

## Abstract

Ultra-short-term (1–4 h) wind power forecasts increasingly combine on-site SCADA history with two exogenous sources: observations from upstream meteorological stations and numerical weather prediction (NWP). How much each source contributes, whether the contributions overlap, and whether the answer depends on the predictor are rarely tested under leakage-free, pre-registered protocols. We study hourly turbine-level forecasting at three UK wind farms (Kelmarsh, Penmanshiel, Hill of Towie; 41 turbines, 2016–2025) with 12–14 upstream MIDAS stations per farm and GFS 0.25° forecasts used only at their real availability time. A 2 × 2 information decomposition (upstream × NWP) is evaluated on held-out years, and the main finding is confirmed once on two sealed years of a third farm that took no part in any design decision. Adding NWP to a model that already uses upstream observations reduces the mean absolute error (MAE) by 8.7 % and 8.6 % on the development farms and by 6.9 % and 7.2 % on the sealed years (all 95 % confidence intervals above zero). Upstream observations reduce MAE by 5.7–11.0 % when NWP is absent, but only by 0.4–1.5 % once NWP is present, and this residual increment is significant in only part of the test sets. The same ordering holds when the neural predictor (TSMixer) is replaced by gradient-boosted trees (LightGBM), which reach a slightly lower MAE (−1 to −8 kW) with an essentially equal RMSE. Four NWP post-processing mechanisms (displacement correction, anchored NWP increments, ratio anchoring, recent-bias correction) and several other candidate methods give no gain over the full-information baseline under pre-registered gates. In a screening of six predictor families on development data (gradient-boosted trees, TSMixer, TCN-GRU with and without domain-adversarial training, CNN, BiLSTM, and their stacking), gradient-boosted trees had the lowest error; a time-series foundation model (Chronos-2) used zero-shot with NWP covariates was 16–49 kW worse than the trained models, and adding its forecasts as inputs gave no gain. The results bound the design space: at 1–4 h, the forecast skill of a turbine-level model is governed by the availability and quality of NWP rather than by architecture or NWP post-processing.

**Keywords:** wind power forecasting; numerical weather prediction; upstream observations; information decomposition; pre-registration; negative results

---

## 1 Introduction

<!-- Story (keep): 1 practical need (GB/T 40607 1-4 h); 2 two exogenous sources and why they should help; 3 gap = no leakage-free, sealed, cross-farm decomposition; claims in literature mostly from short horizons, random splits, single farm; 4 contributions. -->

Grid operators require ultra-short-term wind power forecasts for the next 15 min to 4 h (e.g. GB/T 40607-2021 in China). Beyond the first hour, the turbine's own history loses most of its predictive value and forecasts must draw on exogenous information. Two sources are common. Upstream observations measure the air mass before it reaches the farm and have long been used in spatio-temporal statistical models [Gneiting et al. 2006; Hering & Genton 2010; Tastu et al. 2011]. NWP forecasts provide the expected evolution of the wind field directly, but with coarse resolution, systematic bias and a publication delay of several hours.

Published accuracy figures for wind power forecasting are hard to compare. Many studies forecast 10–15 min ahead, evaluate whole farms instead of turbines, split data randomly so that neighbouring hours appear in training and test, use measured wind at the target time as an input, or evaluate on a few months. Under such settings R² values above 0.95 are common; under a strict hour-ahead protocol they are not attainable (Section 7.1). For the same reason it is unclear whether reported gains from new architectures or NWP corrections survive a leakage-free, out-of-sample evaluation.

This paper asks three questions under a protocol designed to avoid these pitfalls:

1. How much does each exogenous source (upstream observations, NWP) contribute, and how much of the upstream contribution remains once NWP is available?
2. Do the answers hold on a farm that took no part in the design, and with a different family of predictor?
3. Do commonly proposed mechanisms for using NWP better (displacement correction, anchored increments, bias correction), or alternative predictors, add anything beyond a well-specified full-information baseline?

**Contributions.**

1. A leakage-free hourly turbine-level benchmark on three public UK wind farms, combining SCADA, MIDAS upstream stations and GFS forecasts with a causal forecast-cycle rule (Section 3–4).
2. A 2 × 2 information decomposition with block-bootstrap and Newey–West inference, confirmed once on two sealed years of a third farm (Section 5.1–5.2).
3. Evidence that the decomposition is predictor-independent: the same ordering is obtained with TSMixer and with LightGBM (Section 5.3).
4. A set of pre-registered negative results that bound the design space for NWP post-processing and model choice at 1–4 h (Section 6).

---

## 2 Related work

[PENDING, ~1 page. Groups and anchors, all to be verified before submission:]
- Spatio-temporal statistical forecasting with upstream/offsite information: Gneiting et al. 2006 (JASA, regime-switching space–time); Hering & Genton 2010 (JASA, trigonometric direction diurnal); Tastu et al. 2011 (Wind Energy); Messner & Pinson 2019 (IJF, online adaptive lasso VAR).
- NWP-based power forecasting and post-processing: MOS / bias correction; GEFCom2012 wind track (Hong et al. 2014, IJF); NWP displacement/phase errors; 2026 WRF-iTransformer (Energy).
- Deep learning for wind power with exogenous inputs: iTransformer, TimeXer, TSMixer; graph/spatio-temporal models on Kelmarsh/Penmanshiel (e.g. Sci. Rep. 2025 portability study).
- Time-series foundation models with covariates: Chronos-2 (arXiv 2510.15821).
- Comparability of forecast errors and evaluation pitfalls (Fraunhofer "How good is my forecast?").

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

Hourly MIDAS (UK Met Office) synoptic stations within a fixed radius were screened on development years only (completeness, quadrant coverage) and frozen before any model was fitted: [N] stations for Kelmarsh, [N] for Penmanshiel and 12 for HoT. Eleven handcrafted features summarise upwind and regional wind speed, its 3-h change, and the surface pressure gradient; observations are used with a 1-h delay.

### 3.3 NWP

GFS 0.25° 10-m wind is interpolated to the farm centre and to every station. A forecast cycle C is used for issue time t0 only if C + 6 h ≤ t0 (age 6–11 h), so that every value was available at issue time; leads 3–18 h cover all targets. Cycles that failed integrity checks are marked unavailable for all models alike, never replaced by an older cycle. 100-m wind is not used because its level could not be verified for all years.

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

The neural predictor N is a TSMixer [ref] trained with a 0.8 MAE + 0.2 MSE loss, early stopping on the validation year and three seeds. Four information sets are compared: local SCADA only (B0), + upstream features (B), + NWP (N0), and both (N).

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

NWP is the dominant source: alone it removes 12–18 % of the local-only error, and on top of upstream observations it still removes 7–9 %. Upstream observations are valuable when NWP is absent (6–11 %), but once NWP is present the remaining increment is 0.4–1.5 % and its interval excludes zero in only two of the four test sets. The two sources are therefore largely redundant at this horizon; we do not claim equivalence or complementarity.

### 5.2 Confirmation on a third farm

The pre-registered primary family on the sealed HoT years was (Q1h) N vs B and (Qe) a five-seed ensemble vs a single model. Both passed: Q1h gains 14.91 and 15.05 kW (Holm p < 10⁻²⁰), Qe gains 3.28 and 2.93 kW. The NWP gain is positive for all 21 turbines in both years (7–19 kW). Because both years are evaluated with the same fitted models, they constitute one out-of-sample confirmation on a third farm, not two independent replications.

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

Chronos-2 with GFS as a known-future covariate behaves like persistence when given the same 24-h context as the other models and improves with a 168-h context, but remains far behind every trained model. Feeding its quantile forecasts to N as extra inputs (three variants, 27 fits) improved validation MAE by at most 0.26–0.55 %, below the pre-registered 1 % on every farm. [Fine-tuning run failed technically; PENDING decision whether to rerun.]

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

---

## 6 Pre-registered negative results

Every mechanism below was registered with its gate before it was run; selection used development data only.

**Table 3.** Summary.

| Mechanism | Idea | Gate | Outcome |
|---|---|---|---|
| NWP displacement correction (G1) | match upstream observations to NWP to estimate and correct the forecast's phase error | beats an equal-capacity information control by ≥ 1.5 % on both development farms | gain −0.6 % / −1.0 % (wrong sign); learned shifts no better correlated with true phase errors than a negative control |
| Station-network operator (N0/N1) | learn a station-identity-free upstream operator on the MIDAS network, transfer to farms | beats handcrafted features on held-out stations | better than handcrafted features on 0 of 8 held-out stations; transfer failed |
| Regime-weighted conformal transfer (3 rounds) | transfer residual calibration across farms by upstream-regime density ratios | coverage 88–92 % and interval score ≥ 3 % better than matched power-similarity control | no round passed; best variant ≤ 0.3 % better than its matched control, and significantly worse in the reverse transfer direction |
| LLM-generated feature programs (3 rounds) | a local LLM writes feature-encoding programs, selected on validation | beats handcrafted features by ≥ 1.5 % on both farms | no round passed |
| Anchored NWP increment (ANI) on LightGBM | convert the NWP change into a power change and anchor it at current power | ≥ 1 % on every farm and pooled CI > 0 over the full-information model | pooled −0.04 % [−0.34, 0.25] |
| Wind-anchored / ratio / bias-corrected NWP (3 variants, Bonferroni) | anchor the NWP change at observed nacelle wind; ratio form; 6-h bias correction | same gate | pooled +0.14 %, −0.25 %, +0.06 %, none significant |

<!-- DECISIONS stages G1, N0, M2–M4 (C line), L1/M5–M7, X2, X2-A2 -->

Interpretation: once the model sees the NWP forecast together with the current power and recent history, transformations of the same information do not add skill; the remaining error is dominated by NWP amplitude error and by ramp timing.

---

## 7 Discussion

### 7.1 How do these errors compare with the literature?

[Draft from the literature scan, to be tightened:] With the same convention (hourly, 1–4 h, out-of-year test), our 4-h accuracy (HoT ≈ 84 %, turbine level) is close to published values for wind farms without regional NWP correction (e.g. 84.7 % at 4 h for a four-turbine graph model without NWP), and below farm-level results with corrected or regional NWP (87.8–90.9 %). Studies reporting R² ≥ 0.95 on Kelmarsh use 10-min horizons, random splits or target-time wind as input.

### 7.2 Limitations

- Coarse global NWP (GFS 0.25°, 10 m, 6–11 h delay); regional or hub-height NWP would likely raise the ceiling.
- Three farms in one climate region (UK).
- HoT test years are entirely post-retrofit, while most fit data are pre-retrofit.
- The two HoT test years share the same fitted models.
- The 15-min resolution of GB/T 40607 is not available from hourly aggregation.

---

## 8 Conclusion

[PENDING after the B line.]

---

## Data and code availability

Kelmarsh and Penmanshiel SCADA (Zenodo, CC-BY-4.0); Hill of Towie (Zenodo 22662930); MIDAS Open (OGL v3, attribution required); GFS (NOAA, public). Code, protocols, hashes and all result files: [repository, PENDING].

## References (to verify)

- Gneiting T. et al. (2006) Calibrated probabilistic forecasting at the Stateline wind energy center. JASA.
- Hering A.S., Genton M.G. (2010) Powering up with space-time wind forecasting. JASA.
- Tastu J. et al. (2011) Spatio-temporal analysis and modeling of short-term wind power forecast errors. Wind Energy.
- Messner J.W., Pinson P. (2019) Online adaptive lasso estimation in vector autoregressive models for high dimensional wind power forecasting. IJF.
- Hong T., Pinson P., Fan S. (2014) Global Energy Forecasting Competition 2012. IJF.
- Ansari A.F. et al. (2025) Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.
- Chen S.-A. et al. (2023) TSMixer: An all-MLP architecture for time series forecasting. TMLR.
- Ke G. et al. (2017) LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
- Ganin Y. et al. (2016) Domain-adversarial training of neural networks. JMLR.
- Sci. Rep. 15:33891 (2025) Portability of short-term wind power forecasting (Kelmarsh/Penmanshiel).
- GB/T 40607-2021. Technical requirements for wind/PV power forecasting systems on the dispatch side.
