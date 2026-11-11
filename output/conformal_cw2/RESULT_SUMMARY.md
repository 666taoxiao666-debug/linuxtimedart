# CW2 Result Summary (`cw2_b23072d_r1`)

- **Run code**: `b23072dc63344332a17f3df00a4431887dfdeae4` (`exp/M2`)
- **Server run directory**: `/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Conformal/cw2_b23072d_r1`
- **Elapsed**: `961.42 s` (`completed.json` UTC `2026-11-09T07:29:58.637441+00:00`)
- **Selection freeze SHA256**: `4c89fdac1bb54b33742ebc98252c8ae8663d620e0e2325a104d3889b6d22626e`
- **Read-only acceptance (`output/conformal_cw2/verify_saved.py`)**: `PASSED` (`60` `.npz` arrays, `374` input files, `56` source files hashed; `no_inference=True`, `j3_evaluated=False`)
- **Decision**: **`STOP_AT_SELECTION`** (`selected = null`; neither `N-R` nor `N-Rs` beat `min(N-P, B3-A)` on Winkler across the 4 selection cohorts; `J3` was not evaluated and remains untouched)

## 1. Selection Stage Pooled 90% Metrics (3-Seed Mean: 2024, 2025, 2026)

| Cohort | Method | Coverage (%) | Mean Width (kW) | Winkler (kW) | Cap Fraction (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| **Penmanshiel 2022 (`selection_2022`)** | `B0` | 78.5593 | 746.9999 | 1833.2425 | 0.0000 |
| | `B1` | 91.3723 | 1242.1126 | 1639.9801 | 0.0000 |
| | `B2-A` | 89.2763 | 1098.5232 | 1595.9885 | 0.0000 |
| | `B3-A` | 89.4131 | 961.9498 | **1422.8803** | 0.0000 |
| | `N-P` | 89.4884 | 1032.6261 | 1509.1754 | 0.0000 |
| | `N-R` | 89.4971 | 1037.5299 | 1496.0693 | 0.0000 |
| | `N-Rs` | 89.4831 | 1047.8889 | 1513.1282 | 0.0000 |
| **Penmanshiel 2023 (`selection_2023`)** | `B0` | 82.8470 | 796.5351 | 1636.8964 | 0.0000 |
| | `B1` | 94.4837 | 1352.4709 | 1610.8071 | 0.0000 |
| | `B2-A` | 89.3204 | 1006.1294 | 1497.0523 | 0.0000 |
| | `B3-A` | 89.4561 | 855.2339 | **1283.9894** | 0.0000 |
| | `N-P` | 89.5565 | 879.3264 | 1317.5574 | 0.0000 |
| | `N-R` | 89.5834 | 889.4120 | 1297.3899 | 0.0000 |
| | `N-Rs` | 89.5719 | 892.4840 | 1312.5215 | 0.0000 |
| **Penmanshiel 2024 (`J1`, K→P)** | `B0` | 81.8252 | 815.5509 | 1674.4049 | 0.0000 |
| | `B1` | 94.7307 | 1415.8142 | 1648.4258 | 0.0000 |
| | `B2-A` | 89.5147 | 1080.4109 | 1546.0712 | 0.0000 |
| | `B3-A` | 89.6746 | 909.9160 | **1358.6229** | 0.0000 |
| | `N-P` | 89.7031 | 1002.4891 | 1421.9854 | 0.0000 |
| | `N-R` | 89.6591 | 1000.2962 | 1420.0433 | 0.0000 |
| | `N-Rs` | 89.6523 | 987.4048 | 1409.9314 | 0.0000 |
| **Kelmarsh 2024 (`J2`, P→K)** | `B0` | 91.9664 | 985.3706 | 1318.2475 | 0.0000 |
| | `B1` | 93.6059 | 1072.7034 | 1343.8913 | 0.0000 |
| | `B2-A` | 90.0521 | 889.8525 | 1285.5567 | 0.0000 |
| | `B3-A` | 90.2497 | 809.2723 | **1154.7523** | 0.0000 |
| | `N-P` | 90.2278 | 861.6222 | 1206.5280 | 0.0000 |
| | `N-R` | 90.0645 | 838.5600 | 1181.4591 | 0.0000 |
| | `N-Rs` | 90.0909 | 841.7175 | 1190.5440 | 0.0000 |

## 2. Candidate Eligibility Check vs `min(N-P, B3-A)` and vs `N-P`

In all 4 selection cohorts, `min(N-P, B3-A)` is `B3-A`.

- **`N-R` (18 features: power + 5 changes + 2 state + missing + 9 sectors)**:
  - Coverage is inside `[88%, 92%]` on all 4 cohorts (`89.50%`, `89.58%`, `89.66%`, `90.06%`).
  - Relative to **`N-P` alone** (isolated upstream-feature effect inside normalized conformal), `N-R` reduces Winkler in all 4 cohorts: `-13.11 kW` (`2022`), `-20.17 kW` (`2023`), `-1.94 kW` (`J1`), `-25.07 kW` (`J2`).
  - Relative to **`min(N-P, B3-A) = B3-A`**, `N-R` has higher Winkler on all 4 cohorts: `+73.19 kW` (`-5.1437%`), `+13.40 kW` (`-1.0437%`), `+61.42 kW` (`-4.5208%`), `+26.71 kW` (`-2.3128%`); 4-cohort mean relative gain `-3.2552%`. **Not eligible.**
- **`N-Rs` (6 features: power + 5 changes)**:
  - Coverage is inside `[88%, 92%]` on all 4 cohorts (`89.48%`, `89.57%`, `89.65%`, `90.09%`).
  - Relative to **`N-P` alone**, `N-Rs` reduces Winkler on 3 of 4 cohorts (`+3.95 kW` in `2022`, `-5.04 kW` in `2023`, `-12.05 kW` in `J1`, `-15.98 kW` in `J2`).
  - Relative to **`min(N-P, B3-A) = B3-A`**, `N-Rs` has higher Winkler on all 4 cohorts: `+90.25 kW` (`-6.3426%`), `+28.53 kW` (`-2.2221%`), `+51.31 kW` (`-3.7765%`), `+35.79 kW` (`-3.0995%`); 4-cohort mean relative gain `-3.8602%`. **Not eligible.**

Because neither candidate passed the 4-cohort eligibility gate (`Winkler < min(N-P, B3-A)` on all 4 cohorts), `pilot_conformal_cw2.py` recorded `decision = "STOP_AT_SELECTION"` and did not run `J3` (Kelmarsh 2023 remains untouched).

## 3. Feature Importance (`permutation_importance` on Source Calibration Year, 4-Horizon Mean MAE Drop in kW)

- **Source Kelmarsh (`selection_2022`, `selection_2023`, `J1`)**:
  - `N-Rs`: `pred_power` dominates (`39.93–43.42 kW`), followed by `abs_grad_magnitude_change3` (`2.93–3.63 kW`), `abs_upwind_dws3` (`1.99–2.84 kW`), `abs_local_dp3` (`2.03–2.79 kW`), `abs_regional_dws3` (`2.04–2.36 kW`), `abs_local_dws3` (`1.24–1.38 kW`).
  - `N-R`: `pred_power` (`41.18–44.58 kW`), `regional_ws` (`2.29–4.68 kW`), `abs_grad_magnitude_change3` (`2.18–3.17 kW`), `upwind_minus_local` (`1.87–2.59 kW`), `abs_upwind_dws3` (`1.57–2.19 kW`).
- **Source Penmanshiel (`J2`)**:
  - `N-Rs`: `pred_power` (`60.11 kW`), `abs_grad_magnitude_change3` (`4.71 kW`), `abs_local_dp3` (`3.88 kW`), `abs_upwind_dws3` (`3.10 kW`), `abs_regional_dws3` (`2.53 kW`), `abs_local_dws3` (`2.11 kW`).
  - `N-R`: `pred_power` (`58.32 kW`), `regional_ws` (`4.98 kW`), `abs_grad_magnitude_change3` (`3.91 kW`), `abs_local_dp3` (`3.01 kW`), `upwind_minus_local` (`2.51 kW`).

## 4. Descriptive Strata Summary (Ramp `|ΔP| >= 205 kW` vs Calm; Upstream Missing vs Observed)

- **Ramp vs Calm**:
  - `N-P`, `N-R`, and `N-Rs` raise ramp-window coverage by `+0.6%` to `+3.5%` relative to `B3-A` (e.g. `J1` ramp coverage: `B3-A` `76.36%` vs `N-R` `79.88%`, `N-Rs` `79.65%`), showing that normalized scaling widens intervals during ramps.
  - However, multiplying $\hat{\sigma}_h(x)$ by the global normalized quantile $q_h(1-\alpha_t)$ produces heavier tail quantiles than local kNN (`B3-A` directly estimates the conditional quantile in power neighborhoods without a multiplicative constant-shape assumption), inflating width and Winkler across both ramp and calm windows relative to `B3-A`.
