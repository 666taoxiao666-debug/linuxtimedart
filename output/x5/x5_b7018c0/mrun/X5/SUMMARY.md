# X5b: matched coverage, fresh (6 h) vs old (15 h) GFS cycle (LightGBM, seed 2024)

Label on V2: TRUNCATION_CI_ABOVE_0_AND_CYCLE_CI_WITHIN_BOUND (post-hoc development check; descriptive labels, not causal claims; test years exploratory). Farms whose V2 cycle contrast includes natural-gap differences: none.
U = MAE(LG) - MAE(LUG) in kW. A fresh as is, C old as is, D fresh with C's coverage, Cc old with A's coverage. U(C) - U(A) = [U(D) - U(A)] + [U(Cc) - U(D)] + [U(C) - U(Cc)].
cycle (all) includes any natural-gap difference between D and Cc; cycle (agreeing) uses only windows where D and Cc have identical actual availability; labels use the agreeing contrast and a bound of ratio x |truncation point estimate|.

| farm | split | U A / C / Cc / D | truncation U(D)-U(A) [CI] | cycle all [CI] | cycle agreeing [CI] | agree share | old-side | total | label |
|---|---|---|---|---|---|---|---|---|---|
| Kelmarsh | V2 | 2.28 / 6.75 / 6.43 / 6.03 | 3.75 [2.36, 5.39] | 0.40 [-0.82, 1.59] | 0.40 [-0.82, 1.59] | 1.0000 | 0.32 | 4.47 | CI_WITHIN_BOUND (bound 1.87); truncation CI > 0 |
| Kelmarsh | Kelmarsh_2022 | 2.54 / 7.05 / 7.83 / 7.30 | 4.76 [3.55, 5.98] | 0.53 [-0.28, 1.30] | 0.55 [-0.26, 1.30] | 0.9979 | -0.78 | 4.51 |  |
| Penmanshiel | V2 | 2.18 / 10.91 / 9.38 / 7.93 | 5.76 [3.08, 8.37] | 1.45 [-0.47, 3.50] | 1.45 [-0.47, 3.50] | 1.0000 | 1.53 | 8.73 | CI_WIDE (bound 2.88); truncation CI > 0 |
| Penmanshiel | Penmanshiel_2021 | 2.70 / 6.21 / 5.41 / 5.64 | 2.94 [1.54, 4.28] | -0.23 [-1.25, 0.88] | -0.35 [-1.46, 0.82] | 0.9934 | 0.80 | 3.51 |  |
| HoT | V2 | 1.82 / 13.46 / 14.26 / 14.62 | 12.80 [9.20, 16.32] | -0.36 [-1.98, 1.26] | -0.36 [-1.98, 1.26] | 1.0000 | -0.80 | 11.64 | CI_WITHIN_BOUND (bound 6.40); truncation CI > 0 |
| HoT | HoT_2024 | 3.56 / 9.24 / 9.36 / 9.70 | 6.14 [4.65, 7.46] | -0.33 [-1.01, 0.35] | -0.33 [-1.01, 0.35] | 1.0000 | -0.13 | 5.68 |  |
| HoT | HoT_2025 | 6.70 / 13.83 / 13.93 / 13.48 | 6.77 [5.44, 8.21] | 0.45 [-0.23, 1.18] | 0.45 [-0.23, 1.18] | 1.0000 | -0.10 | 7.12 |  |
