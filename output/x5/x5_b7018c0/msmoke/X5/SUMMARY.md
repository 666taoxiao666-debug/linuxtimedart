# X5b: matched coverage, fresh (6 h) vs old (15 h) GFS cycle (LightGBM, seed 2024)

Label on V2: NO_COMMON_PATTERN (post-hoc development check; descriptive labels, not causal claims; test years exploratory). Farms whose V2 cycle contrast includes natural-gap differences: none.
U = MAE(LG) - MAE(LUG) in kW. A fresh as is, C old as is, D fresh with C's coverage, Cc old with A's coverage. U(C) - U(A) = [U(D) - U(A)] + [U(Cc) - U(D)] + [U(C) - U(Cc)].
cycle (all) includes any natural-gap difference between D and Cc; cycle (agreeing) uses only windows where D and Cc have identical actual availability; labels use the agreeing contrast and a bound of ratio x |truncation point estimate|.

| farm | split | U A / C / Cc / D | truncation U(D)-U(A) [CI] | cycle all [CI] | cycle agreeing [CI] | agree share | old-side | total | label |
|---|---|---|---|---|---|---|---|---|---|
| Kelmarsh | V2 | 0.42 / 1.89 / 0.86 / 1.08 | 0.66 [-1.04, 2.34] | -0.22 [-1.77, 1.21] | -0.22 [-1.77, 1.21] | 1.0000 | 1.03 | 1.47 | CI_WIDE (bound 0.33) |
| Kelmarsh | Kelmarsh_2022 | -1.24 / 7.76 / 6.21 / 4.68 | 5.92 [2.88, 9.01] | 1.53 [-0.64, 3.37] | 1.53 [-0.64, 3.37] | 1.0000 | 1.55 | 9.00 |  |
| Penmanshiel | V2 | 1.09 / 5.88 / 4.92 / 6.58 | 5.49 [2.71, 8.51] | -1.66 [-3.61, -0.04] | -1.66 [-3.61, -0.04] | 1.0000 | 0.96 | 4.79 | CI_BELOW_0 (bound 2.74); truncation CI > 0 |
| Penmanshiel | Penmanshiel_2021 | 1.65 / 3.57 / 6.18 / 4.45 | 2.80 [-0.42, 6.07] | 1.73 [-0.78, 4.30] | 1.28 [-1.16, 3.66] | 0.9783 | -2.61 | 1.92 |  |
| HoT | V2 | 0.71 / 6.95 / 5.26 / 5.21 | 4.49 [0.93, 8.04] | 0.05 [-2.21, 2.23] | 0.05 [-2.21, 2.23] | 1.0000 | 1.69 | 6.24 | CI_WITHIN_BOUND (bound 2.25); truncation CI > 0 |
| HoT | HoT_2024 | 3.29 / 5.76 / 5.08 / 4.54 | 1.25 [-4.75, 6.83] | 0.54 [-2.33, 3.60] | 0.54 [-2.33, 3.60] | 1.0000 | 0.68 | 2.47 |  |
| HoT | HoT_2025 | 5.77 / 9.45 / 11.86 / 14.20 | 8.43 [3.58, 13.78] | -2.33 [-5.46, 0.97] | -2.33 [-5.46, 0.97] | 1.0000 | -2.41 | 3.68 |  |
