# X5 blocks 1-2 (local, zero training; LightGBM X2 arms, seed 2024)

Positive = MAE reduction in kW. V1 fits the D threshold; V2 is the primary development read; test years are exploratory (opened before). Descriptive statistics only.

| farm | split | MAE L0/LU/LG/LUG | phi_U | phi_G | U without G | U with G [CI] | interaction |
|---|---|---|---|---|---|---|---|
| Kelmarsh | V1 | 166.3/147.9/138.7/135.2 | 10.94 (6.6%) | 20.12 (12.1%) | 18.39 | 3.49 [2.21, 4.73] | 14.90 |
| Kelmarsh | V2 | 163.0/145.3/134.4/132.1 | 9.98 (6.1%) | 20.91 (12.8%) | 17.68 | 2.28 [1.22, 3.36] | 15.40 |
| Kelmarsh | Kelmarsh_2022 | 181.2/160.6/147.9/145.3 | 11.53 (6.4%) | 24.30 (13.4%) | 20.52 | 2.54 [1.73, 3.45] | 17.98 |
| Penmanshiel | V1 | 228.4/212.4/191.5/188.6 | 9.46 (4.1%) | 30.32 (13.3%) | 16.02 | 2.89 [1.22, 4.58] | 13.13 |
| Penmanshiel | V2 | 224.2/205.2/178.6/176.5 | 10.62 (4.7%) | 37.14 (16.6%) | 19.06 | 2.18 [0.27, 4.23] | 16.89 |
| Penmanshiel | Penmanshiel_2021 | 190.3/176.5/163.7/161.0 | 8.27 (4.3%) | 21.01 (11.0%) | 13.84 | 2.70 [1.47, 3.93] | 11.13 |
| HoT | V1 | 196.4/180.1/170.1/164.9 | 10.83 (5.5%) | 20.74 (10.6%) | 16.38 | 5.27 [3.12, 7.37] | 11.11 |
| HoT | V2 | 244.4/217.2/188.0/186.2 | 14.52 (5.9%) | 43.68 (17.9%) | 27.22 | 1.82 [-0.34, 3.80] | 25.40 |
| HoT | HoT_2024 | 229.2/212.2/197.6/194.0 | 10.27 (4.5%) | 24.87 (10.9%) | 16.98 | 3.56 [2.32, 4.92] | 13.42 |
| HoT | HoT_2025 | 225.7/203.9/193.4/186.7 | 14.23 (6.3%) | 24.73 (11.0%) | 21.76 | 6.70 [5.17, 8.26] | 15.06 |

## C1: GFS-only vs upstream-only forecast disagreement (top 20% by V1 threshold)

| farm | split | u high | u rest | contrast [CI] | share of u in high | u/err high | u/err rest |
|---|---|---|---|---|---|---|---|
| Kelmarsh | V1 | 9.76 | 1.92 | 7.84 [3.21, 13.10] | 0.56 | 3.9% | 1.7% |
| Kelmarsh | V2 | 6.01 | 1.31 | 4.70 [1.28, 8.01] | 0.54 | 2.7% | 1.2% |
| Kelmarsh | Kelmarsh_2022 | 6.24 | 1.48 | 4.76 [2.46, 7.34] | 0.55 | 2.8% | 1.2% |
| Penmanshiel | V1 | 8.43 | 1.51 | 6.92 [2.15, 11.79] | 0.58 | 2.7% | 0.9% |
| Penmanshiel | V2 | 9.06 | 0.42 | 8.64 [2.59, 15.15] | 0.85 | 3.0% | 0.3% |
| Penmanshiel | Penmanshiel_2021 | 10.87 | 0.95 | 9.92 [5.67, 14.41] | 0.71 | 3.5% | 0.7% |
| HoT | V1 | 11.11 | 3.81 | 7.29 [0.42, 14.13] | 0.42 | 3.6% | 2.8% |
| HoT | V2 | 3.98 | 0.91 | 3.07 [-1.94, 8.40] | 0.65 | 1.3% | 0.7% |
| HoT | HoT_2024 | 9.52 | 1.43 | 8.09 [4.54, 11.56] | 0.70 | 3.0% | 0.9% |
| HoT | HoT_2025 | 17.95 | 2.87 | 15.09 [10.22, 20.30] | 0.68 | 5.4% | 2.0% |

## C2: residual upstream value by GFS age (hours since the used cycle)

- Kelmarsh V1: age 6: u 3.10 (n 3722), age 9: u 3.88 (n 3709)
- Kelmarsh V2: age 6: u 2.12 (n 4126), age 9: u 2.45 (n 4123)
- Kelmarsh Kelmarsh_2022: age 6: u 3.15 (n 8482), age 9: u 1.93 (n 8467)
- Penmanshiel V1: age 6: u 2.33 (n 9728), age 9: u 3.46 (n 9685)
- Penmanshiel V2: age 6: u 2.46 (n 8089), age 9: u 1.89 (n 8087)
- Penmanshiel Penmanshiel_2021: age 6: u 2.56 (n 19077), age 9: u 2.84 (n 19070)
- HoT V1: age 6: u 4.54 (n 2494), age 7: u 2.27 (n 2504), age 8: u 6.71 (n 2478), age 9: u 5.34 (n 2489), age 10: u 6.35 (n 2499), age 11: u 6.45 (n 2468)
- HoT V2: age 6: u 2.54 (n 2475), age 7: u 2.94 (n 2533), age 8: u 2.58 (n 2544), age 9: u 1.76 (n 2474), age 10: u 0.42 (n 2535), age 11: u 0.72 (n 2542)
- HoT HoT_2024: age 6: u 4.20 (n 9910), age 7: u 3.96 (n 9913), age 8: u 3.07 (n 9974), age 9: u 2.99 (n 9917), age 10: u 3.82 (n 9911), age 11: u 3.33 (n 9962)
- HoT HoT_2025: age 6: u 7.37 (n 9552), age 7: u 6.87 (n 9268), age 8: u 6.16 (n 9597), age 9: u 5.70 (n 9560), age 10: u 6.13 (n 9281), age 11: u 7.97 (n 9599)
