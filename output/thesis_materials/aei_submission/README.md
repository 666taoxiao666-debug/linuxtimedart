# Reviewer guide: code and results for the AEI submission

This repository started from the TimeDART code base (the README at the repository root describes that earlier project).
For the study under review internally, use the project branch `codex/self-evolving-wiki`. The tag reserved for an author-approved final version is
`aei-submission-v1`. The default branch `main` contains the earlier project.

## What is and is not in the repository

- **Included:** analysis code, frozen protocols, tests, stage results and their audit records.
- **Stored separately:** raw SCADA, MIDAS and GFS files, prepared inputs, prediction arrays and fitted models. The stage-specific launch, freeze and completion records identify the inputs and artifacts checked in each run.
- **Review record:** `docs/review/DECISIONS.md` and `docs/review/*-codex.md` contain the pre-registered decision rules,
  the review of every stage and the independent recomputations.

## Where each result comes from

| Manuscript item | Result file(s) | Code | Protocol |
|---|---|---|---|
| Table 1, Sections 5.1–5.2 (Kelmarsh, Penmanshiel) | `output/g1/results_78/formal/results.json` (B, N), `output/g2a/results81/results.json` (adds B0, N0 and computes paired comparisons) | `scripts/prepare_g1_batches.py`, `scripts/train_g1.py` (G2a uses `--g2a-parent` to reuse B and N) | `configs/g1_protocol.json`, `configs/g2a_protocol.json` |
| Table 1, Sections 5.1–5.2 (Hill of Towie, sealed) | `output/g2b/sealed_eval/server/results.json` | `scripts/prepare_hot_dev.py` (development inputs), `scripts/train_g2b.py` (development fits and freeze), `scripts/evaluate_g2b_sealed.py` (sealed preparation and evaluation) | `configs/g2b_protocol.json` |
| Table 2 | `output/x1/A0/`, `output/x1/A1/`, `output/x1/A2/`, `output/x1/x1_e4819b0_gpu_r1/A3/results.json` | `scripts/run_x1.py` | `configs/x1_protocol.json` |
| Section 5.3 | `output/x2/x2_91d4d20/report_a/` | `scripts/run_x2.py` | `configs/x2_protocol.json` |
| Table 2b | `output/x2/x2_0f24875/report/` | `scripts/run_x2.py` | `configs/x2_protocol.json` |
| Section 5.4 (Chronos-2 rounds) | `output/x3/x3_f8af1b6/report/dev_decisions.json`, `output/x4/x4_1c0a125/report/dev_decisions.json` | `scripts/run_x3.py`, `scripts/run_x4.py` | `configs/x3_protocol.json`, `configs/x4_protocol.json` |
| Section 5.5 | `output/x1/E1/`, `output/x1/E1_hot_31ce5d3_r2/`, `output/x1/E2/results.json` | `scripts/run_x1.py` | `configs/x1_protocol.json` |
| Table 4, Section 5.6 | `output/x5/x5_e01cc6b/run/X5/results.json`, `output/x5/local_blocks12/results.json` | `scripts/run_x5.py` (delay runs), `scripts/analyze_info_value_x5.py` (local statistics from saved X2 arrays) | `configs/x5_protocol.json` |
| Table 4b | `output/x5/x5_b7018c0/mrun/X5/results.json` | `scripts/run_x5.py` (modes `msmoke`, `mrun`) | `configs/x5b_protocol.json` |
| Tables 5–6, Section 5.7 | `output/wjb/wjb_1fe7ab6/report/results.json` | `scripts/run_wjb.py`, `models/vendor_cabin/`, `models/vendor_tsg/` | `configs/wjb_protocol.json` |
| Table 3 (pre-registered negative results) | `docs/review/DECISIONS.md` (G1, N0, M2–M4, L1/M5–M7, X2, X2-A2, X3-result and X5 for X4) | see the stage sections | see the stage sections |

For formal runs, consult the stage launch and completion receipts alongside the results. For the local X5 blocks, the result JSON identifies the saved X2 inputs and the analysis script hash.

## Running the code

- The root `requirements.txt` lists the base project dependencies. The LightGBM baselines use version 4.5.0.
- Chronos-2 stages run in a separate environment and record the model revision they load.
- For X2–X5 and the wind-journal baselines, see `scripts/launch_x2.sh` and `scripts/status_x2.sh`. The G stages use the entry points shown in the table.
- Tests: `python -m unittest tests.test_x2 tests.test_x5 tests.test_wjb` and the other `tests/test_*.py` files.
- For the X-stage scripts, `--logs` points to the experiment log root (`<data root>/outputs/logs`). For G stages, the launch receipts identify the prepared inputs, parent runs and frozen protocols.

## Data sources

- Kelmarsh wind farm data, Zenodo, https://doi.org/10.5281/zenodo.16807551 (CC BY 4.0).
- Penmanshiel wind farm data v3, Zenodo, https://doi.org/10.5281/zenodo.16807304 (CC BY 4.0).
- Hill of Towie wind farm open dataset v2.1.0, Zenodo, https://doi.org/10.5281/zenodo.22662930 (CC BY 4.0).
- MIDAS Open: UK hourly weather observation data, version 202607, CEDA,
  https://doi.org/10.5285/d04207b551674c07801b7d4e6d883e50 (Open Government Licence v3).
- For GFS cycles issued in 2016–2020, the study used monthly subsets from NSF NCAR GDEX d084001, https://doi.org/10.5065/D65D8PWK (CC BY 4.0). For cycles issued in 2021–2026 within the acquisition period, it used NOAA Open Data on AWS (`s3://noaa-gfs-bdp-pds`). The B2 and B1 annual summaries in `output/external_acquisition/gfs_g0/` document the two batches.

## Third-party code

- Cabin: https://github.com/xiufengliu/cabin-wind-forecasting, commit 392f103, MIT licence (`models/vendor_cabin/LICENSE`).
- Adaptive robust regression with learned partitions: https://github.com/akylasstrat/wind-forecast-missing-data,
  commit 409f20f, MIT licence (`models/vendor_tsg/LICENSE`); two unused import lines are commented out.
