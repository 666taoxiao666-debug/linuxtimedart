# N1-PH2 result record

Completed 2026-11-05 21:36:53 Asia/Shanghai; 3273.94 seconds (54.57 minutes). 42 wind fits + 1 PH-mix. Implementation 324e708.

## Decision

STOP: all four primary development comparisons unsupported; IP noninferiority failed. Effective negative result, not a technical failure. No further PH optimization or automatic rerun.

## Source-only selection

- V0: Kelmarsh 2020 validation MAE 175.951855 kW.
- V1: Kelmarsh 2020 validation MAE 175.748402 kW.
- V2: Kelmarsh 2020 validation MAE 172.030383 kW.
- V3: Kelmarsh 2020 validation MAE 171.970877 kW.

Selected V2 (mse, ph+h): V3 is only 0.059506 kW better, within the frozen <0.5 kW tie rule, which prefers V2. Target outcomes were not used to choose.

## Primary T21 comparisons

Positive gain favours the candidate. MAE/RMSE effects and 95% intervals in kW; Holm applies to ZP/ZQ/F1P/F3P only.
- ZP: MAE gain -4.921768, CI [-9.469538, 0.273630]; Holm 0.18219187595416542; RMSE gain 5.727186, CI [-1.695520, 13.836260].
- ZQ: MAE gain 0.300479, CI [-1.919583, 2.855385]; Holm 1.0; RMSE gain -0.303395, CI [-3.304724, 3.151667].
- F1P: MAE gain 0.300479, CI [-1.919583, 2.855385]; Holm 1.0; RMSE gain -0.303395, CI [-3.304724, 3.151667].
- F3P: MAE gain -1.227899, CI [-2.741094, 0.420471]; Holm 0.38947922058076745; RMSE gain 0.108030, CI [-2.366119, 2.887669].
- IP: MAE gain -1.665612, CI [-2.901674, -0.345826]; Holm not applied; RMSE gain -1.677543, CI [-4.445412, 0.846338].

IP lower bound -2.901674 kW is below the -2 kW noninferiority margin.

## All variants: primary zero-shot descriptive metrics

- V0: MAE 201.879525; RMSE 310.832737; R2 0.768018.
- V1: MAE 200.989265; RMSE 309.650253; R2 0.769783.
- V2: MAE 202.680110; RMSE 314.228339; R2 0.762901.
- V3: MAE 202.655464; RMSE 313.495140; R2 0.764053.

## T21: per-arm seed-mean metrics

- K-last10: MAE 197.758342; RMSE 319.998665; R2 0.754180.
- K-upstream: MAE 202.980589; RMSE 313.954402; R2 0.763358.
- P-last10: MAE 195.520902; RMSE 319.861870; R2 0.754390.
- P-upstream: MAE 184.371365; RMSE 299.897797; R2 0.784086.
- FT-1: MAE 202.980589; RMSE 313.954402; R2 0.763358.
- Scratch-1: MAE 314.430823; RMSE 415.731550; R2 0.582269.
- FT-3: MAE 189.094886; RMSE 302.731258; R2 0.779994.
- Scratch-3: MAE 225.778759; RMSE 336.739858; R2 0.727763.
- PersistenceHour: MAE 206.069054; RMSE 348.135583; R2 0.709051.
- PersistenceLast10: MAE 198.684935; RMSE 343.008944; R2 0.717557.
- K-PH: MAE 202.680110; RMSE 314.228339; R2 0.762901.
- FT-1-PH: MAE 202.680110; RMSE 314.228339; R2 0.762901.
- FT-3-PH: MAE 190.322785; RMSE 302.619743; R2 0.780151.
- P-PH: MAE 186.036976; RMSE 301.577354; R2 0.781664.

## T22: per-arm seed-mean metrics

- K-last10: MAE 232.534201; RMSE 355.129737; R2 0.713240.
- K-upstream: MAE 229.996851; RMSE 346.542028; R2 0.726935.
- P-last10: MAE 229.583088; RMSE 353.177056; R2 0.716379.
- P-upstream: MAE 213.630822; RMSE 329.317339; R2 0.753407.
- FT-1: MAE 226.307202; RMSE 339.242387; R2 0.738325.
- Scratch-1: MAE 312.512536; RMSE 430.229278; R2 0.578668.
- FT-3: MAE 220.890818; RMSE 331.872185; R2 0.749569.
- Scratch-3: MAE 269.599527; RMSE 374.711143; R2 0.680716.
- PersistenceHour: MAE 246.391843; RMSE 391.066808; R2 0.652269.
- PersistenceLast10: MAE 240.901730; RMSE 387.614779; R2 0.658381.
- K-PH: MAE 225.359964; RMSE 339.253788; R2 0.738294.
- FT-1-PH: MAE 224.125583; RMSE 337.885108; R2 0.740413.
- FT-3-PH: MAE 218.660260; RMSE 329.716994; R2 0.752813.
- P-PH: MAE 213.176076; RMSE 328.177773; R2 0.755115.

Descriptive only; no confirmatory decision or family Holm claim.
- ZP: MAE gain 7.174237, CI [2.552585, 11.452632]; RMSE gain 15.868543, CI [9.559344, 22.017979].
- ZQ: MAE gain 4.636887, CI [2.217278, 7.004477]; RMSE gain 7.284180, CI [3.985945, 10.529340].
- F1P: MAE gain 2.181619, CI [0.381414, 3.983323]; RMSE gain 1.356240, CI [-0.971360, 3.650800].
- F3P: MAE gain 2.230559, CI [0.841955, 3.615062]; RMSE gain 2.156703, CI [0.458676, 3.910784].
- IP: MAE gain 0.454746, CI [-0.804554, 1.716019]; RMSE gain 1.142485, CI [-0.760099, 3.110127].

## T23: per-arm seed-mean metrics

- K-last10: MAE 206.684280; RMSE 327.399476; R2 0.732596.
- K-upstream: MAE 209.004837; RMSE 325.976058; R2 0.734913.
- P-last10: MAE 203.850214; RMSE 325.935996; R2 0.734979.
- P-upstream: MAE 188.838761; RMSE 301.960294; R2 0.772537.
- FT-1: MAE 202.299961; RMSE 311.099335; R2 0.758542.
- Scratch-1: MAE 289.612979; RMSE 403.395940; R2 0.592711.
- FT-3: MAE 196.854533; RMSE 306.870752; R2 0.765067.
- Scratch-3: MAE 239.718219; RMSE 346.219405; R2 0.700847.
- PersistenceHour: MAE 217.096319; RMSE 356.732495; R2 0.682535.
- PersistenceLast10: MAE 211.948916; RMSE 355.233942; R2 0.685197.
- K-PH: MAE 202.392412; RMSE 314.642106; R2 0.753028.
- FT-1-PH: MAE 195.014668; RMSE 305.143035; R2 0.767711.
- FT-3-PH: MAE 195.347929; RMSE 306.384952; R2 0.765813.
- P-PH: MAE 188.122429; RMSE 300.483723; R2 0.774754.

Descriptive only; no confirmatory decision or family Holm claim.
- ZP: MAE gain 4.291868, CI [-0.281575, 9.012880]; RMSE gain 12.756800, CI [5.843061, 19.884785].
- ZQ: MAE gain 6.612424, CI [3.516015, 10.093771]; RMSE gain 11.334781, CI [6.940609, 16.210268].
- F1P: MAE gain 7.285293, CI [5.143578, 9.505735]; RMSE gain 5.964135, CI [3.255190, 8.752043].
- F3P: MAE gain 1.506604, CI [-0.373939, 3.485576]; RMSE gain 0.487265, CI [-2.144826, 3.181814].
- IP: MAE gain 0.716332, CI [-0.403748, 1.851947]; RMSE gain 1.475528, CI [-0.487698, 3.427755].

## Verification and limits

- Verified 42 wind fit registrations/checkpoints, PH-mix checkpoint, 53 source files, 430 registered artifacts, 61 saved arrays.
- Recomputed all three fold reports and all four primary zero-shot metric sets from saved arrays; exact canonical JSON equality. No training or new predictions.
- Recomputed 19 own-station metrics; station transfer pair metrics are retained from the original run, but individual pair predictions were not persisted, so they were not independently replayed.
- 149 metadata/log files copied and checked against raw SHA256 manifest. No CSV, model weights or prediction arrays copied.
- Metrics are seed means; RMSE contrasts use sqrt(mean seed MSE), so they need not equal differences of mean seed RMSE.
- T21 is posthoc development evidence; T22/T23 are descriptive and rolling years are not independent confirmation.
- No monitoring automation was created for this run; no active run monitor to close.
- Next: Claude review of implementation, selection, negative result, and source/target generalization gap. Merge only after Claude review and user approval.
