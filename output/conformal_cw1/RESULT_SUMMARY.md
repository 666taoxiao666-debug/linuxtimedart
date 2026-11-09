# CW1：挑选选中 W3（分时效组合），但 J1/J2 均通过 C1/C2、未通过 C3

2026-11-09，M2 R2 复审通过后按第 56 次批准正式运行一次。结论 **STOP_CW1（有效负结果，两组判定数据均过 C1/C2 但未过 C3 新意门槛）**。按事先规则，两组都没过 C3，C 线第 2 轮（CW2）应换方向设计。

## 运行与验收

- 运行代码 `0b3eaf9295dfbbc2403fc464534582c113faf25f`，分支 `exp/M2`；工作区 `/home/XT/nuist/pythoncode/TimeDARTFirst_CW1_0b3eaf9`。
- 正式目录 `/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Conformal/cw1_0b3eaf9_r1`，PID 669360，耗时 424.481 秒，2026-11-09 03:45:16 UTC 完成并退出。
- 命令：`python scripts/pilot_conformal_cw1.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Conformal/cw1_0b3eaf9_r1`；CPU，线程数 2。
- 8 项单元测试与服务器 `--smoke`（`r1/smoke/`）通过，覆盖未来后缀反例（R1-B1）、等权 W1=B3-A、W4 延迟反馈归原组、W3 分步长组合、两年挑选与平局规则、起报簇 block bootstrap 重采样索引等。
- `verify_saved.py` 只读验收通过（`verification.json`）：74 个数组文件、374 个输入文件、55 个源文件 SHA256 全部一致，全部种子/步长指标与 J1/J2 bootstrap 95% CI 由保存数组重算一致，未生成新预测。
- `selection_freeze.json` SHA256 `a041b91768004b8bb43ee8aadc9b9c70b47ffa87bd112635d1534d8eef9b51ae`，写入早于 `judgment_started.json`。

## 挑选阶段（Penmanshiel 2022、2023，正向）

四个变体在 2022、2023 两年的覆盖率均在 `[88%, 92%]` 内（3 个 seed 平均，h1–4 合并）：

- **B3-A（匹配对照）**：2022 覆盖 89.4131% / Winkler 1422.8803 kW；2023 覆盖 89.4561% / Winkler 1283.9894 kW。
- **M-A**：2022 覆盖 89.4410% / Winkler 1425.5136 kW；2023 覆盖 89.5002% / Winkler 1280.1556 kW。
- **W1（近邻内情形密度比加权）**：2022 覆盖 89.5306% / Winkler 1504.4468 kW；2023 覆盖 89.6364% / Winkler 1300.0158 kW；两年平均相对 B3-A 改善 **-3.4903%**。
- **W2（逐 h HistGradientBoosting 密度比）**：2022 覆盖 89.5406% / Winkler 1426.4720 kW；2023 覆盖 89.6212% / Winkler 1279.4774 kW；两年平均相对 B3-A 改善 **+0.0495%**。
- **W3（分时效组合）**：按两年各 h 平均 Winkler，h1、h2 选 `B3-A`，h3、h4 选 `M-A`；2022 覆盖 89.4222% / Winkler 1420.2164 kW；2023 覆盖 89.4875% / Winkler 1277.1402 kW；两年平均相对 B3-A 改善 **+0.3603%**。
- **W4（按过去校准年 k-means 情形分组的 ACI）**：2022 覆盖 88.0656% / Winkler 1438.0808 kW（对照 B3-G：88.0647% / 1429.0541 kW）；2023 覆盖 88.4164% / Winkler 1275.3827 kW（对照 B3-G：88.4374% / 1280.4636 kW）；两年平均相对 B3-A 改善 **-0.1990%**。

按事先规则选中 **W3（`["B3-A", "B3-A", "M-A", "M-A"]`）** 进入 J1、J2 判定。

## 判定阶段（J1 正向 Penmanshiel 2024，J2 反向 Kelmarsh 2024）

### J1（正向：源 Kelmarsh 2023，目标月 Penmanshiel 2023-12，评估 Penmanshiel 2024）
- B0：81.8252% / 宽度 815.5509 / Winkler 1674.4049 kW
- B1：94.7307% / 宽度 1415.8142 / Winkler 1648.4258 kW
- B2-A：89.5147% / 宽度 1080.4109 / Winkler 1546.0712 kW
- **B3-A**：89.6746% / 宽度 909.9160 / Winkler 1358.6229 kW / 截顶率 0%
- **W3**：89.6950% / 宽度 918.8813 / Winkler 1361.4740 kW / 截顶率 0.0254%
- **判定**：**C1 通过**（89.6950% ∈ [88%,92%]），**C2 通过**（1361.4740 ≤ 0.95 × 1546.0712 = 1468.7676），**C3 不通过**（W3 比 B3-A 高 2.8511 kW；`B3-A − W3` 均值 -2.8511 kW，95% block bootstrap CI `[-7.7841, +2.3752]` kW）。

### J2（反向：源 Penmanshiel 2023，目标月 Kelmarsh 2023-12，评估 Kelmarsh 2024）
- B0：91.9664% / 宽度 985.3706 / Winkler 1318.2475 kW
- B1：93.6059% / 宽度 1072.7034 / Winkler 1343.8913 kW
- B2-A：90.0521% / 宽度 889.8525 / Winkler 1285.5567 kW
- **B3-A**：90.2497% / 宽度 809.2723 / Winkler 1154.7523 kW / 截顶率 0%
- **W3**：90.3050% / 宽度 843.7908 / Winkler 1185.0708 kW / 截顶率 0%
- **判定**：**C1 通过**（90.3050% ∈ [88%,92%]），**C2 通过**（1185.0708 ≤ 0.95 × 1285.5567 = 1221.2789），**C3 不通过**（W3 比 B3-A 高 30.3185 kW；`B3-A − W3` 均值 -30.3185 kW，95% block bootstrap CI `[-37.1175, -23.6942]` kW）。
