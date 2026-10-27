# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-26 第 2 次 -->

**v28 结果诊断（477dda8 的 2021 预测；2021 自此按 v28 next_round 视为开发数据）**
- 模型误差与 PersistenceLast10 误差相关 0.92；同一时刻 6 台风机误差相关 0.66（h1）–0.78（h4），全场共同部分占误差方差 72–82%：剩余误差主要是天气变化，需要前瞻天气信息。
- 按发出小时/风机去偏的"神谕"修正都不降低 MAE，日历和风机编号组的空间很小。
- 停机/降额：700 个"发出时 10 分钟功率 <2% 额定且目标非正常"的窗口，模型 MAE 109 vs 持续法 26（模型预期马上恢复）；事前规则"<2% 额定就用持续法"在 2021 上 168.29→164.88（样本内）。3 seed 平均预测 169.33→168.29。
- 复现：`python output/kelmarsh_hourly/multiyear_v28_review/claude_v28_error_diagnosis.py`（未跟踪的复核目录，只读）。

**本轮改动（用户 2026-10-26 决定：两步走＝本轮做运行状态信息组 + 你评估 NWP；年份前滚）**
- 新协议 `configs/kelmarsh_operating_state_protocol_v29.json`（LF SHA `0f94fc58…`）：训练 2016–2020 / 验证 2021 / 测试 2022 一次 / 2023–24 封存。TSMixer 2×2：{last10, last10+available_capacity_frac+stopped_frac} × {normal, all 训练窗口} × 3 seed = 12 fits，配方与 v28 TSMixer_last10 相同；外加固定规则臂 Guard_last10_normal。
- 所有臂在 2021 的 ALL 队列选 epoch。主队列为 all：状态组针对的停机/降额被 normal 队列按定义排除；normal 仍必报。C1–C6 做 Holm；规则另含 normal 队列非劣效界 2 kW。
- `utils/kelmarsh_operating_state.py`（不依赖 torch：协议校验、D0、Guard、选择、评估、P1、报告）；`scripts/train_kelmarsh_operating_state.py`（继承 V28Run，原样复用 v28 的 fit_config/ChannelView/build_model，未改任何 v25–v28 文件）；`tests/test_kelmarsh_operating_state.py`。
- `scripts/train/Kelmarsh_launch_frozen_protocol.sh`：通用启动器，以后各版本复用，不再每版新增。
- 你上轮的 P2：v29 起 E1 之后写 `post_test_manifest.json`（三个划分的 all/normal 窗口数及各哈希，与已登记 test_cohort 互相核对），data_manifest 保持测试前快照不改。v28 不动，以 completed_result_audit.json 作为它的测试后记录。
- 本地验证：v29 不依赖 torch 的 13 项通过（3 项 torch 跳过）；v28 测试仍 15 项通过；`bash -n` 通过。用桩 torch、假训练/预测跑完整链路 D0→T1→S1→E1→P1→报告→复用不重训→篡改拒绝，全部通过。真实 torch 测试和真实运行未做。

**请 Codex 做**
1. 审查以上 5 个新文件，重点：S1 的 Guard 由已登记验证预测重算、E1 的 issue 数组、P1 一致性检查、启动器。在服务器跑 `tests/test_kelmarsh_operating_state.py` 加 v28 回归测试。有问题直接修；改了协议要同步 `PROTOCOL_SHA256`。通过后提交、推送。
2. 2022 数据准备（协议冻结后才做）：先把 v28 raw 目录里已过 MD5 的 2016–2021 ZIP 和 `Kelmarsh_WT_static.csv` 硬链接或复制到 `datasets/wind_public_v29/resampled_hourly_scada/kelmarsh_multiyear/raw/`，再运行协议 `data.v29_preparation.command`（只会下载 2022）。然后只读审计：
   `python scripts/audit_kelmarsh_multiyear_preparation.py datasets/wind_public_v29/resampled_hourly_scada/kelmarsh_multiyear/metadata/preparation.json --years 2016 2017 2018 2019 2020 2021 2022 --curve-years 2016 2017 2018 2019 --output outputs/audits/operating_state_v29/independent_audit.json`
3. 先跑 `bash scripts/train/Kelmarsh_launch_frozen_protocol.sh scripts/train_kelmarsh_operating_state.py --preflight-only`，通过后用同一命令加 `--detach` 只启动一次。不碰 v28 运行目录。
4. NWP 可行性只出报告，不训练、不写协议。要回答：2016–2023 年在 Kelmarsh 坐标、发布时刻之前就能拿到的历史数值预报（如 NCAR RDA ds084.1 的 GFS 0.25°、NOAA NCEI 存档；ERA5 再分析不能当预报用）是否可得；有哪些变量（10/100 m 风、阵风、气压、温度）；预报时效 0–12 h、起报频次与发布延迟；许可；下载量与耗时；服务器工具（cfgrib/eccodes）。
5. 结论写进提交正文和下面的「Codex → Claude」。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-27，v29完成并复核；不追加训练 -->

**已完成审查**
- 实现/审核 `fcd1185`、部署记录 `90b2ed5` 已推送并服务器同步；98项服务器测试/Bash语法通过，无跳过。
- v29：24h→4h，train2016–2020/val2021/test2022，2023–24封存；12fits/240epoch；Wiki关闭。
- last10/state 参数4,262/4,906，Guard阈值严格<41kW；审查修正和NWP可行性/来源见 `fcd1185` 正文。
- 协议SHA `b78dd473f6261b4d11de5a77f7bd508424d703f3286952687605f8fabdd50eb7`；源码指纹 `d6893b7306ea964d8b1dd4584b570b4a3f977634e30af631f1105ff49d0cb510`。

**本次结果（2022 ALL主队列，指标为3种子指标均值，非集成）**
- 10:58 CST训练/报告均完成，退出0；12fits、214epoch（预算12/240），没有追加或重训；运行提交 `28455ab`。
- 服务器目录 `outputs/logs/Kelmarsh/20261027/operating_state_v29_vEjzoZ`，指针 `outputs/logs/Kelmarsh/operating_state_v29_latest.txt`。
- 完成清单SHA `da36df218ac10fa616e949baf6740d04040a10abec90c8de3c144b47ce2197a6`；122项服务器哈希全部一致。
- 本地 `output/kelmarsh_hourly/operating_state_v29_review/results/` 已下载非权重报告/预测/图；74项哈希一致，另外48项权重/验证预测仅在服务器验过。
- 本地 `output/kelmarsh_hourly/operating_state_v29_review/audit_results.py` 独立复算456个总体/逐小时/Skill/命中率值，最大差8.45e-10；24个配对中心差最大6.04e-14；Guard逐元素一致，test窗口实例化晚于selection freeze。
- 服务器只读重算12项对比的bootstrap/Newey–West/Holm及RMSE区间，共96值最大差0，四条规则全部一致；未重跑模型推理或训练。
- 2022 ALL=16,949窗口，normal=15,382（事后条件队列）；2023–24封存。原数据审计/身份见 `d25cac3` 正文。
- 验证预选 `Guard_last10_normal`：MAE180.403064、MSE77724.716749、RMSE278.791213、R²0.768418；对Last10持续法MAE/RMSE Skill=8.118764%/10.067549%，±5/10%容量命中48.123291%/69.054221%。
- 两种持续法MAE/RMSE/R²：小时193.872724/301.865838/0.728498；Last10 196.343749/310.000683/0.713668。
- last10_normal=183.972701/277.047429/0.771306；last10_all=180.729162/276.301755/0.772536；state_normal=190.134702/281.633472/0.763663；state_all=180.826624/276.873790/0.771592（MAE/RMSE/R²）。
- Guard逐h1–4 MAE=113.9812/172.3912/205.3646/229.8753；R²=.904959/.795178/.717484/.656244；仅MAE更好不代表所有指标最好。
- C5 Guard vs Last10持续法：MAE收益15.940685kW、95%CI[13.229697,18.506462]、Holm p=1.74e-30；对小时持续法MAE/RMSE下降6.9477%/7.6440%。
- C1 state_all vs last10_all：MAE收益-0.097462kW、CI[-0.505492,0.375810]、Holm p=.66977；新增状态信息不获支持。
- C3 all vs normal训练（state）：MAE收益9.308078、CI[7.158373,11.730700]、Holm p=6.78e-15；last10对应C4收益3.243540、CI[2.280762,4.258053]。
- 冻结结论：state_information_supported=false；all_window_training_supported=true；learned_state_beats_rule=false；beats_naive=true。
- 预选Guard三种子预测平均仅描述性：MAE179.682722/RMSE277.869914/R².769947；不能混同上述种子指标均值或事后更换主模型。
- 【最终评级】B（本轮结果可复现，但不支持新增状态机制/Wiki贡献）；【必须立即修复】无新实证P0/P1；【需重跑/补测试】无，不改冻结代码。已实看首72h的h1/h4预测图，远期峰谷误差仍明显。
- 请Claude读取results/report.md、test_metrics.csv、test_comparisons.csv、regime_metrics.csv及提交正文；2021曾用于开发，不能把v28 R²≈.81与本轮2022直接比较。
- 下一步仅设计评估：沿预选方案研究发布时刻可用NWP；不得对已看2022追加阈值/seed搜索，2023需独立新协议授权，2024继续封存。
