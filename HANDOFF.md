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
<!-- 由 Codex 覆盖。最后更新：2026-10-27 10:17 CST，v29第二次下载断流已安全续传 -->

**已完成审查**
- 实现/审核 `fcd1185`、部署记录 `90b2ed5` 已推送并服务器同步；98项服务器测试/Bash语法通过，无跳过。
- v29：24h→4h，train2016–2020/val2021/test2022，2023–24封存；12fits/240epoch；Wiki关闭。
- last10/state 参数4,262/4,906，Guard阈值严格<41kW；审查修正和NWP可行性/来源见 `fcd1185` 正文。
- 协议SHA `b78dd473f6261b4d11de5a77f7bd508424d703f3286952687605f8fabdd50eb7`；源码指纹 `d6893b7306ea964d8b1dd4584b570b4a3f977634e30af631f1105ff49d0cb510`。

**本次真实状态及处理**
- 前两次续接PID953199/971835已退出；最新于2026-10-27 10:07:17 CST退出，2022包网络断流 `ChunkedEncodingError/IncompleteRead`，不是训练报错。
- 两份历史日志 `outputs/audits/operating_state_v29/launch.log` 和 `resume_20261027_092202.log` 保留；没有preparation.json、v29训练指针、真实fit或新精度。
- 2022包总485,620,228字节，最新分片464,321,794字节（95.61%），剩21,298,434字节；全部保留，不重下已有分片。
- 已复核协议/源码/续接脚本哈希和真实Python参数，确认无同类进程、无训练指针后安全续传。
- 新续接PID `973622`；新日志 `outputs/audits/operating_state_v29/resume_20261027_101658.log`；旧失败日志保留。
- 复用原脚本 `outputs/audits/operating_state_v29/prepare_and_launch_once.sh`，SHA `830f472d5298bd7b91c492d4aa0bc3e841a7b0cb85ce00427ebeff589c9799fa`；没有改代码、协议或哈希约束。
- 唯一流程仍是prepare→独立标签审计→源指纹复核→preflight→一次detach训练；任一步失败停止，不改阈值、不追加搜索。
- 数据目录 `datasets/wind_public_v29/resampled_hourly_scada/kelmarsh_multiyear/`；raw2022官方MD5 `b695d2b21cd5a236b2e690e0a43df5ab`。
- 审计目标 `outputs/audits/operating_state_v29/independent_audit.json`；训练指针 `outputs/logs/Kelmarsh/operating_state_v29_latest.txt`；未出现前不能宣称开训。
- 查看新日志/真实进程和启动器 `--status`；不得因旧launch.log的历史报错再启动副本，不影响其他用户任务。
- 请Claude暂不改冻结源码；结果到齐后再核对2022双队列/逐小时/统计并记录，不能拿v28结果冒充v29。
