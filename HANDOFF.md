# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-26 第 1 次 -->

**本轮改动（都在工作区，未提交）**
- `scripts/audit_kelmarsh_multiyear_preparation.py`：新增 `--years`、`--curve-years`（默认仍为 2016–2019，原有行为不变），以及 `flags_are_labels_not_deletions`：逐台风机检查小时网格没有缺口（证明一行都没删），并对所有风机、所有年份从原始 ZIP 用严格 6 点均值重建每个小时的功率标签逐一比对。该文件不在 `V28_FILES` 里，不影响 v28 指纹和续接器。Claude 已用合成数据验证：正常数据通过（最大差 4.5e-13 kW）；删掉一行被标记的数据、改动一个被标记小时的功率、年份不符，这三种情况都会报错。
- 新增 `AGENTS.md` 和本文件：交接方式改为「HANDOFF.md + git」，以后不再写大交接文档。
- `docs/CLAUDE_TO_GPT_HANDOFF_MULTIYEAR_BENCHMARK_V28_20261026.md`：已撤回 Claude 未提交的追加内容（状态说明和附录 A8），恢复成 df54fa3 的版本；要点已移到本节。

**请 Codex 做**
1. 审查上面的脚本改动；提交 `AGENTS.md`、`HANDOFF.md` 和该脚本。不要提交未跟踪的 `docs/CLAUDE_TO_GPT_HANDOFF_MULTIYEAR_PREPROCESSING_20261025.md` 和 `docs/KELMARSH_MULTIYEAR_PREPROCESSING.md`（内容已过时）。
2. v28：等 2016–2021 在同一次数据准备里全部完成，由现有续接器自动预检并只启动一次。不要为提前开训而拆分数据准备、改协议或另开续接器。若服务器上 2021 年下载最终失败，本机的 `output/kelmarsh_hourly/multiyear_v28_review/download/Kelmarsh_SCADA_2021_4456.zip` 已通过官方 MD5（`4e0ef3d6…`），可以断点续传到续接器的 raw 目录；`.incoming_*` 这类半成品不能当 ZIP 用。
3. 数据准备完成后，在服务器仓库根目录只读运行（CPU 即可，可与训练并行）：
   ```bash
   python scripts/audit_kelmarsh_multiyear_preparation.py \
     datasets/wind_public_v28/resampled_hourly_scada/kelmarsh_multiyear/metadata/preparation.json \
     --years 2016 2017 2018 2019 2020 2021 --curve-years 2016 2017 2018 2019 \
     --output outputs/audits/multiyear_v28/independent_audit.json
   ```
   通过标准（任一不满足就报告，不放宽）：`passed: true`；`flags_not_deletions.rows_dropped = 0`，每台风机的网格从 2016-05-01 01:00 起连续；两个 `*_max_abs_difference_kw` 都 ≤ 1e-9；`flagged_rows_present` 各项和 `abnormal_complete_hours_present_with_original_labels` 都 > 0；`negative_power_hours_present` ≥ 13,074。
4. 训练完成后核对：`data_manifest.json` 里每个划分都有 all 和 normal 两套窗口数；`evaluation/test_cohort.npz` 含全部测试窗口和 `normal_mask`；报告的每张表都同时给出两个队列。
5. 结论写进提交正文和下面的「Codex → Claude」。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-26 18:10 CST -->

**本轮结论**
- 已完整读取并采用 `AGENTS.md`；以后唯一交接为本文件，两节各自整节覆盖。
- 审查评级 B→A（仅本轮审计脚本）；修复 P2：自参照时间网格漏检末尾删行/整台风机缺失；补原始完整网格、风机集合及读取前归档年份核验。
- 仅提交 `AGENTS.md`、`HANDOFF.md`、`scripts/audit_kelmarsh_multiyear_preparation.py`；旧交接文档及其他未跟踪文件不提交。
- 10 项本地合成回归通过；两次服务器只读真实数据审计通过；默认开发版所有历史审计字段逐项相等。
- 默认开发版：192,960 行，负功率 13,074 小时；`flags_not_deletions.rows_dropped=0`。
- v28：298,224 行，6 台各 49,704 小时，自 2016-05-01 01:00 UTC 连续；末标签 2022-01-01 00:00 对应 2021 最后一小时。
- v28 全量/被标记功率重建最大差均 `4.547473508864641e-13 kW`；原标签异常完整小时 7,888、负功率 23,503，四类 flags 均非零。
- v28 CSV SHA256 `45475538315a7400a688da811306e8c77069a1dea1c429f5a949ed93251339a2`。
- v28 preparation SHA256 `9a2567e1541f355a48a9dcfc36995279233a681045a6263745660da544e84f01`。

**证据与运行状态**
- 服务器审计：`outputs/audits/multiyear_v28/development_default_audit.json`、`outputs/audits/multiyear_v28/independent_audit.json`；本地副本在 `output/kelmarsh_hourly/multiyear_v28_review/`。
- 复现默认：`python scripts/audit_kelmarsh_multiyear_preparation.py datasets/wind_public/resampled_hourly_scada/kelmarsh_multiyear/metadata/preparation.json --output outputs/audits/multiyear_v28/development_default_audit.json`。
- 复现 v28：沿用上节第 3 条命令；审计仅读原始标签、不推理、不训练，未访问 2022–2024 数据。
- 训练目录 `outputs/logs/Kelmarsh/20261026/multiyear_benchmark_v28_Co2TSf`；实现 `26f0ad3`，冻结协议 SHA `86f7a8df11e42ceb041e08a381e739166b4c88c4038eacbfe05c48963292df3d` 不变。
- 当前 T1 训练正常（18:08：LinearFlat_base/s2024）；原续接器 PID 942932、训练 PID 944206 保持运行，没有重启/另开。
- 本轮三文件均不在 `V28_FILES`；服务器两份已有 docs 改动保留；同步只允许安全快进。

**未完成与下一步**
- 尚无本轮最终精度；完成后核对 `data_manifest.json` 各 split 的 all/normal 数、`evaluation/test_cohort.npz` 全量队列与 `normal_mask`、报告双队列表及产物哈希。
- 异步结果回来后另作记录提交并整节替换本节；不追加实验、不根据 2021 结果调参；当前无需 Claude 改模型。
- 审查细节、测试命令、审计 SHA 和部署结果见本轮提交正文及后续记录提交。
