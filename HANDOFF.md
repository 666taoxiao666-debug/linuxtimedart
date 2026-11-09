# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 52 次（G0 取消收尾 + 新阶段 M1：C0b 区间优化一轮 / L1 本地大模型编码器搜索） -->
轮次: R1/2 | 阶段: M1 | 分支: exp/M1 | 基线提交: cb090f1
请求: 审方案后实现并运行（无阻断就直接跑，提速规则适用；L1 要等 Cursor 交回模型权重）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_c0b.py, tests/test_pilot_conformal_c0b.py, output/conformal_c0b/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M1-R1-codex.md
本轮决议: G0 已由用户取消（DECISIONS 的 G0 末节），第 50 次的核验请求作废；M1 是新阶段，暂无编号

需要你做的:
1. **提交与合并**：
   - 先在 exp/G0 上提交本轮的 STAGE_PLAN 和 DECISIONS；
   - 用户同意后，把 exp/G0 用 --no-ff 合回主线（只有文档，作为存档）；
   - 再从主线拉 exp/M1。不写 GFS 核验，不下载、不安装。
2. **审 M1 方案**，有阻断按 v2 写审查文件；没有阻断就实现并各运行一次：
   - C0b：CPU，导入冻结的 C0 代码只加变体。严格因果切分，校准标签 ≤ 2022-12-31 23:00。在 2022 上选、在 2023 上评估，C1–C3 按事先设定判定。
   - L1：GPU。等 Cursor 交回权重的 revision 哈希和推理环境后才启动。程序要过 AST 白名单和超时；输入截断到起报 − 1 h；全部提示词、输出和程序都存档；搜索结果冻结后才做正式训练和评估。
3. **交回**：
   - C0b：所选变体、2023 年各方法的覆盖率、宽度、Winkler，以及 C1–C3 判定；
   - L1：各组的搜索最优值，正式阶段的 MAE/RMSE 和 CI，判定结果；
   - 还要交回前 10 名程序的文本，供我人工抽查因果性。

需要用户决定: 是否把 exp/G0 合回主线（只有文档）

说明:
- 用户 2026-11-09 决定：C0 做一轮优化；大模型编码器并行做，用本地开源模型；不下载 GFS。
- 关键是 C3（情形加权比同样修过的 B3 好 ≥3%）和 L1 对 R* 的比较；只赢弱对照，不算方法创新。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-08 第49次：G0 R2通过 -->
轮次: R2/2 | 阶段: G0 | 分支: exp/G0 | 提交号: 本轮见git log
审查结论: 通过，0条阻断，R1-B1关闭；仅复审第49次增量及上一轮意见，见docs/review/G0-R2-codex.md。
已直接修复: 无；Claude方案与决议原样提交，未修改数据规则。
运行信息: 仅本地方案边界算术核对，未下载、安装、实现核验脚本、训练或解码任何数据。
关键数字: 完整周期6档×3必需字段=18条消息；f000排除；100m风可选不计门槛。
关键数字: 年分母1460/1464固定，95%至少1387/1391完整周期；缺周期留分母。
关键数字: 正式延迟6h对应目标时效7–15h；4/5/9h敏感性预留范围也被f003–f018覆盖，不授权本轮运行敏感性分析。
审查结论: C+f封存、跨年边界、两段经度、缺测不回退、H0措辞修改均通过；6h仍是可用性假设，实际历史证据由探查登记。
运行信息: 放行G0-b正式下载清单规则；等Cursor的probe_report.json和manifest.json到位后，再按批准范围写只读核验及合成测试。
需要 Claude 做的: 复核R2通过结论并协调Cursor按六档清单推进；下载预算报告区分开发/封存存储；G1同信息同预算及主张边界留待正式协议。
需要用户决定: 当前无；下载量或存储超已授权预算时仍交用户。本轮不合G0主线，main未动。
