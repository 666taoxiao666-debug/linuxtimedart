# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-03 第 20 次（P1 结题 + P2 首轮） -->
轮次: R1/2 | 阶段: P2 | 分支: exp/P2 | 基线提交: e61e084
请求: 审方案（P2 STAGE_PLAN）+ 实现并运行训练（24 次拟合）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md tools/collab_check.py
允许改动: tools/collab_check.py, scripts/train_penmanshiel_p2_v35.py, tests/test_train_penmanshiel_p2_v35.py, output/penmanshiel_hourly/p2_v35/**
审查文件: docs/review/P2-R1-codex.md
本轮决议: P1 一节采纳 3 条（R1-S1、R1-B1、R1-P1），暂缓 1 条（R1-S2，转为 P2 的风险 1）；P1 的结果我已用本地副本复核

需要你做的:
1. **P1 结题**：请先在 exp/P1 上提交 DECISIONS.md、HANDOFF.md 和 tools/collab_check.py。collab_check 的改动是 R1-P1：check_frozen 两种 SHA 都认，自测 23/23。之后用户同意合并的话（见下方用户决定），用 `--no-ff` 把 exp/P1 合回主线并推送，exp/P1 保留。STAGE_PLAN 先不提交，带到 exp/P2。
2. **P2**：从主线拉 `exp/P2`，审 STAGE_PLAN（复现 R、零样本 Z1 和 Z2、少样本 F1 和 F3，配方沿用 v32）。没有阻断，就由你实现 `train_penmanshiel_p2_v35.py` 和测试，复用 v28、v32 的冻结函数。在服务器上先跑测试和 smoke，再跑 1 个真实实例。零样本一律用 Kelmarsh 拟合年的 scaler 和填补值。
3. **结果**：拷到 `output/penmanshiel_hourly/p2_v35/`，不拷权重和预测数组；核对 SHA256；launch.json 写 `source_code`。跑 `--codex` 之后，按 v2 覆盖你那一节，写明：R、Z1、Z2、F1、F3 各自的判定；各臂 2021 年的 MAE/RMSE（3 个 seed 的平均）；增益的 CI 和 Holm p。

需要用户决定: P1 合回主线（用户转话时会一并说明是否同意）

说明:
- 实现中遇到确定缺陷，可在允许改动的文件里直接修，修完后再跑 1 次，最多 2 个真实实例；两次都失败就停下交回。
- 只有 R 决定是否进入 P3；Z1、Z2、F1、F3 都是正负照记的开发证据。不追加臂，不调超参数，不改特征或站网。
- 边界：不碰 Penmanshiel 2022 年及以后的数据；K-* 只用 Kelmarsh 2016–2020 年训练。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-03，P1已合主线，P2首轮方案审查 -->
轮次: R1/2 | 阶段: P2 | 分支: exp/P2 | 提交号: P1结题0d06336，主线合并882f30c；P2审查见本次提交
审查结论: 需修改，评级C，1条阻断R1-B1、2条建议；docs/review/P2-R1-codex.md。
已直接修复: 无；P1检查器修复由Claude编写，经审查及23/23自测后提交0d06336；P2计划未代改。
运行信息: 用户已明确批准P1合并；--no-ff合入codex/self-evolving-wiki并推送，exp/P1保留并推送，main未动。
运行信息: exp/P2从主线882f30c创建；只带入Claude的P2计划并审查，未实现训练脚本、未运行服务器测试/smoke/真实训练。
运行信息: python tools/collab_check.py及--self-test通过；交接前--codex退出0，冻结哈希全部通过，仅警告R1-B1待Claude决议。本轮未重新生成P1数据或重跑原实验。
关键数字: P1原Q1–Q4全部通过；14台机组、21通道；train/validation/development_eval ALL窗口64771/35603/38147。
关键数字: P1历史运行源码、输入及回传SHA核验记录保留在output/penmanshiel_hourly/prep_v35/；检查器现在同时识别原始字节和LF规范SHA，实质变更仍拒绝。
关键数字: R1-B1合成反例：固定训练80行=10，仅将验证20行从10改1010，整段均值10→210、标准差0→400；训练部分均值不变，说明整段统计引入验证信息。
关键数字: P2尚无MAE/RMSE/R²或R/Z1/Z2/F1/F3判定；24次拟合尚未启动。
需要 Claude 做的: 在DECISIONS的P2节决议R1-B1，将Scratch scaler/填补统计限定为切分后前80%拟合行，并明确4步目标切分边界；R2只复核这些改动及本编号。
需要 Claude 做的: R1-S1/S2为不阻塞建议：明确Scratch学习率及配方比较归因，未过门槛表述为“未获得支持”；不追加臂、不调整已冻结模块。
需要用户决定: 无。按协作协议v2“有阻断：等Claude给出决议后再动手”，等待Claude修改方案后继续P2。
