# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-25 第 114 次（三场期刊对照正式结果复核：数字通过，写法需修改） -->
轮次: R2/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 基线提交: bdb781e（运行提交 1fe7ab6）
请求: 复核结果（按 W1–W4 改三份交付稿后交回；不新增拟合或预测）
待审范围: git diff bdb781e -- docs/review/DECISIONS.md HANDOFF.md；新文件 output/wjb/wjb_1fe7ab6/claude_review/（复算脚本和输出）
允许改动: output/thesis_materials/**（本轮只改 wjb_abstract_conclusion.md、wjb_claim_revision.md，可选 wjb_development_comparisons.md、wind_forecast_score_inventory.md；paper_draft_v0.md 等其余文件保持现状，用户确认前不改不提交）, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, docs/review/wind-journal-baselines-result-R2-codex.md, docs/review/DECISIONS.md, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/wind-journal-baselines-result-R2-codex.md
本轮决议: 数字通过（48 个读数点估计完全一致，区间符号判读全部相同）；交付稿需修改 4 条（W1–W4），建议 1 条（W5）；更正我登记的一处（G2a 属测试年再分析，不是开发期）。见 DECISIONS「阶段 wind-journal-baselines-result」末尾

三个重点的结论（细节在 DECISIONS）:
- Kelmarsh Cabin LU：按规则仍是"未检出差异"，但属于边界情形：换 50 个随机种子，区间上限 46% 落在 0 以下；四个时距都是 LightGBM 更准。差距缩小是因为 Cabin 在 K 从上游多得 4.87 [2.42, 7.70] kW（事后描述，只在 K 成立）。
- HoT TSG 的 U 由负转正：LUG 臂相对 LG 偏高 +34 到 +50 kW，21/21 台机组如此，V1 上已更差（221.08 对 208.50）。增益变化 23.55 是差中差，和 LightGBM 的 23.12 接近，按登记规则照读；绝对 U 不作解读。
- 摘要：X5b 那句要写明模型（LightGBM 单种子、开发段事后分析）；期刊对照那句要补"同输入下 30 个配对中 29 个 LightGBM 更准"。
另发现: TSG 的 LR 底模型在 K、P 数值失败（V1 MAE 836–1120 kW，预测多被截断在 0 或额定），没被选中，结果不变。
需要你做的:
1. 按 W1–W4 改 wjb_abstract_conclusion.md 和 wjb_claim_revision.md；W5 自行决定是否吸收（comparisons 表注、成绩清单表头）。
2. 请反重力只读查 TSG 的 LR 候选在 K、P 的 author_stdout（jobs/<场>/<臂>/cand0、cand1 的上下界损失），在 R2 审查文件里记一句成因；不重跑，不改网格。
3. 提交 claude_review/ 下的脚本和输出（不含数组），再交回；我核对改稿后，交用户决定是否写入 paper_draft_v0.md。
需要用户决定: 1. 改稿通过后，是否把主张修订稿写入论文初稿 paper_draft_v0.md。

## Codex → Claude
<!-- 第 114 次交接，写法修订及线性候选诊断 -->
轮次: R2/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 提交号: 基线 bdb781e，正式运行 1fe7ab6，本轮改稿与交接同一提交。
审查结论: W1–W4 已落实，采纳 W5；见 docs/review/wind-journal-baselines-result-R2-codex.md。
已直接修复: 本轮仅修订摘要、主张稿、开发期表注和成绩清单说明。
运行信息: 本地核对 Claude 的 48 项复算和 TSG 冻结记录。反重力窗口操作未获准，已交用户代转只读诊断，服务器 author_stdout 回执待收。
关键数字: 29/30 个配对中 LightGBM 更准，Kelmarsh Cabin LU 为登记规则下的边界情形。HoT TSG 与其配对 LightGBM 的上游增益差中差为 23.55 / 23.12 kW。
关键数字: K/P 的线性树在根节点已出现稳健验证损失异常，32/32 棵上界超过 2.6×10⁵；本地证据尚不能识别触发异常的具体数值机制。
需要 Claude 做的: 核对独立改稿。服务器原始日志回执收到后补核原因。
需要用户决定: 改稿经 Claude 核对后，是否写入 paper_draft_v0.md。
