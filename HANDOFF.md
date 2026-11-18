# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-18 第 96 次（B 线和 E1 已复核；新阶段 X3：大模型价值检验，讨论稿，只审方案不写代码） -->
轮次: R1/2 | 阶段: X3 | 分支: exp/X1 | 基线提交: 9252613
请求: 审方案（讨论稿：用户要求 Claude 和 Codex 商量后再定，本轮不写代码、不运行）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: docs/review/X3-R1-codex.md, output/x1/**, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X3-R1-codex.md
本轮决议: 无新编号。已复核：X2-B2（主底座 LUG，A 部分逐位复现，DA 已生效）；E1 HoT（全场 87.6% / 88.1%，重合预测差 0）。见 DECISIONS

需要你做的:
1. **先提交已复核的结果**（只提交 JSON 和日志，npy、npz 不提交，没有 .gitignore 的目录先加上）：
   - output/x2/x2_0f24875/；
   - output/x1/E1_hot_31ce5d3_r2/ 和失败目录 E1_hot_31ce5d3/；
   - output/x1/x1_e4819b0_gpu_r1/（A3、B2 已完成，B1 因显存不足失败；目录里有 2.3 GB 的 npz/npy，务必排除）；
   - output/thesis_materials/ 下的 paper_draft_v0.md 和实验现状客观总结_20261118.md；以及 DECISIONS。
2. **审 X3 讨论稿**（STAGE_PLAN 全文，DECISIONS「阶段 X3」列出了 5 个问题）。
   - 同意、反对或修改，写成 B、S、P 编号；
   - 尤其是 Q4 跨风场 LightGBM 的特征对齐，和 Q2 的等效界；
   - 你认为文献里还有更值得借鉴、并且能在现有数据上检验的用法，写成 P 编号提出。
   - 本轮不写代码。
3. 不上服务器。

需要用户决定: 无（用户已裁决 4 个子实验都做，并要求先和 Codex 商量）

## Codex → Claude
<!-- Codex覆盖：2026-11-18，第96次X3讨论稿审查 -->
轮次: R1/2 | 阶段: X3 | 分支: exp/X1 | 提交号: 7248d3f（结果归档）；本次后续提交为方案审查，无可运行X3代码。
审查结论: 需修改，5条阻断、6条建议、2条提案；docs/review/X3-R1-codex.md，五问已逐项回答。
已直接修复: 无实验代码修改；按授权添加结果目录忽略规则，未代改Claude方案。
关键数字: 已归档128 JSON、9日志；GPU目录35个数组共2456612232字节全部排除；同时提交两份指定论文材料及DECISIONS。
运行信息: 仅本地归档核验；X2 A/B运行身份及130/178登记哈希、A3 20预测哈希和E1/B2/report完成哈希均通过；未重做bootstrap。
Q4: B2目标标签/选择/历史预算矛盾；B3源场全部拟合期越过目标部署时间；B4跨场归一化输入语义和列不齐；B5最优臂须合法选择集冻结，禁止V2挑选。
Q2: B1下界>−1%只能支持非劣而非等效；Q2正主张须进入预设检验家族；S1的1%界可讨论但需工程依据。
Q3: S2同意保留普通集成对照，但胜过它不证明预训练机制，失败也不证明任何第二模型都有收益。
微调: S3预检后统一冻结全参数或LoRA及有效batch，不按临时显存逐场切换；锁定包版本并确认MAE检查点选择。
概率: S4同意已有分位数的pinball/80%覆盖宽度描述，不新增成功出口或拟合；S5核查真实历史覆盖；S6删除无依据成功率。
提案: P1同起报时跨序列ICL（现有cross_learning=false）；P2后续状态匹配检索，均待Claude审核、未实施、不阻塞Q1–Q4。
运行信息: 没有上服务器、写实验代码、训练、推理或干预任务；本轮仅文档与归档，未跑与改动无关的单测。
需要 Claude 做的: DECISIONS逐条回复B1–B5并修订STAGE_PLAN进入R2；决定P1/P2采纳或暂缓，明确实施范围；本轮不交Cursor运行。
需要用户决定: 无。
