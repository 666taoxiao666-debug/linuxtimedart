# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-20 第 104 次（X4 方案 R2：修 R1-B1，逐条决议 S1–S4、P1–P3，新增 D0 诊断规则） -->
轮次: R2/2 | 阶段: X4 | 分支: exp/X1 | 基线提交: 0795f39
请求: 方案复审，只审改动；通过后提交推送（连同你的 X4-R1 审查、X4_innovation_review.md）
待审范围: git diff 0795f39 -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: docs/STAGE_PLAN.md, docs/review/DECISIONS.md, docs/review/X4-R1-codex.md, docs/review/X4-R2-codex.md, output/thesis_materials/**, output/x3/**（*.npy 不要提交）, output/x2/**（x2_0f24875_json.tar.gz 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）
审查文件: docs/review/X4-R2-codex.md
本轮决议: DECISIONS「阶段 X4」第103次决议表 + D0 规则

需要你做的:
1. **B1：**
   - 第 1 段的源候选池和冻结 L_1X 完全相同，按 L_1X 早停切点截止，代码重新生成后核对 q4_meta 的窗口数；
   - C_T3 沿用 k = 1 的第 1 段模型，源截止比 L_3X 晚约 2 周，并和目标早停期重叠，所以只作描述并披露。
   请核这个口径是否闭合。
2. **S4 / P3：** 新增描述臂 C_D1（不经源风场，直接目标适配，同预算），写法改成"所检验的两段适配流程"。
3. **P1：** 采纳为后续方法线，但先做 D0（CPU、LightGBM 代理，≤ 30 分钟）。请重点审 D0：
   - 时间块、源截止、事件分组只用起报时可知的量；
   - U(g, e) 的定义和随机等量对照；
   - 两条继续条件，以及 LightGBM 代理的局限写法。
4. **P2 暂缓：** 理由是本项目在这块已有三次负结果，而且可识别性还没核实；等 D0 有结论再说。是否同意？
5. 通过后提交推送；之后我写 run_x4.py（A、B、D0）交代码审查。本轮不授权运行。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-20，第104次X4方案R2后用户裁决落实 -->
轮次: R2/2 | 阶段: X4 | 分支: exp/X1 | 提交号: 见本节所在提交（基线0795f39）；方案记录，不是Cursor执行版本。
审查结论: 通过；R1-B1已关闭，R2-B1/B2经用户裁决修订后关闭；docs/review/X4-R2-codex.md，不进入R3方案审查。
已直接修复: 用户明确批准后，由Codex同步STAGE_PLAN和DECISIONS的D0分组冻结、标签隔离；本节所在提交记录，未修改实验代码或冻结文件。
运行信息: collab_check.py --codex退出0，8项全部通过、无警告；git diff --check通过；15个协议哈希及9个运行审计一致；未上服务器、未运行实验。
关键数字: 每个伪目标的6组中心及缩放只在首块合规源池拟合一次；后续块冻结；目标训练和源窗口全部h1–h4标签严格早于固定评估切点。
用户裁决: “可以继续。多条线一起”；A、B、D0一起编写和交代码审查，不再等待第3轮方案审查。
调度: D0（CPU，≤30分钟）与GPU链并行；A、B在同一GPU队列顺序运行，保留gpu_fraction=0.7、失败停下不重试。
材料: 本轮提交两轮X4审查、X4_innovation_review.md、STAGE_PLAN、DECISIONS及HANDOFF，共6个文档；排除数组、归档包、大清单和其他材料改动。
需要 Claude 做的: 1. 按批准方案一起写run_x4.py、configs/x4_protocol.json、tests/test_x4.py及launch/status的x4链，交代码审查。
需要 Claude 做的: 2. 明确R2-S1/S2/S3：代理负结果边界、C_D1学习率及目标预算口径、空组/常数相关/超时停下规则；不扩实验臂和预算。
需要 Claude 做的: 3. 测试覆盖首块冻结与标签隔离，保留X3复用身份、源/目标样本键、输入掩码、CPU/GPU队列依赖及失败停止审计。
需要用户决定: 无；已按本聊天裁决落实，本轮无需再次批准同一修法。
