# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-07 第 45 次（L0-B 清点复核 + R1-B2 数据规则，只读重跑 r2） -->
轮次: R2/2 | 阶段: L0 | 分支: exp/L0-B | 基线提交: 8885e9c
请求: 审代码后运行（只实现 R1-B2 的规则，只读重跑一次）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/audit_hill_of_towie_e1.py, tests/test_audit_hill_of_towie_e1.py, output/hill_of_towie_e1/**, output/external_acquisition/hill_of_towie/**, scripts/acquire_hill_of_towie.py, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R2-codex.md
本轮决议: 采纳 1（R1-B2：定新数据规则，不排除站 137）/ 反驳 0 / 暂缓 0 / 转实验 0；r1 的门槛、只读模式和测试年登记已复核通过

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。
2. **实现 R1-B2 规则**（只在 audit_hill_of_towie_e1.py 的只读模式里包一层，不改冻结的 screen/utils 加载器）：
   - 同一站、同一时刻、version_num=1 的多行，取值有冲突时，该时刻的风速、风向、气压都记为缺测；取值一致的重复行保留一行。
   - 任一站-年的冲突时刻超过 1% 就报错停下。
   - 测试用合成数据覆盖：冲突时刻整小时变缺测；一致的重复行保留一行；超过 1% 时报错；无冲突的输入，结果与原加载器逐元素相同。
3. **只读重跑**到新目录 l0b_r2，原 l0b_r1 保留不动；交回：
   - 站 137 的逐年完整率、是否合格、有无气压；
   - 各站的冲突时刻数；
   - 最终门槛结果，以及冻结的合格站列表（附 SHA256）。

需要用户决定: 无

说明:
- 为什么不排除站 137：它是离风场最近的站（27.4 km，方位 326°），只因 2 个冲突时刻就排除整站，代价太大，而且会改变以后的特征；记为缺测是最小、也不篡改数据含义的处理。
- 已有结果不受影响：Kelmarsh、Penmanshiel 的加载器遇到冲突会报错，而它们都跑完了，说明那里没有冲突。
- 不训练、不评估、不解封测试年，不开新的方法实验。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-07 第45次：R1-B2已部署，r2唯一实例已启动，结果查询暂受SSH阻塞 -->
轮次: R2/2 | 阶段: L0-B | 分支: exp/L0-B | 提交号: 本轮见git log
审查结论: 通过；R1-B2按第45次明确批准实施，无新增方案阻断，见docs/review/L0-R2-codex.md。
已直接修复: 本轮为获批实现；只读scoped reader对同站同小时version_num=1冲突将风速/风向/气压全记缺测；一致重复由原加载器折叠。
运行信息: 任一站年冲突小时/完整公历年小时数>1%报错；平年8760、闰年8784，与完整率分母一致；原始文件及冻结screen/utils不改。
运行信息: 本地13单测通过，包含4项指定测试（冲突整小时缺测、一致重复保留一行、1%边界与超限报错、干净输入逐元素同原加载器）。
运行信息: 2026-11-07 13:34:07北京时间已启动唯一r2，PID508698；脱离SSH、.launch_once防重复标记已创建；本轮一次只读预算已使用，不重复启动。
运行信息: 新脚本/测试已传输且服务器SHA256与下列值一致，服务器13测试通过；后续多次只读状态查询SSH超时，完成/失败暂未确认。
关键数字: 脚本SHA256=ec2eb4ba69a6bb1f6dc1359c1aef44391b03869cc39a721bea8be4b08cf77e4f。
关键数字: 测试SHA256=31976ba0b7c97a839c7227697d644deaf2220b99947a3201a0e23b237fe54840。
运行信息: r2计划新目录outputs/audits/hill_of_towie_l0b_readonly_r2，本地output/hill_of_towie_e1/l0b_r2；将输出各站年冲突数、137完整率/资格/气压及qualified_stations.json SHA256。
需要 Claude 做的: 可复核本轮规则实现；r2结果尚未取回，不把r1的11站结果当作最终冻结列表。
需要用户决定: 无；连接恢复后仅查PID/日志/结果并取回，不重新运行。启动回执见output/hill_of_towie_e1/l0b_r2/launch_receipt.json。
边界: l0b_r1保留不动；不训练、不评估、不解封2024–2026，不修改下载逻辑；未合主线，main未动。
