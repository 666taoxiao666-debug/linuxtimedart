# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-12 第 74 次（G1b R2：R1-B1 观测校准的退化处置写成代码和测试；S1、S2 一并处理） -->
轮次: R2/2 | 阶段: G1b | 分支: exp/G1 | 基线提交: 09c745b
请求: 审改动（只审 R1-B1、S1、S2 相关修改），在服务器跑测试；不训练，不写数据脚本
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md models/NWPDisplacement.py tests/test_nwp_displacement.py
允许改动: models/NWPDisplacement.py, tests/test_nwp_displacement.py, output/external_acquisition/gfs_g0/**, output/gfs_g0/**
审查文件: docs/review/G1b-R2-codex.md
本轮决议: 采纳 3（R1-B1、R1-S1、R1-S2）/ 反驳 0 / 暂缓 0 / 转实验 0

需要你做的:
1. **只审这三处改动**：
   - R1-B1：fit_affine_calibration 和 apply_calibration（纯 numpy），规则见 STAGE_PLAN C 节"观测校准"。
     - 只用拟合段的有效配对；斜率改用简化主轴 b = sd(o)/sd(g)，对称、不会为 0。
     - 检查 n ≥720、两个标准差 ≥0.5 m/s、r ≥0.5、b ∈ [1/3, 3]。
     - 不过就退回只扣均值（b = 1）；配对 <720 就把该站整体置掩码 0；各臂相同，诊断量写进审计。
   - R1-S1：优化轮 M_self 同步扩到 5 h，不加气压，优化轮的 Q4 只作描述。
   - R1-S2：validate_batch 要求 oF 和 oF_mask 成对出现；arm = "M_self" 时必须有。
   - 这是 G1b 第 2 轮，也是最后一轮；仍有分歧就写明交用户。
2. **测试：** 新增 2 项，你的反例 b = 0 也在里面。服务器跑 `python -m unittest tests.test_nwp_displacement`，共 23 项（本地 20 项通过，3 项 torch 跳过）。
3. 通过后 G1 方案和模块定稿。B1 下完由用户转达后做 B1 正式核验，然后才写 G1 的数据和训练脚本：校准调用这两个函数，配对规则按 STAGE_PLAN。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-12：第74次，G1b R2通过，23项服务器测试通过 -->
轮次: R2/2 | 阶段: G1b | 分支: exp/G1 | 提交号: 本轮见git log；基线09c745b
审查结论: 通过，0条未解决阻断；R1-B1、R1-S1、R1-S2关闭，详见docs/review/G1b-R2-codex.md。仅复审三处改动，不进入第三轮。
已直接修复: R2-B1（本轮提交）：apply_calibration原来接受b=inf或4，现按原定[1/3,3]检查有限性和上下界；原测试内补0/NaN/inf/4拒绝反例，未改阈值。
运行信息: /tmp/g1b_review_74；OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/XT/miniconda3/envs/timedart/bin/python -m unittest -v tests.test_nwp_displacement；CPU合成测试。
关键数字: 服务器23/23通过，0.088秒，无跳过，含3项torch；M_self一致性/梯度/布局与退化校准均通过。
关键数字: 校准仅拟合段有效配对；n≥720、sd≥0.5、r≥0.5、RMA b∈[1/3,3]；失败按决议offset或masked，原零斜率反例已闭合。
关键数字: 优化轮M_self同步5小时但不加气压，Q4只描述；oF/oF_mask成对且M_self必需。
运行信息: 模型SHA256=3ff8536ec9a1d04149f00d856806d8ce2c3813e420342262b81baaf50fa65146；测试SHA256=71fb17cd83b5abc54302d5ef06125b8eb44c40714823b30506987508a1a341df；本地/服务器一致。
运行信息: 未训练、未写数据脚本、不读真实或封存数据、不改原部署；B1正式验收尚未执行。
需要 Claude 做的: 接收G1b R2通过与定稿记录，事后复核R2-B1最小修复；不再开启方案第三轮。
需要 Claude 做的: B1下完经用户转达后先正式验收，再写G1数据和训练脚本；数据阶段核验校准配对日期/周期/掩码，合成测试不代替时间边界审计。
需要用户决定: 无。
