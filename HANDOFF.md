# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-12 第 71 次（G1 模块定稿；GFS B2 已下完，交你做 G0-c 只读核验） -->
轮次: R1/2 | 阶段: G0c | 分支: exp/G1 | 基线提交: 4973a72
请求: 审代码后运行（你写只读核验脚本并在服务器运行；不训练）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/audit_gfs_g0.py, tests/test_audit_gfs_g0.py, output/gfs_g0/**, output/external_acquisition/gfs_g0/**, scripts/acquire_gfs_g0.py
审查文件: docs/review/G0c-R1-codex.md
本轮决议: 无新编号。G1 R2 复核通过、模块定稿；G0c 方向说明见 DECISIONS「阶段 G0c」

需要你做的:
1. **提交** STAGE_PLAN、DECISIONS，以及我从原目录复制过来的 B2 清单（output/external_acquisition/gfs_g0/ 下的 manifest_b2.json、b2_annual_summary.json、progress_b2.json，只有元数据）。
2. **写 scripts/audit_gfs_g0.py 和测试，只读核验 B2**（规格：G0-R2-codex.md，加上 STAGE_PLAN B 节"B2 的核验内容"）。
   - 数据在服务器 datasets/external/gfs_g0/dev/2016…2020/gfs_YYYYMMDDHH_cropped.nc。
   - 要查：逐文件 SHA256 与清单一致；变量、步长 f003–f018、有效时刻 = C + f、43 × 43 网格（0° 经线不重复）；数值有限且在物理范围内；独立重算每年完整率（分母 = 天数 × 4）；列出不完整和缺失的周期。
   - 要确认缺失的那个周期到底是 2018-05-26 18Z 还是 2018-10-30 06Z。
   - 测试用合成 NetCDF：缺步长、错网格、重复 0° 经线、出现 f000、数值越界，都要能报出来。
   - 每次编辑 ≤150 行。用 gfs_g0 环境解码；只读，不改数据。
3. **交回**：每年完整率、不完整和缺失周期清单、网格与变量的核对结果、发现的问题。B1（AWS 2021–2026）到了，用户会另外转达，同一脚本再核一次；有效时刻 ≥ 2024 的封存部分只核字节和 SHA256，不解码。
- 不写 G1 数据和训练脚本，等 B1 也核验通过后再写。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-11：第70次，G1 R2复审通过，18项服务器测试通过 -->
轮次: R2/2 | 阶段: G1 | 分支: exp/G1 | 提交号: 本轮见git log；基线471bf33
审查结论: 通过，0条阻断；R1-B2关闭，详见docs/review/G1-R2-codex.md。仅复审本轮缺测插值改动和新增测试，模块定稿，不进入第三轮。
已直接修复: 无；本轮模型/测试修改均由Claude完成，Codex只写复审和交接；R1-B1最小修复已获Claude复核。
运行信息: 服务器/tmp/g1_review_69；timedart/bin/python -m unittest -v tests.test_nwp_displacement，CPU合成测试，不训练。
关键数字: 服务器18/18通过，无跳过，0.165秒；本地15通过3缺torch跳过，0.221秒。
关键数字: 新增三例覆盖缺一个插值端点、全风场缺测且观测加10、缺测点不在读取范围；torch parity含回退样本，绝对1e-8及梯度/功率通道检查均通过。
运行信息: 模型SHA256=b1bacfd801db1e61c54809cf48e519b14506cf21322ed90426e10b3e18c21d4a；测试SHA256=8613d875d392c6385a0165ea74276be3d797f42063e7b1870f6461020e1df53c；本地与服务器一致。
运行信息: 四步插值有效权重均满足阈值才修正，否则回退原预报乘掩码并清零错位/偏差/置信度；不增通道、不扩大五臂信息范围。
运行信息: 未写数据脚本、未训练、未下载GFS、未读Token、未触封存年；仅隔离测试目录，未改既有实验部署。
需要 Claude 做的: 接收G1模块R2通过记录；等待用户转达GFS B1/B2到位，再推进G0-c及数据/训练脚本审查，R1-S2在该阶段验证。
需要用户决定: 无。
