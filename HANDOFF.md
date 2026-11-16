# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-16 第 89 次（X2 第二版：A 部分 LightGBM 消融 + ANI，B 部分加入用户两个旧模型做底座筛选；只审不跑） -->
轮次: R2/2 | 阶段: X2 | 分支: exp/X1 | 基线提交: 90029d5
请求: 审代码后运行（只限 torch 单测和小样本冒烟；X2 不训练、不评估）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md models/LegacyBases.py tests/test_legacy_bases.py
允许改动: models/LegacyBases.py, tests/test_legacy_bases.py, scripts/run_x1.py, configs/x1_protocol.json, configs/x2_protocol.json, tests/test_x1.py, output/x1/**, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交、不要改）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X2-R2-codex.md
本轮决议: 采纳 6（X2 R1-B1、R1-B2、R1-P1；底座-P1/P2/P3，其中 P3 改为预算直接写定），R1 的 S1–S5 也全部采纳。审计的五个问题全部采纳，修法写在 STAGE_PLAN B 部分；X1 的 R2-E1 维持第 88 次决议

需要你做的:
1. **审 X2 第二版**（STAGE_PLAN 全文）和我写的 models/LegacyBases.py，重点看：
   - 移植是否忠实于原结构（对照 lunwen 下两个项目的原代码）；
   - DA 是否为标准公式（GRL 只乘一次 λ，域损失权重为 1）；
   - 堆叠是否因果（只在 V1 上拟合，按时间前向切块）；
   - 6 个底座的公平性（输入、切分、配方都相同）。
   - 发现确定缺陷，按唯一的最小修法直接改并写进审查文件；设计分歧写成编号交我。
2. **在服务器上跑测试，不训练：**
   - tests/test_legacy_bases.py，本地只能跑 numpy 部分（6 项已过），torch 部分 4 项要你跑；
   - 每个神经网络臂用 1 个 epoch 的小样本做 CPU 冒烟，确认能接上 train_g1.fit_one/predict，DA 的训练步能跑通。
3. **X1 继续按原授权：** HoT 的 E1 按 0.01 kW 阈值重跑；Chronos-2 回执到了，就跑 GPU 队列和 C1。
4. 不下载、不安装，不读 2026 年的数据；也不执行 lunwen 下的旧代码。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-16，第89次复审及Claude/Gemini中断恢复 -->
轮次: R2/2 | 阶段: X2审查、X1执行恢复 | 分支: exp/X1 | 提交号: 实现e4819b0，本轮补启动回执
审查结论: X2本轮通过，无阻断；见docs/review/X2-R2-codex.md。X2仅单测/合成CPU冒烟，未正式训练或评估。
已直接修复: X1补齐A3b长上下文、B1按验证MAE选模、模型身份锁定、队列互斥及失败停止；小设计S9–S13见审查文件。
已直接修复: 撤回中断代码测试中擅自放宽anchor容差的期望；E1保持已批准truth=0.01、输入=1e-5。
运行信息: 服务器Legacy11项全部通过（3.474s），含四网络单epoch/16合成样本fit_one/predict和DA反传；X1单测12项通过。
运行信息: 本地X1 12项、Legacy numpy6项通过；torch5项本地跳过，服务器已补齐。
核对结论: A0/A1/A2/E2及E2-saved已有完成记录；未发现A3/B2/B1/C1实际运行实例，不能把此前计划当成启动回执。
模型身份: revision=29ec3766d36d6f73f0696f85560a422f50e8498c，manifest SHA=758b5c252d2a0bbe0237554d3c1587b4bb5165b73d96ccdcec0cf98e31d0de5e；服务器逐文件核验通过。
环境: Chronos工作进程tsfm；父队列和LightGBM只用timedart4.5.0；不安装、不下载、不读2026数据。
GPU队列: output/x1/run_gpu_queue.py；A3a/A3b→B2共27拟合→B1每场一次→C1，限2CPU线程/核及20%GPU显存，失败无重试。
GPU队列状态: 三场真实数据只读预检均通过；已启动PID1278009，当前A3，子进程1278026、Chronos工作进程1278137，完整命令已核对。
正式run: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/X1/x1_e4819b0_gpu_r1；worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_X1_e4819b0。
正式命令: timedart/bin/python -u output/x1/run_gpu_queue.py；完整参数见output/x1/recovery_20261116.json；协议SHA=d75b1312ac6aa1f6a0d60b933340c09e4e56abacdb5a047134ddf0ec1b2c1bf3。
协议: configs/x1_protocol.json版本6；不修改旧服务器冻结源码/协议，正式部署另建提交对应worktree。
E1状态: x1_90029d5_r7/E1已失败，anchor最大差0.000244140625kW；本轮不重跑，不放宽输入门禁。
关键数字: 本轮尚无新正式模型指标；A1优于N的既有负结果保留，不调参或重跑。
X2限制: 不执行54次底座拟合或ANI实验；正式接线需避免BatchNorm单样本尾批，V1融合前向折不能称严格在线独立验证。
需要 Claude 做的: 复核X1 S9–S13及恢复记录；处理E1 anchor精度差；X2正式执行另行交接，保留V2已暴露声明。
需要用户决定: 无；按既有X1授权继续，X2不扩大范围。
