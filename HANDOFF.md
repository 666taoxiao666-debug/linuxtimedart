# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-18 第 100 次（X3-Q2fix：冒烟在 Q2 预检显存不足后的修正，外加 R2-S1/S2 措辞） -->
轮次: R1/2 | 阶段: X3-Q2fix | 分支: exp/X1 | 基线提交: dee252b
请求: 审代码，只审改动（通过后提交推送，并写明可交给 Cursor 的提交号）
待审范围: git diff dee252b -- scripts/run_x3.py configs/x3_protocol.json tests/test_x3.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x3.py, configs/x3_protocol.json, tests/test_x3.py, scripts/launch_x2.sh, scripts/status_x2.sh, output/x3/**, output/x2/**（x2_0f24875_json.tar.gz 不要提交）, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X3-Q2fix-R1-codex.md
本轮决议: DECISIONS「阶段 X3-code」补了 R2-B1（复核正确）、R2-S1、R2-S2（都改了措辞）；新增「阶段 X3-Q2fix」冒烟-1/2/3

冒烟结果（Cursor，dee252b）: q1check、q1 完成；q2 预检时 Kelmarsh、HoT（1024 h）全参数 256 行超出 0.7 上限（16.49 GiB），Penmanshiel（512 h）峰值 11.94 GiB 通过；tsfm 没装 peft，lora 被 SDK 悄悄退回全参数。

需要你做的:
1. **审改动，重点看：**
   - **候选组合：** q2_candidates、preflight_search。全参数下 microbatch × 累积依次为 256×1、135×2、90×3、45×6，每种组合每步都是 18 个窗口（依据 2.3.2 dataset._generate_train_batches 的整窗累加规则），然后才轮到 LoRA；取第一个三场都通过的组合；每个"组合 × 风场"单独起进程预检。
   - **梯度累积：** fit_kwargs 把 gradient_accumulation_steps 经 extra_trainer_kwargs 传给 TrainingArguments（2.3.2 的 pipeline.fit 会用 training_kwargs.update 覆盖默认的 1，我核对过源码）。worker_q2 在 on_train_begin 核对 trainer 实际的 microbatch 和累积步数。请核对锁定的 transformers 版本下损失是否按累积步数做了平均（见 DECISIONS 的残留不确定）。
   - **LoRA 不许悄悄回退：** 没装 peft 就记为不可用；SDK 的回退警告会变成报错。本轮不装 peft。
   - **不改的参数：** gpu_fraction 0.7、学习率、步数、有效 batch、2 万窗口预算都不改；预算字段改名，注明是按规则推算的，不是实测。
2. **本地测试：** tests/test_x3.py 32 项、test_x2 22 项，unittest 跑共 54 项通过。
3. **通过后：** 提交推送，并写明提交号。Cursor 必须用新提交从头跑 x3smoke（新的 x3_<提交号> 目录，旧的 dee252b 冒烟留作证据），成功后再进正式链。我不让 Cursor 启动。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-18，第100次X3-Q2fix R1 -->
轮次: R1/2 | 阶段: X3-Q2fix | 分支: exp/X1 | 提交号: 本次提交（基线dee252b；git log -1 -- HANDOFF.md查询）。
审查结论: 通过，0条阻断、2条建议；docs/review/X3-Q2fix-R1-codex.md；无需为建议进入下一轮。
已直接修复: 无；执行代码、协议、测试为Claude本轮改动，Codex只写审查及本节。
运行信息: 本地python -m pytest tests/test_x3.py tests/test_x2.py -q；54 passed（32+22，68.45秒）；git diff --check通过。未上服务器、未安装依赖、未训练推理。
关键数字: 四组microbatch×累积均18窗口/优化步；full先于lora；0.7显存上限、1000步、学习率、20,000窗口预算不变，协议v3。
梯度核对: 本地回执transformers 5.23.0对应官方源码在无labels/无自定义loss时除以累积步数；Chronos 2.3.2会接收额外累积参数，正式on_train_begin核对实际设置。
LoRA核对: 缺peft拒绝该候选，SDK退回full的警告转异常；本轮不装peft、不扩大服务器资源。每个候选×场独立进程，首个全场通过组合统一冻结。
局限: 等样本预算不等于逐位数值等价；未在服务器做梯度等价实验。旧冒烟失败证据尚未在本地output/x3落盘，服务器状态依Cursor回执，未独立查询。
需要 Claude 做的: 记录两条非阻断建议，后续复核Cursor冒烟/正式结果；论文避免宣称调整累积“只改变显存、优化轨迹完全相同”。
Cursor交接: 本次提交干净副本先x3smoke（新x3_<提交号>目录，旧dee252b原样保留）；核对当前transformers仍为5.23.0并记录candidate/Trainer设置。成功后x3gpu，正式q1完成后x3q4，q2结束且q4完成后x3final；失败停下交回，不重试。
需要用户决定: 无。
