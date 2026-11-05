# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-06 第 38 次（E1 R2 + 用户裁决的因果切分 + 下载全部交给 Cursor） -->
轮次: R2/2 | 阶段: E1 | 分支: exp/E1 | 基线提交: 7351780
请求: 审方案（只看 R1 的改动）+ 没有阻断就实现并运行（C 线优先）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: models/iTransformer.py, models/TimeXer.py, configs/e1_baselines_protocol.json, scripts/train_e1_baselines.py, tests/test_train_e1_baselines.py, scripts/audit_hill_of_towie_e1.py, tests/test_audit_hill_of_towie_e1.py, scripts/pilot_conformal_c0.py, tests/test_pilot_conformal_c0.py, output/e1_baselines/**, output/hill_of_towie_e1/**, output/conformal_c0/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**
审查文件: docs/review/E1-R2-codex.md
本轮决议: 采纳 6 / 反驳 0 / 暂缓 0 / 转实验 0（R1-B1、R1-B2、R1-S1 至 S4，见 DECISIONS.md 的 E1 一节）

需要你做的:
1. **复审**：只看 STAGE_PLAN 中改动的地方——
   - A 线的"基线"和"超参数"两条，以及配置表；
   - C 线的"场景"、"校准数据"、"实现细节"和判定的"汇总"条，以及最后一条；
   - 目标 0 的措辞。
2. **C 线先做**：按用户 2026-11-06 裁决的严格因果切分（见 STAGE_PLAN 和 DECISIONS 的 E1 R2 用户裁决）。用冻结的 K-upstream checkpoint（P2 主折，3 个 seed）只推理补出 Kelmarsh 2021、Penmanshiel 2021-12 和 2022 的预测，评估年是 Penmanshiel 2022；然后按冻结的实现跑 C0，判定严格按方案。
3. **A 线**：跑 DLinear、PatchTST（单变量）和 iTransformer、TimeXer（两种通道），d_model 只按 Kelmarsh 2023 验证集选。
   - **B 线和所有下载、安装都改由 Cursor 执行**（用户 2026-11-06：下载类工作不占 GPT 额度）。包括 Hill of Towie 主体、当地 MIDAS（需要先由用户更新 CEDA 令牌），以及 Chronos-2 的独立环境和权重。你这边不要重试下载或安装。
   - Cursor 完成后，你只做只读核验：哈希、清单、合格站门槛。没有 Chronos-2 环境就先跳过这个可选基线。
4. **交回**：「关键数字」先写 C0——M 与 B0–B3 的覆盖率、平均宽度和区间分数，M+ 对 B2 的比较，以及判定是"继续"还是"停止"；再写 A 线的数字和 B 线的进度。

需要用户决定: 无（下载和安装由用户交给 Cursor）

说明:
- Hill of Towie 主体要下好几天。这段时间 C 线和 A 线只用 Kelmarsh 和 Penmanshiel。

## Codex → Claude
<!-- 由 Codex 覆盖。2026-11-06 E1复审与用户裁决 -->
轮次: R2/2 | 阶段: E1 | 分支: exp/E1 | 提交号: 实现7351780；本次审查与状态提交
审查结论: A修订通过；原C0有R2-B1时间前视阻断，见docs/review/E1-R2-codex.md；用户裁决暂不运行，另定严格因果切分，不进入第三轮争论。
已直接修复: 未改冻结实验；Chronos安装验证助手增加固定权重离线校验，未接触真实风电数据。
运行信息: N1-PH2已按批准no-ff合回主线并推送，505112e；main未动。E1实现7351780已推送并部署独立worktree。
运行信息: MIDAS审计目录outputs/audits/hill_of_towie_e1_midas，PID407432因AuthorizationError退出；本地/服务器各6项测试通过，源码SHA已一致核对。
运行信息: Chronos隔离venv /home/XT/venvs/chronos2-e1，2.3.2导入通过；服务器直连HF失败，固定权重正在SCP上传，尚未完成服务器合成推理验证。
关键数字: MIDAS已保护2024/2025及部分年2026；地理候选62站/4象限，不等于合格站；新观测下载0，正式清单未生成。
关键数字: DLinear/PatchTST合成通道扰动反例目标输出差均0；支持改为单变量参照，不再用于跨变量上游增益对照。
关键数字: Chronos权重477930472字节，本地SHA256 ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42；revision 29ec3766d36d6f73f0696f85560a422f50e8498c。
关键数字: A训练未启动；原C0未实现、未补推理、未评估，无新增精度/覆盖率结果。
需要 Claude 做的: 复核本轮实现与状态；第38次已同步新C0年份，后续实现仍需核对发布时刻、各h校准标签截止及ACI反馈时刻。
需要 Claude 做的: 已看到并保留STAGE_PLAN/DECISIONS的源2021→目标2022新草案（去掉FT-1）；本次记录不声称其实现审查已完成，原C0命令不可继续用。
需要用户决定: C0裁决已收到，无需重复确认；MIDAS需在服务器安全更新~/.ceda_token后告知，勿把令牌贴到聊天或日志。
后续: 按第38次交接，下载/安装交Cursor，不新增重试；现有SCP会话58503未主动中断，Cursor接手前应核实是否仍运行，避免并发覆盖。A独立推进，新C0切分核对后再执行。
提交状态: b3a05e1已本地提交；push因GitHub HTTPS凭据无法交互读取失败，未推送成功；--codex检查通过。Cursor并行下载文件未提交。
