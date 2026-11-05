# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-06 第 39 次（E1：按用户裁决的因果切分直接实现并运行 C0 和 A 线） -->
轮次: R2/2 | 阶段: E1 | 分支: exp/E1 | 基线提交: dd06aa7
请求: 审代码后运行（方案已由用户裁决定稿，不再审方案；C 线优先，A 线并行）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: models/iTransformer.py, models/TimeXer.py, configs/e1_baselines_protocol.json, scripts/train_e1_baselines.py, tests/test_train_e1_baselines.py, scripts/pilot_conformal_c0.py, tests/test_pilot_conformal_c0.py, output/e1_baselines/**, output/conformal_c0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**
审查文件: docs/review/E1-R2-codex.md（续写实现与运行记录）
本轮决议: R2-B1 已登记为用户裁决（严格因果切分），见 DECISIONS.md 的 E1 一节；你上一轮的实现和状态我已复核，没有问题

需要你做的:
1. **提交**：先提交本轮的 STAGE_PLAN.md 和 DECISIONS.md。新增了 R2-B1 的决议行，以及"因果断言"一条：校准窗口的最后一个标签必须 ≤ 2021-12-31 23:00，跨年的窗口剔除；每个预测都检查 max(校准标签时刻) ≤ 起报时刻。
2. **C0（直接实现并运行，不用再交审）**：
   - 用 P2 主折冻结的 K-upstream（3 个 seed）只做推理，补齐 Kelmarsh 2021、Penmanshiel 2021-12 和 2022 的预测；评估年是 Penmanshiel 2022。
   - 发布时刻、各 h 的标签截止和 ACI 的反馈时刻都按上面的断言实现；任何一条不满足就报错停下，不改口径。
   - 判定严格按 STAGE_PLAN。
3. **A 线**：DLinear、PatchTST（单变量）和 iTransformer、TimeXer（两种通道）照方案跑。Chronos-2 等权重上传、合成验证都通过后再作为可选项；没有就跳过并记录。
4. **交回**：「关键数字」先写 C0——M 与 B0–B3 的覆盖率、平均宽度、区间分数，M+ 对 B2 的比较，以及判定是"继续"还是"停止"；再写 A 线。

需要用户决定: 无。下载由 Cursor 接手（Cursor 的下载脚本和记录不在你的提交范围内，不要提交）；CEDA 令牌和 GitHub 推送凭据由用户自行修复。

说明:
- 确定缺陷可以按提速规则先改再跑。C0 是第一个结果，请尽快交回。

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
