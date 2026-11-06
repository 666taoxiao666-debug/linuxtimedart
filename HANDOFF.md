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
<!-- 由 Codex 覆盖。2026-11-06 第39次实施，C0完成stop，A运行中 -->
轮次: R2/2 | 阶段: E1执行中 | 分支: exp/E1 | 提交号: 协议ce27e1f；C0代码9e98cbd；A代码b5be60a
审查结论: 按用户裁决及第39次明确授权直接实现运行，未重开方案审查；实现评级B，见docs/review/E1-R2-codex.md后半段。
已直接修复: A补齐模型源码哈希登记；统计直接复用paired函数、不计算Holm；未改冻结历史代码或标签。
运行信息: 服务器/home/XT/nuist/pythoncode/TimeDARTFirst；C0独立worktree后缀_E1_9e98cbd，A后缀_E1_b5be60a。
运行信息: C0 PID433910已完成，目录outputs/logs/Conformal/c0_9e98cbd_r1，08:38:00北京时间结束，98.68秒；CPU2线程，不重跑。
运行信息: A PID433991，python scripts/train_e1_baselines.py --data-root <数据根> --out <数据根>/outputs/logs/E1/e1_b5be60a_r1 --device cuda:0；日志/PID同规则，不自动重启。
运行信息: 两边代码经Git精确提交部署；C0服务器5单测+smoke通过；合并9项服务器单测及A六组合1epoch合成fit/replay全部通过。
关键数字: C0=stop；M覆盖81.2799%（门槛88–92%），宽728.7276kW，Winkler1521.0805；分数达标但覆盖不达标，详见output/conformal_c0/RESULT_SUMMARY.md。
关键数字: B0/B1/B3覆盖78.5593/91.3723/80.4843%，Winkler1833.2425/1639.9801/1522.9092；仅B1进入比较，95%分数门槛1557.9811。
关键数字: B2/M+覆盖89.3427/89.4430%，均出现无限宽区间，平均宽/分数=+∞（JSON null）；不截断、不以宽度差宣称改善。15数组哈希及两覆盖率汇总独立复算通过。
关键数字: A仅Kel2023调d_model64/128共10次，正式54拟合，共64；全部冻结后一次评估，历史参照需哈希/窗口/标签逐元素一致。
关键数字: iTransformer参数64/128为101956/400516；TimeXer136836/535812，均与固定官方实现等参数且同权重输出完全一致；last10/upstream相同参数量。
运行信息: Chronos可选项本次暂跳过，启动时未取得完整环境/权重合成验收；下载及安装交Cursor，未重试。
需要 Claude 做的: 请优先复核C0实现、因果断言及stop负结果；A完成后另交描述性对比，不把已打开年份写成新增确认。
需要 Claude 做的: STAGE_PLAN末尾测试条目“校准不用2021标签”为旧措辞，应同步已裁决的2021校准/2022评估；实际实现按正文与因果断言执行。
需要用户决定: 无；MIDAS认证和下载继续由Cursor接手。GitHub已恢复，ce27e1f、9e98cbd、1343756、b5be60a均推送成功。
边界: main未动，无新增定时监控；未提交Cursor下载文件；未补跑历史实验；正常训练不逐次汇报。
