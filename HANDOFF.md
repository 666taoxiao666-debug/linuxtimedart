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
<!-- 由 Codex 覆盖。2026-11-06 第39次结题交接：C0有效负结果，A评估技术失败 -->
轮次: R2/2 | 阶段: E1结果复核 | 分支: exp/E1 | 提交号: 上轮ad64d61；C0代码9e98cbd；A代码b5be60a；本轮见git log
审查结论: C0完成并按冻结标准stop；A的64拟合完成，但最终评估技术失败，无A测试成绩。见docs/review/E1-R2-codex.md末节。
已直接修复: 本轮无；未修改冻结代码、协议、checkpoint或选择结果，未重启/补跑。
运行信息: C0 outputs/logs/Conformal/c0_9e98cbd_r1于北京时间11月6日08:38完成，98.68秒；15预测数组哈希与全部汇总指标独立复算通过。
运行信息: A outputs/logs/E1/e1_b5be60a_r1，PID433991已退出；北京时间11月6日11:20:31最终评估入口FileNotFoundError。
运行信息: 缺失独立worktree下output/kelmarsh_hourly/upstream_v32_review/results/post_test_manifest.json；该历史文件本地存在但未纳入Git，Codex部署漏带且预检未覆盖。
运行信息: 已逐拟合核对trained.json与selection_freeze，并验证init/best/last权重、history、validation数组全部登记哈希；artifact_errors=[]；evaluation下无npz。
关键数字: C0 M覆盖81.2799%（要求88–92%），宽728.7276kW，Winkler1521.0805，分数达标但覆盖失败→stop。
关键数字: B0/B1/B3覆盖78.5593/91.3723/80.4843%，宽746.9999/1242.1126/685.8425kW，Winkler1833.2425/1639.9801/1522.9092。
关键数字: B2/M+覆盖89.3427/89.4430%，宽与分数均+∞（JSON null）；M+增加0.1002个百分点，不宣称有限宽度/分数改善。
关键数字: A=64/64拟合（10搜索+54正式），冻结记录与产物哈希通过；0最终评估数组，没有可报告的A线MAE/RMSE/R²。
运行信息: A失败与核验JSON已拷回output/e1_baselines/r1/；总结output/e1_baselines/RESULT_SUMMARY.md；C0见output/conformal_c0/RESULT_SUMMARY.md。
需要 Claude 做的: 一并复核C0因果实现、有效负结果及无限区间；复核A部署缺陷与64拟合证据；在DECISIONS逐项记录。
需要 Claude 做的: 审核R2-P1仅评估恢复提案（待审核，未实施）：新版本/新输出目录复用冻结64拟合与选参，补齐并锁定参照元数据、前置核验，不训练不重新选参；原失败目录保留。
需要 Claude 做的: STAGE_PLAN末尾“校准不用2021标签”为旧措辞，需同步2021校准/2022评估；本轮未代改作者方案。
需要用户决定: A已写evaluation_started，原协议instances=1/resume=false；任何恢复评估须Claude复核方案后由用户具体授权，不能以一般“继续”自动补跑。
边界: main未动；未合主线；未提交Cursor文件；没有新增定时监控；GitHub上轮推送已恢复，本轮另提交结果记录。
