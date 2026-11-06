# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-06 第 40 次（E1：C0 停止复核 + A 线只恢复评估 R2-P1） -->
轮次: R2/2 | 阶段: E1 | 分支: exp/E1 | 基线提交: 39827e5
请求: 审代码后运行（只实施 R2-P1，即只恢复评估）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/train_e1_baselines.py, tests/test_train_e1_baselines.py, configs/e1_baselines_protocol*.json, output/e1_baselines/**, output/kelmarsh_hourly/upstream_v32_review/results/post_test_manifest.json, output/conformal_c0/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**
审查文件: docs/review/E1-R2-codex.md（续写）
本轮决议: R2-P1 采纳，需用户授权例外；C0 判定 stop，已独立复核；A 线失败原因和 64 次拟合的核验记录已复核（见 DECISIONS.md 的 E1 一节）

需要你做的:
1. **提交**：本轮的 STAGE_PLAN.md（测试清单的旧措辞已改正）和 DECISIONS.md。另外把 `output/kelmarsh_hourly/upstream_v32_review/results/post_test_manifest.json` 纳入 Git：本地文件的 SHA256 是 a8cc5708…，与 v32 自己的 completed.json 登记一致。
2. **R2-P1（用户授权后再实施）**：
   - 新版本配置，新输出目录；原失败目录只读保留；
   - 复用冻结的 64 次拟合和选参结果，不训练、不重新选参；
   - 参照清单按 v32 completed.json 的哈希锁定；写评估标记之前预检全部参照文件，并补一个"缺文件就报错"的合成反例测试；
   - 只评估一次。
3. **交回**：A 线各模型 × 通道在三个设定上的 MAE/RMSE/R²，iTransformer、TimeXer 的上游增益及 CI，以及各模型相对 TSMixer-upstream 的差距（都只作描述）。

需要用户决定: 是否授权 A 线只恢复评估，作为原协议 instances=1、resume=false 的例外

说明:
- C 线已停止。下一个方法方向由用户和导师商量后再定，这一轮不开新的方法实验。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-06 第40次：A线仅评估恢复已获用户具体授权 -->
轮次: R2/2 | 阶段: E1只恢复评估 | 分支: exp/E1 | 提交号: 批准记录8a2eaa5；恢复实现6204100
审查结论: 通过，评级B；R2-P1已获Claude采纳和用户明确例外授权，见docs/review/E1-R2-codex.md末节。
已直接修复: 6204100按批准范围实现新配置/恢复入口；v32参照manifest已纳入Git；不依赖漏部署的seed JSON，改用v32 completed登记的预测哈希。
运行信息: 服务器7项测试通过（含缺参照文件不写评估标记、篡改冻结记录拒绝、恢复路径不调用fit/choose）。
运行信息: 原64拟合及全部登记产物哈希、原源码和未改共享代码身份检查通过；v32 manifest哈希a8cc5708…通过。
运行信息: 新worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_E1_eval_6204100；PID446391，真实完整命令已确认。
运行信息: python -u scripts/train_e1_baselines.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/E1/e1_b5be60a_eval1 --device cuda:0 --recover-evaluation
运行信息: 输出目录固定唯一且拒绝复用；父目录launch_once排他登记已写入，原r1保持只读；已于北京时间13:44:26完成，无failed标记；本次仅一次评估。
关键数字: 复用64拟合（10搜索+54正式），新增训练=0、重新选参=0；三设定/六臂/三seed，评估仅一次。
关键数字: C0维持stop，M覆盖81.2799%未达88–92%；B2/M+无限区间照实保留；Claude已复核，不再优化或补跑。
需要 Claude 做的: 复核output/e1_baselines/RESULT_SUMMARY.md及eval1/result.json、independent_verification.json；各臂指标、增益CI、相对TSMixer差距均已列出，全部仅描述。
关键数字: TSMixer在Kel/T22/T23的MAE=166.2789/213.6308/188.8388（精确值见result），三设定全部新增基线均更差。
关键数字: TimeXer上游MAE增益4.0119/4.6883/4.1757kW，95%CI均>0；iTransformer为4.0338/1.0364/0.3902kW，后两组CI跨0。
运行信息: 57数组哈希、64拟合原产物重新核验；保存数组复算全部指标和24比较，结果见独立核验JSON；无新增定时任务。
需要用户决定: 无；结果待Claude复核，未合主线。
边界: 未改原冻结协议/旧worktree、未重新训练/选参、未合主线、main未动、无新增定时任务、未提交Cursor下载文件。
