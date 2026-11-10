# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-10 第 60 次（T1 RETAIN_EXISTING 已复核、种子集成成立；仍按第 59 次做 M4 R2 + CW3；L1 现在启动） -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 基线提交: 8af6a7d
请求: 审方案后实现并运行（M4 只审改动；无阻断就正式运行 CW3 一次；L1 核对权重后启动，二者并行）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_cw3.py, tests/test_pilot_conformal_cw3.py, output/conformal_cw3/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/t1_tuning/**, output/conformal_cw2/**, output/conformal_cw1/**, output/conformal_c0b/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M4-R2-codex.md
本轮决议: M4 R1 采纳 3（R1-B1、S1、S2，见 DECISIONS 的 M4 一节，第 59 次已写）；T1 RETAIN_EXISTING 已独立复核并接受，5 个 seed 的集成按事先条款成立（DECISIONS 的 M2 末节）

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS，包含第 59 次没提交的 M4 R2 改动和这次的 T1 复核。
2. **M4 R2 只审改动部分**（最后一轮，有分歧交用户）：
   - J3、J4 降级为部分暴露的探索性判定，附暴露清单；
   - 方法主张只能等 Hill of Towie；
   - 结论档位的真值表。
   - 无阻断就实现 CW3 并正式运行一次（CPU、2 个线程），交回内容同第 58 次。
3. **L1 现在启动**（T1 已跑完，GPU 空闲）：
   - 先只读核对 llm_weights/manifest.json 里 49 个文件的 SHA256、revision c03e6d35… 和 llm_l1 环境，再按 `git show a1ebfa0:docs/STAGE_PLAN.md` 的 L1 规则实现并运行（M1 R2 已审定，提速规则适用）。
   - 交回：各组的搜索最优值，正式阶段两个风场的 MAE/RMSE 和 CI、判定，以及前 10 名程序的文本。

需要用户决定: 无（exp/M2 何时合回主线，等 CW3 和 L1 都有结果后再一并问）

## Codex → Claude
<!-- 由Codex覆盖。2026-11-10：M4 R2通过；权重预检完成；实现待完成 -->
轮次: R2/2 | 阶段: M4/CW3与M1/L1 | 分支: exp/M2 | 提交号: 方案f8f764e；本轮见git log
审查结论: M4 R2通过，0条阻断，R1-B1/S1/S2关闭；docs/review/M4-R2-codex.md。
已直接修复: 无；未改冻结文件。Claude第59/60次STAGE_PLAN及DECISIONS已提交。
运行信息: SSH只读检查49个权重文件，大小与SHA256全部匹配交付manifest；codex_verification.json记录结果。
运行信息: llm_l1环境Python3.10.20、torch2.6.0+cu124、transformers5.20.2、accelerate1.15.0，CUDA可用，与交付一致。
关键数字: manifest声明revision=c03e6d358207e414f1eca0bb1891e29f1db0e242；本轮核验本地交付字节，未通过远端Hub重新证明revision来源。
执行状态: CW3、L1均未启动，正式实例预算未消耗；scripts/pilot_conformal_cw3.py及scripts/llm_encoder_search_l1.py尚不存在。
尚未完成: CW3实现/测试/冒烟/正式运行；L1程序沙箱、编码器、搜索和正式评估实现/测试/启动；不能把方案通过或权重核验写成实验完成。
后续批准范围: 用户第60次已授权继续实现、测试、部署及各一次正式运行，无需重复申请许可；保持CPU2线程/GPU分工和冻结阈值。
需要 Claude 做的: 暂无新阻断需决议；风险3残留“新数据”措辞建议与探索性主规则统一，不阻断实施。
需要用户决定: 无；本轮仅完成前置审查与权重预检，实施工作仍未完成。
