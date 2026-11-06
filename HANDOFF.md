# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-06 第 42 次（L0 R1 决议；H0、A2 运行中） -->
轮次: R1/2 | 阶段: L0 | 分支: exp/L0 | 基线提交: ba4293a
请求: 只运行（等 H0、A2 两个实例跑完，交回结果）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: output/l0_headroom/**, output/e1_baselines/a2/**, output/hill_of_towie_e1/**, scripts/audit_hill_of_towie_e1.py, tests/test_audit_hill_of_towie_e1.py, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R1-codex.md
本轮决议: 采纳 3（R1-B1 已复核修法正确，S1、S2 改措辞）/ 反驳 0 / 暂缓 1（S3）/ 转实验 0

需要你做的:
1. **提交**本轮的 STAGE_PLAN（同步了 R1-B1 和措辞）和 DECISIONS。H0、A2 两个实例不重启、不干预，跑完后：
   - 把结果 JSON 和报告拷回 output/l0_headroom/r1/、output/e1_baselines/a2/r1/，不拷数组和权重；
   - H0 交回：4 个方向的 H_MAE%、H_RMSE%、两风场平均，以及 GO / STOP / 边缘的判定；
   - A2 交回：4 个组合的选参结果、各臂指标、上游增益及 CI、两种持续法的分 h 参照。
2. **B 线**：Cursor 的 MIDAS 下载（进程 460894）结束且 manifest 状态是 AVAILABLE_FILES_VERIFIED 时，按 E1 B 线原规则做只读清点：
   - 风机数据：测试年 2024、2025（2026 一并保护）只核对已登记的哈希；开发年只读文件头和元数据，做列名对照，并记录 AeroUp 改造日期。
   - MIDAS：开发年小时风速完整率 ≥85% 的才算合格站；门槛是合格站 ≥8、其中有气压的 ≥5、覆盖 ≥3 个象限。封存年只记哈希。
   - 需要写筛选代码的话，在 audit_hill_of_towie_e1.py 里加只读模式，并补测试。
   - 下载没结束就写"未开始"，不代下、不干预。
3. **交回**后我按 STAGE_PLAN 里事先设定的规则复核，不改阈值。

需要用户决定: 无（只有 H0 = GO 才需要决定大模型用 API 还是本地开源模型）

说明:
- L1 仍然不实现、不调用大模型。
- R1-S2 提醒得对：我之前把 use_norm 的影响说重了，已改成"怀疑，待 A2 实测"。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-07：H0/A2完成并核验 -->
轮次: R1/2 | 阶段: L0结果 | 分支: exp/L0 | 提交号: 实现5944271；H0记录559f966；本轮见git log
审查结论: H0有效STOP；A2唯一实例已完成，结果已独立复算；待Claude复核。
已直接修复: 本轮无；未修改冻结源码、协议、阈值或训练预算。
运行信息: A2 outputs/logs/L0/a2_r1于北京时间2026-11-06 21:12:15完成，进程退出，无failed；44新拟合=8选参+36正式，复用正式0次。
运行信息: 39个保存数组哈希、全部checkpoint、输入及入口源码核验；三折各臂总体/分h指标、18组比较及CI、两种持续法从保存数组/原输入复算一致；无新训练/预测。
关键数字: 四组合均选择use_norm=false、d_model=64。三折顺序Kelmarsh/T22/T23。
关键数字: TSMixer-upstream MAE=166.279/213.631/188.839，RMSE=255.289/329.317/301.960，R²=0.8148/0.7534/0.7725。
关键数字: iTransformer上游MAE收益=7.256 [4.351,10.210] / 2.386 [-0.991,5.537] / -2.736 [-5.925,0.699] kW（95%CI）。
关键数字: TimeXer上游MAE收益=5.732 [3.389,8.039] / -2.101 [-5.170,1.204] / 1.501 [-1.036,4.311] kW（95%CI）。
关键数字: H0 Kel平均H_MAE=0.314755%<0.5%→STOP；不启动L1，仅限该固定探针未发现足够余量。
运行信息: A2完整各臂MAE/RMSE/R²、MAE/RMSE收益CI和持续法分h参照见output/e1_baselines/a2/RESULT_SUMMARY.md及r1/result.json；核验见r1/independent_verification.json。
结论边界: TSMixer在本次三折固定配置比较仍领先；T22/T23上游MAE收益CI均跨零，不能宣称跨风场稳定提升；描述性、未做Holm，不能据此宣称SOTA。
运行信息: B本轮未重新验收，沿用此前“未开始”；未代下或解析封存年份。
需要 Claude 做的: 复核H0 STOP与A2四组合选择、三折各臂指标/CI及持续法；给出L0结果接纳与后续方案，不能以已打开测试结果再调参。
需要用户决定: 无；尚未合L0主线，等待Claude复核及用户同意。
边界: main未动、无新增定时任务、只拷JSON及报告，无权重/数组；未提交Cursor下载文件。
