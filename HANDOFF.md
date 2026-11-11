# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-11 第 68 次（O3 复核通过：STOP_L1，L1 永久关闭；合并 exp/M2） -->
轮次: R2/2 | 阶段: M7 | 分支: exp/M2 | 基线提交: b09d837
请求: 提交 + 合并（按用户同意把 exp/M2 合回主线），不运行任何实验
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: output/l1_llm_encoder/**, scripts/acquire_gfs_g0.py, output/external_acquisition/gfs_g0/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**, .cursor-tmp/**（后四项是本地未跟踪文件，只为通过范围检查，不要改、不要提交）
审查文件: 无（结果复核，无新编号）
本轮决议: 无新编号。O3 的 STOP_L1 复核通过，复算与选择顺序都一致，见 DECISIONS M7 末段

需要你做的:
1. **提交** STAGE_PLAN（A 线标为已完成）和 DECISIONS（O3 复核）。
2. **合并：** 用户 2026-11-11 已同意"O3 跑完后合并"。
   - 把 exp/M2 用 `git merge --no-ff exp/M2` 合进 codex/self-evolving-wiki，合并说明写上 M2–M7 的结论：T1 RETAIN_EXISTING（种子集成进联合确认）；CW1–CW3 STOP（C 线结束）；L1 r3/O2/O3 STOP（L1 关闭）；爬坡 RAMP_REPLICATED；NO_FALLBACK；G1 五臂方案定稿未实施。
   - 推送主线，main 不动。不 reset、不 force push；有冲突就停下交回。
3. **合并后从主线拉 exp/G1**，作为 G 线的工作分支。暂不实现；GFS 的 B1、B2 数据由用户转达到位后，再做 G0-c 只读核验。
4. 交回：合并提交号、推送结果、exp/G1 的起点。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-11：第68次文档提交、M2合主线、创建G1 -->
轮次: R2/2 | 阶段: M7归档/G1待数据 | 分支: exp/G1 | 提交号: 文档f4e2704；合并860cef1；交接提交见git log
审查结论: 接收Claude对O3的STOP_L1复核；无新方案审核，L1关闭。STAGE_PLAN和DECISIONS已随f4e2704提交并推送exp/M2。
已直接修复: 无模型或协议改动；原工作区切换分支因.git/HEAD权限拒绝失败，已确认索引等于旧主线且无未暂存改动后，恢复本次切换产生的文件变化；原工作区已跟踪文件干净。
运行信息: 在隔离副本D:/Users/XT/Desktop/moxing/gpt/v2/TimeDARTFirst_G1完成--no-ff合并，无冲突；合并双亲f4711a7和f4e2704，合并树与exp/M2完全一致。
运行信息: 860cef1已推送codex/self-evolving-wiki；exp/G1从该提交创建，本分支只新增此交接记录。原共享工作区仍在exp/M2，未修改Windows ACL，未删除或移动Cursor及其他未跟踪文件。
关键数字: T1 RETAIN_EXISTING，种子集成进联合确认；CW1–CW3 STOP，C线结束；L1 r3/O2/O3 STOP，L1关闭。
关键数字: 爬坡RAMP_REPLICATED；NO_FALLBACK；G1五臂方案定稿未实施，30次拟合尚未启动。
运行信息: 本轮只做文档和Git归档，无服务器实验、无下载、无Token访问；main保持202d9f6。
需要 Claude 做的: 后续使用TimeDARTFirst_G1的exp/G1，或先解决原目录HEAD写权限再切换已推送的exp/G1；勿在旧exp/M2继续写G1方案。等用户转达GFS B1/B2到位后才做G0-c只读核验。
需要用户决定: 无新的实验决策；原目录HEAD权限问题已绕开完成归档，未擅改权限。
