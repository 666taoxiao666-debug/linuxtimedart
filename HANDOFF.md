# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-11 第 65 次（O2 复核通过；用户裁决：L1 第 3 轮调参 O3 + 恢复 GFS 下载 + G1 方法协议待审） -->
轮次: R1/2 | 阶段: M7 | 分支: exp/M2 | 基线提交: 49596dc
请求: 审方案（A、C 两部分）；A 无阻断就实施并运行一次；C 只审不做
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/llm_encoder_search_l1.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, scripts/acquire_gfs_g0.py, output/external_acquisition/gfs_g0/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**（后三项是本地历史未跟踪文件，只为通过范围检查，不要改）
审查文件: docs/review/M7-R1-codex.md
本轮决议: M6 无编号（R1-S1 只是建议）。接受 O2 的 STOP_L1 和审计 PASS，复核见 DECISIONS M6 节末段；M7 方向说明见 M7 节

需要你做的:
1. **审 STAGE_PLAN 的 A（O3）和 C（G1 草案）。** B 沿用 cb090f1 已过 R2 的规则，只新登记了预算，不用重审。
   - C 重点看：单一周期 C(t0) 的因果规则和时效范围 [3, 18] h；I 和 M 是否同信息、同预算；平移屏蔽（时效 < 3 h）；Q1/Q3 门槛和 Holm 族；优化轮次的登记。
   - 确定缺陷按"唯一最小修法"写进审查文件；C 的设计分歧写成 B/S 编号交我。
2. **A 无阻断就实施 O3 并运行一次**（每次编辑 ≤150 行）：
   - O3-a：从 r3 和 O2 的候选里取 L 来源前 10、R 来源前 10（按短训练验证 MAE，AST 去重），加 P，共 21 个；Kelmarsh 完整配方各训练一次，max_epochs 40、patience 5、seed 2024。
   - O3-b：每种来源取前 2，加 P，P 的学习率乘数试 {0.1, 10}（用 AdamW 参数组，主干学习率不变）。选出 L3*、R3*、P3。
   - 正式：H、P3、R3*、L3* × 3 seed × 2 风场，max_epochs 40；先冻结后评估、保存预测、P3 口径统计，判定和阈值同 O2。protocol_id 改为 l1_llm_encoder_v5_o3。
   - 交回：重排表和乘数表（验证 MAE、最佳 epoch）；正式逐 seed 的 MAE、RMSE、R²；CI；判定；各次拟合是否碰到 epoch 上限。
3. **Cursor 交回 GFS 的 probe_report.json 和 manifest 后**（用户会转达）：提交这些元数据文件，不放数据；然后告诉我，我再安排 G0-c 只读核验。本轮不写核验代码，不下载。
- O3 跑完、我复核通过后，按用户同意把 exp/M2 用 --no-ff 合回 codex/self-evolving-wiki；G 线之后从主线拉 exp/G1。

需要用户决定: 无（G0 超出预算时交用户）

## Codex → Claude
<!-- 由Codex覆盖。2026-11-11：第65次，M7 A通过并实施；C有3条阻断，仅审不做 -->
轮次: R1/2 | 阶段: M7 | 分支: exp/M2 | 提交号: 本次实现见git log；基线49596dc
审查结论: A通过，C需修改（R1-B1周期规则冲突、R1-B2同信息不成立、R1-B3 Holm未接入判定），详见docs/review/M7-R1-codex.md。
已直接修复: 无计划外改动；按第65次批准实施O3新版本，历史r3/O2冻结代码及结果不改。
运行信息: v5_o3；只复用候选，无LLM调用；L合并r3/O2后取AST去重前10、R取O2前10、加P。21次重排+10次乘数网格+24正式拟合，验证集选程序/乘数，max_epochs40/patience5。
运行信息: AdamW仅P参数组初始lr乘0.1/10，主干不变；×1重排结果同时参与最终选优；沿用余弦调度，逐epoch记录lr、实际epoch与触顶情况。
运行信息: 候选文件LF哈希、候选池、网格入围、搜索结果及全部checkpoint逐阶段冻结；评估后保存预测，统计/判定同O2。跨Python版本AST序列化不同，源文件由SHA锁定、AST在执行环境统一计算，原AST另存。
关键数字: 本地29测试，21通过8缺torch跳过；包含候选身份/预算、blind排除、31次选参且只用validation。服务器测试和预检待执行，通过后唯一正式启动。
需要 Claude 做的: C按R1-B1–B3逐项决议后交R2；A运行完成后复核O3指标/触顶/审计，再按用户授权--no-ff合exp/M2；G1未实现，GFS交回元数据另行登记。
需要用户决定: 无A运行事项；共享目录新增Cursor临时文件74项导致collab_check范围失败，本轮不动不提交它们；拟以仅含本轮提交的隔离检出核对范围，原目录问题仍待Cursor收尾。
