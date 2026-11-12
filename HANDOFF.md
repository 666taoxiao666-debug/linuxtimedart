# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-12 第 73 次（G0c 通过；文献核查后，用户同意给 G1 加"本场错位"对照 M_self 和 Q4） -->
轮次: R1/2 | 阶段: G1b | 分支: exp/G1 | 基线提交: 3744f6b
请求: 审方案和代码（G1 协议的改动 + Claude 改的模块和测试），在服务器跑 torch 测试；不训练，不写数据脚本
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md models/NWPDisplacement.py tests/test_nwp_displacement.py
允许改动: models/NWPDisplacement.py, tests/test_nwp_displacement.py, output/external_acquisition/gfs_g0/**, output/gfs_g0/**
审查文件: docs/review/G1b-R1-codex.md
本轮决议: G0c 采纳 R2-S1（我对 B1 的推断错了，已更正；不用 100 m 的决定不变，理由改为层次无法证明）；G0c R2 复核通过。G1b 方向说明见 DECISIONS「阶段 G1b」

需要你做的:
1. **审 G1 协议的改动**（STAGE_PLAN C 节）：
   - 新增 M_self 臂（与 M 结构相同，错位改由风场自身 t0−3…t0−1 观测风估计，也带 28 个站点汇总通道）；
   - 新增 Q4（M 对 M_self），Holm 族改为 {Q1, Q3, Q4}，方法档要 Q3 和 Q4 都过；
   - 观测校准改为拟合年仿射校准 o' = (o − a)/b；共 36 次拟合。
   - 设计分歧写成 B/S 编号交我；确定缺陷按唯一最小修法处理。
2. **审代码**（models/NWPDisplacement.py）：self_site_batch（风场作为唯一"站"，预报取 gF 的前 8 个时刻 t0−5…t0+2，权重为 1）、station_blocks 的 M_self 分支、validate_batch 的可选键 oF/oF_mask、StationBlocks 和 G1TSMixerModel 支持 M_self。
3. **在服务器跑** `python -m unittest tests.test_nwp_displacement`：21 项，含 3 项 torch；一致性和梯度测试已包括 M_self。
4. B1 下完由用户转达后，再做 B1 正式核验：10 m 风身份接受 165/166（B1）或 131/132 加 level-10 过滤（B2）；2024 年以后只核字节和哈希。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-12：第73次，G1b模块21项服务器测试通过，校准方案有1条阻断 -->
轮次: R1/2 | 阶段: G1b | 分支: exp/G1 | 提交号: 本轮见git log；基线3744f6b
审查结论: 模块与Q4扩展通过；方案1条阻断R1-B1，另1条暂缓R1-S1和1条建议R1-S2；docs/review/G1b-R1-codex.md。
已直接修复: 无；本轮模型及测试均为Claude修改，Codex仅审查和合成测试，未代选校准失败策略。
运行信息: /tmp/g1b_review_73；OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 timedart/bin/python -m unittest -v tests.test_nwp_displacement；CPU合成测试，不训练。
关键数字: 服务器21/21通过、无跳过，0.519秒；含3项torch测试及M_self numpy/torch一致性、梯度和功率通道检查。
关键数字: M与M_self均35通道/6结构参数；Q4加入Holm三检验族正确；计划36=6臂×3seed×2风场，本轮未执行拟合。
关键数字: 校准反例g=[1,2,3]、o=[1,0,1]，OLS a=2/3、b=0，逆变换除零；方案尚未定义退化拟合/无效斜率的处置，不能据模块测试通过放行数据准备。
运行信息: 模型SHA256=c6d9a701d4c7d137aa8dd4dbce6287189f797b5583a9f4552994b4ca322ab9f4；测试SHA256=0b60132c7dcb8d0660161e2047025f7c471fa1dfaa85a3cd81fbca552a9a453e；两项本地/服务器一致。
运行信息: 只复制模块/依赖及合成测试至隔离目录；不改既有部署、不读真实/封存数据、不写数据脚本、不做B1全量核验。
需要 Claude 做的: 对G1b R1-B1决议并补全仅用拟合段的配对/退化检查及失败处置；修法不唯一，未先行实施。下一轮只复审该改动。
需要 Claude 做的: R1-S1留到优化阶段处理M_self同步公平性；R1-S2可在后续数据入口按臂检查oF/oF_mask成对必需。B1下载完成后另行正式验收。
需要用户决定: 无。
