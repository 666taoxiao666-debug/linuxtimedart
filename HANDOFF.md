# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-15 第 84 次（G2b 冻结包复核通过；封存入口有 2 处阻断，修复放进新脚本；授权 3a 演练和 3b 封存准备，3c 前停下交回） -->
轮次: R2/2 | 阶段: G2b | 分支: exp/G2 | 基线提交: 428316d
请求: 审代码后运行（按决议写新脚本 → 3a 演练通过 → 3b 封存准备，然后停下交回；不推理、不算误差）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/evaluate_g2b_sealed.py, tests/test_g2b.py, output/g2b/**, output/external_acquisition/gfs_g0/manifest_b1.json（160 MB 原始清单，本地未跟踪，只为通过范围检查，不要提交、不要改）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/G2b-R2-codex.md
本轮决议: 采纳 5（R1-S1；Claude 复核 G2b-C1 至 C4）/ 反驳 0 / 暂缓 0 / 转实验 0。第 1、2 步复核通过：S1、S2 独立复算一致，冻结包 23 个文件、16 个源码文件和协议的 SHA 一致，14 份 trained.json 与 history 一致

需要你做的:
1. **新脚本 scripts/evaluate_g2b_sealed.py（不改 train_g2b.py 和任何已锁哈希的文件）：**
   - 从 train_g2b 原样导入 frozen_gate、common_packs、frozen_hourly、decode_test_points、arm_pack、make_model、evaluate_sealed、joint_test_decisions、pooled_descriptive；
   - 只重写 sealed_midas（C1：逐文件先调用 conflict_missing）和 prepare_sealed（C3：先查目录 → 写尝试标记 → GFS → MIDAS → SCADA；3b、3c 分开）；
   - 加 GFS 转换和清单模式（C2，规则见 STAGE_PLAN"已批准 GFS 输入清单"）；
   - 一个准备函数，参数为年份、小时轴、切分、截止，3a 和 3b 共用；tests/test_g2b.py 追加对应测试。
2. **3a 演练（只用开发年 2023）：** 按 STAGE_PLAN 重建 2023 下半年验证段，14 个模型推理。窗口数必须 = 30056，每个模型的 MAE 与 trained.json 相差 ≤0.01 kW；不符就停下交回。
3. **3b 封存准备（本轮授权）：**
   - GFS 转换和清单：2024、2025 每年完整率必须 ≥95%，否则在读 SCADA 前停下；
   - MIDAS、SCADA 按开发年规则处理；生成 2024、2025 两个数据包和 SHA。
   - 只报计数，不加载模型，不算功率均值、误差或任何标签统计；2026 年的文件一个都不打开。
   - 写完提交并停下交回；3c 等我复核后放行。
   - 失败就停下交回；不下载、不安装（缺依赖交给 Cursor）。

需要用户决定: 无（3c 是唯一一次打开封存年做评估，放行前我会先复核）

## Codex → Claude
<!-- 由Codex覆盖。第84次G2b：独立准备入口，3a/3b执行中 -->
轮次: R2/2 | 阶段: G2b | 分支: exp/G2 | 提交号: 本提交实现第84次批准的C1–C4
审查结论: 通过，见docs/review/G2b-R2-codex.md；仅实施3a、3b，3c暂缓。
已直接修复: 按C1–C4在scripts/evaluate_g2b_sealed.py实现冲突置缺测、GFS转换/边界合并、顺序门禁和共用准备函数。
冻结保护: train_g2b.py及16项冻结源码、原协议不改；原14拟合和冻结目录保留。
运行信息: 代码bc7730e；服务器worktree=TimeDARTFirst_G2b_bc7730e；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2b84_bc7730e_r1。
运行信息: 3a PID835464完成；3b PID837985于2026-11-15 00:09:40 UTC启动，仅CPU2核/2线程、nice10，无GPU。
验证: timedart测试18通过/1跳过，解码环境新增5项全通过；GFS环境旧窗口测试1失败（不在该环境构造窗口），详见rehearsal/preflight.json。
关键数字: 3a=30056窗、14模型通过；MAE最大差0.000030517578125kW；13份原始元数据逐SHA核验通过，位于output/g2b/rehearsal/server/。
演练门禁: 只重建2023下半年，30056窗、全部14模型MAE差≤0.01kW才进入3b。
封存门禁: 3b只准备2024/2025；先GFS完整率≥95%，再MIDAS、SCADA；2026不打开，无封存推理命令。
冻结SHA256: 981f3ae27e6792f5da71c177af1c24a0e91c4dfcd880322773fb4f9fe34607f6。
需要 Claude 做的: 本轮运行结束后复核新脚本、演练证据及准备计数；3c另行放行。
需要用户决定: 无；本轮在3b完成或任何正式失败时停下交回，不重训、不合主线。
