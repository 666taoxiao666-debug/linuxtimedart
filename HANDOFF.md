# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-31 第 13 次 -->

**用户指示（2026-10-31）：** 不要一直在方案上来回对话，要尽快到服务器上跑出结果；对你的审查也要独立判断，不要默认全对。

**我对你第 11 次复审的独立判断：**
- 大部分意见是对的，已在 14.18 修正：q 不能被填充值打开；|d⊥| 丢失左右信息；B0 隐状态会把绝对方向带进查询；H1 必须同头；备选曲线的 v1 可能落在停机箱；K5 的运行模式。
- 有两点不同意：
  - 0.7944 舍入只差 0.04 kW，属于形式问题；
  - D2/K5/曲线这些细节没必要阻塞 K1，因为 K1 只用到已经审定的部分（14.13 的合格 token、零 q 回退、A1 的 D0–D3、κ* 的 J；加上 14.18 的原始有效 q、带符号的 d⊥ 和 u_ref）。
- **因此按用户指示，先放行 K1 的实现与一次运行。** D2 相关条款（K5、曲线、S1/S2、D2 统计）继续由你审 14.18，不影响 K1。

**新代码（我写的，未提交），只用于 K1 = D1 折的 A1：**
- `configs/kelmarsh_arrival_v33_k1.json`：fold D1 = 拟合 2016–20 / 验证 2021 / 开发评估 2022；seed 2024；臂为 B0、B1、B2、A、K0.5、K1、K2、K3、K4（K1 兼作 A-κ1），共 9 次训练，上限 10；决策阈值与 14.13 一致。
- `utils/kelmarsh_station_history_v33.py`（不依赖 torch）：
  - 站点 bank；最新可用 24 h 的 token；
  - 带符号的 d∥/d⊥；u_ref（50 km 内原始有效风向的圆均值）；
  - q 只认原始有效风速和风向；
  - prior_numpy 参考实现；kappa_star；a1_decision。
- `models/UpstreamArrivalResidual.py`：冻结底座上的残差分支。q̃ 只重新分配权重，M 与 B2 相同，头部零初始化，κ 可学习（sigmoid 约束），支持 age0/hflat/geo180 三种变体。
- `scripts/train_kelmarsh_arrival_v33.py`：
  - B0 用原 v28 fit_config 训练（21 通道、平衡损失、按验证 MAE 选模）；
  - 各分支按 J 选 checkpoint，epoch 0 计入；
  - 输出 `k1_decision.json` 和 `k1_summary.md`。
  - 只读 2016–2022 年站点文件，核对 manifest 哈希和 v32 CSV 的 SHA；绝不读 2023/24 年数据。
- `tests/test_kelmarsh_arrival_v33.py`：本地不含 torch 的 9 项全部通过（含你给的两个反例、旋转不变、镜像可区分、严格过去）；3 项 torch 测试需要在服务器上跑（prior 与 NumPy 一致、epoch-0 恒等、置换不变、参数差 1、端到端 smoke）。

**请你做**
1. 审查上述 5 个文件。小问题可以直接修，在提交正文里写明改了什么；改动设计的问题先交审。
2. 在服务器上跑 `python -m unittest tests.test_kelmarsh_arrival_v33`，再跑 `python scripts/train_kelmarsh_arrival_v33.py --smoke --out <tmp>`。
3. 通过后，**真实运行 K1 一次**：`python scripts/train_kelmarsh_arrival_v33.py --out outputs/logs/Kelmarsh/v33_k1_<tag>`。只跑一个实例，不改阈值，不追加臂。
4. 把 `k1_decision.json`、`k1_summary.md`、`data_manifest.json`、各臂的 trained.json 和 history 拷到 `output/kelmarsh_hourly/arrival_v33_k1/`，不要拷权重。在你那一节写出决策分支、各臂的 2022 年 MAE/RMSE、κ*、学到的 κ 和支持率。
5. 提交代码、结果和 HANDOFF。

**审查原则（用户 2026-10-31 澄清，已写进 AGENTS.md）：** 确定缺陷先修再跑；不确定的设计选择用小实验验证后再判断。审查 K1 代码时请写明每条意见属于哪一类。
**边界：** 这是开发证据，不作显著性主张。不读 2023/24 年站点文件，不下载，不碰 Penmanshiel。第 14 节归我；写 HANDOFF 前先从磁盘重读。
## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-31，第13次 K1 审查、执行及结果 -->
**结论：五文件小修后通过；服务器14项测试和独立smoke通过；唯一真实K1完成9/9，决策D3（有效负结果）。**
- 审查分类/证据/修复/验证：`output/kelmarsh_hourly/arrival_v33_k1/review.md`；实现评级B，不表示机制已成立。
- 【确定缺陷，已修/P1】`load_real`补逐文件year，解决旧parser的KeyError；旧冻结加载器未改。
- 【确定缺陷，已修/P2】调用原configure_determinism并登记，恢复TF32关闭等原训练配方。
- 【确定缺陷，已修/P1】scores拒绝非有限/空/错位预测标签，防止通用指标掩码静默缩小ALL队列。
- 【不确定】物理先验效益、κ学习价值、物理阈值均按原方案用K1验证；不因结果修改阈值/网格/臂。
- 上述仅改新runner和测试，依据你第13次“小问题可直接修”授权；proposal第14节未由Codex改写。
- 真实站点限2016–22，36站252文件；窗口训练/验证/2022评估=75024/15680/16949，标签与旧加载器逐项相同。
- 单实例PID4636，UTC06:11:53–06:30:53（北京时间14:11–14:30），9次fit，未重启或追加训练。
- 2022 ALL h1–4合并MAE/RMSE（kW），选中epoch见末列：

| 臂 | MAE | RMSE | best epoch |
|---|---:|---:|---:|
| B0 | 160.911915 | 241.650309 | 11 |
| B1 | 160.911915 | 241.650309 | 0 |
| B2 | 160.911915 | 241.650309 | 0 |
| A | 160.905625 | 241.261684 | 1 |
| K0.5 | 160.911915 | 241.650309 | 0 |
| K1 | 160.911915 | 241.650309 | 0 |
| K2 | 160.911915 | 241.650309 | 0 |
| K3 | 160.911915 | 241.650309 | 0 |
| K4 | 160.911915 | 241.650309 | 0 |

- A对B2/B0：ΔRMSE=0.388625254kW（0.160821335%<1%），ΔMAE=0.006289888kW；B2对B0=0，故D3。
- κ*=1（固定网格验证J均为2，依同分规则选择）；A学到κ=1.010328770，不能当作已识别真实传播速度。
- A物理支持率h1/2/3/4=99.5044/99.6106/99.3333/97.9822%；B0/B1/B2不适用。
- 固定κ支持率（同顺序，%）：K0.5=99.7168/99.7522/99.6106/99.6106；K1=99.6106/99.5398/99.2566/98.0530。
- K2=99.2743/98.0589/89.8401/70.8124；K3=98.9852/89.8814/61.6497/36.3856；K4=98.2064/71.4496/36.8045/13.1040。
- B1/B2/五固定κ均实际训练5epoch后选回epoch0；相同预测是零残差回退，不是漏跑。A训练6epoch选1；B0训练16epoch选11。
- 复算保存预测及B0推理通过；八分支的B0张量未变，A/B2可训参数10515/10514；完整共同队列和选模/决策核验通过。
- 结果21文件已拷回`output/kelmarsh_hourly/arrival_v33_k1/`，逐一SHA256与服务器一致；不含权重/预测数组。
- 详见k1_decision.json、k1_summary.md、data_manifest.json，各臂trained/history及audit/result_verification.json、verify_results.py、测试/启动日志。
- 服务器结果：`/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Kelmarsh/v33_k1_20261031T060631Z`；代码快照/审计在outputs/audits/arrival_v33_k1/20261031T060631Z。
- 审计源码runner SHA256_LF=dd7e03247394f8a1d4cc27259f82659c33ce56219492727c234457bfc3643462；配置318038be…a1384d6，完整身份见audit/launch.json。
- 【后续】按D3只停止从零C1机制臂；请基于负结果决定下一方案。D2/K5/曲线等仍另审，不自动运行K2/K2b或其他路径。
- 这是单seed开发证据，无独立测试/显著性主张；未读2023/24站点、不下载、不碰Penmanshiel；没有新增待实施设计改动。
- 本提交含你新增的AGENTS规则、§14.18、五文件、结果和本节；保留其他未跟踪文件及服务器原有文档改动；提交号/推送状态见git记录与回复。
