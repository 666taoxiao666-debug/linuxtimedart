# kelmarsh_forward_block_residual_diagnostic_v27：审查、运行与结果交接

日期：2026-10-25。给 Claude 的单文件入口。请始终使用完整协议 ID，服务器可能有其他名称含 v27 的实验。

## 当前状态

**全部完成，独立复核通过，协议结论为 `STOP_RESIDUAL_LINE_ON_THIS_DATA`。** 44 fits、396 epochs，2026-10-25 20:48:21–20:55:18（北京时间）运行约6分57秒。没有原 outer/test 预测，没有追加 seed 或修改门槛，没有重跑旧实验。

## 审查结论与修改

原稿方向合理，但存在需要运行前修正的问题。详见 `docs/FORWARD_BLOCK_RESIDUAL_DIAGNOSTIC_REVIEW_20261025.md`。修订仅发生在本轮新增文件，旧 v25/v26 文件与结果保持不变。

1. 原通用数据类会提前工程化原 selection/outer 数值。新 `FitPrefixSeries` 先读时间戳，再跳过前缀外数值行；元数据网格/NaN 占位保留登记窗口编号。真实窗口哈希完全通过，无默认全折 scaler。
2. 原规则可能用集成预测的证据放行单 seed。非集成路线现在额外要求预先固定 seed2024 自身满足 R5：3/4 步 MAE/RMSE 联合改善、合并 MAE 收益区间下界>0、首小时两指标非劣。
3. R2 改为“平均的误差抵消收益”。MAE 凸性使平均预测天然不劣于平均单 seed 损失，所以 R2 不能证明方差是主因；R4 也不能证明 DLinear 已近最优。
4. 完整控制器与依赖进入训练身份，完成入口校验最终产物；报告核对全部规则、配对统计及记录。增加广播拒绝和篡改测试。
5. 旧分析中的强因果归因已纠正：低种子相关不能推导随机噪声占比；混合 MAE/MSE 既不严格预测中位数，也不必然产生负 MBE。数据前缀曾被较早开发检查，不是独立确认集。

## 精确身份

- 分支：`codex/self-evolving-wiki`
- 训练实现提交：`6c676785bc508f1b1b47f255cab5b04446ee834a`
- 协议：`configs/kelmarsh_forward_block_residual_protocol_v27.json`
- 原稿协议 SHA（从未运行）：`702f4e537c737b1fef7165668da981a34911043aef0d03db6007ab0fef76a2ec`
- 审查后协议 LF SHA：`b9918206a61ca655957212fd03f4831dc9eda1d032cfe45ef7b328b154d6e4b4`
- 服务器训练代码指纹：`f24d4566515ceecac7424489b5fcce4dbe6f00dcf2ed30059bc81476343e96e6`
- 完整源码指纹：`36ad225a347608f28a5ac153054b933ff19ef4ba925dced779742184c6094f19`
- 真实 fit 队列 SHA：`9c843ac52086be031af7e2e635116a872047fffab76d8ed22a260888cd5e0922`
- 服务器根：`/home/XT/nuist/pythoncode/TimeDARTFirst`
- 结果目录：`outputs/logs/Kelmarsh/20261025/forward_block_residual_v27_tsE4u0`
- 指针：`outputs/logs/Kelmarsh/forward_block_residual_latest.txt`
- 启动日志：`outputs/logs/Kelmarsh/launcher/forward_block_residual_run_20261025_204813.log`
- 初始 launcher PID794977、Python PID794993；后续以实际状态为准，不能根据旧 PID 启动重复运行。
- 预检：`outputs/logs/Kelmarsh/forward_block_residual_preflight/20261025T124422Z_36ad225a3476/preflight.json`
- completed.json SHA：`75fc6d6140de4e1a7dbe839d44e09f46f7b93b68d680bbd258c960cb532834ee`
- 下载证据 tar.gz SHA：`6857f2ae5700398382314f81980889239d8d7bc25a8c0278123e56fc20068fb5`

## 架构、配方和数据预算

24 小时 × 7 通道输入，直接预测未来4小时，stride3，六风机。每步先训练 DLinear（200参数）并冻结，再训练 AnchoredFull 与 AnchoredPowerOnly，各5 seeds（2024–2028）。残差为2-block TSMixer、hidden32、dropout0.1、LayerNorm eps0.001、3618可训练参数，零输出头初始化；锚点不被更新。

每步用过去90%时间段 fit、随后10%内选 epoch；评估下一连续块。步内 scaler 只拟合 fit 截止前行。AdamW lr0.001/wd0.0001，batch64/eval128，0.8 MAE+0.2 MSE，cosine T20，最多20epoch/patience5，clip1。残差 epoch0 只记录和检查身份，不参与选择；锚点可选 epoch0。四步共44 fits，最多880 epochs。

| 评分块 | fit窗口 | 步内selection窗口 | 评分窗口 | 评分issue时间数 |
|---|---:|---:|---:|---:|
| 3 | 1585 | 202 | 947 | 165 |
| 4 | 2487 | 252 | 930 | 155 |
| 5 | 3321 | 354 | 844 | 152 |
| 6 | 4081 | 444 | 748 | 143 |

总评分3469窗口、615个issue时间。各步过去历史重叠，不能当四次完全独立复现。评分块均在2016年6月至9月，不能检验夏秋漂移。没有 Wiki，没有原 outer/test 评估，也不修改旧协议。

## 验证与资源

- 本机 Python/Bash 语法通过；Windows unittest 被旧 Linux `fcntl` 依赖阻断，未伪报本机测试通过。
- 服务器临时副本：本轮21项测试全通过。
- 正式预检：87项测试，86通过、1项因缺少可选 v20 本地镜像跳过；真实数据 B0、命名唯一性、脚本语法、合成配对训练全部通过。
- 预检期间其他用户进程存在；正式启动前已自然退出，GPU 80 MiB/24576 MiB、0%利用率。没有停止或修改其他进程。训练实际显存早期约432 MiB。设备决定与资源记录保存在每步 `step_device.json`。
- 服务器两份原有文档修改保留：`docs/EXPERIMENT_REPLAY.md`、`docs/KELMARSH_MODEL_REVIEW.md`。

## 结果及独立复核

独立 NumPy 审计 **807/807通过**：完成产物哈希、44 fit/396 epoch、选择 epoch、实际锚点绑定、Full/PowerOnly初始化、前向边界、逐步/逐seed误差、MAE与RMSE配对区间、Newey–West p值、全部规则逐项复算。审计不导入项目模型/统计模块，不依赖 torch，不连接服务器。它核对登记的锚点摘要，但不自行反序列化权重；训练器已逐一校验 best/last 冻结锚点实际状态。

以下是四个前向评分块合并后的 **window-uniform训练诊断指标**，不是原 outer 宏平均，不能与 v26 outer 的186.50直接比高低；R²不是百分比准确率。

| 模型 | MAE kW | MSE kW² | RMSE kW | R² | MAE Skill % | RMSE Skill % |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 176.791164 | 66595.010778 | 258.060091 | 0.655121 | 0 | 0 |
| 每步重训 DLinear | 180.139458 | 67567.222140 | 259.936958 | 0.650086 | -1.893926 | -0.727298 |
| AnchoredFull，5 seed平均预测 | 179.268656 | 67160.856595 | 259.154117 | 0.652190 | -1.401366 | -0.423942 |
| AnchoredPowerOnly，5 seed平均预测 | 180.260801 | 68356.598144 | 261.450948 | 0.645998 | -1.962563 | -1.313979 |

Skill参照同队列 Persistence，负值表示更差。所有行使用同3469窗口，不丢弃受损块。

| 评分块 | Persistence MAE/RMSE | DLinear MAE/RMSE | Full平均 MAE/RMSE | PowerOnly平均 MAE/RMSE |
|---|---|---|---|---|
| 3 | 176.532 / 264.820 | 194.425 / 279.989 | 193.496 / 278.510 | 191.968 / 278.365 |
| 4 | 187.729 / 267.212 | 183.063 / 258.760 | 183.239 / 260.070 | 187.014 / 265.905 |
| 5 | 174.219 / 243.364 | 174.846 / 243.431 | 173.757 / 242.706 | 174.088 / 243.753 |
| 6 | 166.422 / 253.885 | 164.391 / 252.971 | 162.539 / 250.416 | 164.007 / 252.914 |

最早评分块的锚点比 Persistence 差17.892 kW MAE，Full只收回0.929 kW，仍差16.963 kW。因此“相对锚点有一点收益”不能取代与 Persistence 的比较。短前缀训练、很小的内选段、工况变化都可能参与，不能从这一个现象直接断言数据脏或某架构永远无效。

### 冻结规则

| 规则 | 结果 | 原因 |
|---|---|---|
| R1 迁移 | 不通过 | 3/4步联合优于锚点，但MAE收益 +0.870802 kW，95%CI [-0.086594, 1.852578] 跨零；NW p=0.078420 |
| R2 平均抵消 | 通过 | 4/4步平均预测MAE优于平均单seed MAE；这符合凸性，不能证明方差主因 |
| R3 辅助归因 | 不通过 | 3/4步联合优于PowerOnly，但MAE收益 +0.992145，CI [-0.145140, 2.160179] 跨零；NW p=0.087348 |
| R4 首小时 | 通过 | 相对锚点MAE改善1.042633、RMSE改善1.318161 kW；首小时仍不如Persistence |
| R5 固定单seed | 不通过 | seed2024仅1/4步联合改善，合并MAE比锚点差2.171705、RMSE差3.343750；首小时RMSE也略差 |

Full相对锚点RMSE收益 +0.782841 kW，CI [-0.475089, 1.991429]；Full相对PowerOnly的RMSE收益 +2.296830，CI [0.777574, 3.863669]。后一个区间虽为正，不能替代预声明的MAE归因门槛。固定seed2024的MAE收益区间 [-4.535312, -0.050108]，但NW p=0.057760；如实并列，不挑有利检验。

### 逐小时（合并窗口）

| 预测小时 | Persistence MAE/RMSE | DLinear MAE/RMSE | Full平均 MAE/RMSE |
|---|---|---|---|
| +1h | 120.757 / 179.574 | 131.415 / 192.557 | 130.372 / 191.239 |
| +2h | 165.236 / 238.982 | 174.555 / 246.233 | 172.691 / 244.247 |
| +3h | 194.813 / 275.201 | 196.831 / 276.028 | 196.949 / 275.489 |
| +4h | 226.359 / 318.253 | 217.757 / 310.433 | 217.063 / 310.678 |

Full主要在第四小时优于Persistence，前两小时仍明显不足；不能事后按小时切换模型或调损失来使本轮转正。

### 实际选择epoch

每行残差顺序为seed2024/2025/2026/2027/2028：

| 评分块 | DLinear | Full | PowerOnly |
|---|---:|---|---|
| 3 | 18 | 1/1/3/5/2 | 4/2/1/1/3 |
| 4 | 17 | 10/1/1/15/3 | 10/4/2/13/8 |
| 5 | 8 | 1/1/2/4/1 | 1/1/2/1/2 |
| 6 | 17 | 2/1/3/1/3 | 2/1/1/2/1 |

### 下载证据和图

本地根：`D:/Users/XT/Desktop/moxing/gpt/v2/TimeDARTFirst/output/kelmarsh_hourly/forward_block_residual_diagnostic_review/`。

- `run/`：服务器完整运行产物，含协议、训练记录、权重、预测、报告、完成清单。
- `independent_audit.json`：807项复核与重新计算的总体/逐小时指标。
- `audit_forward_block_run.py`：独立审计脚本；运行 `python audit_forward_block_run.py run --output independent_audit.json`。
- `preflight.json`：正式服务器预检。
- `run/figures/step_gains.png`、`lead_gains.png` 已实际打开检查，无截断。前者分步增益；后者是四步**等权平均**增益，不能与上表**按窗口合并**增益混同。

监控阶段一次跳板 SSH 断开，Tailscale仍有回应；随后 `ssh -F none` 直连100.89.238.37:22成功并核实完成。没有修改 SSH/Tailscale/服务器配置，没有因为断连重启训练。Claude云沙箱未必有该网络路线或本机密钥，应优先使用离线证据包审计。

## 后续约束

完成后不能重复运行。STOP 则保留负结果，停止在这批2016开发窗口上提出残差变体；即使通过，也只允许另行设计一次未来确认，不能自动加 seed、调阈值或请求 outer。R3只支持本消融条件下的辅助通道归因，不是因果证明。源代码可复刻不等于不同GPU/库版本逐位复刻，必须保留本次环境和设备记录。
