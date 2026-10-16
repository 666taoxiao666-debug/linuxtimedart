# 实验版本、参数改动与恢复索引

本文件是查找和恢复实验的入口，不替代原始训练日志或冻结协议。
首次整理：2026-10-16。当前已核验范围是Kelmarsh小时数据24→4的v19—v22；
更早的SDWPF、72→2及Wiki试验仍在原报告和日志中，尚未逐一加入此索引。
各轮效果和完整诊断见[数值底座报告](KELMARSH_NUMERIC_FOUNDATION.md)。

## Git实际保存了什么

- 已提交的模型/训练代码、配置JSON、测试和文档均有历史版本。
- `source_commit.txt`记录启动时的提交，而训练后的说明文档可能在另一个提交中。
  恢复模型应使用下文的**运行提交**，不能把最新文档提交误当成训练实现。
- Git不自动保存未提交改动，也不会自动备份被忽略的数据、日志和检查点。
  文件SHA只证明身份和完整性，不等于文件已经备份。
- 原实验目录的`protocol.json`、`source_hashes.json`、`manifest.json`、`trained.json`
  和`completed.json`还记录了有效配置、源码、窗口/Scaler、初始化检查点及产物身份。
  有这些记录仍须保留原CSV和实际`.pt`文件，只有哈希不能重建权重。

## 最近四轮索引（运行提交已只读核对服务器记录）

共同设置：Kelmarsh 2016小时SCADA，24小时输入、4小时输出，stride3，fold0/1/2、
seed2024，原7通道与原始标签，Wiki关闭。下面的“收益/失败”是研究结论，
不等同于进程退出状态；四轮均正常执行完成。

### kelmarsh-24h4h-v19：匹配的小时预测基准

- 运行提交：`ad2b51a8e9b4e181d834c4c4c9650aa869565ac3`。
- 变更：建立完整24×7输入、4步输出的匹配基准；Persistence、DailyPersistence、
  Ridge、DLinear、PatchTST、PromptTimeDART共18个组合。
- 训练分配：outer-train前60%拟合，随后10%内部选epoch，末30%当时保留不用。
- 冻结配置：`configs/kelmarsh_24h4h_protocol_v19.json`。
  SHA256：`2330f209f6af0fdba123244c8d8a997471cff5d2c0ac538da866f0aa24fd3474`。
- 启动入口：`bash scripts/train/Kelmarsh_launch_24h4h.sh`。
- 原结果：`outputs/logs/Kelmarsh/20261016/24h4h_f0-1-2_s2024_y7r1Jh`。
- 完成清单SHA256：`4ab532757c62f7ca064b4d41f15f232c7583b67163f1c3639a25650da557a6e0`。
- 本地最小证据：`output/kelmarsh_hourly/24h4h_v19_review/`。
- 状态：已外层评估，结果保留，不重跑已完成组合。

### kelmarsh-24h4h-v20：补齐近期训练数据

- 运行提交：`b2e72d39ad860e195b67fd1c5a7206d61ac306ee`。
- 对照：v19。唯一机制变化是训练分配改为outer-train前90%拟合、后10%内部选择。
- 不变：Scaler仍用原60%前缀，验证队列、标签、网络尺寸、loss、LR、epoch/patience、
  seed和评估方式均不变。更多拟合样本会增加每epoch优化步数，不能称计算量匹配。
- 冻结配置：`configs/kelmarsh_recent_train_protocol_v20.json`。
  SHA256：`600ef0bf4bc21ae59dc1a6701c6948ab0ac234936e205e3035208e384d9ebc9a`。
- 启动入口：`bash scripts/train/Kelmarsh_launch_recent_numeric.sh`。
- 原结果：`outputs/logs/Kelmarsh/20261016/recent24h4h_f0-1-2_s2024_FbpKqP`。
- 完成清单SHA256：`3963dda096e32d517a6c209721b033596952111023e2a60b16f6bf62e5cfe1d9`。
- 本地最小证据：`output/kelmarsh_hourly/recent_numeric_v20_review/`。
- 依赖：保留v19对照产物。状态：已外层评估，TimeDART退化，其他若干基线改善。

### kelmarsh-24h4h-v21：功率特征恒等旁路

- 运行提交：`5006137126fe7fac9ceafe341507d188d8d31236`。
- 对照：v20 TimeDART。唯一机制变化为预训练功率token恒等旁路加原mixer的零初始化
  残差，不增加参数或随机数抽取。
- 不变：原数据、90/10分段、Scaler、loss、LR、训练预算和epoch选择；相同三折v20
  预训练缓存仅复用，不加载v20微调权重。
- 冻结配置：`configs/kelmarsh_power_anchor_protocol_v21.json`。
  SHA256：`b0c83ac5bf96179865d635d1b1f846756b0d148ab278490b63c13e6c5bdca17b`。
- 启动入口：`bash scripts/train/Kelmarsh_launch_power_anchor.sh`。
- 原结果：`outputs/logs/Kelmarsh/20261016/power_anchor24h4h_f0-1-2_s2024_8FUYCF`。
- 完成清单SHA256：`fb23ab5551255cd93201009e4bddf6caa05e556a24a20970581c76606cc54afc`。
- 本地最小证据：`output/kelmarsh_hourly/power_anchor_v21_review/`。
- 依赖：v20完整原结果及三组预训练；预训练SHA在配置中。
- 状态：内部门控通过后一次评估全部三折；较v20改善，但未胜过DLinear，配对检验
  不支持显著收益。不得把它写成稳定高精度或Wiki贡献。

### kelmarsh-24h4h-v22：冻结预训练表示消融

- 运行提交：`dd7e1d07c1140ad1735796d815fbe200edf3f3b7`。
- 对照：v21。唯一机制变化为冻结662019个表示参数，仅训练134063个预测侧参数。
- 不变：forward、总参数、初始化、dropout模式、数据/Scaler、loss/LR、预算及内部
  选epoch；仍复用相同v20预训练，不加载v21微调权重。
- 冻结配置：`configs/kelmarsh_frozen_transfer_protocol_v22.json`。
  SHA256：`13d36296551a3e744badd491955a5ca44247abe57fad71d1ee212275229ce594`。
- 启动入口：`bash scripts/train/Kelmarsh_launch_frozen_transfer.sh`。
- 原结果：`outputs/logs/Kelmarsh/20261016/frozen_transfer24h4h_f0-1-2_s2024_iP54nZ`。
- 完成清单SHA256：`e154043bcdcb3cf317375ea0075fb232de4bc5f0dcb78feb9e7b6d5ea7eccd6f`。
- 本地最小证据：`output/kelmarsh_hourly/frozen_transfer_v22_review/`。
- 依赖：v21完整原结果以及v20完整原结果/预训练。
- 状态：正常完成但内部门控失败；没有v22外层预测/指标，不替代v21，不追加本轮搜索。

## 每轮以后必须留下的变更记录

开始新训练前先在本索引增加新条目，并确认以下记录已落盘；完成后补结果和决定。
这里只规定记录流程，不会自动修改或启动训练。

1. 唯一实验ID、父实验、科学假设/修改原因、改动文件，以及参数旧值→新值；同样写明
   保持不变的标签、时间切分、Scaler、学习率、预算、种子和对照。
2. 先提交并保存**运行提交完整SHA**，记录工作区是否有未提交的代码/配置改动；
   如有，保存真实补丁及身份，不能只写HEAD假装代码已完全固定。敏感信息不得入库。
3. 保存原始配置和展开后的有效配置、完整启动命令与显式环境覆盖，启动后关联实际PID、
   时间、日志目录；计划命令和实际执行命令必须区分。GPU设备编号不等于可抢占授权。
4. 在**实际训练环境、启动前**保存Python/依赖版本（pip与Conda，含可编辑/VCS来源）、
   PyTorch、CUDA运行时/驱动、设备、确定性开关和线程设置。不得让环境采集初始化CUDA
   抢占等待中的GPU；应使用包元信息及只读设备查询。
5. 保存数据路径及SHA、聚合/清洗规则版本、特征顺序、窗口/切分/Scaler身份、初始化
   检查点SHA和所有缓存依赖；输出best/last与完成清单，并保留实际文件。
6. 完成后关联选择epoch、指标口径/单位/原始预测、对照、统计检验、负结果及下一决定。
   内部结果与outer结果分开；后续文档提交不能改写原运行的代码身份。

这些信息尽量写入原训练目录已有的protocol/manifest/日志，不平行维护另一套参数真值。
若协议把产物目录封存，则补充审计记录应放在独立sidecar，不改原完成清单或历史文件。

## 怎样查看和恢复某一版

只读查看历史，以下命令不切换当前工作区：

```bash
git log --oneline -- configs/kelmarsh_power_anchor_protocol_v21.json models/PowerAnchoredTimeDART.py
git show --stat 5006137126fe7fac9ceafe341507d188d8d31236
git diff b2e72d39ad860e195b67fd1c5a7206d61ac306ee 5006137126fe7fac9ceafe341507d188d8d31236 -- models scripts configs
git show 5006137126fe7fac9ceafe341507d188d8d31236:configs/kelmarsh_power_anchor_protocol_v21.json
```

需要恢复源码时，优先在**独立目录**恢复，保留当前分支和未提交改动。例如：

```bash
git worktree add --detach ../TimeDARTFirst-replay-kelmarsh-v21 5006137126fe7fac9ceafe341507d188d8d31236
```

这是供明确需要恢复时使用的示例，本次没有执行，也不是一键启动训练。
不要使用`git reset --hard`覆盖现在的工作。创建独立源码目录后还需：

1. 按原配置提供相同数据、原缓存和环境；核对SHA，不把最近的新数据或新权重冒充旧版。
2. 检查完整原结果的`completed.json`及各文件哈希。本地review目录是最小证据集，
   未下载的`.pt`或大图不代表服务器文件损坏，但也不能称已在本地完整备份。
3. 已完成的旧实验通常只需读结果/加载原检查点，不要用`--resume`伪装成重训练。
   `--resume`仅用于同一身份、真正未完成的阶段；旧源码哈希锁不得放宽。
4. 若确需从头复现实验，另建结果命名空间并保留所有缓存对照，先核对启动器/指针行为；
   不删除旧指针、旧目录，也不直接覆盖旧结果。旧原始入口可能拒绝重复启动，这是保护。

## 当前可恢复程度与已知缺口

2026-10-16已只读核对服务器四轮`source_commit.txt`，与上方运行提交逐项一致；
本地也逐项核对四份配置SHA、完成清单SHA，原报告中的独立指标/源码核验记录保留。

**源码和冻结配置可恢复；完整数值复现依赖数据、权重、环境仍然可用。** 最近四轮
自定义Kelmarsh目录未找到独立的启动时环境快照，不能补采今天的环境后称作当时环境。
通用`run.py`路径已有`utils/experiment_audit.py`的环境/参数审计，但不应推断所有
独立Kelmarsh启动器都自动接入了它。本文件没有修改这些已冻结的训练实现。

因此目前不能保证从零重训得到逐位相同结果。数值库/GPU差异也可能产生小浮点差异，
例如v22已实测CPU/GPU内部MAE/RMSE有约1e-6 kW量级差异。恢复原检查点预测与从零
重新训练是两种不同任务，应分别报告。数据和`.pt`若只在服务器保留，仍需独立备份，
Git仓库与最小review包都不替代它们；本次未上传大数据/权重或创建额外备份服务。
