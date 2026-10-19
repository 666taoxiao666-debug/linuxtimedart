# 实验版本、参数改动与恢复索引

本文件是查找和恢复实验的入口，不替代原始训练日志或冻结协议。
首次整理：2026-10-16；2026-10-19补记v23最终结果并新增v24（已运行，内部门控失败）。
当前已核验完成范围是Kelmarsh小时数据24→4的v19—v24（v22、v23、v24内部门控均失败）；
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

## 最近实验索引（运行提交已只读核对服务器记录）

共同设置：Kelmarsh 2016小时SCADA，24小时输入、4小时输出，stride3，fold0/1/2、
seed2024，原7通道与原始标签，Wiki关闭。下面的“收益/失败”是研究结论，
不等同于进程退出状态；v19—v23均正常完成（v23原排队曾按明确授权切换为有界共享训练）。

### kelmarsh-24h4h-v24：匹配调度与预训练价值因子实验（已运行，内部门控失败）

- 动机：2026-10-19对v19—v23的独立诊断。（a）`CosineAnnealingLR(T_max=20)`从未走完：
  已记录的运行都在epoch 1—11内停止或达到最优，等效学习率为未规划的高LR平台；
  （b）预训练与“有无预训练”此前混在一起，且只有单seed，预训练价值尚未被证明；
  （c）v21的修正量对残差斜率仅0.48—0.63，说明单次运行方差大于偏差。
- 父对照：v21（架构、数据、loss、LR、预算、选择规则逐项不变）；DLinear取自v20同队列。
- 因子：3臂×3折×3 seed（2024/2025/2026）=27次运行。
  `PretrainedOriginal`（v21原配方，epochs=20，作参照）；`PretrainedMatched`
  （epochs=8，唯一改动是余弦周期与实际运行长度对齐，8取v21三折实际长度8/8/6的中位数，
  实验前声明、不调参）；`ScratchMatched`（与上一臂完全相同，仅不加载预训练权重）。
  预训练仍是固定的seed 2024三折缓存，seed 2025/2026只改变微调阶段，不度量预训练方差。
- 配置：`configs/kelmarsh_matched_recipe_protocol_v24.json`，SHA256
  `0e4a97e5f414289af1ce77ab8eb263110986cf7de04a7a23fbe535be72d570ea`。
- 实现：`scripts/train_kelmarsh_matched_recipe.py`、`tests/test_kelmarsh_matched_recipe.py`、
  `scripts/train/Kelmarsh_launch_matched_recipe.sh`。**未改动任何已冻结文件**，
  `models/`、`layers/`、`utils/`、`data_provider/`及既有训练脚本字节不变，v19—v23源码锁不受影响。
- 入口：`bash scripts/train/Kelmarsh_launch_matched_recipe.sh --check`只做CPU校验
  （协议/参考完成清单/三折预训练SHA/窗口身份），不占GPU；不带参数才会启动训练；
  指针`outputs/logs/Kelmarsh/matched_recipe_latest.txt`，另有`--status`、`--resume`。
- 门控（仅内部选择段，外层之前）：各折三seed均值后，主候选对参照臂至少2折同时改善
  MAE与RMSE，且宏平均MAE/RMSE均优于参照臂与Persistence；DLinear只报告不门控。
  通过才对全部三折评估主候选的三个seed及其seed均值，并与v21、DLinear配对；
  失败则保留负结果，不追加epoch数、seed或阈值。
- 另报告：臂间配对差异（调度效应、预训练效应）、seed间标准差、残差斜率（过度离散）、
  以及`PretrainedOriginal` seed 2024对已完成v21内部指标的复现差（只报告）。
- 第一次尝试（失败，保留不删）：运行提交`a681a9d1291919f15bc23afbf70357289568baa8`，
  目录`outputs/logs/Kelmarsh/20261019/matched_recipe24h4h_f0-1-2_s2024-2025-2026_EChSmc`，
  2026-10-19 15:03（UTC+8）经用户授权启动。27次训练全部正常完成，随后汇总阶段抛出
  `TypeError: float() argument must be ... 'generator'`（`arm_summary`里括号放错，属于代码笔误，
  与协议、数据、训练无关），`status.env`为`FAILED`。失败发生在写`gate.json`、`factorial.json`之前，
  因此该目录没有门控结论、没有外层预测；`inner_records.json`和各臂训练产物保留。
  原目录不`--resume`：修复会改变源码哈希，而源码锁不放宽。
- 处理：修复该笔误并把门控/汇总逻辑抽成`summarize_inner`，补充端到端汇总测试和外层评估
  冒烟测试（测试已确认能复现该`TypeError`）；冻结协议、SHA、门控规则、臂、seed和epochs
  一律不变。随后在新的结果目录从头重跑同一协议，训练结果与失败目录对比可检验可复现性。
  第一次尝试中`PretrainedOriginal` fold0 seed2024的内部MAE/RMSE与v21记录逐位一致。
- **重跑结果（2026-10-19，运行提交`d7bf0a208990554ff6474e5a35aef31178dc6a4a`，服务器只读核对）**：
  目录`outputs/logs/Kelmarsh/20261019/matched_recipe24h4h_f0-1-2_s2024-2025-2026_rerun_MdQSBu`，
  正常完成，`outer_evaluated=false`，完成清单SHA256
  `e1f5015ee9a98e054a579b7f4aeac92efc4cd186b8d9cf97cb48e187bf8111e5`。指针
  `matched_recipe_latest.txt`现指向该目录；第一次失败目录保留。
  - 复现：三折`PretrainedOriginal` seed2024的内部MAE/RMSE与已完成v21记录之差均为0.0，
    两次尝试的数值一致，说明该配方在此环境下可逐位复现。
  - **内部门控失败**：`PretrainedMatched`对参照臂联合改善MAE与RMSE的折数为1/3（仅fold1），
    要求至少2折。宏平均内部MAE/RMSE（kW，三seed单模型均值）：Matched 158.870/238.224；
    Original 158.939/238.247；ScratchMatched 159.437/238.463；Persistence 163.197/247.726；
    DLinear（v20，仅seed2024）160.925/239.629。fold2中三臂多数seed的内部最优为epoch 0
    （即退回Persistence），Matched与Original在该折完全相同。
  - 调度效应：Matched−Original的MAE差−0.069 kW（9个单元差的标准差0.150，3个更好）、
    RMSE差−0.023 kW（标准差0.129）。9个（折，seed）中两臂选出的最优epoch完全相同，
    **没有可检出的调度效应**，这与最优epoch集中在1—4、周期长短几乎不影响前几轮的预判一致。
  - 预训练效应：Matched−Scratch的MAE差−0.567 kW（标准差0.688，9个中6个更好），
    RMSE差−0.239 kW（标准差3.654，5个更好）。预训练在MAE上略有帮助，RMSE上无法区分。
    9个单元共享三段数据，不是独立样本，未做显著性检验，不能写成预训练有稳定价值。
  - seed均值读出（不进门控，仅内部选择段）：Original 157.726/236.524；Matched 157.724/236.579；
    Scratch 159.005/237.858。相对Original单seed宏平均降低约1.2/1.7 kW，说明seed方差可观。
    该数字是在已用于选epoch的同一内部段上度量的，偏乐观，不是外层证据。
  - 已知缺陷（仅诊断，不影响门控和指标）：残差斜率在最优epoch为0的运行上修正量接近0，
    分母趋零使均值失真（出现>1e5的量级），因此只有Pretrained臂的seed均值斜率（约1.46）可读；
    且其方向与此前外层估计0.48—0.63不同，说明斜率依赖数据段和选择过程，不是稳定收缩系数。
  - 决定：按协议是负结果。不追加epoch、seed或阈值，不做外层评估。seed集成方向若要验证，
    必须作为新的、预先声明并带门控的实验，不能作为对v24的补救。

### kelmarsh-24h4h-v23：有界共享训练中的通道交互位置匹配实验

- 父对照：v21及v20 DLinear；新增等参数、等初始化的Early/Late分支，只改交互位置。
  两分支参数均862386，原v21为796082，容量不同由Late对照控制，不冒充完全同模型。
- 不变项：原7通道、24→4/stride3、fold0/1/2 seed2024、90/10分段、Scaler、标签、
  loss/LR/预算、相同三折预训练缓存、零残差头及功率直通；没有新的预训练或NWP。
- 配置：`configs/kelmarsh_channel_timing_protocol_v23.json`，SHA256
  `43f17d453a2b52eda6c1d102a72d9d89f7daba9d78c49378a0afb2ae8e58b6a4`。
- 运行提交：`60e7f725c73980d6f841a240247a036d6b0c4e33`；已推送并服务器快进。
- 入口：`bash scripts/train/Kelmarsh_launch_channel_timing.sh`；指针
  `outputs/logs/Kelmarsh/channel_timing_latest.txt`。本地与服务器9项单测及Bash语法通过。
- 原目录：`outputs/logs/Kelmarsh/20261016/channel_timing24h4h_f0-1-2_s2024_qnI0sb`；
  启动日志：`outputs/launch_logs/channel_timing_v23_20261016_152948.log`。
- 2026-10-16 15:29启动的Bash PID81501/Python PID81508：控制器RUNNING，阶段
  WAITING_FOR_GPU，待Early/fold0/seed2024。当次其他计算任务gyx/PID74417/8944 MiB，
  未停止或并行干扰对方。每60秒自查空闲，单次等待上限24小时；不是已开始GPU训练。
- 实际argv、UTC时间、工厂和Git身份见`runtime_snapshot.json`/`runtime_identity.json`；
  采集到Torch2.6.0+cu124、CUDA构建12.4、101项Python依赖和32项Conda元数据。
  原快照保留，续跑拒绝关键依赖变化，不倒填历史环境，不保证逐位复现。
- 当前没有本轮真实训练指标或完成清单；环境/部署快照不能代替完成产物核验。
- 决策：六个内部候选先通过冻结的MAE/RMSE及强基线门控，才一次评估全部六个外层
  候选；未通过不生成新outer精度，不追加搜索。已看外层仍是开发证据。
- 依赖：v21完整结果、v20完整结果/预训练/同队列DLinear；旧实验不重跑。
- 后续授权更正：用户明确要求有界并行而非idle-only。仅替换自己的未训练排队控制器，
  使用`/home/XT/miniconda3/envs/timedart/bin/python -u scripts/run_kelmarsh_channel_timing_shared.py --output <上述原目录>`。
  他人运行时本进程allocator2 GiB/CPU2/每批让步20ms，至少保留4 GiB空闲；独占时解除，
  显存上限不含上下文/驱动，不保证固定算力或零风险。冻结训练协议和源码锁不变。
  新增`shared_execution_contract.json`、`shared_launch_<pid>.json`和`resource_policy.json`，
  保留原启动提交/快照；共享入口源码哈希单独绑定，恢复时仍须核验，不能隐瞒运行条件变化。
- 共享入口运行提交：`d762b3bee86fc6b3a020a45422b5d6508410f8ed`；本地及服务器25项测试通过。
  原PID81508确认未训练后仅结束自己的队列；共享PID85111于2026-10-16 15:43启动。
  日志`outputs/launch_logs/channel_timing_shared_v23_20261016_154258.log`，原结果目录不变。
  已确认Early/fold0跑出epoch1—4，mode=shared_limited；当次本进程显存464 MiB，
  其他用户PID74417仍运行且显存8944 MiB。以上为运行中的快照，非最终精度。
- **最终结果（2026-10-19只读核对服务器`gate.json`/`result.json`，原目录未改动）**：
  已正常完成，内部门控失败，`outer_evaluated=false`，没有v23外层预测/指标。
  宏平均内部MAE/RMSE（kW）：Early 159.387/239.087；Late 158.522/235.750；
  v21参照 158.788/235.684；DLinear 160.925/239.629；Persistence 163.197/247.726。
  候选是Early，Late是位置对照；门控要求Early在同一折同时优于Late与v21的MAE和RMSE，
  实际0/3折满足，宏平均上Early的MAE/RMSE也都劣于Late与v21。Late相对v21的逐折差
  （MAE/RMSE，kW）：fold0 −1.06/+2.54，fold1 +1.04/−1.09，fold2 −0.79/−1.26，
  量级远小于折间离散，不构成稳定收益。内部选择epoch：Early 2/1/1，Late 2/5/1，
  最优点集中在前几轮，而余弦周期仍是20。
  结论：早期通道交互未通过门控；不得追加本轮搜索，不得把它写成收益。
  这一结果也是v24把余弦周期与实际运行长度对齐的依据之一。

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

## 诊断前恢复点（2026-10-19，v24之前）

按独立诊断修改模型前，已在提交`0e39d731ac32beac8875be2e530c299f827c25ce`（v23运行记录）
处固定恢复点：

- 带注释标签：`pre-diagnosis-v21-0e39d73`；备份分支：`backup/pre-diagnosis-0e39d73`。
- v24是纯新增：没有修改或删除任何已冻结的模型、训练脚本、配置或测试，因此该提交的
  训练实现与v24提交之后逐字节相同。若v24不满意，删除上述三个新增文件和本节/v24条目即可，
  或用`git revert`撤销v24提交，不影响任何已完成的v19—v23结果目录。

只读核对（不切换当前工作区）：

```bash
git show --stat pre-diagnosis-v21-0e39d73
git diff pre-diagnosis-v21-0e39d73 HEAD --stat -- models layers utils data_provider scripts configs
```

第二条的输出应只列出v24新增文件（`scripts/train_kelmarsh_matched_recipe.py`、
`scripts/train/Kelmarsh_launch_matched_recipe.sh`、`configs/kelmarsh_matched_recipe_protocol_v24.json`）。
需要恢复源码时，在独立目录而非覆盖当前工作区：

```bash
git worktree add --detach ../TimeDARTFirst-pre-diagnosis pre-diagnosis-v21-0e39d73
```

服务器上的结果目录（`outputs/logs/Kelmarsh/...`）和检查点从未被本次改动触碰；
恢复点保存的是源码与配置身份，不是数据、权重或日志的备份。

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
