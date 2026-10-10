# SDWPF 精度问题审计与匹配归一化试验

本报告针对 `afa6581` 的 v5 失败结果，及其后的单变量修正。原始实验与标签保留；v1—v5 不重新选择阈值、seed 或最佳 epoch。

## 【最终评级】：C

数据切分、训练集 scaler、单风机连续滑窗、标签对齐的已检路径未发现 P0。存在确定的预训练/预测表示错配，但已完成的 v6 内部配对试验没有带来精度提升，不能把该错配当作主要误差原因。Wiki 原冻结比较的增益很小且统计检验未显著，不能用本次数值底座修正冒充 Wiki 贡献。

## 【必须立即修复 Top 5】

实际确认两项问题，不凑五项：

1. 预训练与预测使用不同的风速归一化约定。新增可选 `consistent_physics_norm`，用同一历史归一化函数，重建逆变换也一致。
2. 仅按张量形状迁移 checkpoint 不能审计这种表示身份。新增方法元数据、加载检查、独立 checkpoint/run 标签，拒绝新旧归一化约定混用。

另有一项试验设计限制：v5 内部 epoch 来自一个训练前缀和内部预训练；外层 refit 使用不同的、历史验证选择后已冻结的预训练。内部第8轮不是外层最佳轮数的保证。禁止根据已看的外层曲线改选第3轮。

## 【需重跑的实验与需补充的单元测试】

仅运行协议 `configs/sdwpf_physics_norm_protocol_v6.json`：fold1/seed2024、输入336、预测12；沿用 v5 前80% fit、随后10%内部留出，全在 outer train 结束前。新建归一化匹配的预训练（最多20轮、patience3），随后固定8轮微调。复用 v5 reference 的真实第8轮指标作为配对对照；不重复训练已完成的参考模型。

唯一机制变化是预训练保留全球训练标准化后的绝对 Wspd，和预测保持一致。MIXED权重0.2、步长权重1、功率权重0、LR1e-6/newLR5e-6、数据、seed、Wiki阈值不变。比较预声明第8轮 MAE/RMSE/R²、MAE Skill、RMSE Skill；不得改挑另一个 epoch。MAE、RMSE 均改善才设计下一匹配外层确认；否则结束该机制，不增加候选。此启动器不会自动评估 outer validation 或 sealed test。

测试覆盖：实际两阶段 patch 输入一致、归一化逆变换、有限重建梯度、旧行为逐位一致、checkpoint约定/特征顺序拒绝、独立标签、固定轮数比较不偷选早期最优结果。

## === 详细问题清单 ===

### 1. 最强物理通道的预训练表示错配

- 严重级别：P2（确定的实现不一致；精度影响需要验证）。
- 位置：`models/TimeDART.py` → `Model.pretrain` 原923—950行 vs `Model.forecast` 原1150—1171行；修正后分别约925、1135行。
- 触发条件及证据：旧预训练把所有通道去均值、除以窗口标准差；预测在 `revin_keep_wind=1` 时将 Wspd 替换回原全球标准化值。原操作点信息在预训练阶段被删除、预测阶段重新出现。同一编码器的最强先验通道（Wspd初始scale2.5）接收不同表示。
- 训练侧证据：201,408 个有效 fit 窗口，截止2023-05-25 09:10；全球标准化风速窗口均值10/50/90分位数分别 -0.5693/-0.0743/0.9370，而旧预训练几乎全部为0；窗口标准差对应0.5184/0.7976/1.1429，旧预训练几乎为1。数据SHA256为 `af02f09128ef39ba3b813c51ef2d610b3fcc3aaff6c73bbaacfb6df439e202e0`。
- 对论文结论的影响：数值底座可能未有效利用预训练的物理表示。当前证据不证明这解释全部预测误差，也不保证统一后提升。
- 修复方案：共用 `utils/physics_normalization.normalize_history`；保留通道的逆变换设mean0/scale1，其余通道仍用历史均值/标准差。默认关闭新开关，旧预测行为保持一致。
- 验证修复的测试：真实 Model 的 pretrain/forecast patch 输入一致；反向重建恒等；重建梯度有限；旧 MS 功率输出逐位相同。
- 置信度：实现不一致为高，精度收益为待实验。

### 2. 迁移身份只检查权重形状不够

- 严重级别：P2。
- 位置：`utils/tools.py` → `transfer_weights`；`run.py` → `pretrain_signature/load_finetuned_model`；`exp/exp_timedart.py` → `_save_pretrain_checkpoint`。
- 触发条件及证据：归一化开关不改变编码器张量形状，旧 checkpoint 可被错误复用为新方法，导致改动未真正实现。
- 对论文结论的影响：新旧方法对照的 provenance 不完整，可能把同一个底座混标为两种方法。
- 修复方案：预训练 checkpoint 写入 `consistent_physics_norm/use_norm/revin_keep_wind/feature_columns`；加载前验证，缺字段的旧 checkpoint 只认作legacy；新方法单独命名，评估必须有匹配manifest。
- 验证修复的测试：新模型加载旧预训练明确报错；同约定可加载；特征顺序或风速约定不同拒绝；run/checkpoint签名独立。
- 置信度：高。

### 3. 内部训练轮数不能直接代表外层泛化最优

- 严重级别：P2（设计限制，不是标签泄漏）。
- 位置：`scripts/tune_sdwpf_accuracy.py` → 内部选择与 outer refit；v5 的 `selected.json/result.json`。
- 触发条件及证据：内部 reference 第8轮 MAE151.1695/RMSE254.1998 最优；外层固定第8轮 MAE129.3257/RMSE211.7757，相比冻结趋势 MAE128.1317/RMSE212.4592，仅 RMSE改善，MAE恶化1.1940kW。两者预训练/训练时间段不同。
- 对论文结论的影响：不能称联合精度提升；不能根据已经观察的 outer 曲线回选更漂亮的轮数。
- 修复方案：v6先只做匹配内部对照，固定终点；外层确认另行明确匹配预训练/轮数选择规则，不混用历史最优结果作为新参数的选择依据。
- 验证修复的测试：固定第8轮比较忽略更好的早期 epoch，持久化协议、历史及 checkpoint 哈希。
- 置信度：现象为高，恶化具体因果为待证据。

## 排除的假设与已核查数据流

“历史功率太平稳导致修正完全失效”不作为本轮改动依据：历史标准差低于全球标准差0.1（43.006kW）的270个窗口，仅占全部训练持续性绝对误差0.01349%；单纯尺度下限很难解决总体误差。

数据流：原SCADA → 同时点物理特征与有界因果ffill → 原功率标签不插补 → 全局时间边界 → scaler仅fit前缀 → 单风机无缺口历史/未来窗 → DataLoader → 历史归一化/编码器/残差头 → 全球原尺度等比例损失 → 原尺度指标。原先真实标签的物理clip和预测clip沿用冻结协议。本次未删异常工况、未改功率标签、未拟合验证或测试目标。

## 操作与产物

服务器启动：`bash scripts/train/SDWPF_launch_physics_norm.sh`；查看：同命令加 `--status`；确认进程已退出、GPU空闲且是真实代码失败后才能 `--resume`。

最新指针：`outputs/logs/SDWPF/physics_norm_latest.txt`。根目录包含 `protocol.json/provenance.json/train_audit.json/progress.json/result.json`；阶段日志在 `inner_pretrain/launcher.log` 和 `candidate/launcher.log`。本地原始训练审计产物为 `output/normalization_optimization/train_audit.json`。

## 2026-10-10：完成结果与下一步只读定位

v6 已完成，固定第8轮 MAE152.4973457 / RMSE254.6750647 / R²0.6372884；同内部段 reference 为151.1694846 / 254.1998380 / 0.6386408。MAE恶化1.3278611kW，RMSE恶化0.4752268kW，联合改善为false。因此停止该机制，不改挑早期 epoch，不启动外层 refit，不重新训练这两个检查点。原始目录：`outputs/logs/SDWPF/20261009/004_physics_norm_h12_f1_s2024_inner_physnorm1_mix0.2_ep8_trainonly`。

下一步仅做训练侧错误归因，使用 v5 reference 与 v6 固定第8轮检查点，在2023-06-12 05:40（内部选择结束）至2023-06-30 02:10（outer train 截止，严格不含）之间的完整前向OOF窗口推理。这段未用于这两个模型拟合或选择 epoch；用于后续开发诊断后，不再当成独立确认集。只构造 train dataset 的数据路径，再筛取该训练期后段；不构造/评估 outer val 或 sealed test loader。

工况定义来自原 `wind_event_factor_wiki.json` 的冻结物理门槛，仅看过去12点风速/功率。这是硬阈值诊断标签，不是 Wiki 的软激活率，不能把它称作 Wiki coverage。事件可能重叠，单事件误差占比不能相加；使用4位组合码做互斥分区，检验总贡献守恒。另报告每步长、每风机、历史涨跌（过去两半均值变化超过5%容量的固定诊断分箱），以及 MAE/RMSE/R²、Persistence Skill、偏差、容量容差命中率、大误差贡献。输出两模型的配对原尺度预测、时间戳、可用性掩码与源检查点/数据哈希；原标签、clip约定和采样 stride12 不改。

实现：`utils/train_error_audit.py`、`scripts/audit_sdwpf_train_errors.py`。启动器 `bash scripts/train/SDWPF_launch_train_error_audit.sh`；查看加 `--status`，实时日志用 `tail -f "$(cat outputs/logs/SDWPF/train_error_audit_latest.txt)/launch.log"`。两线程CPU、nice10、CUDA不可见，避免干扰其他人的GPU任务。单实例锁、已完成推理缓存和哈希检查使 `--resume` 只续未完成阶段。未获得诊断结果前不预设新的模型原因，也不启动下一轮训练。
