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

### 只读误差归因已完成

`2557b68` 已在服务器同步，17项相关测试与Bash语法通过。诊断11800窗口、141600点、127台风机，全部目标晚于内部选择结束并严格早于outer train截止；可用性不足的4.1504%目标点仍按原actual-power协议纳入，没有按误差删数据。产物在 `outputs/logs/SDWPF/20261010/001_train_error_audit_h12_f1_s2024_forward_oof_cpu2_epoch8`，本地复制在 `output/train_error_audit_f1_s2024_20261010`，两个配对预测npz哈希已核对。

- reference总体 MAE111.674994 / RMSE176.323530 / R²0.699585；Persistence111.586996 / 181.900155 / 0.680282。reference的MAE略差，但RMSE改善3.0658%。这些训练期后段指标不能和outer MAE128.13直接比较。
- v6在这段 MAE111.406934 / RMSE175.963100，略好于reference，与先前固定内部选择段的负结果方向不同。这是时间敏感性线索，不改写v6停止决定，也不把新看的段改成epoch/参数选择集。
- reference误差超过300kW的8.4767%预测点贡献63.0790% SSE；最后四步贡献47.8039% SSE。主要问题不是只发生于第一步。
- 未命中四种冻结事件的53.1780%窗口贡献60.9159% SSE；前十高误差风机仅贡献16.2887% SSE。因此只增加异常事件条目或删少数风机不能解决主要总体误差。
- 历史稳定组MAE91.4619，Persistence89.0011，Skill为-2.7649%；历史上升组reference146.7773，Persistence151.8259，Skill为+3.3252%。统一修正策略存在明显工况差异，但这不证明某个新门控必然有效。
- 额定饱和组仅53窗口，reference MAE187.4888 vs Persistence123.7890，确实很差但SSE占比仅0.8519%；高风低功率仅213窗口，不能用其单组改善替代总体改善。

### 新的有限训练侧假设：逐步长残差缩放

以reference固定检查点为底座，预测形式 `last_power + alpha[history_trend, horizon] * (reference - last_power)`。alpha只允许0/0.25/0.5/0.75/1；历史组仍用固定5%容量、过去12点两半均值变化，不用未来涨跌。无截距、不放大残差、不改主干/标签/Wiki，不把此次数值校准声称为Wiki贡献。拟合期为原OOF时间范围的固定前半，后半仅检查，跨中点未来窗口丢弃。每格至少64个fit窗口；只在fit上选择MAE和RMSE均不差于原预测的alpha，以平均两种误差比排序，支持不足或同分回退alpha1。

代码 `utils/residual_scale_calibration.py`、`scripts/train_sdwpf_residual_scale.py`；4项测试覆盖时间隔离、修正幅度、有界性、标签不进入应用接口与不支持状态回退。检查段的整体诊断此前已被观察过，故本轮是探索性开发证据，不是独立确认。禁止基于这段再次选择alpha网格/阈值/seed；没有联合改善则结束。参数绑定来源checkpoint SHA256，禁止把同一校准表移植到不同底座后冒充相同方法。

运行：`python scripts/train_sdwpf_residual_scale.py --audit-dir "$(cat outputs/logs/SDWPF/train_error_audit_latest.txt)"`。最新目录指针 `outputs/logs/SDWPF/residual_scale_latest.txt`；内有 `protocol.json/calibration.json/result.json/check_predictions.npz/forecast_accuracy_overview.png` 及PDF。指标覆盖MAE/RMSE/R²/两类Skill和容量容差命中率；图注明训练期检查边界。单实例锁和已完成拒绝重跑保留既有结果。

### 有界残差校准已完成：对自身底座小幅联合改善，MAE仍未胜Persistence

`e28bad5` 已推送并在服务器快进；新增/相关测试共28项通过。校准仅用前段6668个窗口拟合，固定时间中点2023-06-21 03:55，后段5132窗口检查。来源是原先冻结reference SHA256 `4061d9afb646247c003f30f1d6d64d4277d02ec91c607d022c85e461f94b50dc`，没有重训主干或使用outer验证/封存测试。

同后段reference → calibrated：MAE138.7710269 →137.8845108kW（改善0.8865161kW，0.638834%）；RMSE206.2587483 →205.4807104kW（改善0.7780380kW，0.377215%）；R²0.7091185 →0.7113088。±5%容量（±75kW）命中率44.0147% →44.2453%；±10%容量（±150kW）66.6780% →67.0531%。R²不是准确率。

同后段Persistence的MAE为135.6615123 / RMSE211.2981543kW。校准后的MAE Skill仍为-1.638636%，RMSE Skill为+2.753192%。因此只能说相对自己底座有所改善，不能说已经全面击败Persistence或解决总体精度问题。前述强基线的outer结果属于不同时间范围，不直接与此训练期后段混排。

36个state×horizon参数中只有5格被fit证据缩放为0.75：历史稳定组第4/5/8/9步，上升组第11步；其余alpha1。没有手工把所有稳定工况关掉，也没有以检查段标签决定参数。指标图和两种Skill忠实显示MAE仍为负收益。

服务器目录 `outputs/logs/SDWPF/20261010/002_residual_scale_pilot_h12_f1_s2024_train_midpoint_grid5_cpu`，STATUS=COMPLETED。本地复制 `output/residual_scale_v7_f1_s2024_20261010`。下一步需要固定规则、底座身份和训练/确认边界后建立匹配对照；本轮不自动替换原论文模型、不追加本段参数搜索、不把数值校准收益计入Wiki贡献。原v6监控保持PAUSED，本轮无剩余训练进程。

### 固定校准的后续时间段匹配检查

用户继续授权后，新增 `scripts/confirm_sdwpf_residual_scale.py` 与启动器 `bash scripts/train/SDWPF_launch_residual_scale_followup.sh`。本轮不再拟合校准表、不重训神经网络；只把已保存的36格系数应用于完全相同的inner reference checkpoint SHA256 `4061d9...`。沿用该模型的inner-fit scaler，不加载outer模型预测权重；outer manifest仅提供原fold1验证窗口边界与身份。校准、源协议、原OOF预测、checkpoint和manifest均校验哈希，运行前固化follow-up协议。

目标只能在2023-06-30 02:10至2023-07-08 04:50（严格不含）的原fold1验证段；绝不构造封存test loader，要求所有目标早于test起点2023-07-16 07:30。按原stride12和单风机无缺口窗口重新选择，必须和原manifest的6245窗口、目标最小/最大日期一致。raw labels、可用性掩码、clip规则不变。基线是同checkpoint未经校准的预测和同窗Persistence；历史outer full-train趋势模型训练数据更多，不能冒充同训练预算的对照。

在读取本段目标之前冻结统计设置：按forecast issue time跨风机平均损失差，移动时间块bootstrap144个观测时间点、5000次、seed2024，DM/Newey-West lag143。CI/DM是time-balanced均值的推断，主表MAE是window-weighted；时间网格有缺口时144个点不一定恰为一天。汇总每步长、历史状态、MAE/RMSE/R²/Skills/容量容差命中率，以及校准改变覆盖率和改变窗口伤害率；后两者不是Wiki指标。

该outer段在历史研究中已被查看，故本次只是冻结方案的后续时间开发检查，而非全新独立确认。无论结果正负，不在本段追加系数、epoch、阈值或seed搜索。进程为两线程CPU/nice10，避免干扰其他用户GPU任务；锁与完成缓存防止重复运行。查看用启动器加 `--status`，结果指针 `outputs/logs/SDWPF/residual_scale_followup_latest.txt`。

### 后续时间段结果：停止静态缩放机制

后续检查在2026-10-10 11:21:17完成，目录 `outputs/logs/SDWPF/20261010/003_residual_scale_followup_h12_f1_s2024_frozen_cpu_outerval`，STATUS=COMPLETED/EXIT_CODE=0。首次source检查因旧outer manifest未记录后来加入的 `sdwpf_robust_pitch`，把missing与false误判为不一致，在推理前退出；已核对历史默认false及inner实际median repair=0，只做兼容修复，增加missing=false而true仍拒绝的测试，原目录续跑。另修复续跑时RUNNING状态未覆盖历史FAILED记录的问题，不影响预测或参数。

全部6245个窗口/74940点/127台风机通过身份检查，目标实际从2023-06-30 08:10至2023-07-08 04:40；不可用点3.63491%仍保留。原尺度配对预测SHA256 `a73ebad1fd9b8022d45633534becf3b255d1697c552183b1e6e84f4aa65efc4f`。同inner checkpoint：reference MAE128.4141421 / RMSE212.5590193 / R²0.6590536；calibrated128.5113445 / 213.1031671 / 0.6573057。校准使MAE恶化0.0972024kW、RMSE恶化0.5441477kW，joint_improvement_vs_reference=false，不改挑参数。

同窗Persistence为MAE132.3080376 / RMSE225.0524218 / R²0.6177967。校准后两类Skill为+2.869586% / +5.309543%，但这些正收益来自原底座，校准本身损失精度；未经校准的Skill为+2.943053% / +5.551330%。校准后的±5%/10%容量命中率51.015479%/69.918602%，也略低于reference51.023485%/69.921270%。不要把不同日期训练检查段的R²0.7113与本段0.6573直接当成方法退化幅度。

校准实际改变19.862557%预测点、77.854283%窗口；未改变窗口22.145717%。改变点平均gain=-0.489375kW，改变窗口伤害率37.638832%。这些是校准反事实指标，不是Wiki coverage/伤害指标。

按547个forecast issue times跨风机平均的MAE增益：相对reference均值-0.1933125kW，144点block/5000次bootstrap CI95%[-0.6571930,+0.0377004]kW，DM/Newey-West p=0.324188，不能声称显著提升。相对Persistence的window-weighted gain为3.7966931kW，time-balanced gain5.8602553kW；bootstrap CI[1.6318120,13.9279564]但DM p=0.071901，不能只挑有利检验。该段历史开发暴露和仅单fold仍限制证据范围。

审计评级维持B（数据路径与身份边界可复核，方法泛化收益未成立）。当前最需处理的是时间段依赖的修正收益，而非再调这张36格表。无需重跑完成的v7或本follow-up；新增单测覆盖身份/时间边界、只应用不拟合、精度与伤害如实报告、旧manifest默认值及续跑状态。停止在本fold验证段继续搜索，不自动移植校准表或追加seed，不把数值校准记作Wiki创新贡献。

改变的第4/5/8/9/11步在本段的MAE gain点估计分别为-0.04384/-0.16080/-0.14798/-0.35530/-0.45850kW，方向均为负；这显示静态表的时间泛化不足，但不是证明某个新机制必然有效。补齐的图同时展示校准前同checkpoint、校准后与Persistence三者，Skill面板专门对同checkpoint显示负收益，避免只有Persistence对照掩盖机制损害。`--render-only`仅检查保存预测/校准哈希和指标一致性后重绘，不调用拟合或神经网络推理。新版图已在本地渲染检查，无标注遮挡或裁切；预测NPZ下载后SHA256与服务器一致。原结果JSON不修改。

### 后续训练侧归因：工况构成与条件收益分离

继续定位只复用原train-OOF配对预测与已有v7校准，不加载/运行神经网络，不读取outer validation或sealed test预测。新增 `utils/residual_utility_drift.py`、`scripts/audit_sdwpf_residual_drift.py`，启动器 `bash scripts/train/SDWPF_launch_residual_drift_audit.sh`，查看加 `--status`。只从原train dataset重建过去12点的物理上下文；核对raw data、原scaler/feature order、全部window starts/目标时间戳/风机、历史事件/趋势及last-power anchor，与原预测逐窗一致才计算。

固定功率均值/容量分箱边界0.08/0.3/0.7/0.9、风速均值分箱3/5/10m/s；边界归上箱。这些诊断箱不按预测目标选择，不是Wiki gate或新的模型参数。时间段沿用原v7固定日历中点，跨界目标窗口丢弃。仅分析已存在的5个非恒等修正格，报告每箱前/后支持量、历史物理状态、预测与真实功率变化（后者仅作诊断）、偏差及反事实修正收益。

对每格前后MAE/SSE增益差做对称分解：共同支持箱的工况构成项 `(w_late-w_early)*(g_late+g_early)/2`、条件收益项 `(g_late-g_early)*(w_late+w_early)/2`，单边无支持箱贡献单独记录；三项必须严格还原总变化。该分解是描述性归因，不是时间漂移的因果证明；相关窗口/跨风机观测不当成独立样本。已有训练段摘要此前已被开发观察，本次仍是探索性定位。不给新alpha拟合、不增加候选/seed、不用外层负结果重新选系数；只有训练侧支持明确机制才能另建下一项试验。

产物指针 `outputs/logs/SDWPF/residual_drift_audit_latest.txt`，包括冻结 `protocol.json`、历史上下文 `history_context.npz`、哈希缓存标记和 `report.json`。两线程CPU/nice10/CUDA不可见；单实例和完成哈希避免重复诊断。新增测试覆盖历史分箱、无未来上下文、原窗口身份、构成/条件项守恒（含无共同支持）和负收益如实保留。

诊断在2026-10-10 12:01:39完成，目录 `outputs/logs/SDWPF/20261010/004_residual_drift_audit_h12_f1_s2024_train_midpoint_fixed_bins_cpu2`；本地/服务器5项测试通过。原历史与预测逐窗一致，无新模型推理。低于8%容量且风速低于3m/s的稳定组，前/后段支持2292/920窗口，四个缩放步均正收益；同样低功率但风速3—5m/s，支持134/111窗口，四步均负收益。上升组低功率且风速低于3m/s，支持98/115窗口，第11步前/后gain为-0.7164/-3.0592kW。因此三类历史涨跌把物理工作点不同的修正对象混在一起；不是单纯删极端样本或调一张状态表能解决。五格训练后段gain全部提高，主要条件收益变化为+2.381/+2.783/+4.065/+4.023/+5.024kW；不能据此声称已解释outer负收益的因果原因。

### 有限训练侧改动：双时间块物理支持门控

v8不重新拟合alpha、不训练神经底座/风险头、不改Wiki。仅允许已存在5格缩放在指定物理箱中启用：原OOF日历前50%（原alpha拟合块）与接着25%（门控校准块）都至少64窗口，且各自MAE与MSE增益严格为正；否则完全保留原checkpoint预测。最后25%只检查，跨两个分界的目标窗口丢弃。这是一个预先固定的支持/联合收益规则，没有新系数网格或阈值搜索，不把稀疏/单边有益箱纳入。

代码 `utils/physical_scale_guard.py` 与 `scripts/train_sdwpf_physical_scale_guard.py`。应用接口只接受预测、last-power、历史state与物理bin，不接受未来标签；拟合接口只接受两个训练块，检查块标签不传入。沿用同checkpoint和历史上下文哈希，保存固定协议、guard证据和检查预测。仅原训练期探索开发，已有全段诊断曾被观察，不能当独立确认；不自动再看outer或sealed test。图复用原数值绘图管线，并把同checkpoint基准放入比较，避免Persistence正Skill掩盖改动损害。

运行命令：`python scripts/train_sdwpf_physical_scale_guard.py --drift-dir "$(cat outputs/logs/SDWPF/residual_drift_audit_latest.txt)"`。结果指针 `outputs/logs/SDWPF/physical_scale_guard_latest.txt`。不满足联合改善就停止该规则，不能基于检查结果回调支持量、分箱或门槛。

### v8结果：精度收益不足，暂不采用物理支持门控

提交 `f7b7f70` 已推送并在服务器同步，两项新增测试在本地/服务器通过；v8在2026-10-10 12:06:46完成，无神经网络重训/外层评估。目录 `outputs/logs/SDWPF/20261010/005_physical_scale_guard_h12_f1_s2024_train_50-25-25_min64_fixed_scale_cpu`，本地复制 `output/physical_scale_guard_f1_s2024_20261010`。原早块6668窗口、门控校准块2541、末块检查2506，跨两分界85窗丢弃。检查目标2023-06-25 15:10至06-27 23:50，严格在原outer train内；上下文与预测哈希均绑定原source checkpoint。

末块同窗reference → v8：MAE117.7940869 →117.7809571kW、RMSE181.6945768 →181.6416689kW、R²0.4683565 →0.4686661。联合点估计虽然为正，净增益仅0.0131298/0.0529079kW，不能当作有价值或显著的泛化提升。v8的Persistence Skill为MAE+2.012163%/RMSE+1.530372%；±5%/10%容量命中率49.231844%/72.735435%。这些属于该训练末块，不能与不同outer时间段R²0.659等混排。

重要对照：旧v7静态缩放在完全相同末块是MAE117.5916657 / RMSE181.3790671 / R²0.4702013。v8反而比它恶化MAE0.1892914/RMSE0.2626018kW。因此不能只报与原reference的正差而称新门控提高精度；旧v7的outer负结果也仍保留，不因本段静态正收益恢复采用。v8接受5个物理箱×步长格，改变6.790370%预测点/26.296887%窗口，不介入73.703113%；改变窗口伤害率34.901366%，改变点平均gain0.193359kW。这些是数值缩放门控的反事实指标，不是Wiki coverage。

当前审计评级维持B。没有发现新增时间/标签错位，但本轮条件收益分解和末块开发证据不证明外层收益；当前最需解决的两项是：涨跌状态不能代表相同物理工作点，以及局部残差门控的覆盖/效果不足。无需重复v8训练或追加min64/分箱/系数搜索，原论文模型保持不变，新门控暂不采用、不自动outer refit。训练末块reference bias=-52.6816kW，门控后=-52.5348kW，说明本规则几乎未改变系统性低估；偏差是下一步训练侧定位线索，不是已经证明的全部误差原因，也不授权直接把这个数加到外层预测上。
