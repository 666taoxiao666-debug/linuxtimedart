# 多年份 Kelmarsh 审查、修复与真实执行结果：Claude 单文件交接

日期：2026-10-26，北京时间。任务：本聊天一次性定时审核。此文件是本轮唯一新增的人工总结/交接文档；JSON、CSV 和原始日志是机器证据，不是另外的报告。

## 1. 先读结论

**【最终评级】B（修复后的数据准备组件）；完整训练方案尚不具备可执行条件。**

附件是预处理测试，不是新模型或训练控制器。本地新增的实际实现是 `data_provider/kelmarsh_multiyear.py`、`scripts/prepare_kelmarsh_multiyear.py`、`utils/kelmarsh_enrichment.py` 以及测试。检查时 `configs/` 没有多年份训练协议，`scripts/` 只有多年份 prepare 入口，没有对应 train/report/controller。

Claude 原交接 `docs/CLAUDE_TO_GPT_HANDOFF_MULTIYEAR_PREPROCESSING_20261025.md` 第 0 节第 4 条和第 8 节也明确：先出质量报告，再冻结年份与训练方案。本轮用户授权了修正和部署训练，但尚无可复用的冻结多年份训练任务。因此本轮完成可核验的数据组件修复、测试、推送和服务器数据准备；**没有把旧 v27 重新启动来冒充新训练，也没有新模型 MAE、RMSE 或 R²。**

多年份数据、小时末状态和更清楚的工况区分值得研究；它们能否提高预测能力尚未证明。v25/v26/v27 的失败只否定那些固定配方，不能证明“网络容量必然不是瓶颈”，也不能把一切归因于脏数据。删除停机/限功率目标窗口之后的条件指标不能替代全部样本的部署精度；R² 是否上升也不能预先保证，因为目标方差会随队列改变。

## 2. 输入与代码身份

- 仓库：`D:/Users/XT/Desktop/moxing/gpt/v2/TimeDARTFirst`；分支 `codex/self-evolving-wiki`。
- 审查起点：`80382466a1d48ef7d50b240db5b1c30560448e12`。
- 修复实现提交：`4acaf79`，已推送 GitHub，并服务器安全快进。
- 附件：`C:/Users/XT/.codex/attachments/45118038-e3f9-4882-9928-0f900c25a561/已粘贴的文本.txt`。
- 附件 SHA256：`a79733a8f9b746438076b08ba9cc806755c826dd1110a01ffa786ada4fc27f86`。附件保留不变；修复后的测试以 Git 为准。
- 本轮提交包含上述四个 Python 文件、只读独立核验脚本 `scripts/audit_kelmarsh_multiyear_preparation.py` 及本交接文档；工作区其他未跟踪内容和旧实验均保留。
- 数据准备 `schema_version=2`；实际准备元数据记录脚本、增强函数和复用聚合脚本的 SHA256，以及输入 ZIP 官方 MD5、SHA256、大小、输出 CSV SHA256、曲线、规则和年度质量。**这不是模型训练协议 ID。**

## 3. 【必须立即修复 Top 5】及处理结果

### 3.1 [P1，已修] 全年容量记录比例会改变过去输入

位置：`scripts/prepare_kelmarsh_multiyear.py -> available_capacity_mask/prepare_turbine`。

原逻辑以同风机整年有效容量记录比例是否达到 50% 决定该年所有时刻是否启用容量特征与降额规则。若后半年缺失率变了，年初 `available_capacity_frac` 和 `n_derated_ticks` 也会变化。它违反历史输入因果性，即便不使用预测误差也不成立。

修复：只使用当前 tick 已观测的容量值；缺失仍是未知，不填零。年度比例保留为审计统计，不参与输入开关。原全年 50% 门槛已移除；98% 额定容量的物理规则没有调参。

验证：固定前一天观测，只把未来三天容量改为缺失，前一天全部容量特征、点数、降额标记逐位相同。该测试能复现旧实现的问题。置信度：高。

### 3.2 [P1，已修] 曲线年份没有与实际训练划分绑定

位置：`data_provider/kelmarsh_multiyear.py -> KelmarshMultiYearSeries.__init__`。

原预处理只要求 curve-years 属于“已准备年份”，provider 却允许任意 train/validation/test 划分。调用者可以用包含验证年份标签拟合的曲线决定正常窗口，而 provider 不报错。

修复：文件输入必须带 schema 2 准备元数据且 CSV SHA256 一致；所有曲线拟合年份必须属于实际 train；train 必须早于持出年份，validation 必须早于 test。内存 DataFrame 必须显式传 preparation，审计记录它没有文件哈希验证，不能把它当注册文件。

验证：混入 held-out 曲线年份、倒序训练/验证、缺 provenance、CSV 哈希不匹配都拒绝。置信度：高。

### 3.3 [P1，已修默认行为；评估控制器仍待实现] 自动展开全部年份及隐式筛目标

位置：`scripts/prepare_kelmarsh_multiyear.py -> prepare/main`；`data_provider/kelmarsh_multiyear.py -> windows/audit`。

原 prepare 不给 years 就处理 2016–2024，包括计划封存的年份。provider 默认返回按未来四小时运行状态筛选的 normal 队列，调用者容易误称部署精度。

修复：prepare 默认仅 2016–2019；本次服务器也显式指定这四年。provider 默认返回 all，normal 需要明确请求；audit 标记 normal 使用未来目标状态、全部队列必须报告。保留交接中提出的 normal 条件研究功能，并未把目标标签改掉；交接里对用户偏好的转述不等于授权本轮直接筛选封存测试集。

**剩余限制：数据类只能提供两套队列，尚没有报告控制器强制所有结果同时报告两者。不能再声称“实现已经强制每张表同时输出 all/normal”。** 未来协议需规定相同队列上的基线比较、筛选比例和两套结果，不允许用 normal 的较低误差冒充全样本提升。置信度：高。

### 3.4 [P2，已修] 未知状态和时间异常缺少防护

位置：`utils/kelmarsh_enrichment.py -> validate_tick_index/state_flags/normal_operation`；provider 构造函数。

原数据在功率完整但风速不完整时可能把“无法判定停机/欠发”当正常；原小时末增强入口也没有严格拒绝无时区、重复和亚秒离网 tick。

修复：新增完整风速/功率状态点数校验和 `flag_incomplete_state`；容量观测数单独报告，缺容量仍不能证明没有降额。增强要求带时区、排序唯一、精确 10 分钟网格；provider 要求精确小时标签、正整数窗口参数和无重复额外通道。过滤不改变 power_kw。

验证：缺失风速保留功率标签但不能算已确认正常；微秒偏移、重复和无时区 tick 拒绝。置信度：高。

### 3.5 [P2，已修] 数据身份和重运行保护不足

位置：prepare 的输出检查/元数据，provider 的年份过滤与窗口归属。

修复：既有准备 CSV/质量表/manifest 一律拒绝覆盖，原始 ZIP 仍按官方 MD5 校验复用；只让声明年份及其必要历史进入 provider 特征/网格；窗口归属校验全部目标小时，不只两端年份；核对六个 ZIP 成员的风机身份与静态表一致；非法数字/无穷值拒绝，不能静默变成缺失。离线优先复用原获取流程缓存的 Zenodo 元数据。

验证：修改准备哈希拒绝；重复 prepare 不覆盖；加入未声明未来年份不改变训练窗口、原始输入和 scaler。置信度：高。

## 4. 最终数据架构与参数

本轮不是新的神经网络，**没有新增模型可训练参数、优化器或 checkpoint**。

数据流：官方年度 ZIP → 校验与供应商列解析 → 每风机唯一 UTC 10 分钟 tick → 严格六点小时聚合 + 因果小时末增强 → 训练年份功率曲线与启发式工况标记 → 带哈希元数据的 CSV → 显式年份 provider → all/normal 窗口视图。

- 当前默认窗口：24 小时历史，预测未来 4 个小时，stride=3。尚未冻结为训练实验。
- 基础 7 通道：Wspd、Etmp、Ndir_sin、Ndir_cos、Pab_mean、rotor_speed、power。可选额外特征插在 power 前；power 始终最后。
- 额外特征包括小时末功率/风速、小时内波动、容量、停机点比例、风向与机舱差的 sin/cos、时刻/日序和同时刻场站均值。全场均值要求至少 3 台有值，使用历史时间点，不使用未来实测风速。
- 小时桶为 `[t,t+1h)`，发布标签为 `t+1h`；小时末取 `hh:50`。现有严格六点聚合函数未修改。
- scaler：训练年份的全体可观测行、逐通道均值/标准差；不是 normal 队列上单独拟合。常数通道 scale=1，完全无训练观测的通道拒绝。
- 停机启发式：风速≥5m/s 且功率≤额定 1%；降额：已观测容量<额定98%；欠发：功率<训练曲线期望50%，且期望≥额定10%。规则未根据模型误差调整。
- 曲线：0.5m/s 分档，每档至少20小时，中位数、插值、单调化并封顶额定；这只用于条件标记，**不改原标签**。它并非真实可用功率的因果识别器。
- 两种基线输出接口：最近一小时均值 Persistence、最近 10 分钟功率 Persistence。缺失最后 tick 时后一基线仍可能缺失；未来训练控制器必须预先冻结公共评估队列或明确覆盖率，不能各方法各删一批样本。
- `normal` 仍是“没有检测到异常且功率/风速完整”，缺容量时不能证明真实正常。它还可能把尾流、偏航、结冰等合法困难工况筛走。
- 偏航特征目前沿用“Wind direction − Nacelle position”的假设。官方 signal mapping 仅把 wind direction 描述为 Wind direction，没有在映射中明确其相对/绝对参考系；**需进一步供应商语义证据，不能因为公式常见就称其已被证明。** 下一训练协议未核实前不应启用该可选通道。

## 5. 本轮测试、部署与服务器执行

- 本地初次测试因缺 `requests` 在 import 阶段失败，不是训练失败。安装 requests 后测试通过（未改旧获取脚本）。
- 本地：23 项测试通过，40.418 秒；服务器：相同 23 项通过，26.363 秒，含 torch/sklearn 下新旧窗口规则等价测试。
- 测试覆盖因果性、元数据/曲线年份、完整性、重复执行保护、网格、scaler、合成真实格式 ZIP、基线接口及旧下载断点校验；没有用这些合成数值当真实精度。
- SSH 直连成功，服务器代码快进到 `4acaf79`。服务器原有 `docs/EXPERIMENT_REPLAY.md`、`docs/KELMARSH_MODEL_REVIEW.md` 修改保留。
- 检查时其他用户 gyx 的 GPU 进程为 823602（约1516MiB）、824455（约2886MiB）；本轮不初始化 CUDA，不干预其他任务。
- 数据准备 PID 初值 835354；自身 OMP/BLAS/MKL 各2线程。日志：`outputs/audits/multiyear_20261026/preparation.log`；测试日志：同目录 `tests.log`。
- 执行命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/XT/miniconda3/envs/timedart/bin/python -u scripts/prepare_kelmarsh_multiyear.py \
  --years 2016 2017 2018 2019 --curve-years 2016 2017 2018 2019 --download
```

**实际完成时间：2026-10-26 02:02:23 +08:00。** 准备进程已正常退出；2016 ZIP 复用，新下载2017/2018/2019 ZIP 均通过官方 MD5 和独立 SHA256 核对。总计 192,960 条风机小时记录、189,301 条完整功率小时、185,422 条符合当前 normal 标记的小时。没有读取 2020–2024 SCADA，没有 sealed-test 推理。

| 年份 | 风机小时数 | 完整功率小时 | 当前规则未检出异常的小时 | 占完整功率小时 |
|---|---:|---:|---:|---:|
| 2016（5月起） | 35,280 | 34,401 | 33,717 | 98.0117% |
| 2017 | 52,560 | 51,835 | 50,636 | 97.6869% |
| 2018 | 52,560 | 50,638 | 49,517 | 97.7862% |
| 2019 | 52,560 | 52,427 | 51,552 | 98.3310% |

这张表不是机组真实健康率：标记是启发式，未知容量仍有限制；异常标记可重叠，也可能位于功率缺失小时，不能把各标记计数简单相加作剔除数。现有证据不支持“数据普遍异常，所以只要清洗就会大幅提高精度”的结论。

真实 provider 冒烟只声明 `train=(2016,2017,2018,2019)`，没有持出预测，也没有模型 fit：

| 输入配置（仅可用性核查） | all窗口 | normal窗口 |
|---|---:|---:|
| 基础7通道 | 59,113 | 55,968 |
| 基础 + 小时末功率 | 59,113 | 55,968 |
| 基础 + 小时末功率/风速 | 59,113 | 55,968 |
| 基础 + 可用容量 | 59,113 | 55,968 |
| 基础 + 场站平均功率/风速 | 58,952 | 55,812 |

normal保留基础窗口的94.6797%；加入场站均值会因缺失少161个all窗口。这说明未来消融需要明确公共评估窗口，不能拿各自筛后的指标直接比较。小时末特征在这些开发年份未额外损失基础窗口，具有优先做小规模冻结信息消融的条件；**可用性不是增益证据**。provider检查中 `cuda_initialized=false`。

独立核验已完成：四个ZIP、CSV、准备源码哈希一致；24份供应商文件头全部明确 UTC；24行年度风机质量表逐项重算一致；2016风机1的5,880个小时标签直接从原始ZIP重新聚合，最大差 `4.547473508864641e-13 kW`，与浮点舍入一致。保留13,074个负功率小时、660个略超额定小时，范围 `[-11.6458471,2064.0865479] kW`，没有截断。

关键身份：

- 输出 CSV SHA256：`50134f7b6c23c9608d7ef7926ff67692463e9846fa7204a9515b2b7d473dfd0e`。
- `preparation.json` SHA256：`ff4f6ebd9ed3dd59016295a81bb9b614cbb77dbfe6d3355a3efa170c1ff09aa2`。
- `independent_audit.json` SHA256：`f961c15553e7f4535c6d45a8ad3ece7fa17a01d38815e9d672a897e77f3217b9`。
- `quality_by_year.csv` SHA256：`8c55f66210f57f289ffe123d070451b0d385f7d4322b201c1f0b84baae6f3015`。
- `provider_audit.json` SHA256：`d61a69101d15aabac4b5276a51118acdf84dc9e153232b948bf5534005eee196`。
- 这些机器产物及原始测试/准备日志已下载至本地 `output/kelmarsh_hourly/multiyear_20261026_review/`。大CSV与ZIP留在服务器，没有另生成多份人工报告。

## 6. 最新训练结果与不能声称的内容

**本轮新模型 fit 数：0。新模型 MAE/MSE/RMSE/R²/Skill/容差命中率：未产生。新预测图：未产生。** 数据准备完成也不是训练完成。

最新已完成训练仍为完整 ID `kelmarsh_forward_block_residual_diagnostic_v27`：44 fits、396 epochs，仅训练期前向诊断。Full 五 seed 预测均值的 pooled MAE/RMSE/R²=179.268656/259.154117/0.652190；DLinear=180.139458/259.936958/0.650086；Persistence=176.791164/258.060091/0.655121。Full 相对 DLinear MAE 收益0.870802kW，95%区间[-0.086594,1.852578]跨0；整体还不如 Persistence。R1失败、R2通过、R3失败、R4通过、R5失败，按注册规则 `STOP_RESIDUAL_LINE_ON_THIS_DATA`。这些是旧诊断结果，**不是本轮多年份外层精度**。

不能据此声称“清洗成功”“新模型显著提升”“已证明数据是唯一问题”或“Wiki贡献”。本轮没有训练Wiki。

## 7. Claude 下一步需要补的具体实现

1. 使用本轮真实质量报告，在训练前建立独立、多年份协议 ID 与 SHA；不要复用已结束 v27 名称。明确开发/选择/一次确认年份和封存范围，功率曲线只能属于 fit 范围，内部分块时须重新检查同一约束。
2. 写与此 provider 匹配的训练控制器、预检、恢复锁、checkpoint身份和完整报告。数据类“功率在最后”不等于与全部旧控制器契约自动兼容；需小型端到端冒烟验证归一化反变换、预测形状、baseline/label原尺度和断点恢复。
3. 先固定可解释对照：小时均值 Persistence、last10 Persistence、DLinear。同一数据/同一评估队列；每一新增信息组保留同架构无该信息对照。不要同时换年份、筛标签、换模型、加多类特征后把全部差值归因于其中一项。
4. 研究 normal 条件任务可以继续，但 all 是部署结果的必要报告；披露正常窗口占比、不能判定的状态覆盖率和四个步长指标。是否基于 normal 训练、是否另有 all-trained 对照必须在训练前确定。任何曲线/过滤阈值不得按持出集预测误差挑选。
5. 冻结模型、特征、loss/LR、epoch/seed预算、内选规则、继续/停止规则、共享GPU限制和一次确认访问规则后，才提交本聊天执行训练。当前缺这些内容，不能虚构“方案已完整、只差启动”。

## 8. 证据来源与复现

- 官方数据发布页：<https://zenodo.org/records/16807551>。其 v4 描述明确为六台风机的2016–2024年10分钟SCADA，且不是所有信号在全时段可用；因此可选信号缺失要审计，不能假设所有年份完整。许可证 CC BY 4.0。
- 供应商映射：服务器已有 `datasets/wind_public/resampled_hourly_scada/kelmarsh_2016/raw/Kelmarsh_WT_dataSignalMapping.csv`；本次只读核查，未改。
- 测试命令：`python -m unittest tests.test_kelmarsh_multiyear_preprocessing tests.test_hourly_wind_acquisition`。
- 独立核验命令（服务器仓库根目录，CPU）：`python scripts/audit_kelmarsh_multiyear_preparation.py datasets/wind_public/resampled_hourly_scada/kelmarsh_multiyear/metadata/preparation.json --output outputs/audits/multiyear_20261026/independent_audit.json`。它核验现有产物，不重新准备数据，不训练模型。
- 准备输出目录：`datasets/wind_public/resampled_hourly_scada/kelmarsh_multiyear/`；原数据全部保留。
- Git恢复代码以本轮实现提交为准；复现数据需同一输入ZIP/MD5、规则、curve-years、准备源码哈希和CSV哈希。不要依赖旧交接附录中的源码副本覆盖修复后的工作树。

本次一次性任务完成后取消定时触发，不自动追加周期训练或候选搜索。
