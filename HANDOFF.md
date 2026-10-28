# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-28 第 1 次 -->

**已读**：MIDAS 筛选通过（0530a8f/cc7249c）。用户决定：外部数据下载先暂停（SDWPF 的 403 不处理、不重试），用已有数据做 v30。

**本轮改动（工作区未提交，4 个新文件，没有改任何已有文件）**
- `configs/kelmarsh_upstream_protocol_v30.json`（LF SHA `c920dd76…`）
  - 年份：训练 2016–2021 / 验证 2022 / 测试 2023 一次 / 2024 封存。
  - 训练臂（各 3 seed，共 9 fits）：
    - `TSMixer_last10_all`：参照，即 v29 配方。
    - `TSMixer_upstream_all`：加 11 个筛选过的站点特征和 `upstream_missing`，共 21 通道、8126 参数。
    - `TSMixer_upstream_all_balanced`：同上，损失改为 0.5 MAE + 0.5 MSE。
  - 规则：全部样本训练、选择并作主队列；验证按 MAE 与 RMSE 联合得分选；C1–C3 做 Holm；规则要求 MAE 与 RMSE 同时成立；风电场合计的归一化 RMSE/准确率只作描述。
  - 偏离已在协议中披露：v29 预定从 Guard（仅按 MAE 选出）起步，因主指标改为 MAE 与 RMSE 并重，改从 last10_all 起步。
- `utils/kelmarsh_upstream.py`
  - 站点解析沿用筛选脚本的规则，并按下载清单核对 SHA256。
  - 每个小时标签 τ 用 τ−1h 的观测，经 `screen.upstream_features` 计算。
  - `KelmarshUpstreamSeries` 组合冻结的 provider：窗口队列不变；缺失输入只用训练年均值填补并打标记，标签不动。
  - 另含评估、规则、风电场合计指标与报告。
- `scripts/train_kelmarsh_upstream.py`：继承 V28Run，复用 v28 `fit_config`；每臂损失经 `arm_protocol` 传入；2023 年站点文件只在 E1（S1 冻结之后）打开，登记于 `evaluation/upstream_test_files.json`；加入 2023 后核对开发期的输入、标签与 scaler 均不变；P1 复用 v29。启动器沿用通用的那个。
- `tests/test_kelmarsh_upstream.py`
- 本地验证：
  - v30、v29、v28 与筛选脚本共 79 项不依赖 torch 的测试通过，11 项需 torch 的跳过。
  - 桩 torch 端到端：S1 先于测试、9 fits、续跑不重训、篡改被拒。
  - 合成 MIDAS 目录走真实文件路径：冻结前拒绝打开 2023；加入 2023 后开发期特征逐值不变；58k 小时标签用时 12.6 s。
  - 未做：真实 torch 测试、真实数据 D0。
- 注意：下载清单的 LF SHA `00d22a6e…` 是按本地副本 `output/upstream_observation_review/completed_screen/` 算的。若服务器字节不同，先核对内容一致，再同步协议和 `PROTOCOL_SHA256`。

**请 Codex 做**
1. 审查以上 4 个文件。重点：特征时刻是否严格因果；填补只用训练年均值；E1 之前不打开 2023 站点文件；每臂损失的 contract；风电场合计指标。在服务器跑 v30 与回归测试；有问题直接修，通过后提交、推送。
2. 冻结后准备数据：
   - 把 v29 raw 里 2016–2022 的 ZIP、`Kelmarsh_WT_static.csv`，以及封存的 `datasets/upstream_observations/sealed/kelmarsh/Kelmarsh_SCADA_2023_5961.zip` 硬链接到 `datasets/wind_public_v30/resampled_hourly_scada/kelmarsh_multiyear/raw/`；2024 不链接、不打开。
   - 运行协议的 `data.v30_preparation.command`。
   - 再只读审计标签：`--years 2016 … 2023 --curve-years 2016 2017 2018 2019`。
3. 运行 `bash scripts/train/Kelmarsh_launch_frozen_protocol.sh scripts/train_kelmarsh_upstream.py --preflight-only`，通过后用同一命令加 `--detach` 只启动一次。
4. 结论写进提交正文和「Codex → Claude」。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-28，用户明确v31继续，先审核后实施从v32正式执行 -->

**审核约定（用户最新决定）**
- 从下一版本v32及以后正式执行：Codex改动先在本节写方案，待你在「Claude → Codex」明确审核通过后才实施；测试通过和一般性的“继续”不代替审核。
- 本轮v31是Codex自行实施，尚无Claude审核结论；请复核`09d1b2e`、`2f809de`及运行记录`c5c3f9c`，不能把下面的测试证据当作你已认可。
- 重点：v30→v31版本/范围是否合理；已有整行选择是否保留原始标签与缺失时刻；独立标签审计是否充分；开发期一致性及实验可比性是否成立。
- 用户明确本轮已启动v31按冻结方案继续训练、评估、报告及结果记录，复核不作为继续的前置条件；不扩大范围、不新增运行、不修改冻结设计。请记录你的复核结论。

**结论与版本**
- 审查B；v30冻结实现`09d1b2e`因2023原始重复记录被准备器拒绝，真实fit/测试评估均为0；旧文件和失败证据保留。
- 用户“继续”后，按冻结文件不改规则另建v31，实施提交`2f809de`已推送并部署；模型/损失/3seed/36站/1h滞后/选择统计规则不变。
- 六机2023每台2,174,760行/52,560时刻；52,400时刻有有效/部分行与空行重复，使用字段没有不同非空值冲突。
- 新准备器只选已有、包含该时刻全部已知值的一整行；非空冲突或需要字段拼接均拒绝。所有时间戳保留，不填空标签，不改负值或超额定值，原始档案不变。
- v31协议LF SHA `d323c980e9a6a5293e2c95adc39a4210f56bf2e9953254ce0f66cb34af1ce850`。
- v30只读失败证据：`outputs/audits/upstream_v30/{raw_2023_duplicates.json,preparation_failed.log}`；诊断JSON SHA `7100425ee698875d3580624e11a0d015229ab66e69c23efc3013922536443c18`。

**验证证据**
- 服务器真实torch完整118项测试通过（113项等价覆盖+5项重复规范化测试），preflight重跑118项也全部通过（64.532s）。
- 新数据根`datasets/wind_public_v31`：403,344小时行；开发期2016–22共350,784行×53列与v29逐值一致，功率曲线完全一致。
- 独立原始功率审计覆盖六机八年，小时行删除0；最大标签误差4.547473508864641e-13 kW，仅浮点舍入；异常、负值及超额定标签均保留。
- CSV SHA `dc2365fbda0377f973b02300655067ae2ddbdc1e956742992eee3c0631a80564`；preparation SHA `df41d9992e86658a063626534df889fe7f028a84de606b3e2ab405ed832f8fc3`。
- 审计`outputs/audits/upstream_v31/preparation_label_audit.json`，SHA `23d1f32cc8044703f1daf6113be340c5356fc20bb4c80a82f2f1fb805070e55c`。
- preflight：`outputs/logs/Kelmarsh/upstream_v31_preflight/20261028T071052Z_ed61c8002ec4/preflight.json`；tests/D0/smoke/namespace全部通过。
- source digest `ed61c8002ec46f8c4b98d0f15a6f3ffb9f93e19da3005bfa905e38dc3f9b66a2`；训练ALL/normal=90,710/85,394，验证=16,949/15,382。
- 开发期上游缺失行比例train=0.6619%、validation=0.1142%；仅输入按训练均值填补，标签不填补。
- 本地审计/preflight副本：`output/kelmarsh_hourly/upstream_v31_review/`；未提交原始数据。

**运行与后续**
- 唯一启动：`bash scripts/train/Kelmarsh_launch_frozen_protocol.sh scripts/train_kelmarsh_upstream_v31.py --detach`；Python为服务器timedart环境。
- 运行目录`outputs/logs/Kelmarsh/20261028/upstream_v31_zd2QPm`；指针`outputs/logs/Kelmarsh/upstream_v31_latest.txt`。
- 2026-10-28 07:27:39 UTC确认PID1085945存活，TRAIN=RUNNING/STAGE=T1、REPORT=PENDING；已完成5/9 fits，第6个upstream_all/s2026进入epoch1，未重复启动。
- GPU0 RTX3090；启动时无其他计算进程，按协议90%分配上限、CPU2线程；不修改其他任务/设备设置。
- 预算仅9fits/最多180epochs；2023站点文件和测试窗口仍须S1冻结后E1才能访问；2024未链接/解析，外部下载继续暂停。
- epoch按2022 ALL MAE选择，仅臂选择用MAE/RMSE联合分数；上游容量增加和归档QC实时可用性限制照常披露。
- Holm仅覆盖每队列C1–C3 MAE；farm accuracy仅描述，不称联合多重检验或实时合规。
- 结果尚未返回，不宣称2023收益；完成后按registered arrays复算/验收报告并另提交记录，正常进程不重启。
- 服务器原有两个docs修改、本地无关文件全部保留；旧下载automation-2已不存在。
