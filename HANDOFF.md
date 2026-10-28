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
<!-- 由 Codex 覆盖。最后更新：2026-10-28，v30审查与服务器测试通过，待数据准备/preflight -->

**审查结论：B（可按冻结协议运行；研究结论待2023一次评估）**
- 接收v30四个新文件；未修改v25–v29、筛选脚本或冻结协议。协议LF SHA `c920dd760a404e038182b0bca696c7ed96ba9aa2acf018a7e17890f24a4269be`。
- 已核对1h观测滞后、训练年均值填补、原标签保留、共同窗口、S1后2023站点门禁、每臂损失contract及六机总功率指标。
- 服务器真实torch：原112项测试全部通过；新增1项heldout极值扰动测试通过，确认训练填补/scaler/输入及标签不受影响。
- 测试使用unittest（服务器没有pytest）；日志`/tmp/codex_v30_review_tests.log`；preflight会再次执行完整113项。
- 只新增上述防泄漏测试，未改训练实现或参数。MIDAS清单服务器SHA与协议`00d22a6e…`完全一致。
- 选择规则须准确表述：每个fit按2022 ALL MAE选epoch；三臂之间按MAE/RMSE联合得分选择，不能称epoch联合选择。
- 归因限制：上游组增加12通道/3864参数；归档QC观测的实时1h可用性是假设；2021/22已用于开发筛选。
- Holm仅用于每队列C1–C3的MAE检验；RMSE使用单独CI，不能宣称所有指标联合多重检验已校正。
- 风电场accuracy=100%-归一化RMSE仅描述，非合规结论；既有preparation元数据的旧cohort说明由v30协议ALL主队列覆盖。

**执行范围与下一步**
- 用户此次交接授权v30：2016–21训练、2022选择、S1后2023一次测试；2024仍封存。
- 先提交推送冻结实现，再硬链接2016–23 ZIP和静态文件/缓存元数据，运行原准备与只读标签审计。
- 然后原通用launcher运行preflight，通过后仅启动一个9-fit任务（3臂×3seed、最多180epochs）。
- 外部SDWPF/Penmanshiel暂停；不处理403、不重试。旧automation-2已不存在，未创建重复监控。
- 服务器原有两个docs修改保留；本地无关未跟踪文件不提交。
- 数据审计、preflight与实际启动结果完成后更新本节并另作记录提交；训练结果回来后再记录真实结论。
