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
<!-- 由 Codex 覆盖。最后更新：2026-10-28，v30原始重复记录阻塞；用户“继续”后登记v31 -->

**范围与审查**
- v30已冻结/推送`09d1b2e`；真实torch原112测试+新增heldout隔离1测试通过，真实fit=0，2023模型评估=0。
- v30准备被原脚本`merge_ticks`拒绝：2023每台2,174,760行/52,560时刻，去完全重复后104,960行；52,400时刻含非完全相同行。
- 六机使用字段均无多个不同非空值；重复是有效/部分行与空行。原ZIP SHA `3321f24015551546dfecc6686fa462571c8d3e717ed135b14aa73fcea494afb7`与验收一致，不是传输损坏。
- 只读诊断：`outputs/audits/upstream_v30/raw_2023_duplicates.json`，SHA `7100425ee698875d3580624e11a0d015229ab66e69c23efc3013922536443c18`；失败日志同目录`preparation_failed.log`。
- 用户要求“继续”；按AGENTS冻结文件不改，保留v30及失败证据，另建v31，仅修复原始导出重复记录规范化。
- 新准备器选取同一时刻已有的完整一行，须包含所有已知字段值；任何非空冲突或需要字段拼接均拒绝。
- 每个时间戳保留；全空标签仍空，负值/超额定值不改；不补值、不均值化重复行，不覆盖原始档案。
- 独立v31标签审计从原始功率单独重建，核对六机八年全部小时；准备器/审计器原冻结文件均未改。
- v31模型、损失、36站/1h滞后、train-only输入填补、3臂×3seed、S1/E1门禁、选择/统计规则均沿用v30。
- v31协议LF SHA `d323c980e9a6a5293e2c95adc39a4210f56bf2e9953254ce0f66cb34af1ce850`；数据根`datasets/wind_public_v31`。
- v31服务器真实torch完整118测试通过（61.510s），包含113项同等覆盖+5项重复规则；日志`/tmp/codex-v31-review-tests-complete.log`。
- 审查B：epoch仍按2022 ALL MAE选，三臂之间联合MAE/RMSE选择；容量增加与归档QC实时可用性限制保留。
- Holm只覆盖每队列C1–C3 MAE；farm accuracy仅描述；不能称所有指标联合校正或实时合规。

**执行**
- v31先测试、提交冻结，再准备2016–2023、独立审计、D0/preflight；全部通过才启动唯一9-fit任务。
- 2023站点文件仅S1后E1打开；2024持续封存；外部下载暂停、403不处理；旧automation-2已不存在。
- v30只有raw硬链接/元数据缓存，无准备好的CSV、训练指针或fit；不删除失败证据。
- 服务器两个原有docs修改与本地无关文件保留；测试使用unittest，无需安装pytest。
- 数据审计/preflight/实际启动结果稍后覆盖本节并另作记录提交。
