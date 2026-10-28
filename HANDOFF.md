# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-28 第 3 次 -->

**审核与复核结论**
- `31faad6`（v31 结果记录）：**通过**。我用本地副本独立复算：5 个臂 ALL 队列的 MAE/RMSE/R² 与 decision 逐值一致；C1/C2 点增益独立重算一致；三条规则按规则文本复算与登记一致。
- C2 `balanced_loss_supported=true` **符合冻结规则**：RMSE 下限 0.410 > 0；MAE 下限 −1.269 > −2。须表述为"RMSE 显著下降，MAE 在 2 kW 非劣效界内（点估计 −0.81，整个 CI 都更差）"，不是"两项都改善"。
- 验证选择差距仅 1.763087 对 1.763634（约 0.03%），报告与论文须披露这一点。
- 本轮没有 Codex 待审改动。P3（turbine_id 依赖调用顺序）已在 v32 新的数据脚本中解决，v31 不改。

**用户决定（2026-10-28）**：v32 = 用封存的 2024 年对 v31 设计做一次确认，此后不再有封存年。

**本轮改动（工作区未提交，5 个新文件，没有改任何已有文件）**
- `configs/kelmarsh_upstream_protocol_v32.json`（LF SHA `b9fbdce2…`）
  - 臂、特征、36 站、1 h 滞后、损失、seed、联合选择、规则都与 v31 相同，测试会逐项核对。
  - 年份：训练 2016–2022 / 验证 2023 / 测试 2024 一次；`sealed_years=[]`。
  - 开发期站点文件来自两个根目录：2016–2022 在 development，2023 在 sealed（v31 E1 已打开过）。
- `utils/kelmarsh_upstream_v32.py`：通用部分全部从 v31 导入；只新增 v32 的校验、显式风机编号的准备核验，以及年份取自协议的报告。
- `scripts/train_kelmarsh_upstream_v32.py`：继承 `V31Run`。T1/S1/E1/P1/decide 原样继承；只覆盖 D0（参照 v31 CSV）、站点根目录、名称/指纹、报告。
- `scripts/prepare_kelmarsh_multiyear_normalized.py`：按协议驱动，以后各版本复用；`prepare` 复用冻结的 v31 `canonical_ticks`，风机编号取自准备器自身循环；`audit` 复用冻结的 v31 独立功率审计。
- `tests/test_kelmarsh_upstream_v32.py`
- 本地验证：
  - v32/v31/v30/v29/v28/筛选全部不依赖 torch 的测试通过，16 项需 torch 的跳过。
  - 桩 torch 端到端：9 fits、S1 先于测试、续跑不重训。
  - 合成 MIDAS 两根目录：冻结前拒绝打开 2024；加入 2024 后开发期特征逐值不变。
  - 未做：真实 torch 测试、真实数据 D0。

**请 Codex 做**
1. 审查 5 个文件，在服务器跑 v32 与回归测试。需要修改就按门禁先写方案、标"待 Claude 审核，未实施"；无问题则提交、推送。
2. 冻结后准备数据：
   - 把 v31 raw 里 2016–2023 的 ZIP、`Kelmarsh_WT_static.csv`，以及封存的 `Kelmarsh_SCADA_2024_5962.zip`（MD5 `e9e3318f…`）硬链接到 `datasets/wind_public_v32/resampled_hourly_scada/kelmarsh_multiyear/raw/`。
   - 运行协议里的 `prepare` 命令，再运行 `label_audit` 命令。
3. preflight 通过后，用通用启动器加 `scripts/train_kelmarsh_upstream_v32.py --detach` 只启动一次。
4. 完成后：结果副本放到 `output/kelmarsh_hourly/upstream_v32_review/results/`；在「Codex → Claude」逐项写出 C1–C3（MAE/RMSE 增益与 CI、Holm p）、四条规则的复算、验证选择与风电场准确率。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-29，v32完成与结果复核 -->

**结论与完整性**
- 【最终评级】B（结果复核）；【必须立即修复】无；【需重跑/补测试】无。本轮仅核查/同步/记录，未改冻结实现或新增训练。
- v32的9/9 fits、测试及报告全部完成；2026-10-28 15:04:42 UTC（北京时间23:04:42）正常退出code=0，TRAIN/REPORT均COMPLETED。
- 运行`outputs/logs/Kelmarsh/20261028/upstream_v32_huL1ai`；实现`c2d39da`；准备/审计/preflight及启动证据见`0c9bfda`。
- 服务器completed清单94/94 SHA256一致；结果副本已放`output/kelmarsh_hourly/upstream_v32_review/results/`，42文件/6,318,474字节，不含模型权重。
- 传输归档SHA256=`115c642c97cf70d9d959c87ce6020118ea1365e0b8bfee1ac2f48d598ca4dfdf`与服务器一致；其中38项本地文件再与completed清单一致。
- decision SHA=`461c7ec39d1169109d8abe9ae7e609908aee728f17a47a991ae9b052a4c36a10`；completed SHA=`791435ba6e4a68f8662a98b37b6d5de567ebdff551b8ce7211a3379683f6c563`。
- 已只读调用evaluate_registered并check_against_registered：指标/比较/规则/集成/farm/regimes一致；另独立重算Holm、三条规则、验证均值及联合得分，全部通过。
- S1于15:03:21 UTC冻结；15:03:51才登记打开测试series，15:03:54预测，15:04:11登记decision；测试ALL/normal=16,738/15,478（92.472219%）。
- 复核命令最初误把protocol.json外层记录当协议而报KeyError；改用其中protocol后通过，仅核查命令错误，未修改或重跑实验。

**C1–C3（增益=reference−candidate，kW；CI95；p为MAE Holm）**
- C1=upstream对last10；C2=balanced对upstream；C3=2023验证选中的balanced对PersistenceLast10。

| 队列/比较 | MAE gain [CI95] | Holm p | RMSE gain [CI95] |
|---|---|---|---|
| ALL C1 | 18.199788 [15.244327,21.058694] | 7.789603e-35 | 28.272626 [23.619050,32.647175] |
| ALL C2 | -1.063092 [-1.562083,-0.502714] | 8.832553e-05 | 1.102209 [0.219163,2.055967] |
| ALL C3 | 34.891917 [30.539004,39.394254] | 2.664343e-55 | 66.965539 [59.911429,73.903027] |
| normal C1 | 19.008914 [15.940237,21.983406] | 2.332913e-34 | 29.390110 [24.620032,34.024103] |
| normal C2 | -1.021648 [-1.520220,-0.448805] | 0.000254350 | 0.760646 [-0.030953,1.661553] |
| normal C3 | 36.407268 [32.049448,40.979328] | 1.452945e-54 | 67.018749 [59.841418,73.821105] |

**冻结规则逐项复算与限制**
- upstream_information_supported=true：ALL C1 MAE下限15.244327>0、Holm≤.05、RMSE下限23.619050>0；normal C1 MAE下限15.940237>−2。
- balanced_loss_supported=true：ALL C2 RMSE下限0.219163>0、MAE下限−1.562083>−2；MAE点估计及整个CI均更差，不能称两项都改善。
- beats_naive=true：ALL/normal C3 MAE下限30.539004/32.049448>0、两者Holm≤.05；ALL RMSE下限59.911429>0；三条与decision一致。
- 协议只有3个布尔规则；第4项attribution_limits是文字约束：C1多12通道/3864参数，归档天气假定滞后1h，不能建立Wiki贡献、因果归因或实时可用性。
- normal C2 RMSE的CI跨0，不能称该条件队列RMSE显著改善；normal为事后条件指标。Holm仅覆盖各队列C1–C3 MAE。

**验证选择、风电场与Claude下一步**
- 2023验证选中balanced：三seed均值MAE166.048789、RMSE247.555292，联合分数1.771990046164；upstream=1.773057905639，差约0.06023%，须披露选择接近。
- last10验证参照均值MAE186.610977、RMSE280.618461，分数2；独立重算逐seed均值、归一化求和与排序一致，未按2024改选。
- 六机2627个完整issue时刻、容量12300kW；accuracy=100−NRMSE%：Hour86.393612、Last10 86.255952、TSMixer基线87.708322、upstream88.993900、balanced89.042323%。
- balanced的1–4h accuracy=92.665184/89.685935/88.161259/86.585590%；仅描述，不等同15分钟法规口径或单机R²。
- 2024一次性确认支持冻结规则；此后无剩余封存Kelmarsh年，不追加运行。请Claude复核本地副本、结论措辞和记录后再设计下一轮；新改动仍先审核。
- 无关本地文件与服务器原有docs修改保留，外部403下载未动；本轮没有待Claude审核的实现修改。
