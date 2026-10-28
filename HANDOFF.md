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
<!-- 由 Codex 覆盖。最后更新：2026-10-28，v32准备/审计通过并启动 -->
 
**本轮结论与批准范围**
- 【最终评级】B：实现审查、真实数据审计、D0和preflight通过；真实训练运行中，尚无v32测试结论。
- 【必须立即修复】无；【需重跑/补测试】无新增要求，等待本次冻结运行自然完成；不另开运行。
- 按Claude第3次交接执行；5个文件原样采用，Codex未修改代码/测试/数据处理/实验设计，无待审实现。
- 冻结提交`c2d39da`已推送并部署；上轮网络阻塞记录`95186d1`已推送，本轮SSH重试后恢复，未改网络配置。
- 协议LF SHA256：`b9fbdce22a83dd0e5128aa3d9a489cbcb725c4f298f2feba6fe15f028f7a3588`。
- 训练2016–2022、验证2023、一次性测试2024；模型/损失/36站/1h滞后/seed/联合选择/判定规则均沿用v31。

**准备与独立审计**
- 已按批准硬链接v31的2016–2023原始ZIP/静态表及已验收2024 ZIP至v32 raw；复用缓存的固定Zenodo记录，未追加下载。
- prepare正常完成：456,048行，年份2016–2024、曲线拟合2016–2019；6台风机显式ID，时间戳及非空值保留，填补/冲突值均0。
- CSV SHA256：`96acfcabb8399ddca415a61011b3cf02ca9a675d7fcc31f2ef70db89a3cb37ef`。
- preparation.json SHA256：`6a5a3061a7008f6a4d56f2eba68eb733472aaeb3ed1659594a98463b9d6d7677`。
- 独立审计`outputs/audits/upstream_v32/preparation_label_audit.json`：passed=true，输入/源码哈希均匹配，54个质量行及UTC表头通过。
- 独立原始功率抽核5,880小时，最大差值4.547473508864641e−13 kW；39,756个负值标签、1,798个超额定标签原样保留，未裁剪/插补。
- 准备/审计日志：`outputs/audits/upstream_v32/{prepare_20261028T1420.log,label_audit_20261028T1426.log}`。

**D0、preflight与启动**
- 先前扩展回归143项通过；本轮协议要求的preflight回归127项/79.992秒通过；真实D0、namespace、bash语法、真实torch合成smoke均通过。
- preflight目录：`outputs/logs/Kelmarsh/upstream_v32_preflight/20261028T143148Z_99e4ec27b379`。
- preflight.json SHA256：`841b482b985fef1fa498eb4e81a41f3aaf9e0ed5fcd5c930a9df749257fa521f`。
- D0对照v31：403,344行×53列全部一致，功率曲线一致；test_windows_instantiated=false。
- 开发窗口ALL/normal：train=107,665/100,782；validation=16,716/15,419；未提前读取2024 MIDAS站点或构造测试窗口。
- 2026-10-28 14:35:15 UTC（北京时间22:35:15）通过通用启动器`scripts/train/Kelmarsh_launch_frozen_protocol.sh scripts/train_kelmarsh_upstream_v32.py --detach`仅启动一次。
- 运行目录：`outputs/logs/Kelmarsh/20261028/upstream_v32_huL1ai`；source_commit=c2d39da；launcher PID1127684，Python PID1127736（后续以完整命令确认）。
- 已核实真实Python命令、latest指针及upstream_v32.lock写锁；首个TSMixer_last10_all/s2024已完成epoch1，正常训练中，不是已跑完。
- 按冻结方案9 fits；S1前只用开发期，S1后由原E1打开2024站点/测试窗口并再次验证开发输入、标签、窗口及缩放器不变。

**待结果返回**
- 无需Claude再改代码；用户说“GPT好了”时可先复核本记录，最终结果需等待本次运行及报告结束。
- 完成后同步`output/kelmarsh_hourly/upstream_v32_review/results/`，复算C1–C3 MAE/RMSE区间、MAE Holm、三条布尔规则、验证选择及farm accuracy，另列attribution_limits。
- 不将C2非劣效称为两项都改善；C1额外通道/参数及归档天气实时可用性限制继续披露，normal/farm指标仅描述。
- 2024确认后无剩余封存Kelmarsh年份；不按结果改方案或追加运行。后续Codex修复仍须先提案、待Claude明确审核。
- 无关本地文件和服务器原有docs修改保留；外部403下载未动。
