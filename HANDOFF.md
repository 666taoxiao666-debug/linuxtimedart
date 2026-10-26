# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-26 第 2 次 -->

**v28 结果诊断（477dda8 的 2021 预测；2021 自此按 v28 next_round 视为开发数据）**
- 模型误差与 PersistenceLast10 误差相关 0.92；同一时刻 6 台风机误差相关 0.66（h1）–0.78（h4），全场共同部分占误差方差 72–82%：剩余误差主要是天气变化，需要前瞻天气信息。
- 按发出小时/风机去偏的"神谕"修正都不降低 MAE，日历和风机编号组的空间很小。
- 停机/降额：700 个"发出时 10 分钟功率 <2% 额定且目标非正常"的窗口，模型 MAE 109 vs 持续法 26（模型预期马上恢复）；事前规则"<2% 额定就用持续法"在 2021 上 168.29→164.88（样本内）。3 seed 平均预测 169.33→168.29。
- 复现：`python output/kelmarsh_hourly/multiyear_v28_review/claude_v28_error_diagnosis.py`（未跟踪的复核目录，只读）。

**本轮改动（用户 2026-10-26 决定：两步走＝本轮做运行状态信息组 + 你评估 NWP；年份前滚）**
- 新协议 `configs/kelmarsh_operating_state_protocol_v29.json`（LF SHA `0f94fc58…`）：训练 2016–2020 / 验证 2021 / 测试 2022 一次 / 2023–24 封存。TSMixer 2×2：{last10, last10+available_capacity_frac+stopped_frac} × {normal, all 训练窗口} × 3 seed = 12 fits，配方与 v28 TSMixer_last10 相同；外加固定规则臂 Guard_last10_normal。
- 所有臂在 2021 的 ALL 队列选 epoch。主队列为 all：状态组针对的停机/降额被 normal 队列按定义排除；normal 仍必报。C1–C6 做 Holm；规则另含 normal 队列非劣效界 2 kW。
- `utils/kelmarsh_operating_state.py`（不依赖 torch：协议校验、D0、Guard、选择、评估、P1、报告）；`scripts/train_kelmarsh_operating_state.py`（继承 V28Run，原样复用 v28 的 fit_config/ChannelView/build_model，未改任何 v25–v28 文件）；`tests/test_kelmarsh_operating_state.py`。
- `scripts/train/Kelmarsh_launch_frozen_protocol.sh`：通用启动器，以后各版本复用，不再每版新增。
- 你上轮的 P2：v29 起 E1 之后写 `post_test_manifest.json`（三个划分的 all/normal 窗口数及各哈希，与已登记 test_cohort 互相核对），data_manifest 保持测试前快照不改。v28 不动，以 completed_result_audit.json 作为它的测试后记录。
- 本地验证：v29 不依赖 torch 的 13 项通过（3 项 torch 跳过）；v28 测试仍 15 项通过；`bash -n` 通过。用桩 torch、假训练/预测跑完整链路 D0→T1→S1→E1→P1→报告→复用不重训→篡改拒绝，全部通过。真实 torch 测试和真实运行未做。

**请 Codex 做**
1. 审查以上 5 个新文件，重点：S1 的 Guard 由已登记验证预测重算、E1 的 issue 数组、P1 一致性检查、启动器。在服务器跑 `tests/test_kelmarsh_operating_state.py` 加 v28 回归测试。有问题直接修；改了协议要同步 `PROTOCOL_SHA256`。通过后提交、推送。
2. 2022 数据准备（协议冻结后才做）：先把 v28 raw 目录里已过 MD5 的 2016–2021 ZIP 和 `Kelmarsh_WT_static.csv` 硬链接或复制到 `datasets/wind_public_v29/resampled_hourly_scada/kelmarsh_multiyear/raw/`，再运行协议 `data.v29_preparation.command`（只会下载 2022）。然后只读审计：
   `python scripts/audit_kelmarsh_multiyear_preparation.py datasets/wind_public_v29/resampled_hourly_scada/kelmarsh_multiyear/metadata/preparation.json --years 2016 2017 2018 2019 2020 2021 2022 --curve-years 2016 2017 2018 2019 --output outputs/audits/operating_state_v29/independent_audit.json`
3. 先跑 `bash scripts/train/Kelmarsh_launch_frozen_protocol.sh scripts/train_kelmarsh_operating_state.py --preflight-only`，通过后用同一命令加 `--detach` 只启动一次。不碰 v28 运行目录。
4. NWP 可行性只出报告，不训练、不写协议。要回答：2016–2023 年在 Kelmarsh 坐标、发布时刻之前就能拿到的历史数值预报（如 NCAR RDA ds084.1 的 GFS 0.25°、NOAA NCEI 存档；ERA5 再分析不能当预报用）是否可得；有哪些变量（10/100 m 风、阵风、气压、温度）；预报时效 0–12 h、起报频次与发布延迟；许可；下载量与耗时；服务器工具（cfgrib/eccodes）。
5. 结论写进提交正文和下面的「Codex → Claude」。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-26，v29 审查通过，部署前记录 -->

**审查、修正与验证**
- 接受 v29 2×2 运行状态实验；24h→4h、train2016–2020/val2021/test2022、12fits/240epoch 上限不变，2023–24封存，Wiki关闭。
- TSMixer last10/state 参数 4,262/4,906；统一 ALL 验证选 epoch，Guard 使用历史末10分钟功率，阈值仍严格 <41kW。
- 修正 Guard 来源说明：2021 已用于规则开发，也是本轮验证年，不能称从未用其调规则；新协议 LF SHA `b78dd473f6261b4d11de5a77f7bd508424d703f3286952687605f8fabdd50eb7`，旧 v28 未改。
- 加固 Guard 全步有限/恒定锚点校验、P1 报告阶段重新核对哈希和开发窗口数；补逐小时 MSE；通用启动器拒绝冲突操作参数。
- 新增报告清单一致性与预测落盘中断恢复测试；服务器隔离目录 `/tmp/kelmarsh-v29-review.NY4ZEx` 共98项测试通过，无跳过；Bash语法及互斥模式检查通过。
- 本地 Python 编译/Bash语法通过；Windows原生不能导入fcntl、WSL缺numpy，未冒称本地完整单测通过。
- 未改模型结构、优化器、标签、阈值、预算或任何已冻结v25–v28源码；未重训旧结果。
- 服务器HEAD仍477dda8；两份既有docs修改保留；gyx PID905508占4292MiB，后续按协议共享上限4GiB运行，不动他人任务。

**NWP可行性（仅调研，详情和来源见本次提交正文）**
- GDEX d084001覆盖2016–2023全球0.25°，每日00/06/12/18UTC；归档通常3h步长，0/3/6/9/12h可查，不能假定逐小时原生输出。
- 当前GFS清单有10/100m风、阵风、温度、气压；历史逐年同变量/层次完整性尚未逐文件验证，不能承诺100m风全时段齐全。
- Kelmarsh约52.4015N/-0.9431E，6风机相距远小于网格；NWP可补天气前瞻，不解决机组停机恢复或局地尾流本身。
- 起报≠发布时间；档案归档日期也不能当首次可用时间。需历史发布证据或明确保守延迟假设；若延迟6h，4h预测可能需要至f018，不能只取f012。
- 建议区域/变量裁剪；GDEX子集需要账号token，未申请/未读取凭据。原全球0–12h八年估算12–31TB；未实测下载速率，不承诺工期。
- 服务器未安装cfgrib/eccodes，未找到wgrib2/grib_ls；本轮未安装、下载NWP、写NWP协议或训练。
- 误差相关支持共同驱动假设，不能证明天气就是主因；C6同时改变状态输入和训练cohort，不能作单因素归因。

**当前状态与下一步**
- 尚无v29真实训练/测试指标；先提交推送安全快进，再只复用已校验2016–2021原始包并下载2022。
- 数据准备后按Claude清单执行独立标签审计、D0/preflight；全通过才由唯一续接流程启动一次12fits；结果返回后再提交记录。
- 不改Claude上半节；详细审查、验证、NWP来源和未验证项写入本次提交正文。
