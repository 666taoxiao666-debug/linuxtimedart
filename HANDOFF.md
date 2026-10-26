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
<!-- 由 Codex 覆盖。最后更新：2026-10-26，v29 已部署，2022下载进行中 -->

**审查、修正与验证**
- 实现/审核提交 `fcd1185` 已推送并服务器安全快进；旧v25–v28不改、不重跑，服务器两份既有docs修改保留。
- v29：24h→4h，train2016–2020/val2021/test2022，2023–24封存，Wiki关闭；4配置×3seed=12fits/240epoch上限。
- TSMixer last10/state 参数4,262/4,906；统一ALL验证选epoch；Guard按历史末10分钟功率严格<41kW切Persistence。
- 修正协议的2021来源矛盾：该年已用于规则开发和本轮验证；冻结前措辞修正，阈值/训练配方不变。
- 最终协议 LF SHA `b78dd473f6261b4d11de5a77f7bd508424d703f3286952687605f8fabdd50eb7`；服务器full_source_digest `d6893b7306ea964d8b1dd4584b570b4a3f977634e30af631f1105ff49d0cb510`。
- 加固Guard全步有限/恒定锚点校验、报告P1链接/窗口数复核，补逐小时MSE；启动器拒绝冲突模式。
- 服务器隔离副本共98项测试通过（含真实Torch完整流程、预测落盘中断后禁止optimizer.step、v28回归），无跳过；Bash语法/模式互斥通过。
- 本地Python编译/Bash语法通过；本地Windows缺fcntl、WSL缺numpy，不冒称本地完整测试通过。

**真实部署状态**
- 唯一续接PID `953199`；脚本/日志 `outputs/audits/operating_state_v29/{prepare_and_launch_once.sh,launch.log}`，已脱离SSH。
- 续接脚本SHA `830f472d5298bd7b91c492d4aa0bc3e841a7b0cb85ce00427ebeff589c9799fa`；本地同脚本在 `output/kelmarsh_hourly/operating_state_v29_review/`。
- 已重验并硬链接2016–2021官方MD5/SHA及static；登记 `outputs/audits/operating_state_v29/raw_reuse.json`；2023–24未读取。
- 2022文件 `Kelmarsh_SCADA_2022_4457.zip` 总485,620,228字节，官方MD5 `b695d2b21cd5a236b2e690e0a43df5ab`；最后检查分片已收2,359,296字节，进程正常存活。
- 数据目标 `datasets/wind_public_v29/resampled_hourly_scada/kelmarsh_multiyear/`；未生成训练指针，无v29真实fit/测试指标。
- 续接顺序：prepare→独立原标签审计→源指纹复核→preflight→唯一detach训练；任一步失败即停止，不绕过失败、不重复启动。
- 查看日志和 `bash scripts/train/Kelmarsh_launch_frozen_protocol.sh scripts/train_kelmarsh_operating_state.py --status`；下载中不要再开续接器。
- 后续真实审计 `outputs/audits/operating_state_v29/independent_audit.json`；训练指针 `outputs/logs/Kelmarsh/operating_state_v29_latest.txt`。
- gyx PID905508最后占4292MiB；预定本任务4GiB共享分配器上限/2CPU线程，不影响其进程配置。

**NWP结论（详情/来源在fcd1185正文，仅调研）**
- [GDEX d084001](https://gdex.ucar.edu/datasets/d084001/) 覆盖2016–2023全球0.25°；每日4起报，归档3h步长。当前有10/100m风、阵风、温压；历史逐年层次完整性未逐文件确认。
- 起报≠发布时间，归档日期不证明实时可用。历史发布延迟未证实；若预声明6h延迟，4h目标可能需至f018，不能只取f012；ERA5不能冒充实际预报。
- Kelmarsh约52.4015N/-0.9431E，风场远小于网格；误差相关只是共同驱动线索，不能证明天气主因。
- [GDEX子集](https://projectpythia.org/gdex-cookbook/notebooks/services/api-subset/) 需账号token；全全球0–12h八年约12–31TB，区域变量裁剪才合理；未实测速率/队列，不能承诺工期。
- 当前服务器无cfgrib/eccodes/wgrib2/grib_ls；未安装、注册、下载NWP或写NWP协议。NWP许可/变量/体量估算出处见提交正文。
- 限制：ALL是11通道可用的共同窗口；C1/C2增加644参数，C6还改变训练cohort，不能作纯信息归因。
- 请Claude暂不改本轮冻结源码，等唯一续接完成后复核2022双队列/逐小时/配对统计；结果到齐后Codex另提交记录，不能拿v28数字冒充v29。
