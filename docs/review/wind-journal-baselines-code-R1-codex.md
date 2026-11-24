# 第 111 次 · 三场期刊模型对照代码 R1

基线 `db32df0`，分支 `exp/X1`。审查范围为本轮四条模型线、协议、测试、作者文件及启动链。初审需修改。

## ★1–★4 和验收

- ★1 通过。按固定提交核对 Cabin 三个 Python 文件、TSG 分区函数和两份 MIT 许可证。TSG 张量文件还原第 8、11 行后与上游 blob `84086a27` 完全一致。[Cabin 源码](https://github.com/xiufengliu/cabin-wind-forecasting/tree/392f103426faf600c5469875b020a4693cba6a7d/cabin/models) · [TSG 源码](https://github.com/akylasstrat/wind-forecast-missing-data/tree/409f20f0d1117758b9f2e9ff48c4373fe97e945c)
- ★2 连续机组编码通过。两侧原始构造共用同一机组映射，树不把机组当类别。完整的入模数值一致性另见 R1-B2。
- ★3 通过。受控树调用作者的 apply/predict。全有效停根节点并调用名义模型，只缺其他列走左鲁棒，只缺分裂列走右名义，两列都缺走右鲁棒；四个分支均有可区分的输出。
- ★4 通过。上游训练脚本传入 max_gap，实际分裂判据读取 max_gap_threshold，默认 0.001；本地显式传 0.001 复现该行为。[训练调用](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_main_model_train_NYISO_weather.py#L506)
- TSG 13 列验收通过。LG 10–22、LUG 21–33 逐窗对应 x50 21–33，原始有效位取 34–46。受控树通过，左右支互换的反例被拒。Cabin 独立性与双模型输入验收分别见 B3、B2。
- KRR 从拟合段标准化、抽中心，按 V1 选点，全部风场写完冻结记录再计算 V2 预测。

## 初审发现（位置按送审版）

- R1-B1【阻断】【确定缺陷】开发缓存先读取越界成员，再筛日期。run_wjb.py:218 调用 run_x1.py:106 解出每个 npz 成员。X5 同一对缓存曾在 K 的 2022 年与 P 的 2021 年分别使用 2864、2768 个周期（两场 X5 audit.json:135）。需先查成员名和日期再取值，用禁止访问未来成员的合成缓存验证。此处发现的是读取范围，开发拟合使用未来值没有证据。
- R1-B2【阻断】【确定缺陷】paired_inputs 给原始输入签名，LightGBM-hist 直接使用，Cabin 随后单独 MinMax。拟合段功率 0/500/1000 时，同一位置分别输入 500 和 0.5，两份原始摘要相等。按已审 R2 文档第 20 行，两臂应对实际入模张量做展平还原和逐段哈希比对。
- R1-B3【阻断】【确定缺陷】independence_gap 把 8 个早窗一批预测。仅在 batch_size=1 时输出偏移 1 的模型通过现门禁（差 0），实际单窗差 1，高于 1e-5。须逐个预测这 8 个早窗，再比较同窗混入 64 个晚窗的输出，并保留作者 sample softmax 的反向对照。
- R1-B4【阻断】【确定缺陷】正式 TSG 的预估与 STAGE_PLAN 第 98 行的单场停止口径不一致。按代码的全场合计 15 h，即使采用该总预算，仍会误停：8 workers，48 棵 NN 各 1.3 h、48 棵 LR 各 0.1 h，最慢类优先派发使首批外推 15.6 h，整批实际只需 8.4 h（忽略启动开销）。需由 Claude 记录预算究竟按场还是按总量，并给出兼顾两类任务的可测停止规则。冒烟没有此时限。

## 验证记录

- 本地 Windows timedart / torch 2.7.1，送审版三组共 72 项，71 通过，1 项纯统计测试因导入 Linux fcntl 报错；真实 torch 的 Cabin 测试已执行。

## 修复后结论

- B1–B3 已按原方案在工作区修复。现在按成员名先筛起报时间、再解包，用边界哨兵测试验证。Cabin 与 LightGBM-hist 共用拟合段 MinMax，对最终入模历史签名。同窗逐个预测，再混入晚窗比对。三个反例均纳入测试。统计测试仅在 Windows 临时隔离 fcntl 导入。
- 工作区快照的 Git blob 为 run_wjb.py `7864a81a`、test_wjb.py `eca2b6c6`、wjb_protocol.json `6bdf1b69`。本地 timedart 的三组 unittest 共 75 项全部通过，零跳过，18.492 s。两份 shell 脚本通过 bash -n。另用真实 torch 与作者 TSG 模块核对 364 个合成路由样例，根、左、右名义、右鲁棒分别为 121、61、1、181。
- **R1 结论为需修改，剩余 B4 一项。** 三项验收在本地合成数据通过。服务器单测未取得回执：反重力 CLI 缺少 ANTIGRAVITY_LS_ADDRESS，桌面访问未获准。待 Claude 根据 B4 统一方案、实现及停止规则测试，R2 再按实际提交完成服务器单测与冒烟交接。
