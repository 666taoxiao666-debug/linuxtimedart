# X2-A2 R2/2：第93次修法复审

结论：通过，R1-B1/B2关闭，无新阻断；只复审本轮改动及必要来源接口。

- R1-B1通过：past_history按issue取t0-5至t0的原始历史值；VB要求hist6形状正确，缺少直接失败，不从x50的零值猜缺测。有效值等于均值保留，真实NaN不计入至少3小时门槛。
- hot_test_history小时轴通过：2023-01-01T01:00至2026-01-01左闭右开，最后小时为2025-12-31T23:00；与run_x1.dense_pack HoT路径及既有封存准备小时轴对应，包含2024/2025测试窗口需要的历史上下文。
- 来源通过：读取G2/g2b84_bc7730e_r1/sealed_prep/points.npz；使用同一cached_hourly周期/lead插值与场中心列0；stats来自g2b_4471499_shared_r1/formal并经frozen_gate校验。记录点缓存实际SHA，不读取新SCADA或2026数据。
- 原冻结准备对接近2026截止周期有额外保护；测试包本身按周期截止条件筛过，所需过去小时不越过该截止，复用缓存不为本轮引入新周期信息。
- 每个非训练包先bind_history抽最多20000窗口复现24步历史通道（1e-4），训练包已由farm_curves/bind_hourly核验；past_history对所有窗口查询6小时轴，缺失小时报错。
- R1-B2通过：a2_execute先遍历全部风场prepare，不调用LightGBM拟合，全部成功才写A2/preflight.json；随后逐场重建并核对指纹后才拟合该场。预检末场失败时拟合次数0已测试。
- 指纹覆盖样本键、hist6、两种曲线摘要和缩放参数；第二阶段若后续场发生漂移，允许此前已通过的场完成，但停止漂移场，不声称撤销既有拟合。
- 旧曲线核验通过：按dev_freeze中的SHA读取curves.json，再检查机组集合、来源、样本数和点值（绝对容差1e-9）。
- R1-B3/S1已由Claude复核采纳；VN/VB负风速截0在方案中明确，不改变原判定阈值或三变体预算。

本地验证：python -X utf8 -m unittest tests.test_x2 tests.test_legacy_bases，30项，25通过、5因无torch跳过，8.013秒。
X2 19项全部通过；Legacy numpy6通过。覆盖有效均值/真实NaN、缺hist6失败、历史绑定不一致、全场预检失败0拟合、阶段间漂移和旧曲线篡改。
git diff --check通过；未访问服务器、未训练、未安装、不处理B线。
没有Codex执行代码修改；证据等级B（静态接口和本地合成回归通过，真实数据需Cursor门禁核验）。
交Cursor：批准提交的新干净副本仅运行a2，完成后report2；沿用原方案不另跑冒烟，真实预检失败即停，不重试、不放宽容差。B线留在91d4d20原实例。
