# wind-journal-baselines-result · 第 113 次 R1 结果复核

结论为通过，阻断 0 项。按正式结果返回前登记的两问复核，使用本地归档预测，无新增拟合或预测。

## 身份与独立重算

- 运行提交 `1fe7ab6ff37a1e21cbd8d1895d3fc09140f4686f`。五个正式目录的完成回执 SHA256 均符合反重力交付清单，启动回执、Git 中的脚本和协议摘要一致，report 记录的 9 个供应方文件 blob 与运行提交一致。TSG 日志记录 96/96 棵树，六个风场/信息臂对的 6 h 与 18 h 预测均有差异。
- 已取回的 102 个 V1/V2 预测数组与 24 个键包逐文件符合四条线的 registry。四条线的 V1/V2 起报时刻、机组和真值字节摘要均与 X2 固定队列一致。另核对 X5/X2 的参照完成回执、清单和 30 份 V2 参照预测。训练缓存及模型归服务器保管，本轮复核对象为回传预测和审计记录。
- 四份 V1 冻结记录与 report 所引用的选择一致，30 组所选配置均对应记录中的最低 V1 MAE。Cabin 与 LightGBM-hist 在三场两臂的训练、V1、V2 输入摘要 18/18 相同。
- V2 逐窗核对的窗口数/起报簇数为 Kelmarsh 8,249/1,405、Penmanshiel 16,176/1,256、Hill of Towie 15,103/2,196。用保存的真值和预测独立计算汇总及 h1–h4 的 MAE、RMSE、R² 共 990 项。
- 对模型差、上游增益和增益变化共 81 组逐窗配对，按起报簇做长度 8 的非循环移动块抽样、2,000 次、seed 2024，按簇内窗口数加权。点估计及区间吻合，指标、配对统计与原报告最大绝对差小于 2×10⁻¹³。
- 正式 results.json SHA256 为 `974bd070f8dae47d246765f6b0a687cd32daa874c8cb5b9746ba3bbbdbddfebd`。

## 已登记两问

1. 各自与指定参照配对。核岭回归 12/12 格的同延迟 LightGBM 更准（11.49–44.82 kW）。TSG 12/12 格的混合延迟 LightGBM 更准（16.58–49.92 kW）。Cabin 的同输入 LightGBM-hist 在 5/6 格更准，Kelmarsh LU 未检出差异。三组各自解释，逐格判读按 0.5 kW 与未校正的 95% 区间。
2. 核岭回归三场的 U(18)−U(6) 为 10.80、8.71、18.72 kW；TSG 为 12.46、6.81、23.55 kW。两者分别有 3/3 场方向一致且 CI > 0。HoT 的 TSG 从 −18.13 转为 +5.42 kW。18 h 表示未来目标超出已取 f018，较旧周期的附加效应引用 X5b。

## 写法与交接

[三组主表与上游增益](../../output/thesis_materials/wjb_development_comparisons.md)逐场注明 V2、单种子、TSG 预算和掩码适配、Cabin 输入。确认性主张使用三场 N 对 B 的首次测试。请 Claude 用本地键包和数组独立复算关键差值及区间，复核[主张修订稿](../../output/thesis_materials/wjb_claim_revision.md)和[摘要结论](../../output/thesis_materials/wjb_abstract_conclusion.md)。
