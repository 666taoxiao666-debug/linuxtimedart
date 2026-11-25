# AEI 现有材料核对，第 119 次

主线基线 `10da29e`，首轮通过。核对范围为已有导读、数据出处、版式要求与正式结果状态。

- R1-B1【阻断】【确定缺陷，已修复】导读表 1 漏列 `configs/g2a_protocol.json`。G1 完成 60 次拟合，G2a 以 `train_g1.py --g2a-parent` 复用 B/N，新增 B0/N0 共 12 次。对照 `output/g2a/preflight80/launch_receipt.json`、`output/g2a/results81/launch.json`。
- R1-B2【阻断】【确定缺陷，已修复】表 4 的本地统计来自 `scripts/analyze_info_value_x5.py`，源码哈希与 `output/x5/local_blocks12/results.json` 一致。延迟拟合与表 4b 使用 `scripts/run_x5.py`，分别依赖 X5、X5b 协议。
- 表 1 的 HoT 行分别对应开发准备、G2b 冻结及封存评估；正式评估的入口及参数见 `output/g2b/sealed_eval/server/formal_launch.json`。G1、G2a、HoT 封存结果和期刊对照报告的 JSON 按仓库 LF 字节复算，均与完成回执一致。
- 表 2 和 §5.5 对应 X1 的 A0–A3 与 E1/E2；§5.3、表 2b 分别对应 X2 的首轮 report_a 与 B2 report；§5.4 的两轮适配对应 X3/X4；表 5–6 对应 WJB 正式 report；表 3 逐阶段对应 DECISIONS 登记。表中路径已按 Git 跟踪内容核对。
- Penmanshiel 实际准备记录 `output/penmanshiel_hourly/prep_v35/preparation.json` 指向 16807304，发布页标为 v3。GFS 按起报年区分 2016–2020 的 GDEX 月度子集和 2021–2026 采集期内的 AWS 数据；年份范围见 B2/B1 年度汇总。这是归档来源，已验收的缺失起报不补算。[Penmanshiel v3](https://zenodo.org/records/16807304) · [GDEX d084001](https://gdex.ucar.edu/datasets/d084001/)
- 2026-11-25，浏览器直接查阅 AEI 指南 Abstract、Keywords、Highlights。摘要最多 250 词，要求 1–7 个英文关键词；Highlights 3–5 条，每条最多 85 字符（含空格）。现有材料为 245 词、6 个关键词，独立 DOCX 五条与原稿一致，长度 76、80、79、82、82。[AEI 作者指南](https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors)
- 本次是实验与材料核对记录。下一次起草或续写论文需由用户明确提出，最终标签待用户确认。
