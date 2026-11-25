# wind-journal-baselines-paper 第 2 轮终核

第 118 次，基线 `a160f2c`。摘要、Highlights 和参考文献通过，遗留阻断 0。R1-B1–B3 已登记，逐机组、站网用途及线性候选失败范围的修订均保留。

## 摘要与 Highlights

- 表 1 的 4 组配对支持已有站点输入时 GFS 仍降低 MAE。已有 GFS 时，站点增益为 0.4–1.5%，仅两组的主区间下界大于零。摘要区分首次评估、HoT 两个封存年的预登记确认及其余探索性结果。
- 表 4、4b 支持开发集范围 1.82–2.28、15.50–24.38、3.75–12.80 kW。事后比较限于 6 h 与 15 h，匹配覆盖后换旧周期的区间均跨零，Penmanshiel 的上界仍达 3.50 kW。
- 表 5、6 对应同输入、单种子开发段的适配结果。30 个配对中 29 个的区间支持 LightGBM，余下一个区间含零。两种使用 GFS 的适配模型在三场均重现了站点增益随覆盖减少而上升的方向。§5.4 与表列判据支持摘要所述预登记门槛结果。
- 空白分词为 245 词，关键词按分号计 6 个，五条 Highlights 含空格字符数依次为 76、80、79、82、82。第 2、4、5 条明确了区间、延迟比较及模型适配范围。
- Elsevier 已确认 Highlights 为 3–5 条、每条最多 85 字符、单独上传。AEI 的 ScienceDirect 与 Elsevier 指南入口本轮均返回 HTTP 403；摘要 250 词、关键词最多 7 个沿用第 118 次登记，投稿前应在期刊页面复核。[出版方 Highlights 说明](https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript) · [AEI 作者指南](https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors)

## 参考文献

- Gneiting 等（2006）与 Hering、Genton（2010）的 DOI 和卷期页均与出版记录一致。[Gneiting 等](https://doi.org/10.1198/016214506000000456) · [Hering 与 Genton](https://doi.org/10.1198/jasa.2009.ap08117)
- Peduto 等（2026）为第 3 卷第 4 期，文章号 e2025JH001174。[AGU / Wiley](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH001174)
- Stratigakos、Andrianesis 的正式刊期为 2026 年 1 月，17(1)，806–817。DOI 中的 2025 对应正式刊期前的在线发表。[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11202569)
- Tastu 等（2011）的五位作者顺序核实为 Tastu、Pinson、Kotwa、Madsen、Nielsen。Crossref 出版记录中末位作者名为 Henrik Aa. Nielsen。[Wiley](https://doi.org/10.1002/we.401)
- R2-S1【建议】【确定缺陷，已直接修复】正式发表的 TSMixer 第三作者为 Nathanael C. Yoder，第四作者为 Sercan Ö. Arik。19 条文献与正文作者、年份完成对应检查。[Google Research](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)

下一步由 Claude 整理投稿材料，补齐数据与代码段的真实仓库链接并单列 Highlights。
