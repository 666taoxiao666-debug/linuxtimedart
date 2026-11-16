# X2-code R2/2：第91次逐小时曲线复审

结论：通过，R1-B1关闭，无新增阻断。仅复审本轮修法及其必要上游接口，未重审旧方案。

- HoT排列通过：prepare_hot_dev.prepare_windows按sorted(bounds)逐机组reindex同一hours，concatenate后保存units；train_g2b.common_packs也按tile(hours)/repeat(turbine)解释。load_hourly的reshape(units,hours,-1)与来源一致。
- K/P标签通过：prepare_g1_batches.load_development使用v32重导出的v31.KelmarshUpstreamSeries；base.raw末列原样进入series.raw末列，fit_h_scaler亦保留该列，build_packs取同一列作为truth。dates/turbines继承base，小时索引与机组键一致。
- 拟合范围通过：hourly_samples按hour-end减1小时限制拟合段，每机组每小时最多一个样本；不依赖训练窗口数，NaN不插补、不裁剪标签。
- HoT施工排除通过：小时[t-1h,t)与[first,end)相交才剔除；works_bounds的结束日加1天，与原施工日历定义相同。
- 绑定通过：bind_hourly检查有序唯一小时轴，固定seed抽最多20000个训练窗口，重建24步历史通道及四步标签；容差分别1e-4、1e-3kW，失败停止，不放宽。
- 来源通过：HoT沿用冻结准备文件及施工表SHA；K/P调用已有哈希门禁的数据加载器、核对prepared元数据，并记录点缓存实际SHA；curves.json保存来源、拟合边界、样本数和绑定窗口数。
- 门槛、箱宽/最少样本、回退、臂、年份均未改变；本轮没有Codex代码修改。

验证：python -X utf8 -m unittest tests.test_x2 tests.test_legacy_bases，22项，17通过、5因本地缺torch跳过，3.155秒。
其中X2 11项全通过，包含逐小时边界/施工排除、窗口复制16次曲线不变、历史通道及标签不匹配拒绝、合成A到报告流程。
git diff --check通过；本地未访问真实服务器数据，未训练、未安装、未下载。

运行交接：允许Cursor在批准提交的干净副本先运行真实CPU smoke；真实bind_hourly门禁及torch测试必须通过，再按既定方案运行A/B及报告。
异常处理：启动前错误可能只有日志；进程退出而无完成/失败标记时交回日志，不自动重试，不更改容差。
证据等级：B（代码接口与本地回归通过，真实数据/torch由Cursor执行门禁）；不将合成测试当成真实数据验收。
