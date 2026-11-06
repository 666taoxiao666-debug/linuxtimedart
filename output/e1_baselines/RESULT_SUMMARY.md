# A线 E1：单次仅评估恢复完成

用户于2026-11-06明确授权；代码6204100；新目录e1_b5be60a_eval1；原r1原样保留。北京时间2026-11-06 13:44:26完成。复用64次冻结拟合（10搜索+54正式），新增训练0次、重新选参0次。

全部仅作描述性比较：这些测试年此前已打开；不做Holm，不据此宣称独立确认或普遍SOTA。固定预算及训练配方下的对比不等价于每个架构充分调优后的上限。单变量DLinear/PatchTST与多变量模型信息量不同，不用于单独归因架构优劣。

指标按三个seed分别计算再取均值；RMSE差异CI遵循原paired函数，先聚合平方误差再开方，与seed平均RMSE相减可能略有差异。正收益=参照误差减候选误差。

## Kelmarsh（16738窗口）

各臂MAE / RMSE（kW） / R²：

- TSMixer-upstream: 166.2789 / 255.2892 / 0.814814
- DLinear_power: 196.6191 / 301.4938 / 0.741715
- PatchTST_power: 197.9978 / 301.7314 / 0.741308
- iTransformer_last10: 196.1056 / 299.9909 / 0.744282
- iTransformer_upstream: 192.0719 / 291.6353 / 0.758326
- TimeXer_last10: 195.9821 / 298.1951 / 0.747334
- TimeXer_upstream: 191.9702 / 290.8454 / 0.759637

上游增益与95%区间（kW）：

- iTransformer: MAE 4.0338 [2.1090, 6.0618]；RMSE 8.3538 [5.1910, 11.5970]。
- TimeXer: MAE 4.0119 [2.4449, 5.5699]；RMSE 7.3503 [4.8640, 9.8027]。

各候选相对TSMixer-upstream的MAE差距（候选−TSMixer，正值更差）：

- DLinear_power: 30.3402 kW；95%区间 [26.3235, 34.2572]。
- PatchTST_power: 31.7188 kW；95%区间 [27.4158, 35.8213]。
- iTransformer_last10: 29.8267 kW；95%区间 [25.8042, 33.6565]。
- iTransformer_upstream: 25.7929 kW；95%区间 [22.0512, 29.2138]。
- TimeXer_last10: 29.7032 kW；95%区间 [25.5646, 33.6521]。
- TimeXer_upstream: 25.6913 kW；95%区间 [21.8564, 29.4370]。

## T22（37741窗口）

各臂MAE / RMSE（kW） / R²：

- TSMixer-upstream: 213.6308 / 329.3173 / 0.753407
- DLinear_power: 244.0019 / 379.3210 / 0.672843
- PatchTST_power: 248.7702 / 379.8217 / 0.671973
- iTransformer_last10: 248.4713 / 381.1369 / 0.669672
- iTransformer_upstream: 247.4349 / 376.3246 / 0.677972
- TimeXer_last10: 248.3899 / 376.5363 / 0.677625
- TimeXer_upstream: 243.7016 / 372.4530 / 0.684576

上游增益与95%区间（kW）：

- iTransformer: MAE 1.0364 [-1.2040, 3.2803]；RMSE 4.8191 [1.5289, 8.3033]。
- TimeXer: MAE 4.6883 [2.7866, 6.5387]；RMSE 4.0817 [1.3588, 6.7634]。

各候选相对TSMixer-upstream的MAE差距（候选−TSMixer，正值更差）：

- DLinear_power: 30.3711 kW；95%区间 [25.4639, 34.7987]。
- PatchTST_power: 35.1394 kW；95%区间 [30.0799, 39.8954]。
- iTransformer_last10: 34.8405 kW；95%区间 [29.9698, 39.4051]。
- iTransformer_upstream: 33.8041 kW；95%区间 [29.1131, 38.2680]。
- TimeXer_last10: 34.7590 kW；95%区间 [29.6097, 39.5547]。
- TimeXer_upstream: 30.0708 kW；95%区间 [25.4710, 34.3852]。

## T23（36838窗口）

各臂MAE / RMSE（kW） / R²：

- TSMixer-upstream: 188.8388 / 301.9603 / 0.772537
- DLinear_power: 214.9238 / 346.3564 / 0.700734
- PatchTST_power: 218.8835 / 348.2901 / 0.697377
- iTransformer_last10: 219.5450 / 347.2171 / 0.699244
- iTransformer_upstream: 219.1548 / 347.5738 / 0.698570
- TimeXer_last10: 217.0623 / 344.4095 / 0.704087
- TimeXer_upstream: 212.8866 / 339.2411 / 0.712897

上游增益与95%区间（kW）：

- iTransformer: MAE 0.3902 [-1.7207, 2.5303]；RMSE -0.3890 [-3.7106, 2.9873]。
- TimeXer: MAE 4.1757 [2.6235, 5.7527]；RMSE 5.1659 [2.6589, 7.7508]。

各候选相对TSMixer-upstream的MAE差距（候选−TSMixer，正值更差）：

- DLinear_power: 26.0851 kW；95%区间 [21.5206, 31.0101]。
- PatchTST_power: 30.0448 kW；95%区间 [25.3927, 34.9423]。
- iTransformer_last10: 30.7062 kW；95%区间 [26.1539, 35.4073]。
- iTransformer_upstream: 30.3160 kW；95%区间 [26.0607, 34.8246]。
- TimeXer_last10: 28.2235 kW；95%区间 [23.7354, 32.9349]。
- TimeXer_upstream: 24.0478 kW；95%区间 [19.8464, 28.4600]。

## 核验与交接

服务器7测试通过；57个预测/cohort数组哈希核验；64拟合登记及产物重新核验；从保存数组复算指标与24项paired比较（见eval1/independent_verification.json）。不重新生成预测。只拷JSON，不拷权重/数组。

结论：TSMixer-upstream在本次三设定全部候选中保持最佳；TimeXer上游MAE收益三设定区间均为正，iTransformer仅Kelmarsh的MAE区间为正，不能声称跨模型一致有效。C0仍按预设规则STOP（M覆盖81.2799%低于88%下限），详情见../conformal_c0/RESULT_SUMMARY.md。请Claude一并复核，无新增实验、未合主线。
