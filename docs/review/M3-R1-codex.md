# M3 R1 — 第 57 次，审查 CW2（上游变化特征的误差尺度模型）

结论：通过，0 条阻断；按批准方案实现 `scripts/pilot_conformal_cw2.py` 与 `tests/test_pilot_conformal_cw2.py`，通过测试与 `--smoke` 后在服务器 CPU（限 2 线程）正式运行一次，不干扰 GPU 上正在运行的 T1（PID 674567）。

## 重点核查结论

1. **σ̂ 拟合范围与因果隔离【通过】**：
   - 每个时效 $h \in \{1,2,3,4\}$ 独立拟合一个 `HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=2024, early_stopping=False)`。
   - 拟合样本与目标仅使用源校准年的 `|残差|`，不使用目标风场的任何数据或评估年数据。
   - 下限裁剪 $\hat{\sigma}(x) \leftarrow \max(\hat{\sigma}(x), 0.1 \times \text{median}(|r_{\text{source}, h}|))$；当训练残差中位数非正或非有限时直接报错中止。
2. **三臂特征对齐与公平性【通过】**：
   - `N-P`（匹配对照）：仅用该 $h$ 的预测功率 `prediction[:, h]`（1 维）。
   - `N-Rs`（仅变化类候选）：`prediction[:, h]` + 起报时刻已知的 `|upwind_dws3|, |local_dws3|, |regional_dws3|, |local_dp3|, |grad_magnitude_change3|`（6 维）。
   - `N-R`（完整上游变化+状态候选）：`N-Rs` 的 6 维 + `upwind_minus_local, regional_ws, upstream_missing`（由起报时刻 H11 是否有缺测确定）+ 最近站风向 8 扇区与缺测类（共 18 维）。
   - 三臂除特征外，回归器超参数、源校准年分数 $s = |r|/\hat{\sigma}(x)$、有界 ACI 更新（步长 0.005、范围 `[0.005, 0.5]`、按机组×$h$ 独立状态与真实到达时刻反馈）及规则 F（落在 $+\infty$ 质量上时取源校准集最大分数）完全一致。
3. **4 组挑选与合格条件【通过】**：
   - 在已用过的 4 组开发数据（`selection_2022` 正向、`selection_2023` 正向、`J1` Penmanshiel 2024 正向、`J2` Kelmarsh 2024 反向）上评估 `B0, B1, B2-A, B3-A, N-P, N-R, N-Rs`。
   - 候选 `N-R`、`N-Rs` 合格条件：4 组的覆盖率均在 `[88%, 92%]` 内，且 4 组的 Winkler 均严格小于该组的 `min(N-P, B3-A)`。
   - 若两个候选均合格，按 4 组相对 `min(N-P, B3-A)` 的改善百分比等权均值最大者选择，平局取 `N-Rs`；若无合格候选，写 `selection_freeze.json` 后直接停在挑选阶段（`STOP_AT_SELECTION`），不计算 `J3`。
4. **J3 判定数据与 C1–C3 规则【通过】**：
   - 仅当选出合格候选并冻结 `selection_freeze.json`、核对代码与输入 SHA256 后，才写入 `judgment_started.json` 并计算 `J3`（反向：源 Penmanshiel 2022，目标月 Kelmarsh 2022-12，评估 Kelmarsh 2023，校准截止 `2022-12-31T23:00`）。
   - `C3` 对照为 `J3` 上 `N-P` 与 `B3-A` 中平均 Winkler 较小者，要求候选 Winkler $\le 0.97 \times \min(\text{N-P}, \text{B3-A})$ 且按起报时刻配对 block bootstrap（块长 8、5000 次、seed 2024）的 `对照 − 候选` 95% CI 下界 $> 0$。

## 建议与实现登记（不触发新一轮）

- R1-S1【建议】【不确定】登记描述性分层与特征重要性的无前视口径 —— 变化窗口按目标时刻相对上一小时（$h=1$ 相对起报时刻 `anchor`，$h>1$ 相对前一预测步真值）的真实功率变化 $|\Delta P| \ge 0.10 \times 2050\text{ kW} = 205\text{ kW}$ 分层，仅用于事后描述；特征重要性用源校准集上的 `permutation_importance(..., n_repeats=5, random_state=2024)` 登记，不影响任何预测或选择。
