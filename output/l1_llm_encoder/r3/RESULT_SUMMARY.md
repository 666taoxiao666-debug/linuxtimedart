# L1 r3 结果交回

完成时间：2026-11-10 09:18:51 UTC。原程序判定 STOP_L1，正式拟合30次；不重跑。
以下为3个seed的指标均值，顺序MAE(kW) / RMSE(kW) / R2。

- Kelmarsh：H 160.020 / 242.169 / 0.82526；P 160.225 / 242.632 / 0.82460；R* 159.881 / 242.824 / 0.82432；L* 160.392 / 244.272 / 0.82221；blind 168.270 / 255.817 / 0.80501。
- Penmanshiel：H 184.410 / 299.395 / 0.78480；P 186.033 / 300.970 / 0.78254；R* 185.829 / 301.678 / 0.78152；L* 184.355 / 299.958 / 0.78400；blind 189.880 / 308.398 / 0.77167。
- 搜索R/L/blind各30槽，合法数30/14/5；L和blind触1536 token上限均0。最优程序R_15、L_g2_s1、L_blind_g4_s2。
- L*相对H的集成误差增益及95% CI：Kelmarsh -0.524 [-1.720,0.472]；Penmanshiel -0.418 [-2.000,1.165] kW。
- L*相对blind：Kelmarsh 7.934 [6.079,9.686]；Penmanshiel 5.607 [3.401,7.706] kW。优于blind不等于优于手工H。
- 完整逐seed数值、其余比较与CI见l1_summary.json；CI采用集成预测后按issue聚合的口径，不等于逐seed MAE差。

## 核验和限制

30次拟合的MAE/RMSE/R2均值由保存seed记录复算一致；服务器328个登记输入逐文件SHA256复核无差异。运行用时2小时10分36秒。summary本地与服务器SHA256一致：0727bfba2a9e570e1a196457dac2dd6b8762e36719d2a4f5f503e7ef0a030b45。
run目录未保存正式预测数组或checkpoint，因此没有独立重算CI，也没有重新生成预测。
scripts/llm_encoder_search_l1.py:766在fit_arm内已执行eval，769和777才写formal_selection_freeze及test_evaluation_started；后两标记不能作为真实首次评估时间证据。搜索冻结先于正式阶段；当前没有据测试选参的直接证据。
原summary的stage为M4，是历史标签；本轮交接阶段为M5，原文件不修改。
请Claude复核审计缺口及可报告范围；保持STOP_L1，不启动r4。
