# Kelmarsh v7：可拒绝数值残差与共同底座Wiki

日期：2026-10-15（Asia/Shanghai）。用户明确授权诊断、最小修正、测试、推送、部署和训练。

## 问题与改动

旧f1/seed2024的OriginalTrend实际选中epoch0，预测近似Persistence。数值适配器的训练期
收益随时间减少，outer MAE反而从101.888844升至107.236914kW；大部分损害不在Wiki介入处。
因此本轮不继续只调Wiki阈值，而将冻结数值残差本身视为可拒绝的执行候选。

实现：`utils/certified_numeric_residual.py`。仅用原训练selection拟合每步的凸收缩系数；
目标为平均的归一化MAE与MSE，在二者均不劣的可行区间上求解，不做outer阈值搜索。
原fresh calibration按issue时钟分三段：

1. 数值执行证据：再分三个连续时块，每块至少24窗口、3台风机、4个不同issue，要求
   风机宏平均MAE/MSE收益均正、至少2/3风机MAE受益。任一块失败，对应步长精确回退。
2. Wiki epoch选择：固定20轮，MAE/RMSE均不劣于共同底座，保留epoch0回退。
3. Wiki执行证据：不变的旧收益阈值，但证书重新绑定实际执行的底座门和新专家。

跨边界目标移除；风机6不提供知识/门控标签。第一阶段检查点不重训。数值策略确定后，
用原memory+early训练WikiIncrement和ZeroMemoryIncrement，所有优化设置和参数量匹配；
仅后者的类型化记忆token置零。记忆置零仍保留相同知识支持mask，不能称为完全无知识。

## 预声明验收

协议：`configs/kelmarsh_certified_fallback_protocol_v7.json`。
固定fold1、seed2024，两条20epoch分支；不自动扩折/seed，不重训旧完成任务。

同时比较Persistence、OriginalTrend、冻结NumericContext、旧NumericCalibrated、旧v6 Wiki、
新CertifiedNumeric及新Wiki/匹配置零分支。报告MAE/MSE/RMSE/R²、相对Persistence的双Skill、
容量容差命中率、分步长误差、coverage、selected gain、harm、abstention、时间簇bootstrap/
Newey-West及预测图。所有评估样本保留、功率标签不改、不截负功率，不访问封存test。

Wiki需MAE和RMSE同时优于共同CertifiedNumeric、匹配置零及OriginalTrend/Persistence，才支持
有用的端到端Wiki精度贡献。数值回退获得的收益应单独归因。负结果全部保留。

## 实现自审

【最终评级】：C（工程检查通过但新实验未完成，不具备新的精度结论）。

【必须立即修复Top 5】：原先默认执行不稳定残差已修正；证书/执行对象错配以门ID和检查点
哈希防护；校准与选epoch隔离；相关窗口不冒充独立样本；数值回退收益不计为Wiki贡献。
后两项是报告边界，不代表发现新的代码错误。

【需重跑的实验与需补充单元测试】：只启动新v7的两条分支，不重跑v1-v6。已完成本地41项测试，
包含10项新门控测试及既有预测、Wiki质量、残差、哈希回放和时序支持回归。服务器测试另记。

=== 详细问题清单 ===

- [P1，已加控制] `scripts/train_kelmarsh_wiki_quality.py`的旧对照保留原始NumericContext，
  其outer跨时段退化已有真实指标支持；这不是单纯代码bug。新runner将其作为可拒绝候选，
  失败精确回退，测试覆盖全块通过要求、翻负、稀疏支持与零误差。置信度高。
- [P1，已加控制] `scripts/train_kelmarsh_certified_fallback.py`的prepare/run/evaluate：
  新base会改变增量专家的目标，不能沿用旧Wiki证书。每个新证书绑定gate ID、新增量检查点
  及源检查点，实例化/推理不读取outer标签。测试覆盖身份篡改与证据可用时间。置信度高。
- [P2，保留限制] 内部fresh calibration分段后样本更少，且天气/风机相关。固定块检查不是
  统计保证；outer已被历史查看，本轮只能提供开发证据。须保留所有CI/p值和失败结果。
  不能为正收益改变块数、阈值或seed。置信度高。
- [P2，保留限制] 新流程重新分配内部样本，旧v6与新v7差异不能作为纯Wiki机制消融。
  真正的记忆贡献只在v7内部相同底座/预算的两分支上比较。置信度高。

## 部署记录

待本次推送及服务器校验后补充真实提交、PID、目录和状态。未完成前不填预测收益。
