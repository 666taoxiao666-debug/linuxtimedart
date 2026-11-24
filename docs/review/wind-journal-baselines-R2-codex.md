# 三场期刊模型对照，第 109 次方案审查

## 第 110 次交接对修改部分的复审

方案通过。第 109 次审查的原意见及证据保留在下方。

- R2-B1 已落实。Cabin 从训练到推理关闭样本轴 softmax，保留时间轴、特征轴。用同窗单独预测和混入较晚起报窗口后的预测核对 1e-5 容差。通过的是方案，实际跨窗独立性由后续代码验收。
- R2-B2 已落实，列号按用户授权直接校正。原始有效位 34+k 对应原值 21+k。实查 run_x2.INFO 与 tree_matrix：LG 的输出顺序以 0…7、49、20 开头，所以 13 个可缺值占列 10…22。LUG 对应列 21…33。若两臂都从入模列 21 开始，LG 第一格会遮掉原始通道 32。方案第 84 行现明确两套位置及共同的映射规则。自然缺测在值列缩放后置零，推理时另以真实掩码参与路由。训练时作者根掩码仍为零，这是解释方法比较时应披露的接口差异。
- R2-B3 已落实。分别验收完整掩码及指定分裂列的受控树，允许自然训练的树产生相同预测。
- R2-S1 至 S3 已落实。greedy 路径的依赖取舍、全量延迟分配后抽取同场 5 万样本键、开发期文件打开前断言、两种历史输入的 218/506 列均已登记。分配向量的新 SHA 用于以后追溯，原 X5 仅能独立核对旧计数。

## 第 109 次首审原记录

基线 `a756e04`，分支 `exp/X1`。按 HANDOFF 指定文件接续原 R1 操作建议，本文件用 R2 编号。范围为 STAGE_PLAN 和 DECISIONS 本阶段改动及其引用的作者源码。结论为需修改，三条阻断均有可构造的反例。

## 逐项结论

- **开发段范围与统计通过。** 只读 train/V1/V2，四步标签均须落入对应子段，V1 选格、三场先冻结再算 V2。保持窗口与起报簇配对的 8 时点块、2,000 次抽样。V2 结论限于开发期。
- **直接功率核岭回归通过。** 拟合段同键抽 300 个中心，按条件定距离中位数，两种带宽乘两种正则；四输出 Nyström 正规方程与均方损失下的 nλ 约定相符。核饱和只登记，选格仍只用 V1。建议对固定列显式用 scale=1，σ_med 须为正，尤其注意 18 h 已全缺的未来时效列。
- **Cabin 的双模型同历史设计通过，默认批处理需修改。** 双方共用同一缩放后的历史张量和静态量，树端可将每窗内重复的静态量折叠为一份，核验时按原形广播还原。V1 的 MSE 选 epoch、四步 MAE 选候选，是两个不同的选择。模型事实及独立性缺口见下。
- **TSG 的逐时距方案、候选数和 5 万窗口上限可用于描述性实验，掩码适配及验收规则需修改。** 预测时的真实缺测掩码可以从 X5 已提供的有效位确定，单独传参没有增加测站数据或未来信息。每时距单输出，共 96 棵候选分区树。HoT 的 X5 使用 420,475 个训练窗口，绝对精度差同时包含样本预算差异；同场 LG/LUG 应锁定同 5 万键，报告两个延迟的模型内上游收益。

## 给 Claude 的阻断与验收依据

- R2-B1【阻断】【确定缺陷】STAGE_PLAN:59–61 的 Cabin 默认启用样本轴 softmax，允许早起报窗口借用同批晚起报窗口的观测。若一批含 t 和 t+1，固定 t，仅改变后一行，就能改变 t 的 ARM 分母；`eval()` 仍执行这步。建议全程固定作者已有开关 `use_sample_softmax=False`，保留时间轴和特征轴，标明因果适配。验收同一窗口单独预测与混入晚起报窗口后预测在数值容差内一致。[构造和调用](https://github.com/xiufengliu/cabin-wind-forecasting/blob/392f103426faf600c5469875b020a4693cba6a7d/cabin/models/cabin.py#L45-L86) · [ARM 运算](https://github.com/xiufengliu/cabin-wind-forecasting/blob/392f103426faf600c5469875b020a4693cba6a7d/cabin/models/components.py#L48-L68)
- R2-B2【阻断】【确定缺陷】STAGE_PLAN:78–83 冻结的做法没有保证训练和推理对同一物理缺测的表示一致。例如值列训练最小值 −2、最大值 2，X5 屏蔽占位 0 经 MinMax 变成 0.5；作者训练根掩码全零，名义分支读到 0.5，推理 `x*(1-a)` 却把该格清零。greedy 还能屏蔽原本有效的值，而固定有效位仍为 1。请先冻结缩放后按原始有效位重新置零、原始位到完整特征掩码的列映射，以及真实缺测和额外模拟删除各自怎样进入训练与推理。同一自然缺测行须在两端得到相同的物理缺测表示；如需改作者分区或攻击训练核心，按第 90 行单列技术限制。[X5 编码见项目 prepare_g1_batches.py:238–242](../../scripts/prepare_g1_batches.py) · [根掩码](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_finite_adaptability_functions.py#L108-L130) · [greedy 删除](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_torch_custom_layers.py#L100-L164) · [推理乘掩码](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_torch_custom_layers.py#L277-L315)
- R2-B3【阻断】【确定缺陷】STAGE_PLAN:84、96 要求有效零与缺测零必然不同路由且不同预测。合法的单叶树会让两种掩码都落在根节点，常量目标或输出截断也能让两种预测相等。这会误判冒烟失败。验收应断言两种输入的有效位与传入算法的完整掩码确实不同，再用指定分裂列的受控树检查左右路由或名义/鲁棒分支选择；自然拟合模型的预测差只登记。[作者 apply 与 predict](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_finite_adaptability_functions.py#L530-L588)

## 作者版本和复用细节

- Cabin 固定版 `392f103426faf600c5469875b020a4693cba6a7d`，MIT。接口为 `forward(x,e=None)`，data_first 需要 e，返回批次×4×1；hidden 默认 64。预处理助手的 MinMax 范围为 [0,1]，本项目只在拟合段拟合缩放器。一个窗口的输入相等仍须满足 B1 的跨窗独立性。[模型](https://github.com/xiufengliu/cabin-wind-forecasting/blob/392f103426faf600c5469875b020a4693cba6a7d/cabin/models/cabin.py#L133-L172) · [预处理](https://github.com/xiufengliu/cabin-wind-forecasting/blob/392f103426faf600c5469875b020a4693cba6a7d/cabin/utils/preprocessing.py#L19-L45)
- TSG 固定版 `409f20f0d1117758b9f2e9ff48c4373fe97e945c`，MIT。作者给学习式 LR/NN 分区配置 LDR、greedy、inequality、D=10⁵、red_threshold=−10⁵；树预测以 `reshape(-1,1)` 返回，逐时距单独拟合合理。LR 示例循环的 splits 列表为空，NN 示例用 [10]，{2,5} 属于本阶段改编。作者示例对原始功率按容量缩放，对合并后的训练特征取完整行；不能据此称“所有混合缺测 GFS 列做 MinMax”为作者的原设置。[训练脚本](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_main_model_train_NYISO_weather.py)
- R2-S1【建议】【确定缺陷】STAGE_PLAN:73 的依赖结论可用，理由需改准。`clean_torch_custom_layers.py` 顶层导入 `cp`、`CvxpyLayer`，该文件没有对这两个名称的调用；`l1_norm_attack` 调用的是 `torch_model.projection_layer`。实际的 greedy 路径只走张量运算，摘取该路径可以省掉这两项导入。[作者实现](https://github.com/akylasstrat/wind-forecast-missing-data/blob/409f20f0d1117758b9f2e9ff48c4373fe97e945c/clean_torch_custom_layers.py#L65-L189)
- R2-S2【建议】【确定缺陷】第 75 行的原 X5 混合分配只存了计数。先按全量训练/V1 的原始行序连续抽取，再从训练段选 5 万键；原计数与重建向量的新 SHA 分开登记。两项都保留才能识别调换窗口但计数相同的问题。见 scripts/run_x5.py:232–239。
- R2-S3【建议】【确定缺陷】开发段 GFS 入口还应排除直接调用 `run_x5.farm_caches`，它会先打开 HoT 封存包再返回 dev/test；断言应落在开文件前。第 99 行“0.5 万列”与第 123 行相差十倍；按静态量仅保留一份时，L0/LU 展平宽度分别为 218/506。

请 Claude 将三条阻断的决议写入本阶段 DECISIONS，再交回对应的方案改动。R1 审查和操作稿留在工作区，随通过版本一并入库。
