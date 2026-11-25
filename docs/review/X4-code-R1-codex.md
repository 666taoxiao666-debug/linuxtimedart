# X4-code R1/2：第105次代码审查

基线eab3276；只审声明改动及被新代码调用的冻结接口。本地审查，不上服务器；运行交Cursor。
【最终评级】：B，代码审查通过；无未解决阻断，真实Chronos/LightGBM接口和显存须由同提交服务器冒烟验证。
【必须立即修复】：发现的R1-B1、R1-B2已按第105次“有确定缺陷就按最小修法直接改”授权修复；未改方案阈值、年份、实验臂或拟合预算。
【需重跑的实验与需补充的单元测试】：未启动X4，无实验需要重跑；新增4项回归/身份测试后，X4 22项、X3 38项、X2 22项，共82项通过。

R1-B1【阻断】【确定缺陷】【已直接修复】A历史回执未加载为数组，准备必然失败。
- 严重级别P1；位置scripts/run_x4.py:93，bind_contexts → mode_a → window_arrays → run_x3.gfs_history。
- 证据：save_source返回path/sha256/source，mode_a原样传入window_arrays；gfs_history读取hours，实际复现KeyError: 'hours'。
- 影响：冒烟A无法完成，不是预测精度负结果；修法是在window_arrays入口校验回执哈希并读取历史数组，保持已有内存数组接口；置信度高。
- 验证：用save_source的真实回执调用window_arrays，比较内存数组结果，并验证篡改历史文件被拒绝。

R1-B2【阻断】【确定缺陷】【已直接修复】D0无效诊断边界未按INCOMPLETE收尾，末次拟合超时可能仍产生有效结论。
- 严重级别P1；位置scripts/run_x4.py:778，d0_target的es_split、首次kmeans和timed包装。
- 证据：空目标块在es_split中先触发IndexError，尚未到空块判断；空源池在kmeans取随机行失败；timed只在拟合前检查，最后一轮越过deadline不会被检查。
- 影响：预定的不可判定情形被记技术失败，或超预算结果被错误判成GO/STOP；修法在分割/聚类前判空，并在拟合后及最终判定前核对时限；置信度高。
- 验证：空目标、空源池、非有限源组及最后一次拟合跨时限应停止且不返回有效判定；不增加预算、不改变门槛。

## 重点审查结论

- D0首块冻结：通过。frozen只在首块合规源池拟合；后续块仅assign，top_groups只读U的历史块，当前块的u尚未加入U；中心与缩放输出到d0.json。
- D0标签隔离：通过。先budget_mask，再es_split；该冻结函数要求全部标签<c，源池同样逐标签过滤，两处断言复核；不移动评估段、不修改标签。
- D0无效情形：修后空块/空有限源池/空组/常数U转INCOMPLETE，超预算不能得到GO/STOP；不增加重试。单个正在执行的LightGBM调用不会被强行打断，返回后判超时，不将其结果计为有效结论。
- A绑定：通过。使用已固定SHA的A3上下文、X3 q4_meta与X3相同时间筛选函数，逐目标/k核对target_windows/es_windows/es_cut/fit_windows；bind_source再核对上下文与完整冻结包的issue/turbines及实际标签时间。
- 样本键核验不仅依赖数量：源池按相同冻结集合及切点重建，目标窗口按相同budget_mask/es_split产生，V1按key_positions重排；本轮未读取大数组独立复跑Claude的真实窗口预检，服务器冒烟仍执行这些断言。
- 掩码：通过。先还原缺测标志并去掉上游填充值，再逆变换，最后将目标预算前全部通道和GFS历史设NaN；数组接口与冻结build_inputs的对照测试通过。
- 模型链：通过。基础模型经原receipt/manifest校验；每段保存selected各文件SHA到selection_freeze，下一段和预测经checked_model校验；调用spec自身以SHA传给worker，冻结记录及预测再入registry。新增改动/缺失权重拒绝测试通过。
- B对照：通过。固定q2_spec的SHA再校验原训练npz；新模型重载后与旧Q2预测在相同V1键上选候选，旧候选优先打平；V2/测试不参与选点。
- 报告：通过。A/B1/B2开发判定及D0结论先写dev_decisions.json，再算测试描述；B2要求LUG及普通集成对照两项均满足；退化合并区间不写成功结论。
- 启动链：通过。x4smoke=a/b/d0/report；同提交冒烟report完成才准正式链；x4gpu中a→b，x4d0可CPU并行，x4final需三段完成；mkdir独占启动、失败停止、不自动重试。

## 验证、限制和交接

- python -m pytest tests/test_x4.py tests/test_x3.py tests/test_x2.py -q：82 passed；Git Bash对launch_x2.sh/status_x2.sh执行bash -n均通过。
- 本地X3的3个completed和3个q2_spec，共6份SHA与X4协议完全一致；未读取预测数组或运行模型。
- git diff --check通过；未修改run_x1/run_x2/run_x3或已冻结协议；只修run_x4及对应测试，方案中的测试数同步为22。
- 代码通过不等于服务器冒烟已通过、模型效果已验证或新方法创新已成立；本次无服务器操作，执行由Cursor完成。
- Claude事后复核两项直接修复即可，不因已修问题再开方案审查；后续结果须按已登记的探索性口径记录。
- 协作门禁：collab_check.py --codex退出0，无失败，1项提示为已授权的launch/status高危文件改动；15个冻结协议哈希及9个运行审计一致。
