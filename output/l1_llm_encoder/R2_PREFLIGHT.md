# L1 r2 预检记录

- 代码：6c1895d；日期：2026-11-10；正式搜索和训练未启动。
- 服务器：`/home/XT/nuist/pythoncode/TimeDARTFirst_L1_6c1895d`。
- run：`/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/LLM_L1/l1_6c1895d_r2`。
- timedart 执行 `python -m unittest tests.test_program_encoder tests.test_llm_encoder_search_l1`：18/18，11.922秒，无跳过，含训练smoke。
- 执行 `python -u scripts/llm_encoder_search_l1.py --preflight --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out <run>`，仅一次预检。
- L：seed900001，1031 tokens，围栏闭合；17个INIT_P常数违反上限16，未通过。
- L-blind：seed900002，579 tokens，围栏闭合；解包赋值违反AST白名单，未通过。
- 两臂均未碰1536上限，不能归因为再次截断；也不能当成方法效果负结果。
- 提示中 BLIND_EXAMPLE 自身含解包赋值，是确定缺陷，交回下一版修正及预检范围裁决。
- 没有更换种子或补生成，没有正式实例锁、搜索冻结或评估标记；r1保留。
- 权重49文件、15,303,338,337字节已核验；源码和协议SHA256见preflight.json。
- 取回前后SHA256一致：preflight.json `57ca759f01b93e0060253688bb95de8e6afa97c4365f81441f50db4df3f92ed4`。
- preflight.log为空，SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；生成原文及失败原因完整保存在JSON。
