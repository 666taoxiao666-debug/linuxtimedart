# X4-smokefix R1（基线 efbd0af）

范围：第106次声明的代码/协议差异及只读诊断材料；不访问服务器，不重跑实验。
结论：通过（B，直接修复后通过）；3项确定缺陷已按 AGENTS「先审后改」常设授权在允许文件内最小修复，无未解决阻断；待Claude事后复核。

- R1-B1【阻断】【确定缺陷】完全相同分支会掩盖另一风场的技术失败 —— positive_or_identical 在发现任一 identical 后直接返回；HoT 相同且 Kelmarsh 非相同但 CI 宽度为0时，被误报普通未通过。修法：先对非相同比较逐场执行既有 pooled_effect 门禁，再处理相同特例；不改统计公式。
- R1-B2【阻断】【确定缺陷】相同分支绕过输入有效性检查 —— compare 对两份相同的 +inf 数组返回合法零增益，空数组和非有限 truth 也能绕过 paired 的检查。修法：相同判定前验证有限、非空、同形和起报时间对齐，零增益写入诊断。
- R1-B3【阻断】【确定缺陷】B1完全相同时技术失败仍无定位 —— decide仅写status=IDENTICAL，technical_failures因此输出farm/reason/diagnosis均为空。修法：保留B1技术失败口径，补相同风场列表、首个风场及非劣区间未定义的原因。
- R1-S1【建议】【确定缺陷】新增4项测试在 Windows 因 fcntl 导入失败 —— 仅测试环境隔离 Linux 锁模块，继续使用真实 bootstrap/合并统计；不修改旧统计源码或服务器依赖。
- R1-S2【建议】【确定缺陷】交接计数表述需订正 —— 四组×三场共12项，其中1项相同、11项不相同；A/B1/B2_vs_control合计9项，不是12项。

补证：diagnosis.json 中唯一退化项为 B2_vs_LUG/Kelmarsh；w=1，240窗口/40起报簇，最大预测差0，gain=0，CI=[0,0]，候选/参照非有限计数均0。
诊断脚本本地SHA256与回执一致：1c5d640c2f9cc0d59da1d5d7aa5aee0fa2d45680567a1656a52d329d7667fce8。
证据边界：本轮核对回传JSON、日志、脚本及其哈希，不冒称重新读取服务器原始数组；旧失败现场不动。
统计要求：40簇≥2、200重复≥100、块长8；非相同预测也可能产生相同绝对误差，不能单凭CI宽度0认定预测相同。
口径：严格逐值相同才走登记的零增益/UNDEFINED_IDENTICAL特例；B1沿用非劣门禁，不能用正增益特例替代其技术失败处理。
后续：通过后Cursor仅在新提交干净副本、新目录完整跑x4smoke；旧efbd0af不覆盖；同提交冒烟完成后才按原授权推进正式链。

验证（2026-11-21）：本地 pytest tests/test_x4.py tests/test_x3.py tests/test_x2.py：92 passed（32/38/22，16.87秒）。
新增反例：相同风场不能掩盖另场零CI/零参照损失；相同inf/NaN、空数组/错位/单簇拒绝；不同预测但相同绝对误差仍技术失败；B1失败有定位；report先落盘诊断再失败，不写判定/测试结果。
补证JSON回放：A=false、B1=NEGATIVE、B2=false、B2_vs_LUG=UNDEFINED_IDENTICAL(Kelmarsh)，technical_failures为空；这些只是旧冒烟诊断的门禁回放，不是正式实验结论。
与原基线常规统计规则等价测试通过；本次未改阈值、年份、实验臂、资源预算或旧统计工具。协议v2为Claude本轮已声明的新版本。
交接检查：collab_check.py --codex退出0、0失败、1项待Claude事后复核提示；git diff --check通过。
