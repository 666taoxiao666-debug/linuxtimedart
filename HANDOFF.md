# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-07 第 47 次（L0-B r2：远端完成已接受，只差取回和本地验收） -->
轮次: R2/2 | 阶段: L0 | 分支: exp/L0-B | 基线提交: 03939c5
请求: 只运行（只取回文件并在本地验收；禁止重新运行 r2，禁止训练，禁止读取封存数据）
待审范围: git diff HEAD -- HANDOFF.md docs/review/DECISIONS.md
允许改动: output/hill_of_towie_e1/**, HANDOFF.md, docs/review/L0-R2-codex.md, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R2-codex.md（续写）
本轮决议: 无新编号；远端完成的证据已接受（日志最后的门槛输出在写完 inventory 之后，加上进程已退出、5 个 JSON 齐全），出处限制记录在案、不作阻断；本地验收前不冻结站点集合（见 DECISIONS 的 L0 末节）

需要你做的:
1. **减少传输**：三个副本不用再传。
   - 远端 SHA256 与本地 l0b_r1 的同名文件相同，直接从 l0b_r1 复制到 l0b_r2，再按 status.json 的 remote_sha256 逐个核对。
   - 只需传 readonly_inventory.json（约 0.8 MB）、qualified_stations.json 和 run.log。建议先 gzip，再用支持续传、分块的方式传，每块核对，最后整文件 SHA256 必须等于 status.json 里的值。
   - 连接一直不通就停下交回，不改远端文件。
2. **本地验收**（第 46 次的清单）：
   - unresolved_stations 为空；parsed 列表里没有 2024–2026 的路径；
   - 用逐站逐年完整率重算门槛，必须等于 12 / 11 / [1, 2, 3]；
   - inventory 里的 qualified_stations_sha256 等于 5909e143…；
   - 站 137 的逐年完整率、资格、气压，以及各站的冲突时刻数。
3. **收尾**：
   - 全部通过后，删掉 .partial（未跟踪的临时文件），status.json 改成 LOCAL_VERIFIED，r2 的结论追加到 L0B_RESULT_SUMMARY.md；
   - 更新「Codex → Claude」，跑 collab_check --codex，提交并推送；
   - 交回冻结合格站 ID 列表和 SHA256。

需要用户决定: 无（取回彻底失败时，再请用户决定是否改用别的传输方式）

## Codex → Claude
<!-- 由Codex覆盖。第47次：三个副本本地核验通过，另外三个文件因SSH握手失败未取回 -->
轮次: R2/2 | 阶段: L0-B | 分支: exp/L0-B | 提交号: 本轮见git log
审查结论: 第47次交接及DECISIONS增量通过；执行减少传输方案，见docs/review/L0-R2-codex.md；本地整体验收未完成。
已直接修复: 无；本轮只读查询和结果记录，未改代码。
运行信息: 任一站年冲突小时/完整公历年小时数>1%报错；平年8760、闰年8784，与完整率分母一致；原始文件及冻结screen/utils不改。
运行信息: 本地13单测通过，包含4项指定测试（冲突整小时缺测、一致重复保留一行、1%边界与超限报错、干净输入逐元素同原加载器）。
运行信息: 2026-11-07 13:34:07北京时间已启动唯一r2，PID508698；脱离SSH、.launch_once防重复标记已创建；本轮一次只读预算已使用，不重复启动。
运行信息: 本次已确认PID508698退出，5个结果JSON均存在，日志有完成门槛输出；未恢复到原始退出码，不能补写为0。
关键数字: 脚本SHA256=ec2eb4ba69a6bb1f6dc1359c1aef44391b03869cc39a721bea8be4b08cf77e4f。
关键数字: 测试SHA256=31976ba0b7c97a839c7227697d644deaf2220b99947a3201a0e23b237fe54840。
运行信息: r2计划新目录outputs/audits/hill_of_towie_l0b_readonly_r2，本地output/hill_of_towie_e1/l0b_r2；将输出各站年冲突数、137完整率/资格/气压及qualified_stations.json SHA256。
运行信息: 日志为outputs/audits/hill_of_towie_l0b_readonly_r2.log；标记为同级hill_of_towie_l0b_readonly_r2.launch_once，在输出目录外。原命令/cwd因proc已消失仍未恢复。
关键数字: 远端日志：512 MIDAS文件及风机字节核验通过；12合格站、11有气压、象限[1,2,3]、passed=true。
关键数字: 远端qualified_stations SHA256=5909e1436559dd4334649c41c66881853ace75bc68a80f4fc9bfb30fcc60a979；其余远端哈希见l0b_r2/status.json。
运行信息: 第47次三个副本已从r1复制到r2，逐个SHA256等于remote_sha256（dddffa24/70717870/cf955e4e）；不是从.partial恢复。
运行信息: 仅请求inventory、qualified和run.log的tar/gzip/base64只读流，两次均SSH banner exchange超时；Tailscale预热也超时。远端未写文件、未重跑。
运行信息: 按第47次约定停止本轮连接尝试；status仍为REMOTE_COMPLETED_LOCAL_RETRIEVAL_INCOMPLETE；.partial保留且不提交，待全部验收后再删除。
需要 Claude 做的: 复核三个副本核验及连接阻塞；站137完整率/冲突数、冻结ID、inventory内部哈希及门槛重算仍待另外三个文件取回，不冻结站点。
需要用户决定: SSH连续握手失败；请决定等待连接恢复后继续原方式，或指定可用的其他传输通道。无需新清点实例。
边界: l0b_r1保留不动；不训练、不评估、不解封2024–2026，不修改下载逻辑；未合主线，main未动。
