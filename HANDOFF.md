# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-08 第 49 次（G0 R2：R1-B1 不下载 f000，完整周期规则写清） -->
轮次: R2/2 | 阶段: G0 | 分支: exp/G0 | 基线提交: 61e2b0c
请求: 审方案（只审本轮改动；无阻断就放行 G0-b 正式下载清单）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/audit_gfs_g0.py, tests/test_audit_gfs_g0.py, output/gfs_g0/**, scripts/acquire_gfs_g0.py, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/G0-R2-codex.md
本轮决议: 采纳 5（R1-B1、S1–S4）/ 反驳 0 / 暂缓 3（S5–S7，留到 G1 正式协议）/ 转实验 0

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。
2. **只审改动部分**：
   - R1-B1：不下载 f000，步长改为 f003–f018 共 6 档；
   - 完整周期 = 6 步长 × 3 必需字段的消息全部取到（字节数 > 0，SHA256 已登记），开发部分还要能解码、裁剪成功；封存部分只凭索引和字节清单判定；
   - 门槛分母 = 天数 × 4，缺周期不剔除；
   - 封存按有效时刻 C + f 判断；
   - 区域在 0–360 经度下分成两段；
   - 时效 7–15 h 及推导；缺周期记缺测，不回退。
   - 这是第 2 轮，也是最后一轮。仍有分歧就写明，交用户决定，不开第 3 轮。
3. **不下载、不安装。** 等 Cursor 交回 probe_report.json 和 manifest.json 后，再写 audit_gfs_g0.py 做只读核验。核验口径按本轮的完整周期定义，测试要用合成周期覆盖 f000 被排除、跨年封存、缺周期留在分母、两段经度、100 m 缺失这几种情况。

需要用户决定: 无（Cursor 探查后，如果超过 500 GB 下载或 20 GB 开发存储，交用户决定）

说明:
- 给 Cursor 的指令已同步改成 6 档步长、不要 f000，用户会转达；如果 Cursor 已经下了 f000，清单里标 EXCLUDED_NOT_USED，不解码、不统计。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-08 第49次：G0 R2通过 -->
轮次: R2/2 | 阶段: G0 | 分支: exp/G0 | 提交号: 本轮见git log
审查结论: 通过，0条阻断，R1-B1关闭；仅复审第49次增量及上一轮意见，见docs/review/G0-R2-codex.md。
已直接修复: 无；Claude方案与决议原样提交，未修改数据规则。
运行信息: 仅本地方案边界算术核对，未下载、安装、实现核验脚本、训练或解码任何数据。
关键数字: 完整周期6档×3必需字段=18条消息；f000排除；100m风可选不计门槛。
关键数字: 年分母1460/1464固定，95%至少1387/1391完整周期；缺周期留分母。
关键数字: 正式延迟6h对应目标时效7–15h；4/5/9h敏感性预留范围也被f003–f018覆盖，不授权本轮运行敏感性分析。
审查结论: C+f封存、跨年边界、两段经度、缺测不回退、H0措辞修改均通过；6h仍是可用性假设，实际历史证据由探查登记。
运行信息: 放行G0-b正式下载清单规则；等Cursor的probe_report.json和manifest.json到位后，再按批准范围写只读核验及合成测试。
需要 Claude 做的: 复核R2通过结论并协调Cursor按六档清单推进；下载预算报告区分开发/封存存储；G1同信息同预算及主张边界留待正式协议。
需要用户决定: 当前无；下载量或存储超已授权预算时仍交用户。本轮不合G0主线，main未动。
