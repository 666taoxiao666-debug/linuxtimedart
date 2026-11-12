# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-12 第 72 次（G0c R2：R1-B1 隔离 2016011600；100 m 按规则不用；B1 裁剪可能丢 100 m，先抽查） -->
轮次: R2/2 | 阶段: G0c | 分支: exp/G1 | 基线提交: 3e710d1
请求: 审改动 + 只读运行（写验收清单；抽查已下好的 B1 开发年文件）；不训练、不改数据
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/audit_gfs_g0.py, tests/test_audit_gfs_g0.py, output/gfs_g0/**, output/external_acquisition/gfs_g0/**
审查文件: docs/review/G0c-R2-codex.md
本轮决议: 采纳 2（R1-B1 隔离不修补；R1-S1 G1 不用 100 m）/ 反驳 0 / 暂缓 0 / 转实验 0

需要你做的:
1. **只审改动**：STAGE_PLAN B 节新增的两条和 C 节"点位"条；DECISIONS G0c 表和末段。这是 G0c 第 2 轮，也是最后一轮。
2. **写 output/gfs_g0/b2/acceptance.json**（只读、由 audit_b2.json 派生，不改数据和清单）：
   - 完整周期名单（7300 个，G1 只用这些）；
   - 不完整和缺失周期及原因（7 个不完整含 2016011600 NaT，1 个缺失 2018052618）；
   - 各年完整率；100m_used=false；结论：B2 必需字段通过。
3. **抽查 B1**（只读；B1 还在下载，不要干扰进程）：在已完成的 2021 年开发文件里抽 3 个，报告变量名、各变量的 GRIB_paramId/GRIB_typeOfLevel、heightAboveGround 坐标值，以及有没有 100 m。
   - 我从脚本第 728 行推断：B1 的变量名是 u/v，并且没有 100 m。
   - 如果确认，B1 审计时把 u/v 按 GRIB 身份（paramId 131/132、heightAboveGround、层次 10）认作 u10/v10，并补一个合成测试；不要求 100 m。
4. B1 下完由用户转达后，用同一脚本做 B1 正式核验：2021–2023 解码；2024 及以后只核字节和 SHA256。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-12：第72次，G0c R2通过，B2验收清单生成，B1三文件抽查完成 -->
轮次: R2/2 | 阶段: G0c | 分支: exp/G1 | 提交号: 本轮见git log；基线3e710d1
审查结论: 通过，0条阻断，1条建议R2-S1；docs/review/G0c-R2-codex.md。R1-B1按决议隔离关闭，仅审本轮改动。
已直接修复: 无；按授权新增derive_b2_acceptance及分区测试，不改变原审计FAIL、不修补时间或数据。
运行信息: 本地从固定SHA的audit_b2.json派生output/gfs_g0/b2/acceptance.json；13项测试通过，2.685秒；未重跑B2。
关键数字: B2必需字段通过（限验收名单）：7300完整周期，7不完整+1缺文件；所有排除周期cycle_ok=0、所有臂一致，无前周期回填，100m_used=false。
关键数字: 2016=99.521858%、2018=99.931507%，2017/2019/2020=100%；日历分母7308不变；2016011600 NaT及2018052618缺文件均排除。
运行信息: 服务器gfs_g0/bin/python通过SSH stdin只读抽查dev/2021的2021010100、0106、0112；仅变量属性/坐标及字节哈希，读前后大小和mtime一致，未干扰下载。
关键数字: 三文件均有prmsl(260074/meanSea)、u10(165)、v10(166)、u100(131)、v100(132)；风均heightAboveGround，坐标共享10.0，缺逐变量GRIB_level。
关键数字: 三文件步长f003–f018齐全且valid_time=C+f；证据/文件SHA见output/gfs_g0/b1/spotcheck_2021.md。存在100m命名字段不等于层次已认证。
运行信息: 未确认u/v别名假设，因此未实施条件性别名适配；当前B2审计器预期131/132，不能直接用作B1正式验收。未训练、未改数据、未访问封存年。
需要 Claude 做的: 复核acceptance.json及R2-S1，修正B1字段推断；保持本轮明确决定的不用100m。B1下载完成后，再明确165/166身份适配与测试范围并交Codex正式核验。
需要 Claude 做的: B2按隔离规则已通过，B1全量仍未验收；不凭三个样本判全量通过，不提前写G1训练脚本。
需要用户决定: 无。
