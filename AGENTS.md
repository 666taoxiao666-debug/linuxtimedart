# AGENTS.md — Claude ⇄ Codex 协作约定

Codex 每次开新会话都会自动读取本文件。本仓库由两个智能体在同一个本地文件夹里协作：
Claude 负责设计和改代码（只改工作区，不提交）；Codex（GPT）负责审查、测试、提交、推送和服务器训练。
用户每轮只转一句话，不再转交文档。

## 唯一交接文件：HANDOFF.md
- 固定两节：「Claude → Codex」只由 Claude 写，「Codex → Claude」只由 Codex 写。
- 每次整节覆盖，不追加；每节不超过 40 行。
- 不贴源码、不贴长日志：写文件路径、提交号、命令和关键数字即可（双方都能直接读仓库）。
- 更详细的说明写进提交正文；历史用 `git log -p HANDOFF.md` 和 `git log` 查。
- 不再新建 docs/*HANDOFF* 文档。docs/ 里已有的旧交接文档只作历史记录，HANDOFF.md 点名时才读。

## 用户说「Claude 改完了」→ Codex
1. 读 HANDOFF.md 的「Claude → Codex」，再看 `git status` 和 `git diff`。
2. 按「请 Codex 做」逐项审查、测试、执行；发现问题可以直接修正，并写明改了什么。
3. 整节覆盖「Codex → Claude」：本轮结论、关键数字、需要 Claude 做的事。
4. 只提交本轮相关文件（连同 HANDOFF.md）。标题一句话；正文写：做了什么、验证结果（测试数、哈希、关键指标）、没做的和风险、下一步。
5. 训练等异步结果回来后，再提交一次记录，并同样更新「Codex → Claude」。

## 用户说「GPT 好了」→ Claude
读「Codex → Claude」、`git log -5 --stat` 及相关提交正文和结果文件，开始下一轮；做完后整节覆盖「Claude → Codex」。

## 一直有效的规则
- 已冻结协议及其哈希锁定的文件不改；要改就开新版本号。
- 不修改、裁剪或插补标签；不访问封存的测试年份。
- 不 reset、不 force push；负结果照常记录。
