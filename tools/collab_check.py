#!/usr/bin/env python3
"""协作协议 v2 的交接检查。只用 Python 标准库；规则见 AGENTS.md「协作协议 v2」。

用法（在仓库里任意目录运行均可）:
  python tools/collab_check.py                 Claude 交接前、Codex 开工时运行：8 项检查，并打印待审 diff
  python tools/collab_check.py --codex         Codex 交接前运行：本轮新写的审查意见允许暂未决议，但 Codex 那一节必须是 v2 格式
  python tools/collab_check.py --no-diff       只看检查结果，不打印 diff
  python tools/collab_check.py --init-baseline 把当前已有的未跟踪文件记为历史基线（存在 .git/ 里，只在本机，不提交）
  python tools/collab_check.py --self-test     在临时 git 仓库里造出各种违规，确认每一种都会被拦下

退出码: 0 = 没有失败项（可以有警告）；1 = 至少一项失败；2 = 脚本本身无法运行（例如不在 git 仓库里）。
git 统一带 core.autocrlf=true、core.filemode=false 运行，避免 Windows 换行或权限位造成的假改动。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from fnmatch import fnmatchcase
from pathlib import Path

HANDOFF = "HANDOFF.md"
STAGE_PLAN = "docs/STAGE_PLAN.md"
REVIEW_DIR = "docs/review"
DECISIONS = "docs/review/DECISIONS.md"
BASELINE = "collab_check_baseline.tsv"
LIMITS = {"section": 40, "stage_plan": 150, "review": 80, "blockers": 5, "proposals": 3, "rounds": 2}
ALWAYS_ALLOWED = ("HANDOFF.md", "docs/STAGE_PLAN.md", "docs/review/**")
PROTECTED_BRANCHES = ("codex/self-evolving-wiki", "main", "master")   # 主线（用户 2026-11-01 指定）及旧主分支
HIGH_RISK = ("requirements*.txt", "environment*.yml", "setup.py", "pyproject.toml", "*.sh", "conftest.py", ".github/**")
CLAUDE_FIELDS = ("轮次", "阶段", "分支", "请求", "待审范围", "允许改动")
CODEX_FIELDS = ("轮次", "阶段", "分支", "提交号", "审查结论")
STAGE_HEADINGS = ("阶段名", "目标", "假设与判定标准", "实验配置表", "改动文件清单", "本阶段不做什么", "风险", "存档引用")
DECISION_WORDS = ("转实验", "采纳", "反驳", "暂缓")
EMPTY = {"", "-", "—", "–", "无", "n/a", "N/A"}
GIT = ("git", "-c", "core.autocrlf=true", "-c", "core.filemode=false", "-c", "core.quotepath=false")
ID_RX = re.compile(r"(?<![A-Za-z0-9])R(\d+)-([BSP])(\d+)(?![0-9])")
REVIEW_NAME = re.compile(r"^(?P<stage>.+)-R(?P<round>\d+)-codex\.md$")
SECTION_RX = re.compile(r"^##\s*(Claude|Codex)\s*(?:→|->)\s*(?:Codex|Claude)\s*$")
FIELD_RX = re.compile(r"^\s*(?:[-*]\s*)?\**([一-鿿A-Za-z0-9 ]{1,16}?)\**\s*[:：]\s*(.*?)\s*$")
PASS, WARN, FAIL, SKIP = "通过", "警告", "失败", "跳过"


class GitError(RuntimeError):
    pass


def git(root, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(GIT + ("-C", str(root)) + tuple(args), capture_output=True, env=env)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout.decode("utf-8", "replace")


def git_blobs(root, commit, paths):
    """{路径: 该提交里的 Git 对象原始字节}；对象不存在的路径不出现在结果里。用一个 cat-file 进程批量读取。"""
    if not paths:
        return {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    query = "".join(f"{commit}:{p}\n" for p in paths).encode("utf-8")
    r = subprocess.run(GIT + ("-C", str(root), "cat-file", "--batch"), input=query, capture_output=True, env=env)
    out, pos, blobs = r.stdout, 0, {}
    for p in paths:
        end = out.find(b"\n", pos)
        if end < 0:
            break
        header = out[pos:end].split()
        pos = end + 1
        if len(header) == 3 and header[1] == b"blob":
            size = int(header[2])
            blobs[p] = out[pos:pos + size]
            pos += size + 1
    return blobs


def read_text(path):
    try:
        return Path(path).read_bytes().decode("utf-8-sig", "replace").replace("\r\n", "\n")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None


def n_lines(text):
    t = (text or "").rstrip("\n")
    return t.count("\n") + 1 if t else 0


def sha_lf(path):
    """与仓库现有的 text_sha256_lf 相同：先把 CRLF 换成 LF 再算 SHA256。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def strip_paren(s):
    return re.sub(r"（[^）]*）|\([^)]*\)", " ", s or "")


def _glob_rx(pat):
    out, i = [], 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def in_scope(path, patterns):
    for p in patterns:
        p = p.strip().strip("`")
        if p.startswith("./"):
            p = p[2:]
        if not p:
            continue
        if p.endswith("/"):
            p += "**"
        if _glob_rx(p).match(path):
            return True
        if not any(c in p for c in "*?[") and path.startswith(p + "/"):
            return True
    return False


def is_high_risk(path):
    base = path.rsplit("/", 1)[-1]
    return any(in_scope(path, [p]) if "/" in p else fnmatchcase(base, p) for p in HIGH_RISK)


# ---------------------------------------------------------------- HANDOFF / DECISIONS 解析

def parse_handoff(text):
    secs, cur = {}, None
    for line in (text or "").split("\n"):
        m = SECTION_RX.match(line.strip())
        if m:
            cur = m.group(1).lower()
            secs[cur] = [line]
            continue
        if line.startswith("# ") or line.startswith("## "):
            cur = None
            continue
        if cur:
            secs[cur].append(line)
    for v in secs.values():
        while v and not v[-1].strip():
            v.pop()
    return secs


def parse_fields(lines):
    out = {}
    for line in lines[1:]:
        s = line.strip()
        if s.startswith("<!--") or s.startswith("|"):
            continue
        for part in re.split(r"\s+\|\s+", s):
            m = FIELD_RX.match(part)
            if m:
                out.setdefault(m.group(1).strip(), m.group(2).strip())
    return out


def handoff_info(fields):
    def first(rx, key):
        m = re.search(rx, fields.get(key, ""))
        return m.group(0) if m else None
    rnd = re.search(r"R\s*(\d+)\s*/\s*(\d+)", fields.get("轮次", ""))
    pats = [p.strip("`") for p in re.split(r"[,，、;；\s]+", strip_paren(fields.get("允许改动", ""))) if p.strip("`")]
    return {"stage": first(r"[\w.\-]+", "阶段"), "round": int(rnd.group(1)) if rnd else None,
            "branch": first(r"[\w./\-]+", "分支"), "base": first(r"\b[0-9a-f]{7,40}\b", "基线提交"),
            "request": fields.get("请求", ""), "review": fields.get("待审范围", ""),
            "patterns": [p for p in pats if p not in EMPTY]}


def parse_decisions(text):
    out, stage = {}, None
    for line in (text or "").split("\n"):
        m = re.match(r"^##\s+阶段\s*[:：]?\s*(\S+)", line)
        if m:
            stage = m.group(1)
            out.setdefault(stage, {})
            continue
        if line.startswith("## ") or line.startswith("# "):
            stage = None
            continue
        s = line.strip()
        if stage is None or not s.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in s.strip("|").split("|")]
        if not cells or not ID_RX.fullmatch(cells[0]):
            continue
        cells += [""] * (6 - len(cells))
        word = next((w for w in DECISION_WORDS if w in cells[3]), None)
        out[stage][cells[0]] = {"decision": word, "reason": cells[4], "impl": cells[5]}
    return out


def parse_review_cmd(value):
    m = re.search(r"`([^`]+)`", value or "")
    cmd = (m.group(1) if m else strip_paren(value)).strip()
    if not cmd.startswith("git diff"):
        return None, "「待审范围」必须是一条 git diff 命令"
    if re.search(r"[;&|<>$\n]", cmd):
        return None, "「待审范围」含 shell 元字符，拒绝执行"
    try:
        return shlex.split(cmd)[2:], None
    except ValueError as e:
        return None, f"「待审范围」无法解析：{e}"


def split_diff_args(args):
    if "--" in args:
        i = args.index("--")
        head, paths = args[:i], args[i + 1:]
    else:
        head, paths = args, []
    revs = [a for a in head if not a.startswith("-")]
    rng = next((r for r in revs if ".." in r), None) or (f"{revs[0]}..{revs[1]}" if len(revs) >= 2 else None)
    return rng, paths


# ---------------------------------------------------------------- 改动收集与历史基线

def status_entries(root):
    toks, res, i = git(root, "status", "--porcelain", "-z", "--untracked-files=all").split("\0"), [], 0
    while i < len(toks):
        t = toks[i]
        i += 1
        if len(t) < 4:
            continue
        xy = t[:2]
        res.append((xy, t[3:]))
        if "R" in xy or "C" in xy:          # -z 模式下改名的旧路径单独占一项
            if i < len(toks) and toks[i]:
                res.append((xy, toks[i]))
            i += 1
    return res


def names(out):
    return {n for n in out.split("\0") if n}


def baseline_file(root):
    p = Path(git(root, "rev-parse", "--git-path", BASELINE).strip())
    return p if p.is_absolute() else Path(root) / p


def load_baseline(root):
    out = {}
    for line in (read_text(baseline_file(root)) or "").split("\n"):
        parts = line.split("\t")
        if len(parts) == 4 and parts[1].isdigit() and parts[2].lstrip("-").isdigit():
            out[parts[0]] = (int(parts[1]), int(parts[2]), parts[3])
    return out


def stat_key(root, rel):
    st = (Path(root) / rel).stat()
    return st.st_size, int(st.st_mtime)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unchanged_since_baseline(root, rel, rec):
    """只有大小相同且内容 SHA256 与基线相同，才视为历史文件未改（R1-B2：时间戳不能证明内容没变）。"""
    try:
        size, _ = stat_key(root, rel)
        return size == rec[0] and file_sha(Path(root) / rel) == rec[2]
    except OSError:
        return False


def collect_changes(root, info):
    base = load_baseline(root)
    changed, legacy, touched, notes = {}, 0, [], []
    for xy, path in status_entries(root):
        if xy == "??" and path in base:
            if unchanged_since_baseline(root, path, base[path]):
                legacy += 1
                continue
            touched.append(path)
        changed[path] = xy
    committed = set()
    if info["base"]:
        try:
            committed |= names(git(root, "diff", "--name-only", "-z", f"{info['base']}..HEAD"))
        except GitError:
            notes.append(f"基线提交 {info['base']} 无法解析，未检查已提交的改动")
    args, _ = parse_review_cmd(info["review"])
    if args is not None:
        rng, _ = split_diff_args(args)
        if rng:
            try:
                committed |= names(git(root, "diff", "--name-only", "-z", rng))
            except GitError:
                notes.append(f"待审范围 {rng} 无法解析")
    return {"changed": changed, "committed": committed, "legacy": legacy, "touched": touched,
            "has_baseline": bool(base), "notes": notes}


# ---------------------------------------------------------------- 8 项检查

def check_handoff(text, secs, cf, xf, codex):
    if text is None:
        return FAIL, "缺 HANDOFF.md", []
    fails, warns = [], []
    for key, label in (("claude", "Claude → Codex"), ("codex", "Codex → Claude")):
        if key not in secs:
            fails.append(f"缺「{label}」一节")
        elif len(secs[key]) > LIMITS["section"]:
            fails.append(f"「{label}」{len(secs[key])} 行，超过 {LIMITS['section']} 行")
    miss = [f for f in CLAUDE_FIELDS if not cf.get(f)]
    if "claude" in secs and miss:
        fails.append("Claude 节缺字段：" + "、".join(miss))
    if "codex" in secs:
        if "轮次" not in xf:
            (fails if codex else warns).append("Codex 节还不是 v2 格式" + ("" if codex else "（过渡期，只提示）"))
        else:
            miss = [f for f in CODEX_FIELDS if not xf.get(f)]
            if miss:
                fails.append("Codex 节缺字段：" + "、".join(miss))
    summary = f"Claude 节 {len(secs.get('claude', []))} 行，Codex 节 {len(secs.get('codex', []))} 行"
    return (FAIL if fails else WARN if warns else PASS), summary, fails + warns


def check_stage_plan(root):
    text = read_text(Path(root) / STAGE_PLAN)
    if text is None:
        return FAIL, f"缺 {STAGE_PLAN}", []
    n = n_lines(text)
    heads = re.findall(r"^##\s+(.+?)\s*$", text, re.M)
    missing = [h for h in STAGE_HEADINGS if not any(x.startswith(h) for x in heads)]
    details = [f"缺固定章节：{'、'.join(missing)}"] if missing else []
    if n > LIMITS["stage_plan"]:
        return FAIL, f"{n} 行，超过 {LIMITS['stage_plan']} 行（先删旧内容再加新内容）", details
    return (WARN if missing else PASS), f"{n}/{LIMITS['stage_plan']} 行", details


def check_reviews(root, info, codex):
    root = Path(root)
    files = []
    if (root / REVIEW_DIR).is_dir():
        for p in sorted((root / REVIEW_DIR).iterdir()):
            m = REVIEW_NAME.match(p.name)
            if m:
                files.append((m.group("stage"), int(m.group("round")), p))
    dec_text = read_text(root / DECISIONS)
    decisions = parse_decisions(dec_text)
    fails, warns, pending = [], [], []
    if info["round"] and info["round"] > LIMITS["rounds"]:
        fails.append(f"HANDOFF 轮次 R{info['round']} 超过 {LIMITS['rounds']} 轮：交用户裁决或转实验")
    if dec_text is None:
        (fails if files else warns).append(f"缺 {DECISIONS}")
    newest = {}
    for st, rd, _ in files:
        newest[st] = max(newest.get(st, 0), rd)
    for st, rd, p in files:
        text = read_text(p) or ""
        ids = {(int(a), b, int(c)) for a, b, c in ID_RX.findall(text)}
        own = lambda kind: sorted(i for i in ids if i[0] == rd and i[1] == kind)  # noqa: E731
        if rd > LIMITS["rounds"]:
            fails.append(f"{p.name}：超过 {LIMITS['rounds']} 轮")
        if n_lines(text) > LIMITS["review"]:
            fails.append(f"{p.name}：{n_lines(text)} 行，超过 {LIMITS['review']} 行")
        if len(own("B")) > LIMITS["blockers"]:
            fails.append(f"{p.name}：阻断 {len(own('B'))} 条，超过 {LIMITS['blockers']} 条")
        if len(own("P")) > LIMITS["proposals"]:
            fails.append(f"{p.name}：提案 {len(own('P'))} 条，超过 {LIMITS['proposals']} 条")
        rows = decisions.get(st, {})
        undecided = [f"R{a}-{b}{c}" for a, b, c in sorted(ids) if b in "BP" and f"R{a}-{b}{c}" not in rows]
        if undecided and codex and rd == newest[st]:
            pending += undecided
        elif undecided:
            fails.append(f"{p.name}：{'、'.join(undecided[:6])} 还没有决议（Codex 交接时请加 --codex）")
        for a, b, c in sorted(ids):
            if a < rd and rows.get(f"R{a}-{b}{c}", {}).get("decision") == "反驳":
                warns.append(f"R{a}-{b}{c} 已被反驳，又出现在 {p.name}：转实验或交用户，不再来回")
        asks = [ln for ln in text.split("\n") if ID_RX.search(ln) and ("？" in ln or "?" in ln)]
        if asks:
            warns.append(f"{p.name}：{len(asks)} 条意见带问号——请改成结论，或写成转实验")
    for st, rows in decisions.items():
        for k, row in rows.items():
            if row["decision"] is None:
                fails.append(f"DECISIONS {st} {k}：决议只能是 采纳/反驳/暂缓/转实验")
            elif row["decision"] == "反驳" and row["reason"] in EMPTY:
                fails.append(f"DECISIONS {st} {k}：反驳必须写理由或反例")
            elif row["decision"] == "转实验" and row["impl"] in EMPTY:
                fails.append(f"DECISIONS {st} {k}：转实验必须在「落实」写实验名")
    if pending:
        warns.append(f"待 Claude 决议：{'、'.join(pending[:8])}")
    stage_files = [p.name for st, _, p in files if st == info["stage"]]
    summary = f"阶段 {info['stage']} 的审查文件 {len(stage_files)} 份" + (f"（共 {len(files)} 份）" if len(files) != len(stage_files) else "")
    return (FAIL if fails else WARN if warns else PASS), summary, fails + warns


def check_scope(ch, info):
    pats = info["patterns"]
    files = sorted(set(ch["changed"]) | ch["committed"])
    out = [f for f in files if not in_scope(f, pats + list(ALWAYS_ALLOWED))]
    summary = f"{len(files)} 个改动文件" + (f"；历史未跟踪 {ch['legacy']} 个按基线忽略" if ch["legacy"] else "")
    details = list(ch["notes"])
    if not pats:
        return FAIL, summary + "；「允许改动」为空", details
    if out:
        details.insert(0, "越界：" + "、".join(out[:5]) + (f" 等 {len(out)} 个" if len(out) > 5 else ""))
        if any(f in ch["touched"] for f in out):
            details.append("其中有历史未跟踪文件被改动（大小或时间与基线不同）")
        if not ch["has_baseline"] and any(ch["changed"].get(f) == "??" for f in out):
            details.append("若这些是协议启用前就有的历史文件，先跑 --init-baseline")
        return FAIL, summary + f"，{len(out)} 个不在「允许改动」内", details
    return (WARN if details else PASS), summary + "，都在「允许改动」内", details


def check_branch(root, info, ch, codex=False):
    cur = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    code_or_run = bool(re.search(r"代码|运行|\brun\b|\bcode\b", info["request"], re.I))
    details = []
    if info["branch"] and cur != info["branch"]:
        details.append(f"当前分支 {cur}，HANDOFF 写的是 {info['branch']}")
    if code_or_run and cur in PROTECTED_BRANCHES:
        target = info["branch"] or ""
        exists = False
        if target.startswith("exp/"):
            try:
                git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{target}")
                exists = True
            except GitError:
                pass
        if target.startswith("exp/") and not exists and not codex:
            details.insert(0, f"{target} 还没建：Codex 开工时先从 {cur} 拉出，再提交和运行")
            return WARN, f"当前 {cur}（请求涉及代码或运行，实验分支待创建）", details
        # R1-P1：Codex 审出阻断后只读交回——实验分支没建，且基线提交之后主线上只提交过交接/审查文档——不算违规
        base_ok = bool(info["base"]) and not any(n.startswith("基线提交") for n in ch["notes"])
        docs_only = all(in_scope(f, list(ALWAYS_ALLOWED)) for f in ch["committed"])
        if target.startswith("exp/") and not exists and codex and base_ok and docs_only:
            details.insert(0, f"只读交回：{target} 未创建，基线提交 {info['base']} 之后只提交过交接或审查文档")
            return WARN, f"当前 {cur}（只读交回，没有提交代码）", details
        return FAIL, f"请求涉及代码或运行，但当前在 {cur}：切到 exp/<实验名> 再提交或运行", details
    return (WARN if details else PASS), f"当前 {cur}" + ("（请求涉及代码或运行）" if code_or_run else ""), details


def check_high_risk(ch, handoff_text):
    hits = sorted(f for f in set(ch["changed"]) | ch["committed"] if is_high_risk(f))
    silent = [f for f in hits if f not in (handoff_text or "")]
    if silent:
        return FAIL, f"{len(silent)} 个高危文件被改动，HANDOFF 里没写明", ["、".join(silent[:5])]
    if hits:
        return WARN, f"{len(hits)} 个高危文件被改动（HANDOFF 已写明）", ["、".join(hits[:5])]
    return PASS, "无高危文件改动", []


def check_frozen(root):
    """复用仓库现有的两种哈希锁定：scripts/*.py 的 PROTOCOL + PROTOCOL_SHA256 常量，以及运行审计 audit/launch.json 的 source_code。"""
    root = Path(root)
    fails, warns, n1, n2 = [], [], 0, 0
    for py in sorted((root / "scripts").glob("*.py")) if (root / "scripts").is_dir() else []:
        t = read_text(py) or ""
        m = re.search(r"^PROTOCOL\s*=\s*ROOT\s*/\s*[\"']([^\"']+)[\"']", t, re.M)
        h = re.search(r"^PROTOCOL_SHA256\s*=\s*[\"']([0-9a-f]{64})[\"']", t, re.M)
        if not (m and h):
            continue
        n1 += 1
        f = root / m.group(1)
        if not f.is_file() or sha_lf(f) != h.group(1):
            fails.append(f"{m.group(1)} 与 {py.name} 登记的 PROTOCOL_SHA256 不一致：冻结协议不能改，要改就开新版本")
    out_dir = root / "output"
    launches = sorted({lj for depth in ("*", "*/*", "*/*/*") for lj in out_dir.glob(f"{depth}/audit/launch.json")})
    n_failed = 0
    for lj in launches:
        try:
            data = json.loads(read_text(lj) or "")
            src = data.get("source_code")
        except (ValueError, AttributeError):
            warns.append(f"{lj.relative_to(root).as_posix()} 无法解析")
            continue
        if not isinstance(src, dict):
            continue
        n2 += 1
        run = lj.parent.parent.relative_to(out_dir).as_posix()
        digests = {rel: d for rel, d in src.items() if isinstance(d, str) and re.fullmatch(r"[0-9a-f]{64}", d)}
        if (lj.parent / "failed.json").exists():
            # P3 阶段：技术失败的运行（审计目录里有 failed.json）没有结果可复现，它的源码允许按批准修复后用于重跑。
            # 不放宽哈希：改为逐个核对它登记的提交里的 Git 对象，确认失败运行的身份本身没有被改。
            n_failed += 1
            commit = data.get("git_head") or data.get("commit") or (data.get("identity") or {}).get("git_head")
            if not commit:
                fails.append(f"技术失败的运行 {run} 没有登记提交号，无法核对历史源码")
                continue
            blobs = git_blobs(root, commit, list(digests))
            for rel, digest in digests.items():
                blob = blobs.get(rel)
                if blob is None or digest not in (hashlib.sha256(blob).hexdigest(),
                                                  hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()):
                    fails.append(f"{rel} 与技术失败运行 {run} 登记的提交 {commit[:9]} 不一致：失败运行的历史身份被改动")
            continue
        for rel, digest in digests.items():
            f = root / rel
            # R1-B1（K1b 阶段）：失配一律失败，交接里提到路径也不放行。
            # R1-P1（P1 阶段）：审计可能登记原始字节 SHA，也可能登记换行规范化后的 SHA，两种都认；内容一变，两种都对不上
            if not (f.is_file() and digest in (sha_lf(f), file_sha(f))):
                fails.append(f"{rel} 与运行 {run} 的审计登记不一致：已完成运行的源码不能原地改，要改就新开文件或版本")
    if n1 == 0 and n2 == 0:
        return SKIP, "没找到 PROTOCOL_SHA256 常量或 audit/launch.json，跳过", []
    summary = f"{n1} 个协议哈希（scripts/*.py）+ {n2} 个运行审计（audit/launch.json）" + (f"，其中 {n_failed} 个技术失败运行按其提交核对" if n_failed else "")
    verdict = "有失配" if fails else ("登记哈希一致，但有审计文件无法解析" if warns else "一致")
    return (FAIL if fails else WARN if warns else PASS), f"{summary} {verdict}", fails + warns


def new_file_diff(root, rel):
    data = (Path(root) / rel).read_bytes()
    head = [f"diff --git a/{rel} b/{rel}", "new file (未跟踪)", "--- /dev/null", f"+++ b/{rel}"]
    if b"\0" in data[:8000]:
        return head + [f"Binary file, {len(data)} bytes"]
    lines = data.decode("utf-8", "replace").replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return head + [f"@@ -0,0 +1,{len(lines)} @@"] + ["+" + x for x in lines]


def review_diff(root, info, ch):
    args, err = parse_review_cmd(info["review"])
    if err:
        return None, err, 0
    rng, paths = split_diff_args(args)
    lines = git(root, "diff", "--no-color", *args).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    extra = []
    if rng is None:                         # 工作区 diff 看不到未跟踪的新文件，这里补上
        if paths:
            extra = sorted(names(git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *paths)))
        else:
            extra = sorted(p for p, xy in ch["changed"].items() if xy == "??")
        extra = [p for p in extra if p in ch["changed"]]
        for rel in extra:
            lines += new_file_diff(root, rel)
    return lines, None, len(extra)


# ---------------------------------------------------------------- 主流程

def run_checks(root, codex=False, show_diff=True):
    root = Path(root).resolve()
    text = read_text(root / HANDOFF)
    secs = parse_handoff(text)
    cf, xf = parse_fields(secs.get("claude", [])), parse_fields(secs.get("codex", []))
    info = handoff_info(cf)
    ch = collect_changes(root, info)
    results = [
        ("HANDOFF", check_handoff(text, secs, cf, xf, codex)),
        ("STAGE_PLAN", check_stage_plan(root)),
        ("审查与决议", check_reviews(root, info, codex)),
        ("改动范围", check_scope(ch, info)),
        ("分支", check_branch(root, info, ch, codex)),
        ("高危文件", check_high_risk(ch, text)),
        ("冻结协议", check_frozen(root)),
    ]
    diff, err, n_new = review_diff(root, info, ch)
    if err:
        results.append(("待审 diff", (FAIL, err, [])))
    else:
        cmd = "git diff " + " ".join(parse_review_cmd(info["review"])[0])
        cmd = cmd if len(cmd) <= 90 else cmd[:87] + "..."
        results.append(("待审 diff", (WARN if not diff else PASS,
                                      f"共 {len(diff)} 行（含 {n_new} 个未跟踪新文件）：{cmd}", [])))
    rnd = f"R{info['round']}/{LIMITS['rounds']}" if info["round"] else "?"
    print(f"== collab_check（协作协议 v2{'，Codex 交接模式' if codex else ''}）| 阶段 {info['stage']} | 轮次 {rnd} ==")
    for k, (name, (status, summary, details)) in enumerate(results, 1):
        print(f"[{status}] {k} {name}：{summary}")
        for d in details[:3]:
            print(f"    - {d}")
        if len(details) > 3:
            print(f"    - …另 {len(details) - 3} 条")
    n_fail = sum(r[1][0] == FAIL for r in results)
    n_warn = sum(r[1][0] == WARN for r in results)
    if show_diff and diff:
        print("----- 待审 diff 开始（Codex 只审这一段）-----")
        for line in diff:
            print(line)
        print(f"----- 待审 diff 结束：共 {len(diff)} 行 -----")
    print(f"结果：失败 {n_fail} 项，警告 {n_warn} 项 → 退出码 {1 if n_fail else 0}")
    return 1 if n_fail else 0


def init_baseline(root):
    root = Path(root).resolve()
    info = handoff_info(parse_fields(parse_handoff(read_text(root / HANDOFF)).get("claude", [])))
    pats = info["patterns"] + list(ALWAYS_ALLOWED)
    old, rows = load_baseline(root), []
    for xy, path in status_entries(root):
        if xy == "??" and not in_scope(path, pats):
            size, mt = stat_key(root, path)
            rows.append(f"{path}\t{size}\t{mt}\t{file_sha(root / path)}")
    bf = baseline_file(root)
    bf.write_text("# collab_check 历史未跟踪文件基线：路径<TAB>字节数<TAB>修改时间(秒)<TAB>SHA256\n" + "\n".join(rows) + "\n",
                  encoding="utf-8")
    print(f"已记录 {len(rows)} 个历史未跟踪文件 → {bf}（旧基线 {len(old)} 个）。"
          f"HANDOFF「允许改动」和总是允许的文件不计入。")
    return 0


# ---------------------------------------------------------------- 自测：临时仓库里造违规

_SELFTEST_HANDOFF = """# HANDOFF

## Claude → Codex
轮次: R{round}/2 | 阶段: S1 | 分支: exp/S1 | 基线提交: {base}
请求: 审代码后运行
待审范围: git diff HEAD -- src/a.py
允许改动: src/a.py, tools/**
审查文件: docs/review/S1-R{round}-codex.md
{extra}
## Codex → Claude
{codex}
"""
_CODEX_V2 = "轮次: R1/2 | 阶段: S1 | 分支: exp/S1 | 提交号: abc1234\n审查结论: 有 1 条阻断，见 docs/review/S1-R1-codex.md"
_DEC_HEAD = "# DECISIONS\n\n## 阶段 S1\n\n| 编号 | 类型 | 摘要 | 决议 | 理由 | 落实 |\n|---|---|---|---|---|---|\n"


def _w(repo, rel, text):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def self_test():
    ident = ("-c", "user.name=collab", "-c", "user.email=collab@example.invalid", "-c", "commit.gpgsign=false")

    def sh(repo, *a):
        subprocess.run(("git",) + ident + ("-C", str(repo)) + a, check=True, capture_output=True)

    def review(repo, rnd, n_block, decided):
        _w(repo, f"docs/review/S1-R{rnd}-codex.md",
           "".join(f"R{rnd}-B{i}【阻断】【确定缺陷】结论 —— 反例 src/a.py:1\n" for i in range(1, n_block + 1)))
        _w(repo, DECISIONS, _DEC_HEAD + "".join(f"| R{rnd}-B{i} | 阻断 | x | 采纳 | y | 本轮 |\n"
                                                for i in range(1, n_block + 1) if decided))

    def legacy(repo, mode):
        f = Path(repo) / "notes/old.md"
        _w(repo, "notes/old.md", "AAAA\n")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--root", str(repo), "--init-baseline"],
                       check=True, capture_output=True)
        st0 = f.stat()
        if mode == "longer":
            _w(repo, "notes/old.md", "AAAA，被改长了\n")
        elif mode == "same_size_keep_time":           # R1-B2 的反例：等长改动并保留原时间戳
            _w(repo, "notes/old.md", "BBBB\n")
            os.utime(f, ns=(st0.st_atime_ns, st0.st_mtime_ns))
        elif mode == "same_content_new_time":         # 内容没变、只有时间变了，应当仍算历史文件
            os.utime(f, (st0.st_atime + 100, st0.st_mtime + 100))

    def audit_touch(repo, st, mention):               # R1-B1 的反例：运行审计登记的源码被原地改
        _w(repo, "scripts/frozen.py", "x = 2\n")
        if mention:
            st.update(extra="- 说明：本轮改了 scripts/frozen.py")

    def crlf_audit(repo, change):                     # P1 阶段 R1-P1：CRLF 源码，审计登记的是原始字节 SHA（基础仓库里已建好）
        if change:
            (Path(repo) / "scripts/crlf.py").write_bytes(b"y = 2\r\n")

    def main_commit(repo):                            # R1-P1 的对照：在主线上提交了代码
        sh(repo, "checkout", "-q", "main")
        sh(repo, "branch", "-q", "-D", "exp/S1")
        sh(repo, "add", "src/a.py")
        sh(repo, "commit", "-q", "-m", "code on mainline")

    cases = [  # (名称, 预期失败的检查号 / None 表示应通过, 改动, 额外参数)
        ("合规基线", None, lambda r, s: None, ()),
        ("越界文件", 4, lambda r, s: _w(r, "src/b.py", "y = 1\n"), ()),
        ("历史未跟踪文件未改（应通过）", None, lambda r, s: legacy(r, "none"), ()),
        ("历史未跟踪文件只改了时间（应通过）", None, lambda r, s: legacy(r, "same_content_new_time"), ()),
        ("历史未跟踪文件被改长", 4, lambda r, s: legacy(r, "longer"), ()),
        ("历史未跟踪文件等长改动且保留时间戳", 4, lambda r, s: legacy(r, "same_size_keep_time"), ()),
        ("HANDOFF 一节超 40 行", 1, lambda r, s: s.update(extra="\n".join(f"- 填充 {i}" for i in range(40))), ()),
        ("STAGE_PLAN 超 150 行", 2, lambda r, s: _w(r, STAGE_PLAN, "x\n" * 151), ()),
        ("阻断超过 5 条", 3, lambda r, s: review(r, 1, 6, True), ()),
        ("提案超过 3 条", 3, lambda r, s: (_w(r, "docs/review/S1-R1-codex.md", "".join(
            f"R1-P{i}【提案】改进 —— 依据\n" for i in range(1, 5))), _w(r, DECISIONS, _DEC_HEAD + "".join(
                f"| R1-P{i} | 提案 | x | 暂缓 | y | — |\n" for i in range(1, 5)))), ()),
        ("Claude 交接时阻断未决议", 3, lambda r, s: review(r, 1, 1, False), ()),
        ("Codex 交接时本轮阻断待决议（应通过）", None,
         lambda r, s: (review(r, 1, 1, False), s.update(codex=_CODEX_V2)), ("--codex",)),
        ("超过 2 轮", 3, lambda r, s: s.update(round=3), ()),
        ("在 main 上运行（实验分支已存在）", 5, lambda r, s: sh(r, "checkout", "-q", "main"), ()),
        ("实验分支未建、Claude 在主线交接（应通过）", None,
         lambda r, s: (sh(r, "checkout", "-q", "main"), sh(r, "branch", "-q", "-D", "exp/S1")), ()),
        ("Codex 只读交回：主线上、实验分支未建、没提交代码（应通过）", None,
         lambda r, s: (sh(r, "checkout", "-q", "main"), sh(r, "branch", "-q", "-D", "exp/S1"),
                       s.update(codex=_CODEX_V2)), ("--codex",)),
        ("Codex 在主线上提交了代码", 5, lambda r, s: (main_commit(r), s.update(codex=_CODEX_V2)), ("--codex",)),
        ("高危文件没写进 HANDOFF", 6, lambda r, s: _w(r, "tools/run.sh", "echo hi\n"), ()),
        ("冻结协议被改", 7, lambda r, s: _w(r, "configs/p.json", '{"v": 2}\n'), ()),
        ("运行审计登记的源码被改", 7, lambda r, s: audit_touch(r, s, False), ()),
        ("运行审计登记的源码被改、HANDOFF 里提到了路径", 7, lambda r, s: audit_touch(r, s, True), ()),
        ("审计登记原始字节 SHA 的 CRLF 源码未改（应通过）", None, lambda r, s: crlf_audit(r, False), ()),
        ("审计登记原始字节 SHA 的 CRLF 源码被改", 7, lambda r, s: crlf_audit(r, True), ()),
        ("技术失败运行的源码按批准修复（应通过）", None, lambda r, s: _w(r, "tools/fixme.py", "z = 2\n"), ()),
        ("技术失败运行登记的哈希与其提交不符", 7, lambda r, s: _w(r, "output/run3/audit/launch.json", json.dumps(
            {"identity": {"git_head": git(r, "rev-parse", "HEAD~1").strip()}, "source_code": {"tools/fixme.py": "0" * 64}}) + "\n"), ()),
        ("技术失败运行没有登记提交号", 7, lambda r, s: _w(r, "output/run3/audit/launch.json", json.dumps(
            {"source_code": {"tools/fixme.py": sha_lf(r / "tools/fixme.py")}}) + "\n"), ()),
    ]
    tmp = tempfile.mkdtemp(prefix="collab_selftest_")
    bad = 0
    try:
        for k, (name, expect, mutate, flags) in enumerate(cases):
            repo = Path(tmp) / f"c{k}"
            repo.mkdir()
            sh(repo, "init", "-q")
            sh(repo, "symbolic-ref", "HEAD", "refs/heads/main")
            _w(repo, "src/a.py", "x = 1\n")
            _w(repo, "configs/p.json", '{"v": 1}\n')
            _w(repo, "scripts/train_x.py",
               f'PROTOCOL = ROOT / "configs/p.json"\nPROTOCOL_SHA256 = "{sha_lf(repo / "configs/p.json")}"\n')
            _w(repo, "scripts/frozen.py", "x = 1\n")
            _w(repo, "output/run1/audit/launch.json",
               json.dumps({"source_code": {"scripts/frozen.py": sha_lf(repo / "scripts/frozen.py")}}) + "\n")
            (repo / "scripts/crlf.py").write_bytes(b"y = 1\r\n")
            _w(repo, "output/run2/audit/launch.json",
               json.dumps({"source_code": {"scripts/crlf.py": file_sha(repo / "scripts/crlf.py")}}) + "\n")
            _w(repo, "tools/fixme.py", "z = 1\n")
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "init")
            first = git(repo, "rev-parse", "HEAD").strip()   # P3 阶段：一个技术失败的运行，登记的是这个提交里的源码
            _w(repo, "output/run3/audit/launch.json", json.dumps({"identity": {"git_head": first},
               "source_code": {"tools/fixme.py": sha_lf(repo / "tools/fixme.py")}}) + "\n")
            _w(repo, "output/run3/audit/failed.json", '{"type": "AssertionError"}\n')
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "failed run")
            sh(repo, "checkout", "-q", "-b", "exp/S1")
            _w(repo, "src/a.py", "x = 2\n")
            _w(repo, STAGE_PLAN, "".join(f"## {h}\n内容\n" for h in STAGE_HEADINGS))
            _w(repo, DECISIONS, _DEC_HEAD)
            st = {"round": 1, "base": git(repo, "rev-parse", "--short", "HEAD").strip(), "extra": "", "codex": "旧格式内容。"}
            mutate(repo, st)
            _w(repo, HANDOFF, _SELFTEST_HANDOFF.format(**st))
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--root", str(repo), "--no-diff", *flags],
                               capture_output=True, env=env)
            out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
            failed = sorted({int(x) for x in re.findall(r"^\[失败\] (\d)", out, re.M)})
            ok = (r.returncode == 0 and not failed) if expect is None else (r.returncode == 1 and expect in failed)
            bad += not ok
            want = "应通过" if expect is None else f"应拦在第 {expect} 项"
            print(f"{'ok ' if ok else 'BAD'} {name}：{want}，实际退出码 {r.returncode}，失败项 {failed or '无'}")
            if not ok:
                print(out)
    finally:
        def _force(func, path, _exc):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except OSError:
                pass
        shutil.rmtree(tmp, onerror=_force)
    print(f"自测：{len(cases) - bad}/{len(cases)} 符合预期")
    return 1 if bad else 0


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="仓库根目录（默认取当前目录所在的 git 仓库）")
    ap.add_argument("--codex", action="store_true", help="Codex 交接模式")
    ap.add_argument("--no-diff", action="store_true", help="不打印待审 diff")
    ap.add_argument("--init-baseline", action="store_true", help="记录历史未跟踪文件基线")
    ap.add_argument("--self-test", action="store_true", help="在临时仓库里验证各种违规都会被拦下")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    try:
        root = Path(a.root) if a.root else Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        if a.init_baseline:
            return init_baseline(root)
        return run_checks(root, codex=a.codex, show_diff=not a.no_diff)
    except GitError as e:
        print(f"collab_check 无法运行：{e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
