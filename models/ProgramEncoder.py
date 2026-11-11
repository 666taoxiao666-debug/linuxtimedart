"""L1: sandbox, verification and TSMixer wrapper for learnable upstream-encoder programs.

Written by Claude (2026-11-10) per STAGE_PLAN M1 "L1" rules (`git show a1ebfa0:docs/STAGE_PLAN.md`) and the
interface in configs/l1_llm_encoder_protocol.json. Codex reviews, runs the torch tests on the server, and writes
the search/data script (scripts/llm_encoder_search_l1.py) on top of this module.

Program contract
  INIT_P = [1..16 numeric constants]            # learnable scalars, initial values
  def encode(obs, geo, P): return <tensor [B, 24, K]>, 1 <= K <= 11
  obs [B, 24, S, 7] = ws_z, dir_sin, dir_cos, p_anom_z, dp3_z, valid_ws, valid_p
      (hourly station observations up to issue - 1 h; time causality is guaranteed by the caller's input cut)
  geo [S, 3]        = dist_100km, bearing_sin, bearing_cos (bearing from the farm to the station)
  P   [len(INIT_P)] = learnable scalars

Safety layers
  1. Static AST whitelist (validate_source): no imports, loops, conditionals, defs, lambdas, comprehensions,
     in-place ops or subscript stores; only whitelisted names and attributes; every reduction / cat / stack /
     softmax / unsqueeze / squeeze names a constant dim that is not 0; no full reductions.
  2. Dynamic checks in a separate process (verify_in_subprocess, 60 s timeout, address-space limit): output shape,
     finiteness (incl. all-missing hours), gradient, station-permutation invariance, variable station count,
     batch invariance (alone / shuffled / mixed with other samples). Dynamic checks are authoritative for
     cross-sample leakage that the static rule cannot see (e.g. negative dims that hit the batch axis).
  3. Execution with empty builtins; the only names in scope are torch, F (= torch.nn.functional), lag, wmean,
     safe_div.
torch is imported lazily, so the static parts (validation, hashing, expert program P, grammar R) run without it.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

try:  # torch is optional at import time (static parts are torch-free)
    import torch as _torch
    from torch import nn as _nn
except ImportError:  # pragma: no cover - sandbox without torch
    _torch = None
    _nn = None

SEQ_LEN = 24
OBS_CHANNELS = ("ws_z", "dir_sin", "dir_cos", "p_anom_z", "dp3_z", "valid_ws", "valid_p")
GEO_CHANNELS = ("dist_100km", "bearing_sin", "bearing_cos")
MAX_PARAMS = 16
MAX_K = 11
MAX_SOURCE_CHARS = 10000
MAX_NODES = 1200

ARG_NAMES = ("obs", "geo", "P")
HELPER_NAMES = ("lag", "wmean", "safe_div")
MODULE_NAMES = ("torch", "F")
RESERVED = set(ARG_NAMES) | set(HELPER_NAMES) | set(MODULE_NAMES) | {"INIT_P", "encode"}

TORCH_FUNCS = {
    "exp", "log", "log1p", "sqrt", "abs", "sigmoid", "tanh", "relu", "clamp", "clip", "where", "sum", "mean",
    "amax", "amin", "maximum", "minimum", "stack", "cat", "cos", "sin", "atan2", "square", "sign", "softmax",
    "logsumexp", "pow", "erf", "nan_to_num", "zeros_like", "ones_like", "full_like", "arange", "cumsum", "std",
    "var", "prod", "unsqueeze", "squeeze", "exp2", "expm1", "reciprocal", "rsqrt", "hypot", "lerp",
}
TORCH_CONSTANTS = {"float32", "float64", "pi"}
F_FUNCS = {"softplus", "relu", "sigmoid", "gelu", "silu", "elu", "leaky_relu", "logsigmoid", "softmax", "log_softmax"}
TENSOR_METHODS = {
    "sum", "mean", "amax", "amin", "clamp", "clip", "exp", "log", "log1p", "sqrt", "abs", "sigmoid", "tanh", "relu",
    "unsqueeze", "squeeze", "expand_as", "pow", "square", "sign", "cos", "sin", "softmax", "logsumexp", "cumsum",
    "std", "var", "prod", "nan_to_num", "reciprocal", "rsqrt",
}
TENSOR_ATTRS = {"shape", "ndim", "dtype", "device"}
ALLOWED_KEYWORDS = {"dim", "keepdim", "min", "max", "dtype", "device", "nan", "posinf", "neginf", "beta",
                    "threshold", "negative_slope", "alpha", "k", "correction", "unbiased", "weight"}
DIM_REQUIRED = {"sum", "mean", "amax", "amin", "softmax", "log_softmax", "logsumexp", "cumsum", "std", "var",
                "prod", "stack", "cat", "unsqueeze", "squeeze", "wmean"}
BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
UNARY_OPS = (ast.USub, ast.UAdd)
CMP_OPS = (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)

# Structural rules that validate_source enforces but the r1/r2 prompts never stated (L1 r2 preflight, R2-B1).
# Neutral wording: shown to both L and L-blind.
PROGRAM_STRUCTURE_TEXT = (
    "Inside encode, every statement is either `name = expression` with exactly one plain name on the left "
    "(no tuple unpacking such as `a, b = x, y`) or the single final `return`. "
    "Index P only as P[i] with 0 <= i < len(INIT_P)."
)

PROGRAM_RULES_TEXT = (
    "Rules: define INIT_P (a list of 1-16 numbers) and def encode(obs, geo, P) returning a tensor [B, 24, K] with "
    "1 <= K <= 11. obs is [B, 24, S, 7] with channels ws_z, dir_sin, dir_cos, p_anom_z, dp3_z, valid_ws, valid_p "
    "(invalid entries are 0); geo is [S, 3] with dist_100km, bearing_sin, bearing_cos; P has len(INIT_P) entries. "
    "Available names only: torch, F (torch.nn.functional), lag(x, k) (shift x along the time axis dim=1 by a "
    "constant k in 0..23, causal edge padding), wmean(x, w, dim=...) (weighted mean), safe_div(a, b). "
    "No imports, loops, if, lambdas, comprehensions, nested functions, in-place ops or item assignment. "
    "Every sum/mean/amax/amin/softmax/logsumexp/cumsum/std/var/prod/stack/cat/unsqueeze/squeeze/wmean must pass "
    "an explicit constant dim that is not 0. Never mix samples across the batch axis. The output must not depend "
    "on station order and must work for any S. Use dtype=obs.dtype, device=obs.device when creating tensors. "
    + PROGRAM_STRUCTURE_TEXT
)


class ProgramRejected(ValueError):
    """Static validation failure (counts as a failed program; no regeneration)."""


# --------------------------------------------------------------------------- static validation
def _reject(msg, node=None):
    where = f" (line {getattr(node, 'lineno', '?')})" if node is not None else ""
    raise ProgramRejected(msg + where)


def _number(node):
    """Return the numeric value of a Constant or -Constant node, else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _check_dim(node, call):
    values = node.elts if isinstance(node, (ast.Tuple, ast.List)) else [node]
    for v in values:
        d = _number(v)
        if d is None or isinstance(d, float):
            _reject("dim must be a constant integer", call)
        if d == 0:
            _reject("dim=0 (batch axis) is forbidden", call)


def _call_name(call):
    """('torch'|'F'|'helper'|'method', name) for a validated call target."""
    f = call.func
    if isinstance(f, ast.Name):
        if f.id not in HELPER_NAMES:
            _reject(f"call to {f.id!r} is not allowed", call)
        return "helper", f.id
    if not isinstance(f, ast.Attribute):
        _reject("only direct function or method calls are allowed", call)
    chain, cur = [f.attr], f.value
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    chain.reverse()
    if isinstance(cur, ast.Name) and cur.id == "torch":
        if chain[:2] == ["nn", "functional"] and len(chain) == 3 and chain[2] in F_FUNCS:
            return "F", chain[2]
        if len(chain) == 1 and chain[0] in TORCH_FUNCS:
            return "torch", chain[0]
        _reject(f"torch.{'.'.join(chain)} is not allowed", call)
    if isinstance(cur, ast.Name) and cur.id == "F":
        if len(chain) == 1 and chain[0] in F_FUNCS:
            return "F", chain[0]
        _reject(f"F.{'.'.join(chain)} is not allowed", call)
    if len(chain) == 1 and chain[0] in TENSOR_METHODS:
        return "method", chain[0]
    _reject(f"method {'.'.join(chain)!r} is not allowed", call)


def _validate_call(call):
    kind, name = _call_name(call)
    for kw in call.keywords:
        if kw.arg is None or kw.arg not in ALLOWED_KEYWORDS:
            _reject(f"keyword {kw.arg!r} is not allowed", call)
    if any(isinstance(a, ast.Starred) for a in call.args):
        _reject("starred arguments are not allowed", call)
    if kind == "helper" and name == "lag":
        k = next((kw.value for kw in call.keywords if kw.arg == "k"), call.args[1] if len(call.args) > 1 else None)
        kv = None if k is None else _number(k)
        if kv is None or isinstance(kv, float) or not 0 <= kv < SEQ_LEN:
            _reject("lag(x, k) needs a constant integer k in 0..23", call)
    if name in DIM_REQUIRED:
        dim = next((kw.value for kw in call.keywords if kw.arg == "dim"), None)
        if dim is None:
            pos = {"method": 0, "torch": 1, "F": 1, "helper": 2}[kind]
            dim = call.args[pos] if len(call.args) > pos else None
        if dim is None:
            _reject(f"{name} needs an explicit dim", call)
        _check_dim(dim, call)


def _validate_index(node):
    if isinstance(node, getattr(ast, "Index", ())):  # Python 3.8 wraps indices
        node = node.value
    parts = node.elts if isinstance(node, ast.Tuple) else [node]
    for p in parts:
        if isinstance(p, ast.Slice):
            for bound in (p.lower, p.upper, p.step):
                if bound is not None and _number(bound) is None:
                    _reject("slice bounds must be constant integers", p)
        elif isinstance(p, ast.Constant) and (p.value is None or p.value is Ellipsis):
            continue
        elif _number(p) is None or isinstance(_number(p), float):
            _reject("indices must be constant integers, slices, None or ...", p)


def _validate_expr(node, defined):
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            _reject("unexpected store", node)
        if node.id not in defined:
            _reject(f"name {node.id!r} is undefined or not allowed", node)
        return
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or v is None or v is Ellipsis:
            return
        if isinstance(v, int) and abs(v) <= 10 ** 6:
            return
        if isinstance(v, float) and math.isfinite(v):
            return
        _reject(f"constant {v!r} is not allowed", node)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, BIN_OPS):
            _reject(f"operator {type(node.op).__name__} is not allowed", node)
        if isinstance(node.op, ast.Pow):
            e = _number(node.right)
            if e is not None and abs(e) > 8:
                _reject("constant exponent must satisfy |e| <= 8", node)
            if _number(node.left) is not None and e is not None:
                _reject("constant ** constant is not allowed", node)
        _validate_expr(node.left, defined)
        _validate_expr(node.right, defined)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, UNARY_OPS):
            _reject(f"operator {type(node.op).__name__} is not allowed", node)
        _validate_expr(node.operand, defined)
        return
    if isinstance(node, ast.Compare):
        if not all(isinstance(op, CMP_OPS) for op in node.ops):
            _reject("comparison operator not allowed", node)
        _validate_expr(node.left, defined)
        for c in node.comparators:
            _validate_expr(c, defined)
        return
    if isinstance(node, ast.Call):
        _validate_call(node)
        if isinstance(node.func, ast.Attribute):  # validate the receiver of a method call
            cur = node.func.value
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if not (isinstance(cur, ast.Name) and cur.id in ("torch", "F")):
                _validate_expr(node.func.value, defined)
        for a in node.args:
            _validate_expr(a, defined)
        for kw in node.keywords:
            _validate_expr(kw.value, defined)
        return
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            _reject("private attributes are not allowed", node)
        if isinstance(node.value, ast.Name) and node.value.id == "torch":
            if node.attr not in TORCH_CONSTANTS:
                _reject(f"torch.{node.attr} is only allowed as a call", node)
            return
        if node.attr not in TENSOR_ATTRS:
            _reject(f"attribute {node.attr!r} is not allowed", node)
        _validate_expr(node.value, defined)
        return
    if isinstance(node, ast.Subscript):
        if not isinstance(node.ctx, ast.Load):
            _reject("item assignment is not allowed", node)
        _validate_index(node.slice)
        _validate_expr(node.value, defined)
        return
    if isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            _validate_expr(e, defined)
        return
    _reject(f"syntax {type(node).__name__} is not allowed", node)


def _init_p_values(node):
    if not isinstance(node, (ast.List, ast.Tuple)):
        _reject("INIT_P must be a list of numbers", node)
    vals = [_number(e) for e in node.elts]
    if not vals or len(vals) > MAX_PARAMS or any(v is None for v in vals):
        _reject(f"INIT_P must hold 1..{MAX_PARAMS} numeric constants", node)
    if not all(math.isfinite(float(v)) for v in vals):
        _reject("INIT_P values must be finite", node)
    return [float(v) for v in vals]


def _is_docstring(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def validate_source(source):
    """Static checks. Returns {'n_params', 'init_p', 'hash', 'n_nodes'}; raises ProgramRejected."""
    if not isinstance(source, str) or not source.strip():
        _reject("empty program")
    if len(source) > MAX_SOURCE_CHARS:
        _reject("program too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramRejected(f"syntax error: {exc.msg} (line {exc.lineno})") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_NODES:
        _reject("program has too many AST nodes")
    body = [s for s in tree.body if not _is_docstring(s)]
    assigns = [s for s in body if isinstance(s, ast.Assign)]
    funcs = [s for s in body if isinstance(s, ast.FunctionDef)]
    if len(body) != 2 or len(assigns) != 1 or len(funcs) != 1:
        _reject("module must contain exactly INIT_P = [...] and def encode(obs, geo, P)")
    a, f = assigns[0], funcs[0]
    if len(a.targets) != 1 or not isinstance(a.targets[0], ast.Name) or a.targets[0].id != "INIT_P":
        _reject("the only top-level assignment must be INIT_P", a)
    init_p = _init_p_values(a.value)
    args = f.args
    if f.name != "encode" or f.decorator_list or f.returns is not None:
        _reject("function must be a plain def encode(obs, geo, P)", f)
    if (getattr(args, "posonlyargs", []) or args.vararg or args.kwarg or args.kwonlyargs or args.defaults
            or [x.arg for x in args.args] != list(ARG_NAMES) or any(x.annotation for x in args.args)):
        _reject("signature must be exactly encode(obs, geo, P)", f)
    stmts = list(f.body)
    if stmts and _is_docstring(stmts[0]):
        stmts = stmts[1:]
    if not stmts or not isinstance(stmts[-1], ast.Return) or stmts[-1].value is None:
        _reject("encode must end with a single return <tensor>", f)
    defined = set(ARG_NAMES) | set(HELPER_NAMES) | set(MODULE_NAMES)
    for s in stmts[:-1]:
        if not isinstance(s, ast.Assign):
            _reject(f"statement {type(s).__name__} is not allowed (only name = expression and return)", s)
        if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
            _reject("assignment target must be a single name", s)
        target = s.targets[0].id
        if target in RESERVED or target.startswith("_"):
            _reject(f"cannot assign to {target!r}", s)
        _validate_expr(s.value, defined)
        defined.add(target)
    _validate_expr(stmts[-1].value, defined)
    return {"n_params": len(init_p), "init_p": init_p, "hash": canonical_hash(source), "n_nodes": n_nodes}


class _Canon(ast.NodeTransformer):
    """Rename encode's local variables in order of first assignment; drop docstrings."""

    def __init__(self):
        self.names = {}

    def visit_FunctionDef(self, node):
        node.body = [s for s in node.body if not _is_docstring(s)]
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in RESERVED:
            return node
        if isinstance(node.ctx, ast.Store) and node.id not in self.names:
            self.names[node.id] = f"v{len(self.names)}"
        return ast.copy_location(ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node)


def canonical_hash(source):
    tree = ast.parse(source)
    tree.body = [s for s in tree.body if not _is_docstring(s)]
    tree = _Canon().visit(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def extract_code(text):
    """First ```python fenced block of an LLM reply (or the whole text when there is no fence).

    An opening fence without a closing one (reply cut at max_new_tokens) yields everything after the
    opening fence, so the program is judged on its own text instead of failing on the fence line (L1 r1).
    """
    m = re.search(r"```(?:python|py)?[ \t\r]*\n(.*?)(?:```|\Z)", text, flags=re.S)
    return (m.group(1) if m else text).strip() + "\n"


# --------------------------------------------------------------------------- L1 round 2 search feedback (no torch)
FEEDBACK_CHANGE = 0.10  # change cell: |truth - anchor| >= 10% of rated power (same as scripts/ramp_diagnostics.py)


def feedback_breakdown(pred, truth, anchor, rated=2050.0, change=FEEDBACK_CHANGE):
    """Validation error breakdown shown to the LLM in round 2 (anchor = last observed hourly power, [window, lead])."""
    import numpy as np
    pred, truth, anchor = (np.asarray(a, dtype=np.float64) for a in (pred, truth, anchor))
    if pred.ndim != 2 or pred.shape != truth.shape or truth.shape != anchor.shape:
        raise ValueError("pred, truth and anchor must be aligned [window, lead] arrays")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all() and np.isfinite(anchor).all()):
        raise ValueError("non-finite values in feedback inputs")
    err = np.abs(pred - truth)
    chg = np.abs(truth - anchor) >= change * float(rated)
    total = float(err.sum())
    return {"mae": float(err.mean()), "mae_by_lead": [float(v) for v in err.mean(axis=0)],
            "mae_calm": float(err[~chg].mean()) if (~chg).any() else None,
            "mae_change": float(err[chg].mean()) if chg.any() else None,
            "change_cell_share": float(chg.mean()),
            "change_error_share": float(err[chg].sum()) / total if total > 0 else None}


def _fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render_feedback(target_mae, shown, failures, max_failures=6, max_reason_chars=160):
    """Feedback block appended to every round-2 L prompt (fixed wording, deterministic).

    target_mae: validation MAE (kW) of the hand-feature arm H, short-trained in the same run.
    shown: [(program_id, val_mae, breakdown)] for the programs whose code is shown in the prompt, best first.
    failures: [(program_id, failure_reason)] of the previous generation in slot order (they used slots too).
    """
    lines = [f"Target to beat: the hand-crafted 11-feature encoder H reaches validation MAE {float(target_mae):.2f} kW "
             "with the same backbone and training. Only programs clearly below this value are useful."]
    for pid, val, b in shown:
        lead = "/".join(f"{v:.1f}" for v in b["mae_by_lead"])
        share = b["change_error_share"]
        lines.append(f"{pid}: validation MAE {float(val):.2f} kW; by lead h1/h2/h3/h4 = {lead}; calm cells "
                     f"{_fmt(b['mae_calm'])}; change cells (|power change| >= 10% of rated) {_fmt(b['mae_change'])}, "
                     f"{100 * b['change_cell_share']:.0f}% of cells and "
                     f"{'n/a' if share is None else f'{100 * share:.0f}%'} of absolute error.")
    if failures:
        lines.append("Programs of the previous generation that failed (each still used a slot); avoid these errors:")
        for pid, reason in list(failures)[:max_failures]:
            lines.append(f"- {pid}: {' '.join(str(reason).split())[:max_reason_chars]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- execution helpers (torch)
def lag(x, k):
    """Causal shift along the time axis (dim=1) with edge padding: out[:, t] = x[:, max(t - k, 0)]."""
    k = int(k)
    if not 0 <= k < x.shape[1]:
        raise ValueError("lag k out of range")
    if k == 0:
        return x
    pad = x[:, :1].expand(*([x.shape[0], k] + list(x.shape[2:])))
    return _torch.cat([pad, x[:, :-k]], dim=1)


def wmean(x, w, dim):
    """Weighted mean along dim; 0 where all weights are 0."""
    if dim == 0:
        raise ValueError("wmean over the batch axis is forbidden")
    return (x * w).sum(dim=dim) / (w.sum(dim=dim) + 1e-6)


def safe_div(a, b):
    b = _torch.as_tensor(b)
    return a / _torch.where(_torch.abs(b) > 1e-6, b, _torch.full_like(b, 1e-6))


def compile_program(source):
    """Validate, then exec with empty builtins. Returns (init_p list, encode function, static info)."""
    if _torch is None:
        raise RuntimeError("torch is required to compile a program")
    info = validate_source(source)
    namespace = {"__builtins__": {}, "torch": _torch, "F": _torch.nn.functional,
                 "lag": lag, "wmean": wmean, "safe_div": safe_div}
    exec(compile(ast.parse(source), "<l1-program>", "exec"), namespace)  # noqa: S102 - AST-whitelisted
    return list(info["init_p"]), namespace["encode"], info


# --------------------------------------------------------------------------- dynamic verification
def _synthetic(batch, stations, generator, dtype):
    t = _torch
    ws = t.randn(batch, SEQ_LEN, stations, generator=generator, dtype=dtype)
    ang = t.rand(batch, SEQ_LEN, stations, generator=generator, dtype=dtype) * 2 * math.pi
    pan = t.randn(batch, SEQ_LEN, stations, generator=generator, dtype=dtype)
    dp3 = t.randn(batch, SEQ_LEN, stations, generator=generator, dtype=dtype)
    vw = (t.rand(batch, SEQ_LEN, stations, generator=generator, dtype=dtype) < 0.85).to(dtype)
    vp = (t.rand(batch, SEQ_LEN, stations, generator=generator, dtype=dtype) < 0.8).to(dtype)
    vw[:, 5] = 0.0  # one hour with every station missing
    vp[:, 5] = 0.0
    obs = t.stack([ws * vw, t.sin(ang) * vw, t.cos(ang) * vw, pan * vp, dp3 * vp, vw, vp], dim=-1)
    dist = 0.2 + 1.3 * t.rand(stations, generator=generator, dtype=dtype)
    bear = t.rand(stations, generator=generator, dtype=dtype) * 2 * math.pi
    geo = t.stack([dist, t.sin(bear), t.cos(bear)], dim=-1)
    return obs, geo


def _close(a, b, tol):
    return a.shape == b.shape and bool(_torch.allclose(a, b, rtol=1e-6, atol=tol))


def check_program(source, *, tol=1e-5, seed=0, stations=9, batch=6):
    """Dynamic checks in the current process. Returns a JSON-serialisable dict with status OK or FAIL."""
    t = _torch
    try:
        init_p, encode, info = compile_program(source)
    except ProgramRejected as exc:
        return {"status": "REJECTED", "reason": str(exc)[:500]}
    t.set_num_threads(1)
    g = t.Generator().manual_seed(seed)
    dt = t.float64
    try:
        obs, geo = _synthetic(batch, stations, g, dt)
        P = t.tensor(init_p, dtype=dt, requires_grad=True)
        out = encode(obs, geo, P)
        if not isinstance(out, t.Tensor) or out.ndim != 3 or tuple(out.shape[:2]) != (batch, SEQ_LEN):
            return {"status": "FAIL", "reason": f"output shape {getattr(out, 'shape', None)} != [B, 24, K]"}
        k = int(out.shape[-1])
        if not 1 <= k <= MAX_K:
            return {"status": "FAIL", "reason": f"K={k} outside 1..{MAX_K}"}
        if not bool(t.isfinite(out).all()):
            return {"status": "FAIL", "reason": "non-finite output"}
        grad_nonzero = 0
        if out.requires_grad:  # a program may ignore P; it is then a fixed (non-learnable) encoder
            out.square().mean().backward()
            grad = P.grad
            if grad is None or not bool(t.isfinite(grad).all()):
                return {"status": "FAIL", "reason": "non-finite gradient for P"}
            grad_nonzero = int((grad.abs() > 0).sum())
        with t.no_grad():
            P0 = P.detach()
            ref = encode(obs, geo, P0)
            for i in range(batch):  # each sample alone
                if not _close(encode(obs[i:i + 1], geo, P0), ref[i:i + 1], tol):
                    return {"status": "FAIL", "reason": f"batch invariance: sample {i} alone differs"}
            perm = t.randperm(batch, generator=g)
            if not _close(encode(obs[perm], geo, P0), ref[perm], tol):
                return {"status": "FAIL", "reason": "batch invariance: shuffled batch differs"}
            other, _ = _synthetic(batch + 3, stations, g, dt)
            if not _close(encode(t.cat([obs, other], dim=0), geo, P0)[:batch], ref, tol):
                return {"status": "FAIL", "reason": "batch invariance: mixing with other samples changes output"}
            sp = t.randperm(stations, generator=g)
            if not _close(encode(obs[:, :, sp], geo[sp], P0), ref, tol):
                return {"status": "FAIL", "reason": "station permutation changes output"}
            for s in (11, 36):
                o2, g2 = _synthetic(2, s, g, dt)
                y2 = encode(o2, g2, P0)
                if tuple(y2.shape) != (2, SEQ_LEN, k) or not bool(t.isfinite(y2).all()):
                    return {"status": "FAIL", "reason": f"fails with S={s} stations"}
            y32 = encode(obs.float(), geo.float(), P0.float())
            if not bool(t.isfinite(y32).all()):
                return {"status": "FAIL", "reason": "non-finite output in float32"}
        # training-size probe (float32, Kelmarsh station count, with backward): catches memory blow-ups such as
        # repeated station-axis outer products; under the child's address-space limit these end as CRASH.
        ob, gb = _synthetic(32, 36, g, t.float32)
        Pb = t.tensor(init_p, dtype=t.float32, requires_grad=True)
        yb = encode(ob, gb, Pb)
        if yb.requires_grad:
            yb.square().mean().backward()
        if not bool(t.isfinite(yb).all()):
            return {"status": "FAIL", "reason": "non-finite output at training size"}
    except Exception as exc:  # program raised
        return {"status": "FAIL", "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
    return {"status": "OK", "K": k, "n_params": info["n_params"], "grad_nonzero": grad_nonzero,
            "hash": info["hash"], "n_nodes": info["n_nodes"]}


def verify_in_subprocess(source, *, timeout_sec=60, memory_mb=8192, tol=1e-5, seed=0):
    """Static check here, dynamic checks in a child process with timeout and address-space limit.

    status: OK | REJECTED | FAIL | TIMEOUT | CRASH (crash covers OOM / memory-limit kills). Every non-OK status
    is a failed program that still uses its budget slot.
    """
    try:
        validate_source(source)
    except ProgramRejected as exc:
        return {"status": "REJECTED", "reason": str(exc)[:500]}
    payload = json.dumps({"source": source, "tol": tol, "seed": seed})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def limit():  # pragma: no cover - runs in the child
        import resource
        cap = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    try:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify"], input=payload,
                           capture_output=True, text=True, timeout=timeout_sec, env=env,
                           preexec_fn=limit if os.name == "posix" else None)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "reason": f"exceeded {timeout_sec} s"}
    lines = [x for x in r.stdout.strip().splitlines() if x.strip()]
    if r.returncode != 0 or not lines:
        return {"status": "CRASH", "returncode": r.returncode, "reason": r.stderr[-1500:]}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"status": "CRASH", "reason": "unparseable child output"}


# --------------------------------------------------------------------------- expert program P and grammar R
CONE_KERNELS = {
    "sigmoid": "torch.sigmoid(torch.exp(P[0]) * (align - P[1]))",
    "softplus_cos": "F.softplus(torch.exp(P[0]) * (align - P[1])) / (F.softplus(torch.exp(P[0]) * (1.0 - P[1])) + 1e-6)",
    "exp_align": "torch.exp(torch.exp(P[0]) * (align - 1.0))",
    "quadratic_upwind": "torch.clamp(align - P[1], min=0.0) ** 2",
}
DISTANCE_KERNELS = {
    "exp_aniso": "torch.exp(-dist / (F.softplus(P[2]) + 0.05) - cross / (F.softplus(P[3]) + 0.05))",
    "gaussian_iso": "torch.exp(-(dist / (F.softplus(P[2]) + 0.05)) ** 2)",
    "inverse_multiquadric": "1.0 / torch.sqrt(1.0 + (dist / (F.softplus(P[2]) + 0.05)) ** 2)",
    "bi_scale_exp": "0.5 * torch.exp(-dist / (F.softplus(P[2]) + 0.05)) + 0.5 * torch.exp(-dist / (F.softplus(P[3]) + 0.5))",
}
_GAUSS = "torch.softmax(-(lags - tau) ** 2 / (2.0 * (F.softplus(P[6]) + 0.3) ** 2), dim=-1)"
ADVECTION_KERNELS = {
    "gaussian_arrival": _GAUSS,
    "laplace_arrival": "torch.softmax(-torch.abs(lags - tau) / (F.softplus(P[6]) + 0.3), dim=-1)",
    "flux_weighted": "torch.softmax(-(lags - tau) ** 2 / (2.0 * (F.softplus(P[6]) + 0.3) ** 2) - F.softplus(P[7]) * lags, dim=-1)",
    "dual_speed_band": "0.5 * " + _GAUSS + " + 0.5 * torch.softmax(-(lags - tau * torch.exp(P[7])) ** 2 / (2.0 * (F.softplus(P[6]) + 0.3) ** 2), dim=-1)",
}
FEATURE_ORDER = ("upwind_advected_ws", "upwind_ws", "upwind_dws", "upwind_minus_local", "regional_ws",
                 "regional_dws", "upwind_dp3", "regional_dp3", "pressure_grad_east", "pressure_grad_north",
                 "upwind_weight_mass")
# P[0] cone sharpness (log), P[1] cone offset, P[2]/P[3] distance scales, P[4]/P[5] advection speed slope/bias,
# P[6] arrival width, P[7] extra advection shape, P[8] nearest-station temperature.
INIT_P_TEMPLATE = [1.386, 0.3, 0.0, 0.0, 0.1, -1.0, 0.0, 0.0, -2.0]
EXPERT_SPEC = {"cone": "sigmoid", "distance": "exp_aniso", "advection": "gaussian_arrival", "lag": 3, "k": 11}

_TEMPLATE = '''INIT_P = {init_p}


def encode(obs, geo, P):
    """Grammar program: cone={cone}, distance={distance}, advection={advection}, lag={lag}, K={k}."""
    ws = obs[..., 0]
    dsin = obs[..., 1]
    dcos = obs[..., 2]
    pan = obs[..., 3]
    dp3 = obs[..., 4]
    vw = obs[..., 5]
    vp = obs[..., 6]
    dist = geo[:, 0]
    bsin = geo[:, 1]
    bcos = geo[:, 2]
    rsin = wmean(dsin, vw, dim=2)
    rcos = wmean(dcos, vw, dim=2)
    rnorm = torch.sqrt(rsin * rsin + rcos * rcos) + 1e-6
    align = bsin * (rsin / rnorm).unsqueeze(-1) + bcos * (rcos / rnorm).unsqueeze(-1)
    cross = dist * torch.sqrt(torch.clamp(1.0 - align * align, min=0.0) + 1e-9)
    cone = {cone_expr}
    decay = {distance_expr}
    wup = cone * decay * vw
    speed = F.softplus(P[4] * ws + P[5]) + 0.05
    tau = (dist * torch.clamp(align, min=0.0) / speed).unsqueeze(-1)
    lags = torch.arange(6, dtype=obs.dtype, device=obs.device)
    kern = {advection_expr}
    hist = torch.stack([lag(ws, 0), lag(ws, 1), lag(ws, 2), lag(ws, 3), lag(ws, 4), lag(ws, 5)], dim=-1)
    vhist = torch.stack([lag(vw, 0), lag(vw, 1), lag(vw, 2), lag(vw, 3), lag(vw, 4), lag(vw, 5)], dim=-1)
    ws_adv = (kern * hist * vhist).sum(dim=-1) / ((kern * vhist).sum(dim=-1) + 1e-6)
    nearw = torch.exp(-dist / (F.softplus(P[8]) + 0.02))
    local = wmean(ws, nearw * vw, dim=2)
    up = wmean(ws, wup, dim=2)
    upadv = wmean(ws_adv, wup, dim=2)
    dws = ws - lag(ws, {lag})
    vd = vw * lag(vw, {lag})
    updws = wmean(dws, cone * decay * vd, dim=2)
    reg = wmean(ws, vw, dim=2)
    regdws = wmean(dws, vd, dim=2)
    updp = wmean(dp3, cone * decay * vp, dim=2)
    regdp = wmean(dp3, vp, dim=2)
    x = dist * bsin * torch.ones_like(pan)
    y = dist * bcos * torch.ones_like(pan)
    dx = x - wmean(x, vp, dim=2).unsqueeze(-1)
    dy = y - wmean(y, vp, dim=2).unsqueeze(-1)
    dpan = pan - wmean(pan, vp, dim=2).unsqueeze(-1)
    gx = wmean(dpan * dx, vp, dim=2) / (wmean(dx * dx, vp, dim=2) + 1e-3)
    gy = wmean(dpan * dy, vp, dim=2) / (wmean(dy * dy, vp, dim=2) + 1e-3)
    mass = wup.sum(dim=2) / (vw.sum(dim=2) + 1e-6)
    feats = torch.stack([upadv, up, updws, up - local, reg, regdws, updp, regdp, gx, gy, mass], dim=-1)
    return feats[..., :{k}]
'''


def render_program(cone, distance, advection, lag_steps, k):
    """Source text of one grammar program (used for the expert P and the random-search arm R)."""
    if not 1 <= int(k) <= MAX_K or not 1 <= int(lag_steps) < SEQ_LEN:
        raise ValueError("invalid grammar choice")
    return _TEMPLATE.format(init_p=repr(INIT_P_TEMPLATE), cone=cone, distance=distance, advection=advection,
                            lag=int(lag_steps), k=int(k), cone_expr=CONE_KERNELS[cone],
                            distance_expr=DISTANCE_KERNELS[distance], advection_expr=ADVECTION_KERNELS[advection])


EXPERT_PROGRAM = render_program(EXPERT_SPEC["cone"], EXPERT_SPEC["distance"], EXPERT_SPEC["advection"],
                                EXPERT_SPEC["lag"], EXPERT_SPEC["k"])


def sample_grammar_programs(grammar, seed=2024, budget=30):
    """Arm R: `budget` distinct grammar combinations drawn without replacement (expert combination excluded).

    grammar = protocol["grammar_R"] with cone_kernels, distance_kernels, advection_kernels, lag_steps,
    channel_counts. Deterministic for a given seed and grammar.
    """
    combos = [c for c in itertools.product(grammar["cone_kernels"], grammar["distance_kernels"],
                                           grammar["advection_kernels"], grammar["lag_steps"],
                                           grammar["channel_counts"])
              if c != (EXPERT_SPEC["cone"], EXPERT_SPEC["distance"], EXPERT_SPEC["advection"],
                       EXPERT_SPEC["lag"], EXPERT_SPEC["k"])]
    if budget > len(combos):
        raise ValueError("budget exceeds grammar size")
    picks = random.Random(seed).sample(combos, budget)
    out = []
    for cone, dist, adv, lag_steps, k in picks:
        spec = {"cone": cone, "distance": dist, "advection": adv, "lag": int(lag_steps), "k": int(k)}
        out.append({"spec": spec, "source": render_program(cone, dist, adv, lag_steps, k)})
    return out


# --------------------------------------------------------------------------- TSMixer wrapper
if _torch is not None:

    class ProgramTSMixerModel(_nn.Module):
        """encode(obs, geo, P) -> fixed-normalized channels -> TSMixerBasicSmall, trained end to end.

        x is [B, 24, n_scada + 2] in the order [8 SCADA, upstream_missing, power] (the H layout without the
        11 hand features). The backbone input is [8 SCADA, K encoded, upstream_missing, power], so with K = 11
        it has the same 21-channel layout and parameter count as the H arm.
        """

        def __init__(self, source, geo, *, blocks=2, ff_dim=32, dropout=0.1, norm_eps=1e-3, seq_len=SEQ_LEN,
                     pred_len=4, n_scada=8):
            super().__init__()
            from models.TSMixerBasicSmall import Model as TSMixer
            init_p, encode, info = compile_program(source)
            self.encode_fn = encode
            self.program_hash = info["hash"]
            self.n_scada = int(n_scada)
            self.P = _nn.Parameter(_torch.tensor(init_p, dtype=_torch.float32))
            self.register_buffer("geo", _torch.as_tensor(geo, dtype=_torch.float32).clone())
            with _torch.no_grad():
                probe = encode(_torch.zeros(1, seq_len, self.geo.shape[0], len(OBS_CHANNELS)), self.geo, self.P)
            self.k = int(probe.shape[-1])
            if not 1 <= self.k <= MAX_K:
                raise ValueError("program output width outside 1..11")
            self.register_buffer("enc_mean", _torch.zeros(self.k))
            self.register_buffer("enc_std", _torch.ones(self.k))
            self.register_buffer("normalized", _torch.zeros((), dtype=_torch.bool))
            self.backbone = TSMixer(seq_len=seq_len, channels=self.n_scada + self.k + 2, pred_len=pred_len,
                                    blocks=blocks, ff_dim=ff_dim, dropout=dropout, norm_eps=norm_eps,
                                    target_slot=self.n_scada + self.k + 1, input_mode="full")

        @_torch.no_grad()
        def fit_encoder_normalization(self, obs_fit, batch=1024):
            """Fixed per-channel mean/std of the encoder output at INIT_P on fit-period obs only (call once)."""
            total, s1, s2 = 0, _torch.zeros(self.k, dtype=_torch.float64), _torch.zeros(self.k, dtype=_torch.float64)
            for i in range(0, obs_fit.shape[0], batch):
                chunk = obs_fit[i:i + batch].to(self.geo.device, self.geo.dtype)
                y = self.encode_fn(chunk, self.geo, self.P).double().reshape(-1, self.k)
                if not bool(_torch.isfinite(y).all()):
                    raise ValueError("non-finite encoder output on fit data")
                total, s1, s2 = total + y.shape[0], s1 + y.sum(0).cpu(), s2 + (y * y).sum(0).cpu()
            mean = s1 / total
            std = (s2 / total - mean * mean).clamp_min(0.0).sqrt().clamp_min(1e-3)
            self.enc_mean.copy_(mean.float())
            self.enc_std.copy_(std.float())
            self.normalized.fill_(True)

        def forward(self, x, obs):
            if not bool(self.normalized):
                raise RuntimeError("call fit_encoder_normalization on fit-period data first")
            enc = (self.encode_fn(obs, self.geo, self.P) - self.enc_mean) / self.enc_std
            h = _torch.cat([x[..., :self.n_scada], enc, x[..., self.n_scada:]], dim=-1)
            return self.backbone(h)


if __name__ == "__main__" and "--verify" in sys.argv:
    _req = json.loads(sys.stdin.read())
    print(json.dumps(check_program(_req["source"], tol=float(_req.get("tol", 1e-5)), seed=int(_req.get("seed", 0)))))
