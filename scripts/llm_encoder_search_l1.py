"""L1: LLM-discovered differentiable upstream encoder search and two-farm evaluation."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_pe_spec = importlib.util.spec_from_file_location("program_encoder_l1", ROOT / "models" / "ProgramEncoder.py")
pe = importlib.util.module_from_spec(_pe_spec)
_pe_spec.loader.exec_module(pe)

from scripts import pilot_conformal_c0b as cb
from utils import paired_forecast_statistics as pfs

save, sha = cb.save, cb.sha


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(path: Path, expected_sha256: str, identities: dict) -> Path:
    path = Path(path)
    digest = sha(path)
    if digest != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {digest} != {expected_sha256}")
    identities[str(path)] = digest
    return path
PROTOCOL = ROOT / "configs/l1_llm_encoder_protocol.json"
SCADA_SLC = [0, 1, 2, 3, 4, 5, 6, 7, 19, 20]  # 8 SCADA + upstream_missing + power from 21-ch upstream series


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_hourly_obs_and_geo(observations, stations, hours, fit_years):
    """Build lag-1h hourly station tensor [T, S, 7] and station geometry [S, 3] using fit_years only."""
    hours = pd.DatetimeIndex(hours)
    lagged = hours - pd.Timedelta(hours=1)
    fit_mask = np.isin(hours.year, list(fit_years))
    if not fit_mask.any():
        raise ValueError("Empty fit_years mask for hourly station tensor")

    def mat(col):
        return np.column_stack([
            observations[int(s["src_id"])][col].reindex(lagged).to_numpy(dtype=float)
            if int(s["src_id"]) in observations else np.full(len(hours), np.nan)
            for s in stations
        ])

    ws, wdir, pres = mat("wind_speed"), mat("wind_direction"), mat("msl_pressure")
    ws = np.where(np.isfinite(ws) & (ws >= 0.0), ws, np.nan)
    wdir = np.where(np.isfinite(wdir) & (wdir >= 0.0) & (wdir <= 360.0) & (ws > 0.0), wdir % 360.0, np.nan)
    pres = np.where(np.isfinite(pres) & (pres > 0.0), pres, np.nan)
    dp3 = np.full_like(pres, np.nan)
    dp3[3:] = pres[3:] - pres[:-3]

    ws_fit = ws[fit_mask]
    ws_mean = float(np.nanmean(ws_fit))
    ws_std = float(np.nanstd(ws_fit)) or 1.0
    p_fit = pres[fit_mask]
    p_smean = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan for c in p_fit.T])
    p_anom = pres - p_smean[None, :]
    p_std = float(np.nanstd(p_anom[fit_mask])) or 1.0
    dp3_std = float(np.nanstd(dp3[fit_mask])) or 1.0

    vw = np.isfinite(ws)
    vp = np.isfinite(p_anom) & np.isfinite(dp3)
    vd = vw & np.isfinite(wdir)
    rad = np.radians(np.where(vd, wdir, 0.0))
    obs = np.stack([
        np.where(vw, (ws - ws_mean) / ws_std, 0.0),
        np.where(vd, np.sin(rad), 0.0),
        np.where(vd, np.cos(rad), 0.0),
        np.where(vp, p_anom / p_std, 0.0),
        np.where(vp, dp3 / dp3_std, 0.0),
        vw.astype(np.float32),
        vp.astype(np.float32),
    ], axis=-1).astype(np.float32)

    dist = np.array([float(s["distance_km"]) / 100.0 for s in stations], dtype=np.float32)
    bear = np.radians(np.array([float(s["bearing_deg"]) for s in stations], dtype=np.float32))
    geo = np.column_stack([dist, np.sin(bear), np.cos(bear)]).astype(np.float32)
    stats = {"ws_mean": ws_mean, "ws_std": ws_std, "p_std": p_std, "dp3_std": dp3_std}
    return obs, geo, stats


class L1FarmData:
    """Hold 21-ch H series, 10-ch base slices, and lag-1h station tensors for one farm."""

    def __init__(self, series, hours, obs_hourly, geo, stats):
        self.series = series
        self.hours = pd.DatetimeIndex(hours)
        self.obs_hourly = obs_hourly
        self.geo = geo
        self.stats = stats
        row_hour = self.hours.get_indexer(pd.DatetimeIndex(series.dates))
        if (row_hour < 0).any():
            raise ValueError("Series timestamp missing from hourly station grid")
        self.row_hour = row_hour.astype(np.int64)
        fit_b = np.isin(self.hours.year, list(series.splits["train"]))
        valid_starts = [t - 23 for t in range(23, len(self.hours)) if fit_b[t - 23 : t + 1].all()]
        self.fit_hour_windows = np.asarray(valid_starts, dtype=np.int64)[:, None] + np.arange(24, dtype=np.int64)

    def split_pack(self, split_name: str):
        w = self.series.windows(split_name)
        starts = w.window_starts
        idx24 = starts[:, None] + np.arange(24, dtype=np.int64)
        x21 = self.series.values[idx24].astype(np.float32)
        x10 = x21[:, :, SCADA_SLC]
        h_idx = self.row_hour[idx24]
        truth, anchor = w.truth_and_anchor()
        return {
            "windows": w,
            "x21": x21,
            "x10": x10,
            "h_idx": h_idx,
            "truth": truth.astype(np.float32),
            "anchor": anchor.astype(np.float32),
            "times": w.target_times(),
            "issue": w.target_times()[:, 0] - np.timedelta64(1, "h"),
            "turbines": w.turbines[starts],
            "power_mean": float(self.series.scaler_mean[-1]),
            "power_scale": float(self.series.scaler_scale[-1]),
        }


class _HourWindowObs:
    """Lazy slicer over hourly station tensor [T, S, 7] for fit_encoder_normalization."""

    def __init__(self, obs_hourly_t, win_idx):
        self.obs = obs_hourly_t
        self.idx = win_idx
        self.shape = (len(win_idx), 24, obs_hourly_t.shape[1], obs_hourly_t.shape[2])

    def __getitem__(self, slc):
        return self.obs[self.idx[slc]]


def load_real_farm(data_root: Path, farm: str, splits: dict, identities: dict) -> L1FarmData:
    from scripts import train_penmanshiel_p2_v35 as p2
    u = p2.u
    old = read_json(ROOT / "configs/n1ph_protocol.json")
    recipe = read_json(p2.PROTOCOL)
    last_year = max(y for ys in splits.values() for y in ys)
    if farm == "Kelmarsh":
        ref = recipe["data"]["v31_registered_preparation"]
        kcsv = checked(data_root / ref["csv"], ref["csv_sha256"], identities)
        mpath = kcsv.parent.parent / "metadata/preparation.json"
        checked(mpath, ref["preparation_json_sha256"], identities)
        meta = read_json(mpath)
        frame = pd.concat([c[c.label_year <= last_year] for c in pd.read_csv(kcsv, chunksize=50000)], ignore_index=True)
        spec = recipe["data"]["upstream"]
        manifest = data_root / spec["download_manifest"]
        if u.text_sha256_lf(manifest) != spec["download_manifest_sha256_lf"]:
            raise ValueError("MIDAS manifest hash mismatch")
        identities[str(manifest)] = sha(manifest)
        hashes = u.manifest_hashes(read_json(manifest))
        stations = spec["stations"]
        ids = [int(s["src_id"]) for s in stations]
        files = []
        for item in spec["development_roots"]:
            ys = [y for y in item["years"] if y <= last_year]
            if ys:
                files += u.station_files(data_root / item["root"], ids, ys, hashes, True)[0]
        for e in files:
            checked(e["path"], e["sha256"], identities)
        obs_dict = u.load_station_observations(files, ids)
        hours = pd.date_range("2016-05-01", f"{last_year + 1}-01-01", freq="h")
        up_frame = u.hourly_upstream(obs_dict, stations, hours, 1)
        start_iso = "2016-05-01"
    elif farm == "Penmanshiel":
        prepared = data_root / old["p3_run"] / "prepared"
        expected = read_json(ROOT / "output/penmanshiel_hourly/p3_v35/r2/prepared/prepared_complete.json")
        for name, digest in expected["artifacts"].items():
            checked(prepared / name, digest, identities)
        frame = pd.concat([c[c.label_year <= last_year] for c in pd.read_csv(prepared / "penmanshiel.csv", chunksize=50000)], ignore_index=True)
        up_frame = pd.read_csv(prepared / "upstream.csv", index_col="timestamp_hour_end_utc")
        up_frame.index = pd.to_datetime(up_frame.index)
        up_frame = up_frame[up_frame.index < pd.Timestamp(f"{last_year + 1}-01-01T01:00:00")]
        meta = read_json(prepared / "preparation.json")
        part_a = read_json(ROOT / p2.farm.D2_SUBDIR / "part_A.json")
        identities[str(ROOT / p2.farm.D2_SUBDIR / "part_A.json")] = sha(ROOT / p2.farm.D2_SUBDIR / "part_A.json")
        stations = [s for s in part_a["midas"]["stations"] if int(s["src_id"]) in p2.farm.STATIONS]
        reg = read_json(prepared / "input_hashes.json")
        files = []
        for s in stations:
            sid = int(s["src_id"])
            for y in range(2016, last_year + 1):
                matches = [(p, e) for p, e in reg.items() if Path(p).parent.name == str(sid) and Path(p).name.endswith(f"_qcv-1_{y}.csv")]
                if len(matches) != 1:
                    raise ValueError(f"Station file mismatch for {sid} year {y}")
                rel = matches[0][0].split("/datasets/", 1)[1]
                fpath = data_root / "datasets" / rel
                checked(fpath, matches[0][1]["sha256"], identities)
                files.append({"path": fpath, "year": y})
        ids = [int(s["src_id"]) for s in stations]
        obs_dict = u.load_station_observations(files, ids)
        hours = pd.date_range("2016-05-01", f"{last_year + 1}-01-01", freq="h")
        start_iso = meta["start"]
    else:
        raise ValueError(f"Unknown farm: {farm}")

    series = u.KelmarshUpstreamSeries(
        frame, up_frame, splits=splits, input_len=24, horizon=4, stride=3,
        extras=("power_last10", "wind_last10"), start=start_iso, end=f"{last_year + 1}-01-01T01:00:00Z", preparation=meta
    )
    obs_hourly, geo, stats = build_hourly_obs_and_geo(obs_dict, stations, hours, splits["train"])
    return L1FarmData(series, hours, obs_hourly, geo, stats)


def calc_metrics(pred_kw: np.ndarray, truth_kw: np.ndarray) -> dict:
    if pred_kw.shape != truth_kw.shape or not np.isfinite(pred_kw).all():
        raise ValueError("Non-finite or misaligned predictions")
    diff = pred_kw - truth_kw
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.square(diff).mean()))
    denom = float(np.square(truth_kw - truth_kw.mean()).sum())
    r2 = float(1.0 - np.square(diff).sum() / denom) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def fit_arm(farm_data: L1FarmData, source: str | None, seed: int, train_cfg: dict, device_str: str, eval_splits=("validation",)):
    import torch
    from models.TSMixerBasicSmall import Model as TSMixer
    from scripts import train_penmanshiel_p2_v35 as p2
    trainer = p2.trainer

    device = torch.device(device_str)
    trainer.v26run.seed_all(seed)
    tr = farm_data.split_pack("train")
    va = farm_data.split_pack("validation")
    y_tr = ((tr["truth"] - tr["power_mean"]) / tr["power_scale"]).astype(np.float32)

    obs_gpu = torch.as_tensor(farm_data.obs_hourly, dtype=torch.float32, device=device)
    if source is None:
        model = TSMixer(seq_len=24, channels=21, pred_len=4, blocks=2, ff_dim=32, dropout=0.1, norm_eps=1e-3, target_slot=20, input_mode="full").to(device)
    else:
        model = pe.ProgramTSMixerModel(source, farm_data.geo, blocks=2, ff_dim=32, dropout=0.1, norm_eps=1e-3).to(device)
        model.fit_encoder_normalization(_HourWindowObs(obs_gpu, farm_data.fit_hour_windows), batch=512)

    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg["patience"])
    batch_size = int(train_cfg["batch_size"])
    w_mse = float(train_cfg["loss_mse_weight"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

    def predict_pack(pack):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(pack["truth"]), 512):
                if source is None:
                    xb = torch.as_tensor(pack["x21"][i : i + 512], dtype=torch.float32, device=device)
                    yb = model(xb)[..., -1]
                else:
                    xb = torch.as_tensor(pack["x10"][i : i + 512], dtype=torch.float32, device=device)
                    ob = obs_gpu[pack["h_idx"][i : i + 512]]
                    yb = model(xb, ob)[..., -1]
                preds.append(yb.detach().cpu().numpy())
        raw_p = np.concatenate(preds, axis=0)
        return raw_p * pack["power_scale"] + pack["power_mean"]

    val_pred0 = predict_pack(va)
    val_m0 = calc_metrics(val_pred0, va["truth"])
    best_mae, best_epoch, bad = val_m0["mae"], 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "val_mae": best_mae, "val_rmse": val_m0["rmse"]}]

    n_tr = len(y_tr)
    for ep in range(1, max_epochs + 1):
        if bad >= patience:
            break
        model.train()
        g = torch.Generator()
        g.manual_seed(int(seed) + ep)
        perm = torch.randperm(n_tr, generator=g).numpy()
        ep_loss, steps = 0.0, 0
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            yt = torch.as_tensor(y_tr[idx], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            if source is None:
                xb = torch.as_tensor(tr["x21"][idx], dtype=torch.float32, device=device)
                out = model(xb)[..., -1]
            else:
                xb = torch.as_tensor(tr["x10"][idx], dtype=torch.float32, device=device)
                ob = obs_gpu[tr["h_idx"][idx]]
                out = model(xb, ob)[..., -1]
            diff = out - yt
            loss = (1.0 - w_mse) * diff.abs().mean() + w_mse * diff.square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            steps += 1
        sched.step()
        vp = predict_pack(va)
        vm = calc_metrics(vp, va["truth"])
        history.append({"epoch": ep, "train_loss": ep_loss / max(steps, 1), "val_mae": vm["mae"], "val_rmse": vm["rmse"]})
        if vm["mae"] < best_mae:
            best_mae, best_epoch, bad = vm["mae"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

    model.load_state_dict(best_state, strict=True)
    outputs = {}
    for sp in eval_splits:
        pk = va if sp == "validation" else farm_data.split_pack(sp)
        pr = predict_pack(pk)
        outputs[sp] = {"prediction": pr, "metrics": calc_metrics(pr, pk["truth"])}
    learned_p = model.P.detach().cpu().tolist() if source is not None else None
    return {
        "best_epoch": best_epoch,
        "validation_mae": best_mae,
        "history": history,
        "learned_P": learned_p,
        "outputs": outputs,
        "best_state": best_state,
    }

def verify_llm_weights(data_root: Path, protocol: dict) -> dict:
    manifest_path = ROOT / protocol["llm"]["manifest_path"]
    manifest = read_json(manifest_path)
    rev = manifest.get("revision_commit_sha") or manifest.get("revision")
    if rev != protocol["llm"]["revision"]:
        raise ValueError("LLM manifest revision mismatch")
    wsum = manifest.get("weights_summary", manifest)
    weights_dir = data_root / protocol["llm"]["weights_dir"]
    verified = 0
    for item in wsum["files"]:
        rel = item.get("relative_path") or item["path"]
        fpath = weights_dir / rel
        if not fpath.is_file() or fpath.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Missing or size-mismatched weight file: {fpath}")
        if sha(fpath) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch on weight file: {fpath}")
        verified += 1
    return {
        "manifest_sha256": sha(manifest_path),
        "weights_dir": str(weights_dir),
        "revision": rev,
        "verified_files": verified,
        "total_bytes": int(wsum.get("total_size_bytes") or wsum["total_bytes"]),
    }


BLIND_EXAMPLE = """INIT_P = [0.5, -0.5, 0.2, 0.0]

def encode(obs, geo, P):
    x0, x1, x2, x3, x4 = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3], obs[..., 4]
    m0, m1 = obs[..., 5], obs[..., 6]
    g0, g1, g2 = geo[:, 0], geo[:, 1], geo[:, 2]
    w0 = torch.exp(-g0 / (F.softplus(P[0]) + 0.05)) * m0
    w1 = torch.sigmoid(torch.exp(P[1]) * (g1 * wmean(x1, m0, dim=2).unsqueeze(-1) + g2 * wmean(x2, m0, dim=2).unsqueeze(-1) - P[2])) * w0
    c0 = wmean(x0, w0, dim=2)
    c1 = wmean(x0, w1, dim=2)
    c2 = wmean(x0 - lag(x0, 3), w1 * lag(m0, 3), dim=2)
    c3 = wmean(x3, m1, dim=2)
    c4 = wmean(x4, w1 * m1, dim=2)
    c5 = wmean(x0 * lag(x0, 1), m0 * lag(m0, 1), dim=2) + P[3] * c0
    return torch.stack([c0, c1, c2, c3, c4, c5], dim=-1)
"""


def build_llm_prompt(protocol: dict, blind: bool, gen: int, slot: int, history: list[dict]) -> tuple[str, str]:
    sys_msg = protocol["prompts"]["L_blind_system"] if blind else protocol["prompts"]["L_system"]
    valid_hist = [r for r in history if r.get("status") == "COMPLETE" and r.get("val_mae") is not None and math.isfinite(r["val_mae"])]
    valid_hist.sort(key=lambda r: (r["val_mae"], r["program_id"]))
    top_k = valid_hist[:3]

    if blind:
        rules = (
            "Rules: define INIT_P (a list of 1-16 numbers) and def encode(obs, geo, P) returning a tensor [B, 24, K] "
            "with 1 <= K <= 11. obs is [B, 24, S, 7] where channels 0..4 are continuous node signals and channels 5..6 "
            "are binary validity masks (0 or 1); geo is [S, 3] node coordinates; P has len(INIT_P) entries. "
            "Available names only: torch, F (torch.nn.functional), lag(x, k) (shift x along dim=1 by constant integer "
            "k in 0..23), wmean(x, w, dim=...) (weighted mean), safe_div(a, b). No imports, loops, if, lambdas, "
            "comprehensions, nested functions, in-place ops or item assignment. Every sum/mean/amax/amin/softmax/"
            "logsumexp/cumsum/std/var/prod/stack/cat/unsqueeze/squeeze/wmean must pass an explicit constant dim != 0. "
            "Never mix samples across dim=0. Output must be invariant to permutations of dim=2 and work for any S."
        )
        lines = [
            f"Generation {gen}, candidate {slot} (variation style {slot % 3 + 1}).",
            "Design a new differentiable tensor reduction program `encode(obs, geo, P)` that lowers validation MAE.",
            "Explore distinct nonlinear interactions, multi-step lag differences (via `lag(x, k)`), soft gating over `geo`, and masked moments.",
        ]
        if not top_k:
            lines += ["Reference valid program:", "```python", BLIND_EXAMPLE.strip(), "```"]
        else:
            lines.append("Best programs discovered so far (lower validation_mae is better):")
            for rank, item in enumerate(top_k, 1):
                lines += [f"Rank {rank} (val_mae={item['val_mae']:.4f}):", "```python", item["source"].strip(), "```"]
        lines += [rules, "Return ONLY one ```python ... ``` code block containing `INIT_P = [...]` and `def encode(obs, geo, P):`."]
        return sys_msg, "\n\n".join(lines)

    lines = [
        f"Generation {gen}, candidate {slot} (variation style {slot % 3 + 1}).",
        "Task: Design a differentiable upstream meteorological station encoder `encode(obs, geo, P)` for 1-4h ahead wind farm power forecasting.",
        "Physical inputs: `obs` [B, 24, S, 7] = (ws_z, dir_sin, dir_cos, p_anom_z, dp3_z, valid_ws, valid_p) up to issue_time - 1h; "
        "`geo` [S, 3] = (dist_100km, bearing_sin, bearing_cos).",
        "Explore physical mechanisms such as soft upwind cones, anisotropic distance decay, multi-lag advection arrival kernels, "
        "front/ramp acceleration, pressure-gradient dipole alignment, directional shear, or cross-station dispersion.",
    ]
    if not top_k:
        lines += ["Baseline expert program P:", "```python", pe.EXPERT_PROGRAM.strip(), "```"]
    else:
        lines.append("Top-performing valid programs so far (lower validation_mae in kW is better):")
        for rank, item in enumerate(top_k, 1):
            lines += [f"Rank {rank} (val_mae={item['val_mae']:.4f} kW):", "```python", item["source"].strip(), "```"]
    lines += [
        pe.PROGRAM_RULES_TEXT,
        "Produce a NOVEL program distinct from all shown examples. Return ONLY one ```python ... ``` code block.",
    ]
    return sys_msg, "\n\n".join(lines)


def _llm_worker_main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = json.loads(sys.stdin.read())
    wdir = payload["weights_dir"]
    tok = AutoTokenizer.from_pretrained(wdir, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        wdir, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, local_files_only=True
    )
    model.eval()
    results = []
    for req in payload["requests"]:
        seed = int(req["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        msgs = [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["user"]}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(payload["temperature"]),
                top_p=float(payload["top_p"]),
                max_new_tokens=int(payload["max_new_tokens"]),
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
        raw_out = tok.decode(gen_ids, skip_special_tokens=True)
        results.append({"slot": req["slot"], "seed": seed, "raw_output": raw_out})
    print("LLM_WORKER_JSON:" + json.dumps(results))
    return 0


def run_llm_worker_batch(weights_dir: Path, requests: list[dict], temperature: float, top_p: float, max_new_tokens: int, llm_python: str) -> list[dict]:
    """Run Qwen2.5-Coder-7B-Instruct in an isolated child process so GPU VRAM is 100% freed afterward."""
    payload = json.dumps({
        "weights_dir": str(weights_dir),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "requests": requests,
    })
    r = subprocess.run(
        [llm_python, str(Path(__file__).resolve()), "--llm-worker"],
        input=payload, capture_output=True, text=True, timeout=900
    )
    if r.returncode != 0:
        raise RuntimeError(f"LLM worker failed (rc={r.returncode}): {r.stderr[-1500:]}")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("LLM_WORKER_JSON:")]
    if not lines:
        raise RuntimeError(f"LLM worker produced no JSON output: {r.stdout[-500:]}")
    return json.loads(lines[-1][len("LLM_WORKER_JSON:"):])
def evaluate_candidate(
    program_id: str,
    arm: str,
    slot_index: int,
    raw_text: str,
    seen_hashes: set[str],
    farm_k: L1FarmData,
    train_cfg: dict,
    device: str,
    seed: int = 2024,
    prompt_sys: str | None = None,
    prompt_user: str | None = None,
    spec_meta: dict | None = None,
) -> dict:
    t0 = time.time()
    rec = {
        "program_id": program_id,
        "arm": arm,
        "slot_index": int(slot_index),
        "spec": spec_meta,
        "prompt_system": prompt_sys,
        "prompt_user": prompt_user,
        "raw_output": raw_text,
        "source": None,
        "ast_sha256": None,
        "verify": None,
        "status": "FAILED",
        "failure_reason": None,
        "val_mae": None,
        "best_epoch": None,
        "learned_P": None,
        "elapsed_sec": 0.0,
    }
    try:
        src = pe.extract_code(raw_text)
        rec["source"] = src
    except Exception as exc:
        rec["failure_reason"] = f"extract_error: {exc}"
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec

    ver = pe.verify_in_subprocess(src, timeout_sec=60, memory_mb=8192, tol=1e-5, seed=0)
    rec["verify"] = ver
    if ver.get("status") != "OK":
        rec["failure_reason"] = f"verify_{ver.get('status')}: {ver.get('reason')}"
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec

    ast_hash = ver["hash"]
    rec["ast_sha256"] = ast_hash
    if ast_hash in seen_hashes:
        rec["failure_reason"] = "duplicate_ast_sha256"
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec
    seen_hashes.add(ast_hash)

    try:
        fit = fit_arm(farm_k, source=src, seed=seed, train_cfg=train_cfg, device_str=device, eval_splits=("validation",))
        vmae = float(fit["validation_mae"])
        if not math.isfinite(vmae):
            rec["failure_reason"] = "non_finite_val_mae"
        else:
            rec["status"] = "COMPLETE"
            rec["val_mae"] = vmae
            rec["best_epoch"] = int(fit["best_epoch"])
            rec["learned_P"] = fit["learned_P"]
    except Exception as exc:
        rec["failure_reason"] = f"train_error: {type(exc).__name__}: {exc}"
    rec["elapsed_sec"] = round(time.time() - t0, 3)
    return rec


def select_best_record(records: list[dict], fallback_source: str) -> dict:
    valid = [r for r in records if r.get("status") == "COMPLETE" and r.get("val_mae") is not None and math.isfinite(r["val_mae"])]
    valid.sort(key=lambda r: (float(r["val_mae"]), int(r["slot_index"])))
    if valid:
        best = valid[0]
        return {
            "program_id": best["program_id"],
            "slot_index": int(best["slot_index"]),
            "val_mae": float(best["val_mae"]),
            "ast_sha256": best["ast_sha256"],
            "source": best["source"],
            "n_channels": int(best["verify"]["K"]),
            "n_params": int(best["verify"]["n_params"]),
            "all_failed_fallback": False,
        }
    info = pe.validate_source(fallback_source)
    return {
        "program_id": "FALLBACK_EXPERT_P",
        "slot_index": 999,
        "val_mae": float("inf"),
        "ast_sha256": info["hash"],
        "source": fallback_source,
        "n_channels": 11,
        "n_params": int(info["n_params"]),
        "all_failed_fallback": True,
    }


def decide_l1(summary_by_farm: dict) -> str:
    """Apply Stage M4 pre-registered decision rule across Kelmarsh and Penmanshiel."""
    both_beat_h = True
    both_beat_rp = True
    for fk in ("Kelmarsh", "Penmanshiel"):
        fsum = summary_by_farm[fk]
        l_mae = float(fsum["arms"]["L_star"]["mean_mae"])
        h_mae = float(fsum["arms"]["H"]["mean_mae"])
        p_mae = float(fsum["arms"]["P"]["mean_mae"])
        r_mae = float(fsum["arms"]["R_star"]["mean_mae"])
        gain_vs_h_pct = (h_mae - l_mae) / h_mae * 100.0
        gain_vs_p_pct = (p_mae - l_mae) / p_mae * 100.0
        gain_vs_r_pct = (r_mae - l_mae) / r_mae * 100.0
        fsum["relative_gains_pct"] = {
            "L_star_vs_H": gain_vs_h_pct,
            "L_star_vs_P": gain_vs_p_pct,
            "L_star_vs_R_star": gain_vs_r_pct,
            "L_star_vs_L_blind_star": (float(fsum["arms"]["L_blind_star"]["mean_mae"]) - l_mae) / float(fsum["arms"]["L_blind_star"]["mean_mae"]) * 100.0,
        }
        if gain_vs_h_pct < 1.5 - 1e-12:
            both_beat_h = False
        if gain_vs_p_pct < 0.5 - 1e-12 or gain_vs_r_pct < 0.5 - 1e-12:
            both_beat_rp = False
    if both_beat_h and both_beat_rp:
        return "CONTINUE"
    if both_beat_h:
        return "SEARCH_ONLY_STOP_L1"
    return "STOP_L1"


def run_search_phase(
    farm_k: L1FarmData,
    protocol: dict,
    out: Path,
    device: str,
    llm_python: str,
    weights_dir: Path,
    generator_fn=None,
) -> dict:
    s_cfg = protocol["search_phase"]
    llm_cfg = protocol["llm"]
    g_cfg = protocol["grammar_R"]
    seed = int(s_cfg["seed"])
    gen_fn = generator_fn or run_llm_worker_batch

    h_fit = fit_arm(farm_k, source=None, seed=seed, train_cfg=s_cfg, device_str=device, eval_splits=("validation",))
    p_rec = evaluate_candidate("P_expert", "P", 0, pe.EXPERT_PROGRAM, set(), farm_k, s_cfg, device, seed=seed)

    r_items = pe.sample_grammar_programs(g_cfg, seed=int(g_cfg["seed"]), budget=int(g_cfg["budget"]))
    r_seen: set[str] = set()
    r_records = []
    for idx, item in enumerate(r_items, 1):
        rec = evaluate_candidate(
            f"R_{idx:02d}", "R", idx, item["source"], r_seen, farm_k, s_cfg, device,
            seed=seed, spec_meta=item["spec"]
        )
        r_records.append(rec)
    save(out / "search_R_candidates.json", r_records)

    arm_records = {"L": [], "L_blind": []}
    for arm_key, blind in (("L", False), ("L_blind", True)):
        seen: set[str] = set()
        slot_counter = 0
        base_seed = int(llm_cfg["base_seed_L_blind"] if blind else llm_cfg["base_seed_L"])
        for g in range(1, int(llm_cfg["generations"]) + 1):
            reqs = []
            prompts_meta = {}
            for s in range(1, int(llm_cfg["population_per_generation"]) + 1):
                slot_counter += 1
                sys_m, usr_m = build_llm_prompt(protocol, blind=blind, gen=g, slot=s, history=arm_records[arm_key])
                c_seed = base_seed + g * 100 + s
                reqs.append({"slot": slot_counter, "seed": c_seed, "system": sys_m, "user": usr_m})
                prompts_meta[slot_counter] = (g, s, sys_m, usr_m, c_seed)
            batch_out = gen_fn(
                weights_dir, reqs,
                float(llm_cfg["temperature"]), float(llm_cfg["top_p"]),
                int(llm_cfg["max_new_tokens"]), llm_python
            )
            by_slot = {int(item["slot"]): item["raw_output"] for item in batch_out}
            for slot_idx in sorted(prompts_meta):
                g, s, sys_m, usr_m, _ = prompts_meta[slot_idx]
                raw_txt = by_slot.get(slot_idx, "")
                pid = f"{arm_key}_g{g}_s{s}"
                rec = evaluate_candidate(
                    pid, arm_key, slot_idx, raw_txt, seen, farm_k, s_cfg, device,
                    seed=seed, prompt_sys=sys_m, prompt_user=usr_m,
                )
                arm_records[arm_key].append(rec)
            save(out / f"search_{arm_key}_candidates.json", arm_records[arm_key])

    r_star = select_best_record(r_records, pe.EXPERT_PROGRAM)
    l_star = select_best_record(arm_records["L"], pe.EXPERT_PROGRAM)
    lb_star = select_best_record(arm_records["L_blind"], pe.EXPERT_PROGRAM)

    all_valid = [r for r in ([p_rec] + r_records + arm_records["L"] + arm_records["L_blind"]) if r.get("status") == "COMPLETE" and r.get("val_mae") is not None]
    all_valid.sort(key=lambda r: (float(r["val_mae"]), r["program_id"]))
    top10 = [
        {
            "rank": i + 1,
            "program_id": r["program_id"],
            "arm": r["arm"],
            "val_mae": float(r["val_mae"]),
            "ast_sha256": r["ast_sha256"],
            "n_channels": int(r["verify"]["K"]),
            "n_params": int(r["verify"]["n_params"]),
            "source": r["source"],
        }
        for i, r in enumerate(all_valid[:10])
    ]
    save(out / "top10_programs.json", top10)

    search_summary = {
        "H_short_val_mae": float(h_fit["validation_mae"]),
        "P_short_val_mae": float(p_rec["val_mae"]),
        "R_valid_count": sum(1 for r in r_records if r["status"] == "COMPLETE"),
        "L_valid_count": sum(1 for r in arm_records["L"] if r["status"] == "COMPLETE"),
        "L_blind_valid_count": sum(1 for r in arm_records["L_blind"] if r["status"] == "COMPLETE"),
        "R_star": r_star,
        "L_star": l_star,
        "L_blind_star": lb_star,
    }
    save(out / "search_selection_freeze.json", {"frozen_at_utc": now_utc(), **search_summary})
    return search_summary


def run_formal_phase(farms: dict[str, L1FarmData], protocol: dict, out: Path, search_summary: dict, device: str) -> dict:
    f_cfg = protocol["formal_phase"]
    seeds = [int(s) for s in f_cfg["seeds"]]
    arm_sources = {
        "H": None,
        "P": pe.EXPERT_PROGRAM,
        "R_star": search_summary["R_star"]["source"],
        "L_star": search_summary["L_star"]["source"],
        "L_blind_star": search_summary["L_blind_star"]["source"],
    }

    trained_runs = {fk: {ak: [] for ak in arm_sources} for fk in ("Kelmarsh", "Penmanshiel")}
    for fk in ("Kelmarsh", "Penmanshiel"):
        fdata = farms[fk]
        for ak, src in arm_sources.items():
            for sd in seeds:
                res = fit_arm(fdata, source=src, seed=sd, train_cfg=f_cfg, device_str=device, eval_splits=("validation", "eval"))
                trained_runs[fk][ak].append({"seed": sd, **{k: v for k, v in res.items() if k != "best_state"}})

    save(out / "formal_selection_freeze.json", {
        "frozen_at_utc": now_utc(),
        "total_fits": sum(len(trained_runs[fk][ak]) for fk in trained_runs for ak in trained_runs[fk]),
        "validation_summary": {
            fk: {ak: [r["validation_mae"] for r in trained_runs[fk][ak]] for ak in arm_sources}
            for fk in ("Kelmarsh", "Penmanshiel")
        },
    })
    save(out / "test_evaluation_started.json", {"started_at_utc": now_utc()})

    summary_by_farm = {}
    for fk in ("Kelmarsh", "Penmanshiel"):
        eval_pack = farms[fk].split_pack("eval")
        truth = eval_pack["truth"]
        issues = eval_pack["issue"]
        arm_out = {}
        mean_preds = {}
        for ak in arm_sources:
            runs = trained_runs[fk][ak]
            preds = np.stack([r["outputs"]["eval"]["prediction"] for r in runs], axis=0)
            mean_preds[ak] = np.mean(preds, axis=0)
            seed_metrics = [r["outputs"]["eval"]["metrics"] for r in runs]
            arm_out[ak] = {
                "mean_mae": float(np.mean([m["mae"] for m in seed_metrics])),
                "std_mae": float(np.std([m["mae"] for m in seed_metrics])),
                "mean_rmse": float(np.mean([m["rmse"] for m in seed_metrics])),
                "mean_r2": float(np.mean([m["r2"] for m in seed_metrics])),
                "mean_val_mae": float(np.mean([r["validation_mae"] for r in runs])),
                "seeds": [{"seed": r["seed"], "best_epoch": r["best_epoch"], "val_mae": r["validation_mae"], **r["outputs"]["eval"]["metrics"], "learned_P": r["learned_P"]} for r in runs],
            }
        comparisons = {}
        cand_err = np.abs(mean_preds["L_star"] - truth).mean(axis=1)
        for ref in ("H", "P", "R_star", "L_blind_star"):
            ref_err = np.abs(mean_preds[ref] - truth).mean(axis=1)
            diff_df = pd.DataFrame({"issue": issues, "gain": ref_err - cand_err}).groupby("issue", sort=True)["gain"].mean()
            series = diff_df.to_numpy(dtype=np.float64)
            nw = pfs.newey_west_mean_test(series, max_lag=12)
            bt = pfs.moving_block_bootstrap_mean(series, block_length=12, replicates=2000, seed=2024)
            comparisons[f"L_star_vs_{ref}"] = {"newey_west": nw, "bootstrap": bt}
        summary_by_farm[fk] = {"arms": arm_out, "comparisons": comparisons}

    decision = decide_l1(summary_by_farm)
    return {"farms": summary_by_farm, "decision": decision}


class _SynthWindows:
    def __init__(self, starts, truth, anchor, times, turbines):
        self.window_starts = starts
        self._truth = truth
        self._anchor = anchor
        self._times = times
        self.turbines = turbines

    def truth_and_anchor(self):
        return self._truth, self._anchor

    def target_times(self):
        return self._times


class _SynthSeries:
    def __init__(self, values, dates, splits, win_map):
        self.values = values
        self.dates = dates
        self.splits = splits
        self._win_map = win_map
        self.scaler_mean = np.zeros(21, dtype=np.float32)
        self.scaler_scale = np.ones(21, dtype=np.float32) * 500.0

    def windows(self, split_name):
        return self._win_map[split_name]


def make_synthetic_farm(seed: int = 0, n_stations: int = 6) -> L1FarmData:
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2020-01-01", periods=240, freq="1h")
    values = rng.normal(size=(240, 21)).astype(np.float32)
    obs = rng.normal(size=(240, n_stations, 7)).astype(np.float32)
    obs[..., 5:] = (rng.uniform(size=(240, n_stations, 2)) > 0.1).astype(np.float32)
    geo = np.column_stack([
        rng.uniform(0.1, 1.2, size=n_stations),
        np.sin(np.linspace(0, 5, n_stations)),
        np.cos(np.linspace(0, 5, n_stations)),
    ]).astype(np.float32)
    turbines = np.array(["T1"] * 240)
    win_map = {}
    for sp, s0, s1 in (("train", 0, 60), ("validation", 80, 120), ("eval", 140, 180)):
        starts = np.arange(s0, s1, 2, dtype=np.int64)
        truth = rng.uniform(100.0, 900.0, size=(len(starts), 4)).astype(np.float32)
        anchor = truth[:, :1].copy()
        times = np.column_stack([hours.to_numpy()[starts + k] for k in range(1, 5)])
        win_map[sp] = _SynthWindows(starts, truth, anchor, times, turbines)
    series = _SynthSeries(values, hours, {"train": [2020], "validation": [2020], "eval": [2020]}, win_map)
    return L1FarmData(series, hours, obs, geo, {"ws_mean": 5.0})


def smoke(out: Path) -> int:
    protocol = read_json(PROTOCOL)
    sp = copy.deepcopy(protocol)
    sp["grammar_R"].update({"budget": 2})
    sp["llm"].update({"generations": 1, "population_per_generation": 2})
    sp["search_phase"].update({"max_epochs": 1, "patience": 1, "batch_size": 16})
    sp["formal_phase"].update({"seeds": [2024], "max_epochs": 1, "patience": 1, "batch_size": 16})
    fk = make_synthetic_farm(seed=1, n_stations=6)
    fp = make_synthetic_farm(seed=2, n_stations=4)
    fake_gen = lambda wdir, reqs, temp, tp, mtok, py: [
        {"slot": r["slot"], "seed": r["seed"], "raw_output": f"```python\n{pe.EXPERT_PROGRAM}\n```"} for r in reqs
    ]
    s_sum = run_search_phase(fk, sp, out, "cpu", sys.executable, ROOT, generator_fn=fake_gen)
    f_sum = run_formal_phase({"Kelmarsh": fk, "Penmanshiel": fp}, sp, out, s_sum, "cpu")
    save(out / "smoke_summary.json", {"search": s_sum, "formal": f_sum})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "output/l1_llm_encoder")
    ap.add_argument("--llm-python", type=str, default="/home/XT/miniconda3/envs/llm_l1/bin/python")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--llm-worker", action="store_true")
    args = ap.parse_args(argv)
    if args.llm_worker:
        return _llm_worker_main()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        return smoke(args.out)

    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    protocol = read_json(PROTOCOL)
    w_info = verify_llm_weights(args.data_root.resolve(), protocol)
    identities: dict[str, str] = {}
    f_splits = protocol["formal_phase"]["farms"]
    farms = {
        "Kelmarsh": load_real_farm(args.data_root.resolve(), "Kelmarsh", f_splits["Kelmarsh"], identities),
        "Penmanshiel": load_real_farm(args.data_root.resolve(), "Penmanshiel", f_splits["Penmanshiel"], identities),
    }
    code_hashes = {
        "protocol_sha256": sha(PROTOCOL),
        "script_sha256": sha(Path(__file__).resolve()),
        "program_encoder_sha256": sha(ROOT / "models/ProgramEncoder.py"),
    }
    save(args.out / "audit/launch.json", {"started_at_utc": now_utc(), "device": dev, "llm_weights": w_info, "code_hashes": code_hashes})
    s_sum = run_search_phase(farms["Kelmarsh"], protocol, args.out, dev, args.llm_python, Path(w_info["weights_dir"]))
    f_sum = run_formal_phase(farms, protocol, args.out, s_sum, dev)
    full = {
        "stage": "M4",
        "experiment": "L1",
        "completed_at_utc": now_utc(),
        "device": dev,
        "code_hashes": code_hashes,
        "llm_weights": w_info,
        "input_hashes": identities,
        "search": s_sum,
        "formal": f_sum["farms"],
        "decision": f_sum["decision"],
    }
    save(args.out / "l1_summary.json", full)
    save(args.out / "audit/completed.json", {"completed_at_utc": full["completed_at_utc"], "decision": full["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
