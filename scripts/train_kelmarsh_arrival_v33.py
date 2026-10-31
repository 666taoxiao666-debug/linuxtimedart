#!/usr/bin/env python3
"""v33 stage K1 (= A1 on development fold D1): frozen TSMixer base B0 + residual branches B1, B2, A and the fixed-kappa
grid {0.5, 1, 2, 3, 4} (K1 doubles as A-kappa1). Proposal sections 14.13 / 14.17 / 14.18; run config
configs/kelmarsh_arrival_v33_k1.json. Development evidence only: every Kelmarsh year is already exposed.

  python scripts/train_kelmarsh_arrival_v33.py --smoke --out /tmp/v33_smoke          # synthetic, minutes, torch only
  python scripts/train_kelmarsh_arrival_v33.py --out outputs/logs/Kelmarsh/v33_k1_<tag>  # real K1 (Codex, server)

Outputs (run directory): data_manifest.json, B0/ (v28 fit_config artifacts), arms/<arm>/{trained.json, best.pt,
predictions.npz, history.json}, k1_decision.json, k1_summary.md. A completed arm is reused only when its contract
(config + code + data identity) is identical; nothing is re-run with another configuration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from models.UpstreamArrivalResidual import ArrivalResidual, trainable_parameters  # noqa: E402
from scripts import train_kelmarsh_anchored_residual as v26run  # noqa: E402
from scripts import train_kelmarsh_multiyear_benchmark as v28  # noqa: E402
from utils import kelmarsh_station_history_v33 as hist  # noqa: E402
from utils import kelmarsh_upstream_v31 as u31  # noqa: E402
from utils.metrics import forecast_metrics  # noqa: E402

CONFIG = ROOT / "configs/kelmarsh_arrival_v33_k1.json"
CODE_FILES = ("scripts/train_kelmarsh_arrival_v33.py", "models/UpstreamArrivalResidual.py",
              "utils/kelmarsh_station_history_v33.py", "configs/kelmarsh_arrival_v33_k1.json")
STATION_ARMS = ("B2", "A", "K0.5", "K1", "K2", "K3", "K4")


def sha_text(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_digest():
    return {name: sha_text(ROOT / name) for name in CODE_FILES}


def now():
    return pd.Timestamp.utcnow().isoformat()


# --------------------------------------------------------------------------- data
def load_real(cfg):
    p = json.loads((ROOT / cfg["data_source_protocol"]).read_text(encoding="utf-8"))
    d, up = p["data"], p["data"]["upstream"]
    csv = ROOT / d["v32_preparation"]["csv"]
    if u31.file_sha256(csv) != cfg["prepared_csv_sha256"]:
        raise ValueError("Prepared CSV differs from the registered v32 preparation")
    manifest_path = ROOT / up["download_manifest"]
    if u31.text_sha256_lf(manifest_path) != up["download_manifest_sha256_lf"]:
        raise ValueError("Registered MIDAS download manifest changed")
    hashes = u31.manifest_hashes(u31.read_json(manifest_path))
    stations = [dict(s) for s in up["stations"]]
    ids = [int(s["src_id"]) for s in stations]
    years = cfg["station_years"]
    if max(years) > 2022:
        raise ValueError("K1 never opens 2023/2024 station files")
    root = next(r["root"] for r in up["development_roots"] if set(years) <= set(r["years"]))
    files = []
    for year in years:
        year_files, _ = u31.station_files(ROOT / root, ids, [year], hashes, required=True)
        files.extend({**entry, "year": int(year)} for entry in year_files)
    observations = u31.load_station_observations(files, ids)
    fold = cfg["fold"]
    splits = {"train": fold["train"], "validation": fold["validation"], "test": fold["development_eval"]}
    start = pd.Timestamp(d["start_utc"]).tz_convert("UTC").tz_localize(None) - pd.Timedelta(hours=d["input_len"])
    labels = pd.date_range(start, pd.Timestamp(f"{max(years) + 1}-01-01 01:00"), freq="h", inclusive="left")
    upstream = u31.hourly_upstream(observations, stations, labels, cfg["lag_hours"])
    series = u31.KelmarshUpstreamSeries(csv, upstream, splits=splits, input_len=d["input_len"], horizon=d["pred_len"],
                                        stride=d["stride"], extras=tuple(d["extras_in_common_series"]), start=d["start_utc"],
                                        end=f"{max(fold['development_eval']) + 1}-01-01T01:00:00Z")
    identity = {"prepared_csv_sha256": cfg["prepared_csv_sha256"], "manifest_sha256_lf": up["download_manifest_sha256_lf"],
                "station_files": {f"{e['src_id']}/{e['path'].name}": e["sha256"] for e in files}, "stations": ids}
    return p, series, observations, stations, identity


def load_smoke(cfg):
    """Synthetic SCADA (v28/v31 generators) + synthetic stations; NEVER real data."""
    from scripts import train_kelmarsh_upstream_v31 as v31
    p = json.loads((ROOT / cfg["data_source_protocol"]).read_text(encoding="utf-8"))
    splits = {"train": [2016], "validation": [2017], "test": [2018]}
    series = v31.synthetic_series({"splits": splits, "input_len": 24, "horizon": 4, "stride": 3,
                                   "extras": ("power_last10", "wind_last10")}, seed=0)
    rng = np.random.default_rng(1)
    hours = pd.date_range(pd.DatetimeIndex(series.dates).min() - pd.Timedelta(hours=40),
                          pd.DatetimeIndex(series.dates).max(), freq="h")
    stations, observations = [], {}
    for i in range(12):
        sid = 900000 + i
        stations.append({"src_id": sid, "distance_km": float(rng.uniform(20, 150)), "bearing_deg": float(rng.uniform(0, 360))})
        ws = np.clip(6 + np.cumsum(rng.normal(0, .3, len(hours))) * .1 + rng.normal(0, 1, len(hours)), 0, None)
        dirn = (240 + np.cumsum(rng.normal(0, 5, len(hours)))) % 360
        pres = 1010 + np.cumsum(rng.normal(0, .2, len(hours)))
        frame = pd.DataFrame({"wind_speed": ws, "wind_direction": dirn, "msl_pressure": pres}, index=hours)
        frame.iloc[rng.random(len(hours)) < .05] = np.nan
        observations[sid] = frame
    return p, series, observations, stations, {"synthetic": True}


# --------------------------------------------------------------------------- batches
class Split:
    """Windows of one split (ALL cohort) with their station-token rows; batches gathered vectorised."""

    def __init__(self, series, split, bank, lag, channel_indices):
        self.windows = v28.ChannelView(series.windows(split, "all"), channel_indices)
        starts = self.windows.window_starts
        issue = pd.DatetimeIndex(series.dates[starts + series.input_len - 1])
        self.rows, self.ref_rows, self.ages = hist.token_rows(bank, issue, lag)
        self.bank, self.truth = bank, self.windows.truth_and_anchor()[0]
        self.scale, self.mean = float(self.windows.scaler.scale_[-1]), float(self.windows.scaler.mean_[-1])

    def __len__(self):
        return len(self.windows)

    def batch(self, index, device):
        xs, ys = zip(*[self.windows[i][:2] for i in index])
        content, phys, valid = hist.gather(self.bank, self.rows[index], self.ref_rows[index], self.ages)
        to = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        return to(np.stack(xs)), to(np.stack(ys)), to(content), to(phys), to(valid, torch.bool)

    def batches(self, size, device, seed=None):
        order = np.arange(len(self))
        if seed is not None:
            order = torch.randperm(len(self), generator=torch.Generator().manual_seed(int(seed))).numpy()
        for i in range(0, len(order), size):
            yield self.batch(order[i:i + size], device)


@torch.no_grad()
def predict(model, split, device, batch=512):
    model.eval()
    out, support = [], []
    for x, _, content, phys, valid in split.batches(batch, device):
        y, diag = model(x, content, phys, valid, return_diagnostics=True)
        out.append(y[..., -1].cpu().numpy())
        if diag["support"] is not None:
            support.append(diag["support"].float().cpu().numpy())
    pred = np.concatenate(out).astype(float) * split.scale + split.mean
    return pred, (np.concatenate(support) if support else None)


def scores(pred, truth, rated):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape or not pred.size or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("K1 requires finite, aligned predictions and labels on the complete ALL cohort")
    m = forecast_metrics(pred, truth, rated_power=rated)
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m.get("r2", np.nan))}


# --------------------------------------------------------------------------- branch training
def build_branch(arm, base, cfg):
    phys, br = cfg["physics"], cfg["branch"]
    common = dict(width=br["width"], query_width=br["query_width"], lead_width=br["lead_embedding"],
                  score_clamp=br["score_clamp"], kappa_bounds=tuple(phys["kappa_bounds"]), kappa_init=phys["kappa_init"],
                  min_speed=phys["min_speed_ms"], speed_floor=phys["speed_floor_ms"], cross_scale=phys["crosswind_scale_km"],
                  sigma_min=phys["sigma_min_h"], sigma_frac=phys["sigma_fraction"], support_eps=phys["support_epsilon"],
                  content_dim=len(hist.CONTENT_FEATURES))
    spec = cfg["arms"][arm]
    if arm == "B1":
        return ArrivalResidual(base, tokens="base_rows", prior="none", **common)
    if spec["prior"].startswith("none"):
        return ArrivalResidual(base, tokens="stations", prior="none", **common)
    return ArrivalResidual(base, tokens="stations", prior="arrival", kappa=spec["kappa"], **common)


def fit_branch(arm, base, cfg, train, val, ev, ref_val, device, out, max_epochs=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contract = {"arm": arm, "spec": cfg["arms"][arm], "seed": cfg["seed"], "code": code_digest(),
                "base_state_sha256": v26run.state_digest(v26run.cpu_state(base))}
    if (out / "trained.json").exists():
        old = json.loads((out / "trained.json").read_text())
        if old["contract"] != contract:
            raise ValueError(f"{arm}: registered arm has a different contract; refuse to overwrite")
        return old
    t = cfg["training"]
    v26run.seed_all(cfg["seed"])
    model = build_branch(arm, base, cfg).to(device)
    count = trainable_parameters(model)
    if count > cfg["branch"]["max_parameters"]:
        raise ValueError(f"{arm}: {count} trainable parameters exceed the cap")
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["scheduler_t_max"])
    rated = 2050.0

    def assess():
        pred, _ = predict(model, val, device, t["eval_batch_size"])
        s = scores(pred, val.truth, rated)
        return hist.j_score(s["mae"], s["rmse"], ref_val["mae"], ref_val["rmse"]), s

    j, s = assess()
    best, best_epoch, bad = j, 0, 0
    best_state = v26run.cpu_state(model)
    history = [{"epoch": 0, "J": j, **s}]
    w = float(t["loss_mse_weight"])
    for epoch in range(1, int(max_epochs or t["max_epochs"]) + 1):
        if bad >= int(t["patience"]):
            break
        model.train()
        total, n, begin = 0.0, 0, time.monotonic()
        for x, y, content, phys, valid in train.batches(t["batch_size"], device, seed=cfg["seed"] + epoch):
            opt.zero_grad(set_to_none=True)
            diff = model(x, content, phys, valid)[..., -1] - y[..., -1]
            loss = (1 - w) * diff.abs().mean() + w * diff.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite branch loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip_norm"])
            opt.step()
            total, n = total + loss.item(), n + 1
        sched.step()
        j, s = assess()
        kappa = model.kappa()
        history.append({"epoch": epoch, "train_loss": total / n, "J": j, **s, "seconds": time.monotonic() - begin,
                        "kappa": None if kappa is None else float(kappa)})
        print(f"[EPOCH] {arm} epoch={epoch} J={j:.5f} val_MAE={s['mae']:.3f} val_RMSE={s['rmse']:.3f}", flush=True)
        if j < best:
            best, best_epoch, bad, best_state = j, epoch, 0, v26run.cpu_state(model)
        else:
            bad += 1
    model.load_state_dict(best_state)
    pv, sup_v = predict(model, val, device, t["eval_batch_size"])
    pe, sup_e = predict(model, ev, device, t["eval_batch_size"])
    kappa = model.kappa()
    v26run.atomic_save({"model": best_state, "epoch": best_epoch, "contract": contract}, out / "best.pt")
    np.savez_compressed(out / "predictions.npz", validation=pv, development_eval=pe,
                        validation_starts=val.windows.window_starts, eval_starts=ev.windows.window_starts)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    trained = {"contract": contract, "arm": arm, "best_epoch": best_epoch, "validation_J": best,
               "validation": scores(pv, val.truth, rated), "development_eval": scores(pe, ev.truth, rated),
               "trainable_parameters": count, "kappa": None if kappa is None else float(kappa),
               "support_rate_eval_by_h": None if sup_e is None else sup_e.mean(axis=0).tolist(),
               "completed_utc": now()}
    (out / "trained.json").write_text(json.dumps(trained, indent=2))
    return trained


# --------------------------------------------------------------------------- run
def run(out, smoke=False, device=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    determinism = v26run.configure_determinism()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    p, series, observations, stations, identity = (load_smoke if smoke else load_real)(cfg)
    lag = int(cfg["lag_hours"])
    issue_all = pd.DatetimeIndex(series.dates)
    hours = pd.date_range(issue_all.min() - pd.Timedelta(hours=lag + 24 + 4), issue_all.max(), freq="h")
    last_train = max(series.splits["train"])
    bank = hist.build_bank(observations, stations, hours, hours.year <= last_train,
                           radius_km=cfg["reference_direction"]["radius_km"])
    channels = list(range(len(series.channels)))
    train, val, ev = (Split(series, s, bank, lag, channels) for s in ("train", "validation", "test"))
    manifest = {"identity": identity, "bank_stats": bank.stats, "windows": {"train": len(train), "validation": len(val),
                "development_eval": len(ev)}, "code": code_digest(), "smoke": smoke, "utc": now(),
                "determinism": determinism}
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_epochs = 1 if smoke else None

    # B0: unchanged v28 fit_config, balanced loss, 21 channels, validation MAE selection
    b = cfg["base"]
    q = copy.deepcopy(p)
    q["training"] = {**p["training"], "loss_mae_weight": b["loss_mae_weight"], "loss_mse_weight": b["loss_mse_weight"]}
    if smoke:
        q["training"]["max_epochs"] = 1
    base_cfg = {"arm": "B0", "family": "TSMixerBasicSmall", "channel_set": "upstream", "channel_indices": channels,
                "channels": len(channels), "train_cohort": "all", "selection_cohort": "all",
                "expected_parameters": p["families"]["TSMixerBasicSmall"]["parameters"]["upstream"],
                "args": dict(p["families"]["TSMixerBasicSmall"]["args"])}
    contract = {"run": cfg["run_id"], "arm": "B0", "config": base_cfg, "seed": cfg["seed"], "training": q["training"],
                "fit_window_hash": u31.window_hash(train.windows.window_starts),
                "selection_window_hash": u31.window_hash(val.windows.window_starts), "code": code_digest()}
    v28.fit_config(base_cfg, q, fit=train.windows, selection=val.windows, contract=contract, seed=cfg["seed"],
                   device=device, out=out / "B0")
    base, _ = v28.build_model(base_cfg)
    base.load_state_dict(torch.load(out / "B0" / "best.pt", map_location="cpu", weights_only=False)["model"])
    base.to(device).eval()
    rated = 2050.0
    b0 = {}
    for name, split in (("validation", val), ("development_eval", ev)):
        pred = v26run.predict(base, split.windows, cfg["seed"], device, 512)
        b0[name] = scores(pred, split.truth, rated)
    persistence = {name: scores(split.windows.truth_and_anchor()[1], split.truth, rated)
                   for name, split in (("validation", val), ("development_eval", ev))}

    arms = {}
    for arm in cfg["budget"]["planned"][1:]:
        arms[arm] = fit_branch(arm, base, cfg, train, val, ev, b0["validation"], device, out / "arms" / arm, max_epochs)
    grid = {float(cfg["arms"][f"K{g:g}"]["kappa"]): arms[f"K{g:g}"]["validation_J"] for g in cfg["kappa_star"]["grid"]}
    k_star = hist.kappa_star(grid, cfg["kappa_star"]["tie_tolerance"])
    eval_metrics = {"B0": b0["development_eval"], **{a: arms[a]["development_eval"] for a in arms}}
    decision = hist.a1_decision(eval_metrics, cfg["decision"])
    a = arms["A"]
    report = {"run_id": cfg["run_id"], "smoke": smoke, "decision": decision, "kappa_star": k_star,
              "kappa_grid_validation_J": grid, "learned_kappa": a["kappa"],
              "A_beats_fixed_kappa": {"vs_K1": [a["development_eval"]["mae"] < arms["K1"]["development_eval"]["mae"],
                                                a["development_eval"]["rmse"] < arms["K1"]["development_eval"]["rmse"]],
                                      "vs_kappa_star": [a["development_eval"]["mae"] < arms[f"K{k_star:g}"]["development_eval"]["mae"],
                                                        a["development_eval"]["rmse"] < arms[f"K{k_star:g}"]["development_eval"]["rmse"]]},
              "metrics": {"B0": b0, "Persistence": persistence,
                          **{k: {"validation": v["validation"], "development_eval": v["development_eval"]} for k, v in arms.items()}},
              "support_rate_eval_by_h": {k: v["support_rate_eval_by_h"] for k, v in arms.items()},
              "trainable_parameters": {k: v["trainable_parameters"] for k, v in arms.items()},
              "scope": "development evidence on fold D1 (fit 2016-2020, validation 2021, development eval 2022); "
                       "point-estimate screen only, not a significance test", "utc": now()}
    (out / "k1_decision.json").write_text(json.dumps(report, indent=2))
    lines = [f"# v33 K1 ({'SMOKE' if smoke else 'real'})", "", f"Decision: **{decision['branch']}**", "",
             "| arm | eval MAE | eval RMSE | val J | kappa |", "|---|---:|---:|---:|---:|",
             f"| B0 | {b0['development_eval']['mae']:.2f} | {b0['development_eval']['rmse']:.2f} | 2.0000 | |",
             f"| Persistence | {persistence['development_eval']['mae']:.2f} | {persistence['development_eval']['rmse']:.2f} | | |"]
    for k, v in arms.items():
        lines.append(f"| {k} | {v['development_eval']['mae']:.2f} | {v['development_eval']['rmse']:.2f} | "
                     f"{v['validation_J']:.4f} | {'' if v['kappa'] is None else round(v['kappa'], 3)} |")
    lines += ["", f"kappa* (validation J) = {k_star}", "", "Development evidence only; no significance claim."]
    (out / "k1_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "kappa_star": k_star, "learned_kappa": a["kappa"]}, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    run(args.out, smoke=args.smoke, device=args.device)


if __name__ == "__main__":
    main()
