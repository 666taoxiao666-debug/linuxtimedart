#!/usr/bin/env python3
"""v33 stage K1b: END-TO-END retest after K1 (pre-registered here, before any K1b run).

Why: K1 (frozen-base residual, run v33_k1_20261031T060631Z) ended in D3, but every branch - including the
information-free control B1 - overfit the frozen base's IN-SAMPLE residuals from epoch 1 (train loss 0.2175 -> 0.203
while validation J rose monotonically; B0's own final train loss was 0.2232). K1 therefore could not separate the
physics prior from a training-scheme defect. K1 stays recorded as D3 for the frozen-residual design; K1b changes ONE
thing - base and branch are trained jointly from the same seeded base initialisation - and re-asks the question.

Arms (fold D1: fit 2016-2020, validation 2021, development eval 2022; seed 2024; 6 fits):
  T9      TSMixer on the 9 SCADA channels (no upstream information)
  T21     TSMixer on 21 channels = SCADA + the 11 hand-crafted upstream features + missing flag
  T9+B2   T9 base + station branch, generic attention (q_tilde = 1)          trained jointly
  T9+A    T9 base + station branch, arrival prior with learned kappa         trained jointly
  T21+B2  T21 base + generic station branch                                   trained jointly
  T21+A   T21 base + arrival station branch                                   trained jointly
All arms: balanced loss 0.5 MAE + 0.5 MSE on standardized power, AdamW 1e-3 / wd 1e-4, batch 64, <= 20 epochs,
patience 5, cosine schedule, clip 1; checkpoint by validation J_p = MAE/MAE_pers + RMSE/RMSE_pers (kW, ALL cohort,
persistence = last hourly power), epoch 0 included, ties keep the earlier epoch. The branch is the unchanged K1 module.

Pre-registered decision on 2022 (ALL, h1-4 pooled, kW), mutually exclusive:
  E0 invalid   a required arm missing or a metric non-finite (technical failure, no conclusion)
  E1 prior     for base b in {T9, T21}: b+A beats b+B2 by >= 1 % RMSE with MAE >= -1 kW AND b+A beats b by >= 1 % RMSE
               with MAE >= 0                                   (holds for at least one b)
  E2 fusion    not E1, but some b+B2 or b+A beats b by >= 1 % RMSE with MAE >= 0
  E3 none      everything else
Descriptive only: T9+A / T9+B2 versus T21 (station encoder instead of the hand-crafted features). Development evidence,
single seed, no significance claim. Station files 2016-2022 only.

  python scripts/train_kelmarsh_arrival_v33_k1b.py --smoke --out /tmp/v33_k1b_smoke
  python scripts/train_kelmarsh_arrival_v33_k1b.py --out outputs/logs/Kelmarsh/v33_k1b_<tag>
"""
from __future__ import annotations

import argparse
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
from torch import nn  # noqa: E402

from models.UpstreamArrivalResidual import ArrivalResidual, base_forward, trainable_parameters  # noqa: E402
from scripts import train_kelmarsh_anchored_residual as v26run  # noqa: E402
from scripts import train_kelmarsh_arrival_v33 as k1  # noqa: E402
from scripts import train_kelmarsh_multiyear_benchmark as v28  # noqa: E402
from utils import kelmarsh_station_history_v33 as hist  # noqa: E402

K1B = {
    "run_id": "kelmarsh_arrival_v33_k1b",
    "channel_sets": {"T9": "last10", "T21": "upstream"},
    "arms": {"T9": None, "T21": None, "T9+B2": "B2", "T9+A": "A", "T21+B2": "B2", "T21+A": "A"},
    "decision": {"prior_rmse_pct_min": 1.0, "prior_mae_kw_min": -1.0, "gain_rmse_pct_min": 1.0, "gain_mae_kw_min": 0.0},
    "budget_fits": 6,
}
RATED = 2050.0


class BaseOnly(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, content=None, phys=None, valid=None, return_diagnostics=False):
        y = self.base(x)
        return (y, {"support": None, "mass": None}) if return_diagnostics else y

    def kappa(self):
        return None


class JointArrival(ArrivalResidual):
    """The unchanged K1 branch, but the base is NOT frozen: base and branch are optimised jointly."""

    def __init__(self, base, **kw):
        super().__init__(base, **kw)
        for q in self.base.parameters():
            q.requires_grad_(True)

    def train(self, mode=True):
        nn.Module.train(self, mode)            # the base trains too (its dropout is active in training)
        return self

    def forward(self, x, content=None, phys=None, valid=None, return_diagnostics=False):
        base_pred, hidden = base_forward(self.base, x)
        b = x.shape[0]
        z = self.query(hidden.reshape(b, -1))
        tok_in, valid = content, valid.bool()
        t = self.token_mlp(tok_in)
        lead = self.lead.weight
        qv = self.qproj(torch.cat([z.unsqueeze(1).expand(-1, self.horizon, -1), lead.unsqueeze(0).expand(b, -1, -1)], -1))
        e = torch.clamp(self.score(torch.tanh(self.key(t).unsqueeze(2) + qv.unsqueeze(1))).squeeze(-1), -self.clamp, self.clamp)
        expe = torch.exp(e) * valid.unsqueeze(-1).to(e)
        n_valid = valid.sum(1).unsqueeze(-1).to(e)
        s = expe.sum(1) / n_valid.clamp(min=1)
        mass = s / (1 + s)
        if self.prior == "arrival":
            _, support, q_tilde = self.arrival_q(phys, valid)
        else:
            support, q_tilde = None, torch.ones_like(expe)
        w = expe * q_tilde
        norm = w.sum(1, keepdim=True)
        w = torch.where(norm > 0, w / torch.where(norm > 0, norm, torch.ones_like(norm)), torch.zeros_like(w))
        c = torch.einsum("bkh,bkw->bhw", w, self.value(t))
        delta = mass * self.head(torch.cat([c, z.unsqueeze(1).expand(-1, self.horizon, -1),
                                            lead.unsqueeze(0).expand(b, -1, -1)], -1)).squeeze(-1)
        delta = torch.where(n_valid > 0, delta, torch.zeros_like(delta))
        out = (base_pred + delta).unsqueeze(-1)
        return (out, {"support": support, "mass": mass}) if return_diagnostics else out


def build(arm, p, cfg, channels):
    """Seed-identical base first (so T9 and T9+B2/T9+A share the base initialisation), then the branch."""
    channels = [int(index) for index in channels]  # NumPy view indices must serialize in the registered contract.
    v26run.seed_all(cfg["seed"])
    fam = p["families"]["TSMixerBasicSmall"]
    set_name = K1B["channel_sets"][arm.split("+")[0]]
    base_cfg = {"arm": arm, "family": "TSMixerBasicSmall", "channel_set": set_name, "channel_indices": channels,
                "channels": len(channels), "expected_parameters": fam["parameters"][set_name], "args": dict(fam["args"])}
    base, _ = v28.build_model(base_cfg)
    kind = K1B["arms"][arm]
    if kind is None:
        return BaseOnly(base), base_cfg
    phys, br = cfg["physics"], cfg["branch"]
    common = dict(width=br["width"], query_width=br["query_width"], lead_width=br["lead_embedding"],
                  score_clamp=br["score_clamp"], kappa_bounds=tuple(phys["kappa_bounds"]), kappa_init=phys["kappa_init"],
                  min_speed=phys["min_speed_ms"], speed_floor=phys["speed_floor_ms"], cross_scale=phys["crosswind_scale_km"],
                  sigma_min=phys["sigma_min_h"], sigma_frac=phys["sigma_fraction"], support_eps=phys["support_epsilon"],
                  content_dim=len(hist.CONTENT_FEATURES))
    if kind == "B2":
        return JointArrival(base, tokens="stations", prior="none", **common), base_cfg
    return JointArrival(base, tokens="stations", prior="arrival", kappa="learned", **common), base_cfg


def j_p(s, ref):
    return hist.j_score(s["mae"], s["rmse"], ref["mae"], ref["rmse"])


def fit_joint(arm, p, cfg, splits, ref_val, device, out, max_epochs=None, *, run_contract):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, ev = splits
    model, base_cfg = build(arm, p, cfg, list(train.windows.indices))
    contract = {"run": K1B["run_id"], "arm": arm, "base": base_cfg, "branch": K1B["arms"][arm], "seed": cfg["seed"],
                "run_contract": run_contract,
                "code": {**k1.code_digest(), "scripts/train_kelmarsh_arrival_v33_k1b.py":
                         k1.sha_text(ROOT / "scripts/train_kelmarsh_arrival_v33_k1b.py")}}
    if (out / "trained.json").exists():
        old = json.loads((out / "trained.json").read_text())
        if old["contract"] != contract:
            raise ValueError(f"{arm}: registered arm has a different contract; refuse to overwrite")
        return old
    model.to(device)
    t = cfg["training"]
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["scheduler_t_max"])
    w = float(t["loss_mse_weight"])

    def assess():
        pred, _ = k1.predict(model, val, device, t["eval_batch_size"])
        s = k1.scores(pred, val.truth, RATED)
        return j_p(s, ref_val), s

    j, s = assess()
    best, best_epoch, bad, best_state = j, 0, 0, v26run.cpu_state(model)
    history = [{"epoch": 0, "J_p": j, **s}]
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
                raise ValueError("Non-finite joint loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip_norm"])
            opt.step()
            total, n = total + loss.item(), n + 1
        sched.step()
        j, s = assess()
        kappa = model.kappa()
        history.append({"epoch": epoch, "train_loss": total / n, "J_p": j, **s, "seconds": time.monotonic() - begin,
                        "kappa": None if kappa is None else float(kappa)})
        print(f"[EPOCH] {arm} epoch={epoch} J_p={j:.5f} val_MAE={s['mae']:.3f} val_RMSE={s['rmse']:.3f}", flush=True)
        if j < best:
            best, best_epoch, bad, best_state = j, epoch, 0, v26run.cpu_state(model)
        else:
            bad += 1
    model.load_state_dict(best_state)
    pv, _ = k1.predict(model, val, device, t["eval_batch_size"])
    pe, sup_e = k1.predict(model, ev, device, t["eval_batch_size"])
    kappa = model.kappa()
    v26run.atomic_save({"model": best_state, "epoch": best_epoch, "contract": contract}, out / "best.pt")
    np.savez_compressed(out / "predictions.npz", validation=pv, development_eval=pe,
                        validation_starts=val.windows.window_starts, eval_starts=ev.windows.window_starts)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    trained = {"contract": contract, "arm": arm, "best_epoch": best_epoch, "validation_J_p": best,
               "validation": k1.scores(pv, val.truth, RATED), "development_eval": k1.scores(pe, ev.truth, RATED),
               "trainable_parameters": trainable_parameters(model), "kappa": None if kappa is None else float(kappa),
               "support_rate_eval_by_h": None if sup_e is None else sup_e.mean(axis=0).tolist(), "completed_utc": k1.now()}
    (out / "trained.json").write_text(json.dumps(trained, indent=2))
    return trained


def k1b_decision(m, d=K1B["decision"]):
    arms = tuple(K1B["arms"])
    if any(a not in m for a in arms) or any(not np.isfinite(m[a].get(k, np.nan)) for a in arms for k in ("mae", "rmse")):
        return {"branch": "E0", "reason": "missing arm or non-finite metric (technical failure)"}

    def dr(x, y):
        return 100.0 * (m[y]["rmse"] - m[x]["rmse"]) / m[y]["rmse"]

    def dm(x, y):
        return m[y]["mae"] - m[x]["mae"]

    def gain(x, y):
        return dr(x, y) >= d["gain_rmse_pct_min"] and dm(x, y) >= d["gain_mae_kw_min"]
    numbers = {f"{x}|{y}": {"dRMSE_pct": dr(x, y), "dMAE_kw": dm(x, y)} for x, y in
               [("T9+A", "T9+B2"), ("T21+A", "T21+B2"), ("T9+A", "T9"), ("T9+B2", "T9"), ("T21+A", "T21"),
                ("T21+B2", "T21"), ("T21", "T9"), ("T9+A", "T21"), ("T9+B2", "T21")]}
    prior = [b for b in ("T9", "T21") if dr(f"{b}+A", f"{b}+B2") >= d["prior_rmse_pct_min"]
             and dm(f"{b}+A", f"{b}+B2") >= d["prior_mae_kw_min"] and gain(f"{b}+A", b)]
    fusion = [f"{b}+{k}" for b in ("T9", "T21") for k in ("B2", "A") if gain(f"{b}+{k}", b)]
    branch = "E1" if prior else ("E2" if fusion else "E3")
    return {"branch": branch, "prior_supported_on": prior, "branch_gains_on": fusion, "comparisons": numbers}


def run(out, smoke=False, device=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    determinism = v26run.configure_determinism()
    cfg = json.loads(k1.CONFIG.read_text(encoding="utf-8"))
    p, series, observations, stations, identity = (k1.load_smoke if smoke else k1.load_real)(cfg)
    lag = int(cfg["lag_hours"])
    issue_all = pd.DatetimeIndex(series.dates)
    hours = pd.date_range(issue_all.min() - pd.Timedelta(hours=lag + 24 + 4), issue_all.max(), freq="h")
    bank = hist.build_bank(observations, stations, hours, hours.year <= max(series.splits["train"]),
                           radius_km=cfg["reference_direction"]["radius_km"])
    sets = {"last10": p["data"]["channel_sets"]["last10"], "upstream": list(range(len(series.channels)))}
    if len(series.channels) != 21 or sets["last10"][-1] != 20:
        raise ValueError("Unexpected channel layout")
    splits = {name: tuple(k1.Split(series, s, bank, lag, idx) for s in ("train", "validation", "test"))
              for name, idx in sets.items()}
    val, ev = splits["upstream"][1], splits["upstream"][2]
    pers = {name: k1.scores(sp.windows.truth_and_anchor()[1], sp.truth, RATED) for name, sp in (("validation", val), ("development_eval", ev))}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_contract = {"smoke": bool(smoke), "identity": identity, "config": cfg, "source_protocol": p,
                    "k1b": K1B, "max_epochs": 1 if smoke else cfg["training"]["max_epochs"],
                    "code": {**k1.code_digest(), "scripts/train_kelmarsh_arrival_v33_k1b.py":
                             k1.sha_text(ROOT / "scripts/train_kelmarsh_arrival_v33_k1b.py")}}
    manifest_path = out / "data_manifest.json"
    if manifest_path.exists():
        registered = json.loads(manifest_path.read_text())
        if registered.get("run_contract") != run_contract:
            raise ValueError("K1b output has a different run contract; refuse to overwrite or reuse")
    manifest_path.write_text(json.dumps({"identity": identity, "bank_stats": bank.stats, "smoke": smoke,
                                                        "run_contract": run_contract,
                                                        "determinism": determinism, "persistence": pers,
                                                        "windows": {k: len(v) for k, v in zip(("train", "validation", "development_eval"), splits["upstream"])},
                                                        "utc": k1.now()}, indent=2, default=str))
    if len(K1B["arms"]) > K1B["budget_fits"]:
        raise ValueError("K1b budget exceeded")
    results = {}
    for arm in K1B["arms"]:
        set_name = K1B["channel_sets"][arm.split("+")[0]]
        results[arm] = fit_joint(arm, p, cfg, splits[set_name], pers["validation"], device, out / "arms" / arm,
                                 1 if smoke else None, run_contract=run_contract)
    decision = k1b_decision({a: r["development_eval"] for a, r in results.items()})
    report = {"run_id": K1B["run_id"], "smoke": smoke, "decision": decision,
              "learned_kappa": {a: r["kappa"] for a, r in results.items() if r["kappa"] is not None},
              "metrics": {"Persistence": pers, **{a: {"validation": r["validation"], "development_eval": r["development_eval"],
                                                      "best_epoch": r["best_epoch"]} for a, r in results.items()}},
              "support_rate_eval_by_h": {a: r["support_rate_eval_by_h"] for a, r in results.items()},
              "trainable_parameters": {a: r["trainable_parameters"] for a, r in results.items()},
              "scope": "development evidence on fold D1, single seed 2024; pre-registered point-estimate screen",
              "utc": k1.now()}
    (out / "k1b_decision.json").write_text(json.dumps(report, indent=2))
    lines = [f"# v33 K1b ({'SMOKE' if smoke else 'real'}) - joint training", "", f"Decision: **{decision['branch']}**", "",
             "| arm | 2022 MAE | 2022 RMSE | best epoch | kappa |", "|---|---:|---:|---:|---:|",
             f"| Persistence | {pers['development_eval']['mae']:.2f} | {pers['development_eval']['rmse']:.2f} | | |"]
    for a, r in results.items():
        lines.append(f"| {a} | {r['development_eval']['mae']:.2f} | {r['development_eval']['rmse']:.2f} | {r['best_epoch']} | "
                     f"{'' if r['kappa'] is None else round(r['kappa'], 3)} |")
    lines += ["", "Development evidence only (single seed); no significance claim."]
    (out / "k1b_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision["branch"], "prior_supported_on": decision.get("prior_supported_on"),
                      "branch_gains_on": decision.get("branch_gains_on")}, indent=2))
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
