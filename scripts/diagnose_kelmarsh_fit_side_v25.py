#!/usr/bin/env python3
"""Read-only fit-side diagnostics of the FROZEN v25 TSMixer checkpoints and the frozen v20 DLinear (no training).

Answers the two training-side questions of the v25 handoff without touching any selection decision:
  (1) on the SAME kW scale and in eval mode, how do fit-window (training) errors compare with internal-selection errors
      for DLinear, TSMixerFull and TSMixerPowerOnly (the dropout-on running train MAE in v25 logs is not comparable);
  (2) per lead, per physical power regime, per historical-ramp regime and per turbine, where do Full and DLinear differ,
      and with which bias (MBE), on fit windows AND on the internal selection windows.
Regimes are frozen before any error is computed (physical rated-power fractions; ramp terciles from fit windows only).
It reads only fit/selection windows (never outer validation or the sealed test), never writes into the v25 or v20
run directories, never resumes v25 and never retrains anything. Output: a new folder with diagnostic.json + .md.
Run on the training host (needs torch, the prepared CSV and the registered run directories):
  python scripts/diagnose_kelmarsh_fit_side_v25.py [--v25-dir outputs/logs/Kelmarsh/20261024/backbone_information_v25_f0-1-2_grq4Ps]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data_provider.kelmarsh_hourly import KelmarshSeries  # noqa: E402
from models.DLinear import Model as DLinearModel  # noqa: E402
from models.TSMixerBasicSmall import Model as TSMixerModel  # noqa: E402
from utils import kelmarsh_backbone_information as v25  # noqa: E402
from utils import kelmarsh_fit_side_diagnostics as diag  # noqa: E402

V25_PROTOCOL = ROOT / "configs/kelmarsh_backbone_information_protocol_v25.json"
V25_PROTOCOL_SHA256 = "e111a7947004c5a58b486d57c4544a97e14192f966a33740523b16857b947653"
V26_PROTOCOL = ROOT / "configs/kelmarsh_anchored_residual_protocol_v26.json"
ARMS = ("TSMixerFull", "TSMixerPowerOnly")
REGISTERED_TOLERANCE_KW = 1e-3


@torch.no_grad()
def predict(model, dataset, batch=128):
    model.eval()
    chunks = []
    for x, _, _, _ in DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0):
        chunks.append(model(x.float())[..., -1].numpy())
    scaled = np.concatenate(chunks).astype(np.float64)
    return scaled * dataset.scaler.scale_[-1] + dataset.scaler.mean_[-1]


def load_models(p, fold, v25_dir, reference_root):
    """Hash-verified frozen models of one fold. Nothing is trained or written back."""
    ref = p["reference"]
    spec = next(item for item in ref["dlinear_checkpoints"] if item["fold"] == fold)
    run = reference_root / ref["dir"] / "runs" / f"f{fold}_s{ref['seed']}" / "DLinear"
    if v25.file_sha256(run / "checkpoint.pt") != spec["checkpoint_sha256"]:
        raise ValueError(f"v20 DLinear f{fold} checkpoint hash differs")
    dlinear = DLinearModel(SimpleNamespace(**p["arms"]["DLinear"]["model_args"]))
    dlinear.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["model"], strict=True)
    models = {"DLinear": dlinear}
    completed = v25.read_json(v25_dir / "completed.json")
    c = p["candidate"]
    for arm in ARMS:
        name = f"runs/f{fold}_s2024/{arm}/best.pt"
        if v25.file_sha256(v25_dir / name) != completed[name]:
            raise ValueError(f"v25 {name} differs from its completed.json registration")
        model = TSMixerModel(seq_len=p["input_len"], channels=len(p["feature_columns"]), pred_len=p["pred_len"],
                             blocks=c["blocks"], ff_dim=c["feature_hidden_dim"], dropout=c["dropout"], norm_eps=c["norm_eps"],
                             target_slot=p["target_slot"], input_mode=p["arms"][arm]["input_mode"])
        model.load_state_dict(torch.load(v25_dir / name, map_location="cpu", weights_only=False)["model"], strict=True)
        models[arm] = model
    return models


def registered_selection(p, fold, v25_dir):
    out = {"DLinear": next(item for item in p["reference"]["dlinear_checkpoints"] if item["fold"] == fold)["internal_mae_kw"]}
    for arm in ARMS:
        out[arm] = v25.read_json(v25_dir / f"runs/f{fold}_s2024/{arm}/trained.json")["selection_best"]["mae"]
    return out


def fold_diagnostic(p, fold, v25_dir, reference_root):
    series = KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=p["input_len"], horizon=p["pred_len"],
                            stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
    data = v25.inner_datasets(series)  # fit + internal selection only; validation is never instantiated
    models = load_models(p, fold, v25_dir, reference_root)
    raw_power = series.raw[:, -1]
    _, fit_ramp = diag.last_power_and_ramp(raw_power, data["fit"].window_starts, p["input_len"])
    edges = diag.ramp_edges_from_fit(fit_ramp)
    registered = registered_selection(p, fold, v25_dir)
    out = {"fold": fold, "ramp_edges_kw_from_fit": edges, "windows": {k: len(v) for k, v in data.items()}, "splits": {}}
    for split, dataset in data.items():
        truth, persistence = dataset.truth_and_anchor()
        last, ramp = diag.last_power_and_ramp(raw_power, dataset.window_starts, p["input_len"])
        labels = diag.regime_labels(last, ramp, p["rated_power_kw"], edges)
        turbines = dataset.turbines[dataset.window_starts]
        block = {"Persistence": {"overall": diag.error_summary(persistence, truth)}}
        for arm, model in models.items():
            prediction = predict(model, dataset)
            summary = diag.error_summary(prediction, truth)
            block[arm] = {"overall": summary,
                          "by_power_regime": diag.by_group(prediction, truth, labels["power"], diag.POWER_LABELS),
                          "by_ramp_regime": diag.by_group(prediction, truth, labels["ramp"], diag.RAMP_LABELS),
                          "by_turbine": diag.by_group(prediction, truth, turbines, sorted(set(turbines.tolist())))}
            if split == "selection":
                # read-only tool: record (never abort on) the CPU-replay vs registered difference
                gap = summary["mae"] - registered[arm]
                block[arm]["replay_minus_registered_mae_kw"] = gap
                if abs(gap) > REGISTERED_TOLERANCE_KW:
                    out.setdefault("warnings", []).append(f"f{fold} {arm}: CPU replay MAE differs from registered by {gap} kW")
        block["support"] = {"power_regime": {k: int((labels["power"] == k).sum()) for k in diag.POWER_LABELS},
                            "ramp_regime": {k: int((labels["ramp"] == k).sum()) for k in diag.RAMP_LABELS}}
        out["splits"][split] = block
    out["generalization_gap_mae_kw"] = {arm: out["splits"]["selection"][arm]["overall"]["mae"]
                                        - out["splits"]["fit"][arm]["overall"]["mae"] for arm in ("DLinear",) + ARMS}
    return out


def markdown(result):
    lines = [f"# Read-only fit-side diagnostic of frozen v25/v20 models ({result['utc']})", "",
             "No training, no epoch/threshold choice, no outer or sealed data. Fit windows are training data for every model "
             "(training-internal errors); the two periods differ in difficulty, so gaps are descriptive only.", "",
             "| fold | split | model | MAE | RMSE | MBE | lead-1 MAE | lead-1 MBE |", "|---|---|---|---|---|---|---|---|"]
    for fold in result["folds"]:
        for split, block in fold["splits"].items():
            for arm in ("Persistence", "DLinear") + ARMS:
                o = block[arm]["overall"]
                lines.append(f"| {fold['fold']} | {split} | {arm} | {o['mae']:.3f} | {o['rmse']:.3f} | {o['mbe']:.3f} | "
                             f"{o['per_lead'][0]['mae']:.3f} | {o['per_lead'][0]['mbe']:.3f} |")
    lines += ["", "Selection-minus-fit MAE gap (kW) per fold: "
              + "; ".join(f"f{f['fold']} {f['generalization_gap_mae_kw']}" for f in result["folds"]), "",
              "Regime tables (power, ramp, turbine) with their support are in diagnostic.json."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v25-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if v25.text_sha256_lf(V25_PROTOCOL) != V25_PROTOCOL_SHA256:
        raise ValueError("v25 protocol bytes changed")
    p = v25.read_json(V25_PROTOCOL)
    historical = v25.read_json(V26_PROTOCOL)["v25_historical"]
    v25_dir = (args.v25_dir or ROOT / historical["dir"]).resolve()
    if v25.file_sha256(v25_dir / "completed.json") != historical["completed_sha256"]:
        raise ValueError("v25 completed.json identity differs from the registered run")
    if v25.file_sha256(ROOT / p["data_path"]) != p["data_sha256"]:
        raise ValueError("Prepared Kelmarsh source fingerprint changed")
    torch.set_num_threads(2)
    output = args.output or ROOT / "outputs/logs/Kelmarsh" / f"fit_side_diagnostic_v25_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    if v25_dir in output.resolve().parents or output.resolve() == v25_dir:
        raise ValueError("Never write into the v25 run directory")
    output.mkdir(parents=True, exist_ok=False)
    result = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "v25_dir": str(v25_dir),
              "v25_completed_sha256": historical["completed_sha256"], "device": "cpu", "training": "none",
              "splits_read": ["fit", "selection"], "regime_rule": "power: rated fractions 0.1/0.3/0.6/0.9 of the last "
              "observed hour; ramp: terciles of |P(t)-P(t-3h)| on fit windows",
              "folds": [fold_diagnostic(p, fold, v25_dir, ROOT) for fold in p["folds"]]}
    v25.write_json(output / "diagnostic.json", result)
    (output / "diagnostic.md").write_text(markdown(result), encoding="utf-8")
    print(f"[FIT-SIDE] written {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
