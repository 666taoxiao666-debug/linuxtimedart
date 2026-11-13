"""G1 ten-arm runner: immutable selection freeze before any test prediction."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import subprocess
import datetime as dt

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_g1_batches import digest, save_json, nd

BATCH_KEYS = ("oS", "oS_mask", "gS", "gS_mask", "gF", "gF_mask", "dS", "dS_mask", "oF", "oF_mask", "age", "cycle_ok")


def select_information_control(validation, candidates=("I", "I_w", "R_sp")):
    scores = {arm: float(np.mean(validation[arm])) for arm in candidates}
    if any(len(validation[a]) != 3 for a in candidates) or not np.isfinite(list(scores.values())).all():
        raise ValueError("Three finite seed validation MAEs per information control required")
    return min(candidates, key=lambda a: scores[a]), scores


def joint_decisions(results, alpha=0.05, method_gain=0.015):
    from utils.kelmarsh_backbone_information import holm_adjust
    questions = ("Q1", "Q3", "Q4", "Q5")
    farms = ("Kelmarsh", "Penmanshiel")
    p = [max(results[f][q]["newey_west"]["p_value_two_sided"] for f in farms) for q in questions]
    adj = holm_adjust(p)
    out = {q: {"p_max": p[i], "p_holm": adj[i], "passed": bool(adj[i] <= alpha and all(
        results[f][q]["gain"] > 0 and results[f][q]["ci95_low"] > 0 for f in farms))}
        for i, q in enumerate(questions)}
    gains = {f: results[f]["Q3"]["gain"] / results[f]["Q3"]["reference_mean_loss"] for f in farms}
    out["relative_Q3_gain"] = gains
    out["enter_G2"] = bool(all(out[q]["passed"] for q in questions) and min(gains.values()) >= method_gain)
    out["stop_G"] = not out["Q1"]["passed"]
    return out


def check_freeze(freeze, expected, root):
    if len(freeze["fits"]) != expected or len({x["key"] for x in freeze["fits"]}) != expected:
        raise ValueError("Incomplete or duplicate fit freeze")
    for item in freeze["fits"]:
        if digest(Path(root) / item["checkpoint"]) != item["sha256"]:
            raise ValueError("Frozen checkpoint identity changed")


def tensors(pack, indices, device):
    import torch
    return {k: torch.as_tensor(pack[k][indices], dtype=torch.float32, device=device) for k in BATCH_KEYS}


def make_model(arm, pack, fit_config, device):
    return nd.G1TSMixerModel(arm, 21 if arm == "B" else 50, pack["geo"],
        blocks=fit_config["blocks"], ff_dim=fit_config["ff_dim"], dropout=fit_config["dropout"],
        norm_eps=fit_config["norm_eps"], ws_scale=float(pack["ws_scale"])).to(device)


def predict(model, pack, device):
    import torch
    model.eval()
    xs, outputs = pack["x21" if model.arm == "B" else "x50"], []
    with torch.no_grad():
        for lo in range(0, len(xs), 512):
            sl = slice(lo, lo+512)
            pred = model(torch.as_tensor(xs[sl], dtype=torch.float32, device=device), tensors(pack, sl, device))
            outputs.append(pred.detach().cpu().numpy().reshape(-1, 4))
    return np.concatenate(outputs)*float(pack["power_scale"])+float(pack["power_mean"])


def fit_one(arm, seed, train, val, config, directory, device):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = make_model(arm, train, config, device)
    model.fit_station_normalization(tensors(train, slice(i, i+512), device) for i in range(0, len(train["truth"]), 512))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"], eta_min=config["eta_min"])
    best = float(np.abs(predict(model, val, device)-val["truth"]).mean())
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epoch_best, stale, history = 0, 0, [{"epoch": 0, "val_mae": best}]
    xs = train["x21" if arm == "B" else "x50"]
    y = (train["truth"]-float(train["power_mean"]))/float(train["power_scale"])
    for epoch in range(1, config["max_epochs"]+1):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(len(xs))
        for lo in range(0, len(xs), config["batch_size"]):
            ix = order[lo:lo+config["batch_size"]]
            target = torch.as_tensor(y[ix], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(torch.as_tensor(xs[ix], dtype=torch.float32, device=device), tensors(train, ix, device)).reshape(-1, 4)
            error = pred-target
            loss = config["mae_weight"]*error.abs().mean()+config["mse_weight"]*error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
        scheduler.step()
        score = float(np.abs(predict(model, val, device)-val["truth"]).mean())
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        history.append(dict(epoch=epoch, val_mae=score))
        if score < best:
            best, epoch_best, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory/"selected.pt"
    torch.save(state, checkpoint)
    result = dict(arm=arm, seed=seed, validation_mae=best, best_epoch=epoch_best, epochs=len(history)-1,
                  sha256=digest(checkpoint), history=history)
    save_json(directory/"trained.json", result)
    return result


def evaluate_arrays(predictions, truth, issue, choice, stats):
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    losses = {a: np.mean([np.abs(p-truth) for p in ps], axis=0) for a, ps in predictions.items()}
    pairs = {"Q1": ("N", "B"), "Q3": ("M", choice), "Q4": ("M", "M_self"), "Q5": ("M", "M_amp")}
    results, sensitivity = {}, {}
    for q, (candidate, reference) in pairs.items():
        kwargs = dict(replicates=stats["replicates"], seed=stats["seed"])
        results[q] = paired_loss_statistics(losses[candidate], losses[reference], issue,
                                           block_length=stats["block_length"], **kwargs)
        sensitivity[q] = paired_loss_statistics(losses[candidate], losses[reference], issue,
                                           block_length=stats["sensitivity_block_length"], **kwargs)
    metrics = {}
    for a, ps in predictions.items():
        denom = float(np.square(truth-truth.mean()).sum())
        metrics[a] = [{"mae": float(np.abs(p-truth).mean()), "rmse": float(np.sqrt(np.square(p-truth).mean())),
                       "r2": float(1-np.square(p-truth).sum()/denom),
                       "lead_mae": np.abs(p-truth).mean(axis=0).tolist()} for p in ps]
    return dict(primary=results, sensitivity=sensitivity, metrics=metrics, seed_aggregation="mean losses, not mean predictions")


def mechanism_diagnostics(pack, model, geo, ws_scale):
    """Development-only oracle comparison. No choices derived here affect fitting."""
    from scipy.stats import spearmanr
    batch = {k: pack[k] for k in BATCH_KEYS}
    params = {k: getattr(model.station, k).detach().cpu().numpy() for k in ("kp", "t_raw", "l_raw")}
    encoded, details = nd.displacement_diagnostics_numpy(batch, geo, params, ws_scale)
    truth, valid = pack["truth_wind"], pack["truth_mask"].all(axis=1).astype(bool)
    baseline = pack["gF"][:, 6:10]
    valid &= pack["gF_mask"][:, 6:10].all(axis=1).astype(bool)
    costs, amp_costs = [], []
    shifts = np.arange(-2, 4)
    for shift in shifts:
        ix = np.arange(6, 10)+shift
        forecast = pack["gF"][:, ix]
        usable = pack["gF_mask"][:, ix].all(axis=1).astype(bool) & (shift >= 6-pack["age"])
        diff = truth-forecast
        costs.append(np.where(usable, np.abs(diff).mean(axis=1), np.inf))
        amp_costs.append(np.where(usable, np.abs(diff-np.median(diff, axis=1)[:, None]).mean(axis=1), np.inf))
    costs, amp_costs = np.stack(costs, 1), np.stack(amp_costs, 1)
    best_shift = shifts[np.argmin(costs, axis=1)]
    # Deterministic control pairing; cyclic roll within season+age, excluding same day.
    import pandas as pd
    issue = pd.DatetimeIndex(pack["issue"].astype("datetime64[ns]"))
    perm, paired = np.arange(len(issue)), np.zeros(len(issue), dtype=bool)
    season = issue.month % 12 // 3
    for s in range(4):
        for age in range(6, 12):
            ix = np.flatnonzero((season == s) & (pack["age"] == age))
            days = issue[ix].normalize()
            for n, dest in enumerate(ix):
                choices = ix[days != days[n]]
                if len(choices):
                    perm[dest], paired[dest] = choices[np.searchsorted(choices, dest) % len(choices)], True
    negative = dict(batch)
    for key in ("oS", "oS_mask", "gS", "gS_mask", "dS", "dS_mask"):
        negative[key] = batch[key][perm]
    shuffled, _ = nd.displacement_diagnostics_numpy(negative, geo, params, ws_scale)
    def corr(x, y, mask):
        if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
            return None
        return float(spearmanr(x[mask], y[mask]).statistic)
    delta = encoded[:, 4]
    mask = valid & paired
    if not valid.any():
        return {"status": "NO_VALID_CALIBRATED_FUTURE", "windows": 0}
    residual = truth-baseline
    return dict(windows=int(valid.sum()), paired_windows=int(mask.sum()),
        no_correction_mae=float(np.abs(residual[valid]).mean()),
        best_shift_mae=float(costs.min(1)[valid].mean()),
        best_amplitude_mae=float(np.abs(residual-np.median(residual, axis=1)[:, None])[valid].mean()),
        best_both_mae=float(amp_costs.min(1)[valid].mean()),
        shift_rank_correlation=corr(delta, best_shift, mask),
        negative_rank_correlation=corr(shuffled[:, 4], best_shift, mask),
        applied_fraction=float((encoded[:, 6][valid] > 0).mean()),
        interpretation="descriptive oracle; effective shift is not identified physical travel time")


def source_snapshot():
    names = subprocess.check_output(["git", "ls-files", "*.py", "configs/g1_protocol.json"], cwd=ROOT, text=True).splitlines()
    return {n: digest(ROOT/n) for n in names if (ROOT/n).is_file()}


def formal_gate(prepared, protocol_sha256):
    path = ROOT/"configs/g1_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != protocol_sha256 or protocol["status"] != "frozen":
        raise ValueError("Protocol is not frozen with the supplied SHA256")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "configs/g1_protocol.json"], cwd=ROOT):
        raise ValueError("Protocol must be committed before running")
    complete = json.loads((Path(prepared)/"complete.json").read_text(encoding="utf-8"))
    if complete["protocol_sha256"] != protocol_sha256 or not complete["b1_acceptance_sha256"]:
        raise ValueError("Preparation/protocol/B1 identity mismatch")
    if complete["path"] != "T":
        raise ValueError("Path A extra descriptive fit count needs reconciliation")
    for farm in protocol["farms"]:
        folder = Path(prepared)/farm
        if digest(folder/"prepared.json") != complete["farms"][farm]:
            raise ValueError("Preparation identity mismatch")
        meta = json.loads((folder/"prepared.json").read_text(encoding="utf-8"))
        for name, expected in meta["artifacts"].items():
            if digest(folder/name) != expected:
                raise ValueError("Prepared artifact changed: " + name)
        for name, expected in meta["inputs"].items():
            if digest(name) != expected:
                raise ValueError("Prepared input changed: " + name)
    return protocol


def run_formal(prepared, directory, protocol_sha256, device):
    import torch
    import fcntl
    prepared, directory = Path(prepared), Path(directory)
    protocol = formal_gate(prepared, protocol_sha256)
    # A persistent global launch receipt prevents a second directory from bypassing no-resume.
    receipt = prepared/"formal_launch.json"
    with (prepared/"formal.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with receipt.open("x", encoding="utf-8") as f:
            json.dump(dict(directory=str(directory.resolve()), utc=dt.datetime.now(dt.timezone.utc).isoformat()), f)
        directory.mkdir(parents=True, exist_ok=False)
        sources = source_snapshot()
        save_json(directory/"launch.json", dict(protocol_sha256=protocol_sha256, source_code=sources,
            prepared_sha256=digest(prepared/"complete.json"), device=device, fits=60))
        try:
            fits, selections = [], {}
            for farm in protocol["farms"]:
                train, val = [dict(np.load(prepared/farm/(s+".npz"), allow_pickle=False)) for s in ("train", "val")]
                validation = {a: [] for a in protocol["arms"]}
                for arm in protocol["arms"]:
                    for seed in protocol["seeds"]:
                        key = f"{farm}/{arm}/{seed}"
                        result = fit_one(arm, seed, train, val, protocol["fit"], directory/key, device)
                        fits.append(dict(key=key, checkpoint=key+"/selected.pt", sha256=result["sha256"]))
                        validation[arm].append(result["validation_mae"])
                        print(json.dumps(dict(completed=len(fits), total=60, key=key)), flush=True)
                choice, scores = select_information_control(validation)
                selections[farm] = dict(arm=choice, scores=scores)
                del train, val
            freeze = dict(fits=fits, selections=selections, protocol_sha256=protocol_sha256, source_code=sources)
            check_freeze(freeze, 60, directory)
            if source_snapshot() != sources:
                raise ValueError("Source changed during fit")
            save_json(directory/"selection_freeze.json", freeze)
            # Recheck every input immediately before the one and only test evaluation.
            formal_gate(prepared, protocol_sha256)
            with (directory/"test_evaluation_started.json").open("x", encoding="utf-8") as f:
                json.dump(dict(freeze_sha256=digest(directory/"selection_freeze.json")), f)
            farm_results, prediction_hashes, diagnostics = {}, {}, {}
            for farm in protocol["farms"]:
                test = dict(np.load(prepared/farm/"test.npz", allow_pickle=False))
                predictions = {}
                for arm in protocol["arms"]:
                    predictions[arm] = []
                    for seed in protocol["seeds"]:
                        folder = directory/farm/arm/str(seed)
                        model = make_model(arm, test, protocol["fit"], device)
                        model.load_state_dict(torch.load(folder/"selected.pt", map_location=device, weights_only=True))
                        pred = predict(model, test, device)
                        if pred.shape != test["truth"].shape or not np.isfinite(pred).all():
                            raise ValueError("Invalid test prediction")
                        np.save(folder/"prediction.npy", pred)
                        prediction_hashes[str(folder.relative_to(directory)/"prediction.npy")] = digest(folder/"prediction.npy")
                        predictions[arm].append(pred)
                        if arm == "M":
                            for split in ("train", "val"):
                                pack = dict(np.load(prepared/farm/(split+"_diagnostics.npz"), allow_pickle=False))
                                diagnostics[f"{farm}/{seed}/{split}"] = mechanism_diagnostics(pack, model, test["geo"], float(test["ws_scale"]))
                farm_results[farm] = evaluate_arrays(predictions, test["truth"], test["issue"].astype("datetime64[ns]"), selections[farm]["arm"], protocol["statistics"])
            decision = joint_decisions({f: r["primary"] for f, r in farm_results.items()})
            save_json(directory/"results.json", dict(farms=farm_results, decision=decision, diagnostics=diagnostics))
            check_freeze(freeze, 60, directory)
            if source_snapshot() != sources:
                raise ValueError("Source changed during evaluation")
            save_json(directory/"completed.json", dict(status="COMPLETED", fits=60,
                results_sha256=digest(directory/"results.json"), predictions=prediction_hashes,
                freeze_sha256=digest(directory/"selection_freeze.json")))
        except Exception as exc:
            save_json(directory/"failed.json", dict(type=type(exc).__name__, message=str(exc),
                evaluation_started=(directory/"test_evaluation_started.json").exists(), restart_allowed=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_formal(args.prepared, args.run_dir, args.protocol_sha256, args.device)


if __name__ == "__main__":
    main()
