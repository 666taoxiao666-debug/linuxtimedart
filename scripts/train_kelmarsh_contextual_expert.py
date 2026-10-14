#!/usr/bin/env python3
"""Sequential matched-capacity Numeric/Wiki CUDA training, with safe resume."""
from __future__ import annotations
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_kelmarsh_hourly import read, write, sha, safe, seed_all, model_args, predict, block, report, model_metrics, gpu_available_to_this_process
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_provider.kelmarsh_hourly import KelmarshSeries, KelmarshWindows
from models import PromptTimeDART
from models.ContextualWindResidual import ContextualWindResidual
from utils import kelmarsh_episode_wiki as memory
from utils.paired_forecast_statistics import paired_forecast_statistics


def digest_arrays(pack):
    h = hashlib.sha256()
    for key in ("history", "memory", "change", "anchor_ratio", "reference", "truth", "times", "turbines", "overlap"):
        h.update(np.ascontiguousarray(pack[key]).tobytes())
    return h.hexdigest()


def split_late(series):
    late = series.subset("calibration_late")
    lo, hi = series.bounds["calibration_early_end"], series.bounds["train_end"]
    cut = (lo + (hi - lo) // 2).astype("datetime64[h]")
    times = late.target_times()
    masks = (times[:, -1] < cut, times[:, 0] >= cut)
    if any(not mask.any() for mask in masks):
        raise ValueError("Empty chronological adapter selection/calibration")
    return [KelmarshWindows(series, late.window_starts[mask], name) for mask, name in zip(masks, ("adapter_selection", "benefit_calibration"))], cut


def pack_data(dataset, reference, expert, contract, rules, donors, memory_ready=True):
    query = block(dataset, reference, rules)
    # Query labels are explicitly excluded from case retrieval inputs.
    features = {key: value for key, value in query.items() if key not in ("truth", "turbines")}
    count = len(memory.legacy.EVENTS)
    capacity = contract["rated_power_kw"]
    if memory_ready:
        out = memory.candidates(expert, features, contract)
        _, physical, overlap = memory.legacy.candidates(expert["memory"], features, contract)
        summary = np.concatenate(((out["prediction"][..., :count] - reference[..., None]) / capacity,
            out["alpha"][..., :count], np.log1p(np.sqrt(out["variance_kw2"][..., :count]) / capacity),
            physical[..., :count], overlap[..., :count]), -1).astype(np.float32)
    else:
        summary = np.zeros((*reference.shape,count*5),np.float32)
        overlap = np.zeros((*reference.shape,count),bool)
    rows = dataset.window_starts[:, None] + np.arange(dataset.seq_len)[None]
    history = dataset.series.values[rows].transpose(0, 2, 1).copy()
    result = dict(query, history=history, memory=summary, change=((reference-query["anchor"])/capacity).astype(np.float32),
        anchor_ratio=(query["anchor"][:, :1]/capacity).astype(np.float32), overlap=overlap[..., :count])
    if donors:
        mask = result["turbines"] != 6
        result = {key: ({k: v[mask] for k, v in value.items()} if isinstance(value, dict) else value[mask]) for key, value in result.items()}
    if not np.isfinite(result["history"]).all() or not np.isfinite(result["memory"]).all():
        raise ValueError("Nonfinite causal adapter inputs")
    return result


def concatenate_packs(first, second):
    return {key: ({name:np.concatenate((value,second[key][name])) for name,value in data.items()}
                  if isinstance(data,dict) else np.concatenate((data,second[key]))) for key,data in first.items()}


def tensor_data(pack, capacity):
    arrays = [pack[key] for key in ("history", "change", "anchor_ratio", "memory")]
    arrays.append(((pack["truth"] - pack["reference"]) / capacity).astype(np.float32))
    return TensorDataset(*(torch.from_numpy(np.ascontiguousarray(x)).float() for x in arrays))


@torch.no_grad()
def infer(model, pack, capacity, device, use_memory):
    model.eval()
    outputs = []
    for batch in DataLoader(tensor_data(pack, capacity), batch_size=256, shuffle=False):
        x, change, anchor, tokens, _ = [value.to(device) for value in batch]
        outputs.append(model(x, change, anchor, tokens, use_memory=use_memory).cpu().numpy().astype(float))
    return pack["reference"] + capacity * np.concatenate(outputs)


def joint_score(metrics, reference):
    if metrics["mae"] > reference["mae"] + 1e-10 or metrics["rmse"] > reference["rmse"] + 1e-10:
        return float("inf")
    return .5 * (metrics["mae"] / reference["mae"] + metrics["rmse"] / reference["rmse"])


def train_arm(name, fit, selection, capacity, spec, directory, identity, device):
    gpu_available_to_this_process()
    cfg, use_memory = spec["training"], name == "WikiContext"
    seed_all(spec["seed"])
    model = ContextualWindResidual(width=cfg["width"], dropout=cfg["dropout"], bound=cfg["correction_bound_capacity"]).to(device)
    if (directory / "trained.json").exists():
        done = read(directory / "trained.json")
        if done["identity"] != identity:
            raise ValueError("Completed adapter contract changed")
        for file, expected in done["artifacts"].items():
            if sha(directory / file) != expected:
                raise ValueError("Completed adapter artifact changed")
        model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"])
        print(f"[REUSE] {name} epoch={done['best_epoch']}", flush=True)
        return model, done
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    base_metrics = model_metrics(selection["reference"], selection["truth"], selection["anchor"], capacity)
    history, start, best_epoch, best = [dict(epoch=0, selection=base_metrics)], 0, 0, 1.
    if (directory / "last.pt").exists():
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        if last["identity"] != identity:
            raise ValueError("Interrupted adapter contract changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        schedule.load_state_dict(last["scheduler"])
        history, start, best_epoch, best = last["history"], last["epoch"], last["best_epoch"], last["best"]
        torch.set_rng_state(last["rng_cpu"])
        torch.cuda.set_rng_state_all(last["rng_cuda"])
        print(f"[RESUME] {name} epoch={start}", flush=True)
    else:
        torch.save(dict(model=model.state_dict(), identity=identity), directory / "checkpoint.pt")
    for epoch in range(start + 1, cfg["epochs"] + 1):
        model.train()
        generator = torch.Generator().manual_seed(spec["seed"] + epoch)
        total, count, begin = 0., 0, time.monotonic()
        for batch in DataLoader(tensor_data(fit, capacity), batch_size=cfg["batch_size"], shuffle=True, generator=generator):
            x, change, anchor, tokens, target = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            residual = model(x, change, anchor, tokens, use_memory=use_memory)
            error = residual - target
            loss = (1 - cfg["mix_mse_weight"]) * error.abs().mean() + cfg["mix_mse_weight"] * error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite residual training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        metrics = model_metrics(infer(model, selection, capacity, device, use_memory), selection["truth"], selection["anchor"], capacity)
        score = joint_score(metrics, base_metrics)
        history.append(dict(epoch=epoch, train_loss=total/count, selection=metrics, seconds=time.monotonic()-begin))
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), identity=identity), directory / "checkpoint.pt")
        schedule.step()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
            history=history, epoch=epoch, best=best, best_epoch=best_epoch, identity=identity,
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()), directory / "last.pt")
        write(directory / "training_history.json", history)
        print(f"[EPOCH] {name} epoch={epoch}/{cfg['epochs']} loss={total/count:.6f} selection_MAE={metrics['mae']:.6f} selection_RMSE={metrics['rmse']:.6f} best_epoch={best_epoch}", flush=True)
    model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"])
    write(directory / "manifest.json", dict(identity=identity, parameters=sum(p.numel() for p in model.parameters()),
        use_memory=use_memory, training=cfg, best_epoch=best_epoch, selection_scope="chronological original training only"))
    done = dict(identity=identity, best_epoch=best_epoch, epochs_completed=cfg["epochs"],
        artifacts={name: sha(directory/name) for name in ("checkpoint.pt", "manifest.json", "training_history.json")})
    write(directory / "trained.json", done)
    return model, done


def clip_relative(wiki, numeric, capacity, bound):
    return numeric + np.clip(wiki - numeric, -capacity * bound, capacity * bound)


def numeric_calibration_mask(pack, numeric, settings):
    """Non-Wiki label-budget control: global per-lead benefit on the same labels."""
    if np.any(pack["turbines"]==6):
        raise ValueError("Held-out knowledge labels are forbidden")
    if len(numeric)<settings["min_windows"] or len(np.unique(pack["turbines"]))<settings["min_turbines"]:
        return np.zeros(numeric.shape[1],bool)
    gain = np.abs(pack["reference"]-pack["truth"])-np.abs(numeric-pack["truth"])
    squared = (pack["reference"]-pack["truth"])**2-(numeric-pack["truth"])**2
    ids = np.unique(pack["turbines"])
    means = np.array([gain[pack["turbines"]==unit].mean(0) for unit in ids])
    mse = np.array([squared[pack["turbines"]==unit].mean(0) for unit in ids])
    return (means.mean(0)>0)&(mse.mean(0)>0)&((means>0).mean(0)>=settings["positive_fraction"])


def fit_certificate(pack, numeric, candidate, execution, settings):
    if np.any(pack["turbines"] == 6):
        raise ValueError("Held-out knowledge turbine cannot calibrate benefit")
    gain, squared = np.abs(numeric-pack["truth"])-np.abs(candidate-pack["truth"]), (numeric-pack["truth"])**2-(candidate-pack["truth"])**2
    entries = {}
    for level, groups in (("event", 1), ("power", 5), ("joint", 20)):
        strata = np.zeros(len(numeric), int) if level == "event" else pack["context"][level+"_bin"]
        for group in range(groups):
            for h in range(numeric.shape[1]):
                for k in range(pack["overlap"].shape[-1]):
                    mask = (strata == group) & pack["overlap"][:, h, k]
                    ids = np.unique(pack["turbines"][mask])
                    gains = [gain[mask & (pack["turbines"] == unit), h].mean() for unit in ids]
                    mse = [squared[mask & (pack["turbines"] == unit), h].mean() for unit in ids]
                    supported = mask.sum() >= settings["min_windows"] and len(ids) >= settings["min_turbines"]
                    accepted = supported and np.mean(gains) > 0 and np.mean(mse) > 0 and np.mean(np.asarray(gains) > 0) >= settings["positive_fraction"]
                    entries[f"{level}:{group}:{h}:{k}"] = dict(supported=bool(supported), accepted=bool(accepted), windows=int(mask.sum()),
                        turbines=len(ids), gain_kw=float(np.mean(gains)) if gains else None, mse_gain=float(np.mean(mse)) if mse else None,
                        score_kw=float(np.mean(gains)) if accepted else 0.)
    policy = dict(execution=execution, settings=settings, entries=entries, calibration_sha256=digest_arrays(pack),
        actual_outputs_sha256=hashlib.sha256(np.ascontiguousarray(np.stack((numeric,candidate),-1)).tobytes()).hexdigest(),
        outcomes_available_ns=int(pack["times"].astype("datetime64[ns]").astype(np.int64).max()),
        interpretation="One later training period, not a temporal transfer guarantee")
    policy["policy_id"] = memory.legacy.history.digest(policy)
    return policy


def route(policy, pack, numeric, candidate, execution):
    if policy["execution"] != execution or policy["policy_id"] != memory.legacy.history.digest({k:v for k,v in policy.items() if k != "policy_id"}):
        raise ValueError("Utility evidence does not match both actual executing checkpoints")
    origins = pack["times"][:,0].astype("datetime64[ns]").astype(np.int64) - 3600 * 10**9
    if origins.min() < policy["outcomes_available_ns"]:
        raise ValueError("Utility outcomes were not available at issue time")
    score = np.zeros_like(pack["overlap"], dtype=float)
    for h in range(score.shape[1]):
        for k in range(score.shape[2]):
            remaining = np.ones(len(numeric), bool)
            for level, bins in (("joint", pack["context"]["joint_bin"]),("power",pack["context"]["power_bin"]),("event",np.zeros(len(numeric),int))):
                for group in np.unique(bins):
                    row = policy["entries"][f"{level}:{group}:{h}:{k}"]
                    if row["supported"]:
                        mask = remaining & (bins == group)
                        score[mask,h,k] = row["score_kw"]
                        remaining[mask] = False
    available = pack["overlap"] & (score > 0)
    active = available.any(-1) & (np.abs(candidate-numeric)>1e-9)
    best = np.where(available,score,-np.inf).argmax(-1)
    return dict(prediction=np.where(active,candidate,numeric), action=np.where(active,best+1,0),
        coverage_window_pct=float(100*active.any(1).mean()), abstention_window_pct=float(100*(~active.any(1)).mean()))


def run(protocol_path, output, device):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/"suite.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec = read(protocol_path)
        if spec["protocol_id"] != "kelmarsh_h2_history_conditioned_executable_wiki_v4":
            raise ValueError("Unknown research protocol")
        saved = output/"protocol.json"
        if saved.exists() and sha(saved) != sha(protocol_path):
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol_path.read_bytes())
        write(output/"status.json",dict(status="RUNNING",pid=os.getpid()))
        source = ROOT/spec["source_directory"]
        if sha(source/"protocol.json") != spec["source_protocol_sha256"]:
            raise ValueError("Frozen source protocol changed")
        p = read(source/"protocol.json")
        if sha(ROOT/p["data_path"]) != p["data_sha256"]:
            raise ValueError("Raw prepared data changed")
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        result_rows = []
        for fold in spec["folds"]:
            target, original = output/"runs"/f"f{fold}_s{spec['seed']}", source/"runs"/f"f{fold}_s{spec['seed']}"
            if (target/"completed.json").exists():
                done = read(target/"completed.json")
                if (done["protocol_sha256"] != sha(saved) or done["runner_sha256"] != sha(Path(__file__)) or
                        done["model_sha256"] != sha(ROOT/"models/ContextualWindResidual.py")):
                    raise ValueError("Completed protocol identity changed")
                for path, expected in done["artifacts"].items():
                    if sha(target/path) != expected:
                        raise ValueError("Completed result changed")
                result_rows.extend(read(target/arm/"result.json")|dict(fold=fold,seed=spec["seed"]) for arm in ("Trend","NumericContext","NumericCalibrated","WikiRaw","WikiPhysical","WikiSelective"))
                print(f"[REUSE_FOLD] {fold}",flush=True)
                continue
            gpu_available_to_this_process()
            write(output/"progress.json",dict(fold=fold,stage="prepare_train_inputs"))
            series = KelmarshSeries(ROOT/p["data_path"],fold=fold,input_len=p["input_len"],horizon=p["pred_len"],stride=p["train_stride"],start=p["start_utc"],end=p["end_utc_exclusive"])
            contract = read(original/"Wiki_full/utility_policy.json")["contract"]
            if sha(original/"PromptTimeDART/checkpoint.pt") != contract["base_checkpoint_sha256"] or sha(original/"PromptTimeDART/manifest.json") != contract["base_manifest_sha256"]:
                raise ValueError("Frozen numerical base changed")
            base = PromptTimeDART.Model(model_args("PromptTimeDART",series.subset("fit"),p,device)).to(device)
            base.load_state_dict(torch.load(original/"PromptTimeDART/checkpoint.pt",map_location=device,weights_only=False)["model"],strict=True)
            state = read(original/"PromptTimeDART/trained.json")
            np.testing.assert_array_equal(state["contract"]["scaler_mean"],series.scaler.mean_)
            np.testing.assert_array_equal(state["contract"]["scaler_scale"],series.scaler.scale_)
            ds = series.subset("memory"); _,anchor = ds.truth_and_anchor()
            expert = memory.make_model(read(original/"Wiki_full/expert.json"),block(ds,anchor,p["rules"],donors=True),contract,block_hours=spec["memory"]["weather_block_hours"])
            (selection_ds,calibration_ds),cut = split_late(series)
            datasets = dict(fit=series.subset("calibration_early"),selection=selection_ds,calibration=calibration_ds)
            packs = {name:pack_data(ds,predict(base,ds,spec["seed"],device),expert,contract,p["rules"],True) for name,ds in datasets.items()}
            memory_ds = series.subset("memory")
            memory_fit = pack_data(memory_ds,predict(base,memory_ds,spec["seed"],device),expert,contract,p["rules"],True,memory_ready=False)
            packs["fit"] = concatenate_packs(memory_fit,packs["fit"])
            if packs["fit"]["times"].max() >= packs["selection"]["times"].min() or packs["selection"]["times"].max() >= packs["calibration"]["times"].min():
                raise ValueError("Chronological adapter partitions overlap")
            write(target/"case_expert.json",expert)
            shared = dict(protocol_sha256=sha(saved),base=contract,memory_expert_id=expert["expert_id"],
                model_source_sha256=sha(ROOT/"models/ContextualWindResidual.py"),runner_sha256=sha(Path(__file__)),
                fit_sha256=digest_arrays(packs["fit"]),selection_sha256=digest_arrays(packs["selection"]),cut_utc=str(cut))
            trained = {}
            for arm in spec["arms"]:
                write(output/"progress.json",dict(fold=fold,stage="neural_training",arm=arm))
                trained[arm] = train_arm(arm,packs["fit"],packs["selection"],p["rated_power_kw"],spec,target/arm,dict(shared,arm=arm),device)
            execution = dict(shared,numeric_checkpoint_sha256=sha(target/"NumericContext/checkpoint.pt"),wiki_checkpoint_sha256=sha(target/"WikiContext/checkpoint.pt"))
            cal = packs["calibration"]
            numeric = infer(trained["NumericContext"][0],cal,p["rated_power_kw"],device,False)
            numeric_mask = numeric_calibration_mask(cal,numeric,spec["utility"])
            write(target/"numeric_calibration.json",dict(keep_leads=numeric_mask,calibration_sha256=digest_arrays(cal)))
            candidate = clip_relative(infer(trained["WikiContext"][0],cal,p["rated_power_kw"],device,True),numeric,p["rated_power_kw"],spec["training"]["correction_bound_capacity"])
            certificate = fit_certificate(cal,numeric,candidate,execution,spec["utility"])
            write(target/"benefit_certificate.json",certificate)
            write(target/"provenance.json",dict(execution=execution,fit_windows=len(packs["fit"]["truth"]),selection_windows=len(packs["selection"]["truth"]),calibration_windows=len(cal["truth"]),
                neural_base_retrained=False,adapter_neural_training=True,sealed_test_evaluated=False,outer_labels_used_for_fit=False))
            write(output/"progress.json",dict(fold=fold,stage="frozen_outer_evaluation"))
            validation = series.subset("validation")
            reference = predict(base,validation,spec["seed"],device)
            with np.load(original/"PromptTimeDART/raw_predictions.npz") as cache:
                np.testing.assert_allclose(reference,cache["prediction"],rtol=1e-8,atol=1e-7)
                np.testing.assert_array_equal(validation.target_times(),cache["times"])
            evaluation = pack_data(validation,reference,expert,contract,p["rules"],False)
            numeric = infer(trained["NumericContext"][0],evaluation,p["rated_power_kw"],device,False)
            raw = infer(trained["WikiContext"][0],evaluation,p["rated_power_kw"],device,True)
            candidate = clip_relative(raw,numeric,p["rated_power_kw"],spec["training"]["correction_bound_capacity"])
            physical = np.where(evaluation["overlap"].any(-1),candidate,numeric)
            selected = route(certificate,evaluation,numeric,candidate,execution)
            numeric_calibrated = np.where(numeric_mask,numeric,reference)
            for name,prediction in (("Trend",reference),("NumericContext",numeric),("NumericCalibrated",numeric_calibrated),("WikiRaw",raw),("WikiPhysical",physical),("WikiSelective",selected["prediction"])):
                result = report(name,prediction,validation,target/name,p["rated_power_kw"])
                result.update(fold=fold,seed=spec["seed"],best_epoch=(state["best_epoch"] if name=="Trend" else trained["NumericContext" if name.startswith("Numeric") else "WikiContext"][1]["best_epoch"]),
                    development_evidence="Historical outer validation, not independent confirmation")
                result["paired_vs_numeric"] = paired_forecast_statistics(prediction,numeric,evaluation["truth"],pd.DataFrame({"forecast_start":evaluation["times"][:,0]}),**spec["statistics"])
                result["paired_vs_numeric_calibrated"] = paired_forecast_statistics(prediction,numeric_calibrated,evaluation["truth"],pd.DataFrame({"forecast_start":evaluation["times"][:,0]}),**spec["statistics"])
                result["by_knowledge_transfer_group"] = {}
                for group,mask in (("donors_1_to_5",evaluation["turbines"]!=6),("knowledge_held_out_6",evaluation["turbines"]==6)):
                    if mask.any():
                        result["by_knowledge_transfer_group"][group] = model_metrics(prediction[mask],evaluation["truth"][mask],evaluation["anchor"][mask],p["rated_power_kw"])
                if name=="WikiSelective":
                    active = selected["action"]>0; windows=active.any(1)
                    gains=np.abs(numeric-evaluation["truth"])-np.abs(prediction-evaluation["truth"])
                    result["wiki"] = dict(coverage_window_pct=selected["coverage_window_pct"],abstention_window_pct=selected["abstention_window_pct"],
                        physical_candidate_window_pct=float(100*evaluation["events"].any(1).mean()),
                        physical_supported_window_pct=float(100*evaluation["overlap"].any((1,2)).mean()),
                        selected_gain_kw=float(gains[active].mean()) if active.any() else None,
                        harm_window_pct=float(100*(gains.mean(1)[windows]<0).mean()) if windows.any() else None)
                    result["wiki"]["by_event"] = {}
                    for k,event in enumerate(memory.legacy.EVENTS):
                        selected_points = selected["action"]==k+1
                        event_windows = selected_points.any(1)
                        result["wiki"]["by_event"][event] = dict(selected_points=int(selected_points.sum()),
                            selected_gain_kw=float(gains[selected_points].mean()) if selected_points.any() else None,
                            harm_window_pct=float(100*(gains.mean(1)[event_windows]<0).mean()) if event_windows.any() else None)
                    np.savez_compressed(target/name/"interventions.npz",action=selected["action"])
                write(target/name/"result.json",result); result_rows.append(result)
            retained = {"checkpoint.pt","trained.json","manifest.json","training_history.json","result.json","raw_predictions.npz","interventions.npz","provenance.json","benefit_certificate.json","case_expert.json","numeric_calibration.json"}
            write(target/"completed.json",dict(protocol_sha256=sha(saved),runner_sha256=sha(Path(__file__)),model_sha256=sha(ROOT/"models/ContextualWindResidual.py"),
                artifacts={str(path.relative_to(target)):sha(path) for path in target.rglob("*") if path.name in retained}))
            pd.DataFrame([dict(model=r["model"],fold=r["fold"],seed=r["seed"],best_epoch=r["best_epoch"],**r["metrics"],**r.get("wiki",{})) for r in result_rows]).to_csv(output/"metrics_all_runs.csv",index=False)
            del base,trained;gc.collect();torch.cuda.empty_cache()
            print(f"[FOLD_COMPLETE] {fold}",flush=True)
        write(output/"status.json",dict(status="COMPLETED",exit_code=0,pid=os.getpid()))
        write(output/"progress.json",dict(stage="COMPLETED"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/kelmarsh_contextual_expert_protocol_v4.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    args=parser.parse_args()
    try:
        run(args.protocol.resolve(),args.output.resolve(),torch.device(f"cuda:{args.gpu}"))
    except BlockingIOError:
        print("[BUSY] Existing suite not modified",flush=True)
    except Exception as error:
        write(args.output/"status.json",dict(status="FAILED",error=f"{type(error).__name__}: {error}"))
        raise
