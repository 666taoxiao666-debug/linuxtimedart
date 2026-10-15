#!/usr/bin/env python3
"""Train case-response retrieval; keep all frozen numerical/control artifacts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.ResponseCaseAttention import ResponseCaseAttention, tensor_inputs
from scripts import train_kelmarsh_short_dynamics as v9
from utils import response_case_view as case_view
from utils.gpu_runtime_policy import GPURuntimePolicy

old, frozen, guard, previous = v9.old, v9.frozen, v9.guard, v9.previous
PROTOCOL_ID = "kelmarsh_h2_cross_unit_response_attention_v10"


def prepare(spec, output, protocol_sha):
    source = ROOT / spec["source_directory"]
    if old.sha(source / "protocol.json") != spec["source_protocol_sha256"]:
        raise ValueError("Frozen v9 protocol changed")
    previous.verify_directory(source, spec["source_protocol_sha256"])
    prior = old.read(source / "protocol.json")
    if spec["training"] != prior["training"] or spec["utility"] != prior["utility"]:
        raise ValueError("Unapproved training/utility budget change")
    prepared = old.read(source / "prepared.json")
    v7 = ROOT / prior["source_directory"]
    v7spec = old.read(v7 / "protocol.json")
    v5root, v5, v4root, v4, baseroot, p = frozen.source_configuration(old.read(ROOT / v7spec["source_directory"] / "protocol.json"))
    suffix = Path("runs/f1_s2024")
    v5run, v4run, baserun = [path / suffix for path in (v5root, v4root, baseroot)]
    for name, path in dict(base=baserun / "PromptTimeDART/checkpoint.pt", numeric=v4run / "NumericContext/checkpoint.pt",
                           old_wiki=v5run / "WikiQualityContext/checkpoint.pt").items():
        if old.sha(path) != prepared["identity"]["checkpoints"][name]:
            raise ValueError("Frozen numerical/source model changed")
    if old.sha(ROOT / p["data_path"]) != p["data_sha256"]:
        raise ValueError("Input dataset changed")
    bank, contract = old.read(source / "short_case_expert.json"), prepared["identity"]["base"]
    # This pilot has one actual active event/state expert. Do not certify a
    # combined multi-event correction using just one constituent's evidence.
    if [(g["event"], g["state"]) for g in bank["groups"] if g["active"]] != [(0, 5)]:
        raise ValueError("This protocol requires the single frozen active case group")
    packs = {name: frozen.load_pack(source / f"train_{name}.npz") for name in
             ("memory_fit", "early_fit", "selection", "calibration")}
    series = v9.source_series(p)
    ledger = v9.quality_ledger(series, p, v5)
    views = {}
    for name, pack in packs.items():
        guard.clock_support(pack["times"], pack["turbines"], train_end=prepared["train_end"])
        if guard.array_digest(pack) != prepared["identity"]["transformed_pack_hashes"][name]:
            raise ValueError("Saved donor pack changed")
        views[name] = case_view.build(pack, v9.training_info(series, ledger, pack), bank, contract, ready=name != "memory_fit")
    numerical = old.read(v7 / "numerical_gate.json")
    if numerical["gate_id"] != prepared["identity"]["numerical_gate_id"]:
        raise ValueError("Common numerical gate changed")
    code = ("models/ResponseCaseAttention.py", "utils/response_case_view.py", "scripts/train_kelmarsh_response_attention.py",
            "utils/kelmarsh_short_dynamics_wiki.py", "utils/certified_numeric_residual.py", "utils/gpu_runtime_policy.py")
    identity = dict(protocol_sha256=protocol_sha, source_completed_sha256=old.sha(source / "completed.json"),
        source_prepared_sha256=old.sha(source / "prepared.json"), source_case_id=bank["expert_id"],
        checkpoints=prepared["identity"]["checkpoints"], base=contract, numerical_gate_id=numerical["gate_id"],
        pack_hashes={k: guard.array_digest(v) for k, v in packs.items()},
        view_hashes={k: guard.array_digest(v) for k, v in views.items()},
        source_sha256={name: old.sha(ROOT / name) for name in code})
    record = dict(identity=identity, train_end=prepared["train_end"], windows={k: len(v["truth"]) for k, v in packs.items()},
        same_unit_excluded_at_fit_and_inference=True, old_controls_retrained=False, added_case_labels=0)
    if (output / "prepared.json").exists() and old.read(output / "prepared.json") != record:
        raise ValueError("Response attention inputs/executor changed on resume")
    old.write(output / "prepared.json", record)
    old.write(output / "training_view_audit.json", dict(
        supported_windows={name: int(v["supported"].any((1, 2)).sum()) for name, v in views.items()},
        source_group_cases=[g["windows"] for g in bank["groups"] if g["active"]],
        source_units=[g["units"] for g in bank["groups"] if g["active"]],
        case_outputs_not_fabricated=True, non_wiki_inputs_unchanged=True, outer_used_for_fit=False))
    return packs, views, identity, numerical, (source, v7, ROOT / prior["anchored_control_directory"], p, v5, v4run, bank, contract)


def make_model(spec, device, fixed=False):
    c, a = spec["training"], spec["attention"]
    old.seed_all(spec["seed"])
    return ResponseCaseAttention(features=a["features"], width=c["width"], embedding=a["embedding"], dropout=c["dropout"],
                                 bound=c["correction_bound_capacity"], fixed=fixed).to(device)


@torch.no_grad()
def infer(model, view, device, *, trace=False):
    model.eval()
    names = ("delta", "proposal", "weights", "group_proposal") if trace else ("delta", "proposal")
    outputs = {key: [] for key in names}
    for batch in DataLoader(TensorDataset(*tensor_inputs(view)), batch_size=128, shuffle=False):
        part = model.components(*[value.to(device) for value in batch])
        for key in outputs:
            outputs[key].append(part[key].cpu().numpy().astype(float))
    return {key: np.concatenate(value) for key, value in outputs.items()}


def fit_alpha(model, view, pack, capacity, device, train_end):
    proposal = infer(model, view, device)["proposal"]
    fit_pack = dict(pack, reference=pack["numeric"], numeric=pack["numeric"] + capacity * proposal)
    fitted = guard.fit_shrinkage(fit_pack, train_end=train_end)
    model.set_alpha(fitted["alpha"])
    return fitted


def train(spec, output, fit, fit_view, selection, selection_view, identity, capacity, device, resources, train_end, *, fixed=False):
    output.mkdir(parents=True, exist_ok=True)
    model = make_model(spec, device, fixed)
    marker = output / "trained.json"
    if marker.exists():
        done = old.read(marker)
        if done["identity"] != identity or any(old.sha(output / k) != v for k, v in done["artifacts"].items()):
            raise ValueError("Completed response expert changed")
        model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
        print(f"[REUSE_TRAINED] {output.name}", flush=True)
        return model, done
    c = spec["training"]
    baseline = old.model_metrics(selection["numeric"], selection["truth"], selection["anchor"], capacity)
    history, start, best, best_epoch = [dict(epoch=0, selection=baseline)], 0, 1., 0
    alpha_record = None
    if fixed:
        alpha_record = fit_alpha(model, fit_view, fit, capacity, device, train_end)
        frozen.save_torch(output / "checkpoint.pt", dict(model=model.state_dict(), identity=identity))
        metrics = old.model_metrics(selection["numeric"] + capacity * infer(model, selection_view, device)["delta"],
                                    selection["truth"], selection["anchor"], capacity)
        history.append(dict(epoch=None, fitting_only_analytic_alpha=model.alpha.cpu().tolist(), selection=metrics))
        best_epoch = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c["epochs"])
        if (output / "last.pt").exists():
            last = torch.load(output / "last.pt", map_location=device, weights_only=False)
            if last["identity"] != identity:
                raise ValueError("Interrupted response expert identity changed")
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            start, history, best, best_epoch = last["epoch"], last["history"], last["best"], last["best_epoch"]
            torch.set_rng_state(last["rng_cpu"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([x.cpu() for x in last["rng_cuda"]])
        else:
            frozen.save_torch(output / "checkpoint.pt", dict(model=model.state_dict(), identity=identity))
        target = torch.from_numpy(((fit["truth"] - fit["numeric"]) / capacity).astype(np.float32))
        data = TensorDataset(*tensor_inputs(fit_view), target)
        for epoch in range(start + 1, c["epochs"] + 1):
            resources.check()
            model.train()
            total, count = 0., 0
            generator = torch.Generator().manual_seed(spec["seed"] + epoch)
            for batch in DataLoader(data, batch_size=c["batch_size"], shuffle=True, generator=generator):
                *features, label = [value.to(device) for value in batch]
                optimizer.zero_grad(set_to_none=True)
                error = model.components(*features)["proposal"] - label
                loss = (1 - c["mix_mse_weight"]) * error.abs().mean() + c["mix_mse_weight"] * error.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite response loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
                optimizer.step()
                resources.after_step()
                total += len(label) * float(loss.detach())
                count += len(label)
            alpha_record = fit_alpha(model, fit_view, fit, capacity, device, train_end)
            metrics = old.model_metrics(selection["numeric"] + capacity * infer(model, selection_view, device)["delta"],
                                        selection["truth"], selection["anchor"], capacity)
            score = old.joint_score(metrics, baseline)
            history.append(dict(epoch=epoch, train_loss=total / count, fitting_alpha=model.alpha.cpu().tolist(), selection=metrics))
            if score < best:
                best, best_epoch = score, epoch
                frozen.save_torch(output / "checkpoint.pt", dict(model=model.state_dict(), identity=identity))
            scheduler.step()
            frozen.save_torch(output / "last.pt", dict(model=model.state_dict(), identity=identity, optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, history=history, best=best, best_epoch=best_epoch,
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
            old.write(output / "training_history.json", history)
            print(f"[EPOCH] {output.name} {epoch}/{c['epochs']} selection={metrics['mae']:.6f}/{metrics['rmse']:.6f} best={best_epoch} alpha={model.alpha.tolist()}", flush=True)
        model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)["model"])
    old.write(output / "training_history.json", history)
    old.write(output / "manifest.json", dict(identity=identity, model_type=type(model).__name__, fixed_kernel=fixed,
        best_epoch=best_epoch, proposal_epochs=0 if fixed else c["epochs"], parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        executing_alpha=model.alpha.cpu().tolist(), alpha_fit_scope="memory+early fitting labels only",
        alpha_solver="joint-convex fitting-only solver; no searched validation threshold", value_head=False,
        numerical_forecaster_in_optimizer=False))
    done = dict(identity=identity, best_epoch=best_epoch, epochs_completed=0 if fixed else c["epochs"],
        artifacts={key: old.sha(output / key) for key in ("checkpoint.pt", "training_history.json", "manifest.json")})
    old.write(marker, done)
    return model, done


def evaluate(spec, output, arms, executions, numerical, information, device):
    source, v7, v8, p, v5, v4run, bank, contract = information
    validation = v9.source_series(p).subset("validation")
    truth, anchor = validation.truth_and_anchor()
    times, units = validation.target_times(), validation.series.turbines[validation.window_starts]
    reports = v9.control_reports(v7, v8)
    reports.update({name: source / "reports" / name for name in ("DynamicsWikiRaw", "DynamicsWikiSelective")})
    predictions = dict(Persistence=anchor)
    for name, directory in reports.items():
        with np.load(directory / "raw_predictions.npz", allow_pickle=False) as saved:
            for key, expected in dict(truth=truth, times=times, turbines=units).items():
                np.testing.assert_array_equal(saved[key], expected)
            predictions[name] = saved["prediction"].copy()
    common = guard.apply(numerical, predictions["OriginalTrend"], predictions["NumericContextFrozen"], times, numerical["execution"])
    np.testing.assert_array_equal(common, predictions["CertifiedNumeric"])
    pack = old.pack_data(validation, predictions["OriginalTrend"], old.read(v4run / "case_expert.json"), contract, p["rules"], False)
    pack["numeric"] = common
    view = case_view.build(pack, v9.quality_ledger(validation.series, p, v5).context(validation), bank, contract)
    cpack, actions = case_view.certificate_pack(pack, view), {}
    np.savez_compressed(output / "outer_case_identity_trace.npz", case_indices=view["case_indices"], case_units=view["case_units"],
                        supported=view["supported"], times=times, turbines=units, values=view["values"], base_change=view["base_change"])
    for name, (model, trained) in arms.items():
        actual = infer(model, view, device, trace=True)
        candidate = common + p["rated_power_kw"] * actual["delta"]
        routed = old.route(old.read(output / name / "benefit_certificate.json"), cpack, common, candidate, executions[name])
        predictions[name + "Raw"], predictions[name + "Selective"] = candidate, routed["prediction"]
        actions[name + "Raw"] = (abs(candidate - common) > 1e-9).astype(int)
        actions[name + "Selective"] = routed["action"]
        np.savez_compressed(output / name / "executed_trace.npz", **actual, alpha=model.alpha.cpu().numpy())
    refs = ("Persistence", "CertifiedNumeric", "DynamicsWikiSelective", "CaseAnchoredWikiSelective", "FreeWikiSelective",
            "ZeroMemorySelective", "FixedKernelSelective")
    rows = []
    for name, prediction in predictions.items():
        if name in reports:
            result = old.read(reports[name] / "result.json")
        else:
            result = old.report(name, prediction, validation, output / "reports" / name, p["rated_power_kw"])
            if name in actions:
                arm = name.removesuffix("Raw").removesuffix("Selective")
                result["best_epoch"] = arms[arm][1]["best_epoch"]
                result["wiki"] = previous.interventions(prediction, common, truth, view["supported"].any(-1))
                np.savez_compressed(output / "reports" / name / "interventions.npz", action=actions[name])
            result["paired"] = {ref: old.paired_forecast_statistics(prediction, predictions[ref], truth,
                pd.DataFrame({"forecast_start": times[:, 0]}), **spec["statistics"]) for ref in refs if ref != name}
            old.write(output / "reports" / name / "result.json", result)
        rows.append(dict(model=name, best_epoch=result.get("best_epoch"), **result["metrics"], **result.get("wiki", {})))
    values = {row["model"]: row for row in rows}
    success = all(values["ResponseAttentionSelective"][key] < values[ref][key] for ref in refs for key in ("mae", "rmse"))
    old.write(output / "result.json", dict(results=rows, primary_joint_success=success, added_case_labels=0,
        old_controls_retrained=False, numerical_inputs_modified=False, sealed_test_evaluated=False,
        interpretation="One historically inspected development fold; response weighting is a falsifiable transfer hypothesis"))
    pd.DataFrame(rows).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output, shared_gpu=True):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = old.read(protocol)
        if spec["protocol_id"] != PROTOCOL_ID or spec["folds"] != [1] or spec["seed"] != 2024:
            raise ValueError("Unknown response-attention pilot")
        saved = output / "protocol.json"
        if saved.exists() and old.sha(saved) != old.sha(protocol):
            raise ValueError("Frozen protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            previous.verify_directory(output, old.sha(saved))
            print("[REUSE_SUITE] no repeated fitting", flush=True)
            return
        resources = GPURuntimePolicy(shared=shared_gpu,
            on_change=lambda state: old.write(output / "resource_policy.json", dict(state, protocol_sha256=old.sha(saved))))
        resources.configure()
        old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="cross_unit_view_preflight"))
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        packs, views, identity, numerical, information = prepare(spec, output, old.sha(saved))
        fit = old.concatenate_packs(packs["memory_fit"], packs["early_fit"])
        fit_view = {key: np.concatenate((views["memory_fit"][key], views["early_fit"][key])) for key in views["memory_fit"]}
        device, capacity = torch.device("cuda:0"), information[3]["rated_power_kw"]
        arms, executions = {}, {}
        cal, calview = packs["calibration"], views["calibration"]
        calpack = case_view.certificate_pack(cal, calview)
        calibration = dict(numeric=cal["numeric"], truth=cal["truth"], times=cal["times"], turbines=cal["turbines"],
                           overlap=calview["supported"])
        for name, fixed in (("ResponseAttention", False), ("FixedKernel", True)):
            old.write(output / "progress.json", dict(stage="fit_" + name))
            model, done = train(spec, output / name, fit, fit_view, packs["selection"], views["selection"], identity,
                capacity, device, resources, old.read(output / "prepared.json")["train_end"], fixed=fixed)
            actual = infer(model, calview, device, trace=True)
            candidate = cal["numeric"] + capacity * actual["delta"]
            calibration[name] = candidate
            execution = dict(identity, executor=name, fixed_kernel=fixed, checkpoint_sha256=old.sha(output / name / "checkpoint.pt"))
            certificate = old.fit_certificate(calpack, cal["numeric"], candidate, execution, spec["utility"])
            old.write(output / name / "benefit_certificate.json", certificate)
            old.write(output / name / "training_benefit_by_day.json", frozen.replay_diagnostic(calpack, cal["numeric"], candidate,
                certificate, train_end=old.read(output / "prepared.json")["train_end"]))
            np.savez_compressed(output / name / "calibration_trace.npz", **actual, alpha=model.alpha.cpu().numpy())
            arms[name], executions[name] = (model, done), execution
        np.savez_compressed(output / "training_calibration_predictions.npz", **calibration)
        old.write(output / "progress.json", dict(stage="frozen_outer_report"))
        resources.check()
        evaluate(spec, output, arms, executions, numerical, information, device)
        artifacts = {str(path.relative_to(output)): old.sha(path) for path in output.rglob("*") if path.is_file() and path.name
            not in ("suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json")}
        old.write(output / "completed.json", dict(protocol_sha256=old.sha(saved), artifacts=artifacts))
        old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] response encoder + analytic fixed kernel; numerical controls reused", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_response_attention_protocol_v10.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), not args.idle_only)
    except BlockingIOError:
        print("[BUSY] this experiment is already running", flush=True)
    except Exception as error:
        old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
