#!/usr/bin/env python3
"""Train only the Wiki arm; numerical controls, labels and cohort stay frozen."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from scripts import train_kelmarsh_contextual_expert as old
from utils import kelmarsh_wiki_quality as quality


def subset_info(info,mask):
    return {key:({k:v[mask] for k,v in value.items()} if isinstance(value,dict) else value[mask]) for key,value in info.items()}


def quality_pack(dataset,reference,expert,contract,rules,ledger,old_expert,*,ready=True,donors=True):
    # The numerical history, target, anchor and reference are never repaired.
    pack=old.pack_data(dataset,reference,old_expert,contract,rules,False,memory_ready=ready)
    info=ledger.context(dataset)
    if ready:
        query={key:pack[key] for key in ("reference","anchor","context","events","times")}
        output=quality.candidates(expert,query,info,contract)
        pack["memory"]=np.concatenate(((output["prediction"]-reference[...,None])/contract["rated_power_kw"],
            output["alpha"],np.log1p(np.sqrt(output["variance_kw2"])/contract["rated_power_kw"]),
            output["physical"],output["overlap"]),-1).astype(np.float32)
        pack["overlap"]=output["overlap"]
    if donors:
        mask=pack["turbines"]!=6
        pack=subset_info(pack,mask)
        info=subset_info(info,mask)
    return pack,dict(windows=len(pack["truth"]),repaired_history_hours_with_repeats=int(info["repaired_hours"].sum()),
        unresolved_history_hours_with_repeats=int(info["unresolved_hours"].sum()),unknown_context_windows=int((~info["good"]).sum()),
        observed_state_counts={str(k):int((info["state"]==k).sum()) for k in np.unique(info["state"])})


def load_contextual(path,cfg,device):
    model=old.ContextualWindResidual(width=cfg["width"],dropout=cfg["dropout"],bound=cfg["correction_bound_capacity"]).to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["model"],strict=True)
    return model


def verify_completed(directory):
    done=old.read(directory/"completed.json")
    for path,expected in done["artifacts"].items():
        if old.sha(directory/path)!=expected:
            raise ValueError("Untreated completed control artifact changed")


def run(protocol,output,device):
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    with (output/"suite.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=old.read(protocol)
        if spec["protocol_id"]!="kelmarsh_h2_wiki_only_observed_quality_v5":
            raise ValueError("Unknown quality protocol")
        saved=output/"protocol.json"
        if saved.exists() and old.sha(saved)!=old.sha(protocol):
            raise ValueError("Quality protocol changed on resume")
        if not saved.exists():saved.write_bytes(protocol.read_bytes())
        old.write(output/"status.json",dict(status="RUNNING",pid=os.getpid()))
        source=ROOT/spec["source_directory"]
        if old.sha(source/"protocol.json")!=spec["source_protocol_sha256"]:
            raise ValueError("Old neural trial protocol changed")
        v4=old.read(source/"protocol.json");v1=ROOT/v4["source_directory"];p=old.read(v1/"protocol.json")
        if old.sha(v1/"protocol.json")!=v4["source_protocol_sha256"] or old.sha(ROOT/p["data_path"])!=p["data_sha256"]:
            raise ValueError("Original numerical base data/protocol changed")
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=False,True
        meta=pd.read_csv(ROOT/p["data_path"],usecols=["timestamp_hour_end_utc","turbine_id","pitch_a_deg","pitch_b_deg","pitch_c_deg","available_capacity_kw"])
        all_rows=[]
        for fold in spec["folds"]:
            target=output/"runs"/f"f{fold}_s{spec['seed']}"
            if (target/"completed.json").exists():
                done=old.read(target/"completed.json")
                if done["protocol_sha256"]!=old.sha(saved) or done["runner_sha256"]!=old.sha(Path(__file__)) or done["quality_source_sha256"]!=quality.implementation_digest():
                    raise ValueError("Completed quality trial identity changed")
                verify_completed(target)
                print(f"[REUSE_FOLD] {fold}",flush=True)
                all_rows.extend(old.read(f) for f in target.glob("*/result.json"))
                continue
            old.gpu_available_to_this_process()
            old.write(output/"progress.json",dict(fold=fold,stage="prepare_wiki_only_inputs"))
            original=source/"runs"/f"f{fold}_s{spec['seed']}"
            verify_completed(original)
            base_dir=v1/"runs"/f"f{fold}_s{spec['seed']}"
            series=old.KelmarshSeries(ROOT/p["data_path"],fold=fold,input_len=p["input_len"],horizon=p["pred_len"],stride=p["train_stride"],start=p["start_utc"],end=p["end_utc_exclusive"])
            ledger=quality.ObservedQuality(meta,series,spec["quality"])
            contract=old.read(base_dir/"Wiki_full/utility_policy.json")["contract"]
            if old.sha(base_dir/"PromptTimeDART/checkpoint.pt")!=contract["base_checkpoint_sha256"]:
                raise ValueError("Original forecast checkpoint changed")
            base=old.PromptTimeDART.Model(old.model_args("PromptTimeDART",series.subset("fit"),p,device)).to(device)
            base.load_state_dict(torch.load(base_dir/"PromptTimeDART/checkpoint.pt",map_location=device,weights_only=False)["model"],strict=True)
            previous_expert=old.read(original/"case_expert.json")
            memory=series.subset("memory");_,anchor=memory.truth_and_anchor()
            bank=old.block(memory,anchor,p["rules"],donors=True)
            info=subset_info(ledger.context(memory),series.turbines[memory.window_starts]!=6)
            expert=quality.fit_memory(bank,info,contract,previous_expert["memory"]["settings"],spec["quality"])
            old.write(target/"quality_expert.json",expert)
            (selection,calibration),_=old.split_late(series)
            datasets=dict(memory_fit=memory,early_fit=series.subset("calibration_early"),selection=selection,calibration=calibration)
            packs,audits={},{}
            for name,dataset in datasets.items():
                reference=old.predict(base,dataset,spec["seed"],device)
                packs[name],audits[name]=quality_pack(dataset,reference,expert,contract,p["rules"],ledger,previous_expert,ready=name!="memory_fit")
            fit=old.concatenate_packs(packs["memory_fit"],packs["early_fit"])
            if fit["times"].max()>=packs["selection"]["times"].min() or packs["selection"]["times"].max()>=packs["calibration"]["times"].min():
                raise ValueError("Train target partitions overlap")
            identity=dict(protocol_sha256=old.sha(saved),quality_expert_id=expert["expert_id"],base=contract,
                numerical_checkpoint_sha256=old.sha(original/"NumericContext/checkpoint.pt"),
                unprocessed_wiki_checkpoint_sha256=old.sha(original/"WikiContext/checkpoint.pt"),
                runner_sha256=old.sha(Path(__file__)),training_source_sha256=old.sha(ROOT/"scripts/train_kelmarsh_contextual_expert.py"),
                model_source_sha256=old.sha(ROOT/"models/ContextualWindResidual.py"),
                fit_sha256=old.digest_arrays(fit),selection_sha256=old.digest_arrays(packs["selection"]))
            old.write(output/"progress.json",dict(fold=fold,stage="neural_training_wiki_only"))
            model,trained=old.train_arm("WikiContext",fit,packs["selection"],p["rated_power_kw"],v4,target/"WikiQualityContext",identity,device)
            numerical=load_contextual(original/"NumericContext/checkpoint.pt",v4["training"],device)
            cal=packs["calibration"]
            numeric=old.infer(numerical,cal,p["rated_power_kw"],device,False)
            candidate=old.clip_relative(old.infer(model,cal,p["rated_power_kw"],device,True),numeric,p["rated_power_kw"],v4["training"]["correction_bound_capacity"])
            execution=dict(identity,wiki_checkpoint_sha256=old.sha(target/"WikiQualityContext/checkpoint.pt"))
            certificate=old.fit_certificate(cal,numeric,candidate,execution,v4["utility"])
            old.write(target/"benefit_certificate.json",certificate)
            old.write(target/"train_quality_audit.json",audits)
            old.write(target/"provenance.json",dict(execution=execution,numerical_retrained=False,numerical_preprocessed=False,labels_modified=False,
                cohort_changed=False,outer_labels_used_for_fit=False,sealed_test_evaluated=False,new_wiki_best_epoch=trained["best_epoch"]))
            old.write(output/"progress.json",dict(fold=fold,stage="frozen_outer_evaluation"))
            validation=series.subset("validation");truth,_=validation.truth_and_anchor()
            cached={}
            for arm in ("Trend","NumericContext","WikiRaw","WikiSelective"):
                with np.load(original/arm/"raw_predictions.npz") as f:
                    np.testing.assert_array_equal(f["truth"],truth);np.testing.assert_array_equal(f["times"],validation.target_times())
                    cached[arm]=f["prediction"].copy()
            evaluation,audit=quality_pack(validation,cached["Trend"],expert,contract,p["rules"],ledger,previous_expert,donors=False)
            numeric=cached["NumericContext"]
            # Check numerical prediction identity separately; never feed it repaired inputs.
            np.testing.assert_allclose(old.infer(numerical,evaluation,p["rated_power_kw"],device,False),numeric,rtol=1e-12,atol=1e-9)
            raw=old.infer(model,evaluation,p["rated_power_kw"],device,True)
            untreated_model=load_contextual(original/"WikiContext/checkpoint.pt",v4["training"],device)
            replay=old.infer(untreated_model,evaluation,p["rated_power_kw"],device,True)
            candidate=old.clip_relative(raw,numeric,p["rated_power_kw"],v4["training"]["correction_bound_capacity"])
            routed=old.route(certificate,evaluation,numeric,candidate,execution)
            for name,prediction in (("OriginalTrend",cached["Trend"]),("NoWikiUnchanged",numeric),("WikiUntreatedRaw",cached["WikiRaw"]),
                ("WikiUntreatedSelective",cached["WikiSelective"]),("WikiQualitySameWeights",replay),("WikiQualityRaw",raw),("WikiQualitySelective",routed["prediction"])):
                result=old.report(name,prediction,validation,target/name,p["rated_power_kw"])
                result.update(fold=fold,seed=spec["seed"],best_epoch=trained["best_epoch"] if name in ("WikiQualityRaw","WikiQualitySelective") else None,
                    development_evidence="Historically inspected outer validation; not independent confirmation")
                for label,reference in (("paired_vs_unchanged_numeric",numeric),("paired_vs_untreated_wiki",cached["WikiRaw"] if name in ("WikiQualityRaw","WikiQualitySameWeights") else cached["WikiSelective"])):
                    result[label]=old.paired_forecast_statistics(prediction,reference,truth,pd.DataFrame({"forecast_start":evaluation["times"][:,0]}),**v4["statistics"])
                if name=="WikiQualitySelective":
                    active=routed["action"]>0;windows=active.any(1)
                    gain=np.abs(numeric-truth)-np.abs(prediction-truth)
                    result["wiki"]=dict(coverage_window_pct=routed["coverage_window_pct"],abstention_window_pct=routed["abstention_window_pct"],
                        physical_candidate_window_pct=float(100*evaluation["events"].any(1).mean()),quality_supported_window_pct=float(100*evaluation["overlap"].any((1,2)).mean()),
                        selected_gain_kw=float(gain[active].mean()) if active.any() else None,harm_window_pct=float(100*(gain.mean(1)[windows]<0).mean()) if windows.any() else None,
                        input_quality=audit)
                    np.savez_compressed(target/name/"interventions.npz",action=routed["action"])
                old.write(target/name/"result.json",result);all_rows.append(result)
            retained={"checkpoint.pt","trained.json","manifest.json","training_history.json","result.json","raw_predictions.npz","interventions.npz","quality_expert.json","benefit_certificate.json","provenance.json","train_quality_audit.json"}
            old.write(target/"completed.json",dict(protocol_sha256=old.sha(saved),runner_sha256=old.sha(Path(__file__)),quality_source_sha256=quality.implementation_digest(),
                artifacts={str(f.relative_to(target)):old.sha(f) for f in target.rglob("*") if f.name in retained}))
            pd.DataFrame([dict(model=r["model"],fold=r["fold"],seed=r["seed"],**r["metrics"],**r.get("wiki",{})) for r in all_rows]).to_csv(output/"metrics_all_runs.csv",index=False)
            del base,model,numerical,untreated_model;gc.collect();torch.cuda.empty_cache()
            print(f"[FOLD_COMPLETE] {fold}; numeric path unchanged",flush=True)
        old.write(output/"status.json",dict(status="COMPLETED",exit_code=0,pid=os.getpid()))
        old.write(output/"progress.json",dict(stage="COMPLETED"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/kelmarsh_wiki_quality_protocol_v5.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:run(args.protocol.resolve(),args.output.resolve(),torch.device("cuda:0"))
    except BlockingIOError:print("[BUSY] Existing suite not modified",flush=True)
    except Exception as error:
        old.write(args.output/"status.json",dict(status="FAILED",error=f"{type(error).__name__}: {error}"));raise
