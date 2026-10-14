#!/usr/bin/env python3
"""Read-only train-partition diagnostics of frozen contextual residual experts."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import train_kelmarsh_contextual_expert as trial


def summarize(pack, prediction, capacity):
    reference, truth = pack["reference"], pack["truth"]
    delta = prediction-reference
    return dict(metrics=trial.model_metrics(prediction,truth,pack["anchor"],capacity),
        mae_gain_kw=float((np.abs(reference-truth)-np.abs(prediction-truth)).mean()),
        predicted_delta_mean_kw=delta.mean(0),predicted_delta_std_kw=delta.std(0),
        target_residual_mean_kw=(truth-reference).mean(0),target_residual_median_kw=np.median(truth-reference,axis=0),
        bias_kw=(prediction-truth).mean(0),
        positive_correction_fraction=(delta>0).mean(0))


def run(suite,output,device):
    trial.gpu_available_to_this_process()
    spec=trial.read(suite/"protocol.json")
    source=ROOT/spec["source_directory"]
    p=trial.read(source/"protocol.json")
    if trial.sha(source/"protocol.json")!=spec["source_protocol_sha256"] or trial.sha(ROOT/p["data_path"])!=p["data_sha256"]:
        raise ValueError("Source protocol or data changed")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=False,True
    rows=[]
    for fold in spec["folds"]:
        directory=suite/"runs"/f"f{fold}_s{spec['seed']}"
        done=trial.read(directory/"completed.json")
        for path,expected in done["artifacts"].items():
            if trial.sha(directory/path)!=expected:
                raise ValueError("Completed trial artifact changed")
        original=source/"runs"/f"f{fold}_s{spec['seed']}"
        series=trial.KelmarshSeries(ROOT/p["data_path"],fold=fold,input_len=p["input_len"],horizon=p["pred_len"],stride=p["train_stride"],start=p["start_utc"],end=p["end_utc_exclusive"])
        base=trial.PromptTimeDART.Model(trial.model_args("PromptTimeDART",series.subset("fit"),p,device)).to(device)
        base.load_state_dict(torch.load(original/"PromptTimeDART/checkpoint.pt",map_location=device,weights_only=False)["model"],strict=True)
        expert=trial.read(directory/"case_expert.json")
        contract=trial.read(original/"Wiki_full/utility_policy.json")["contract"]
        (selection,calibration),_=trial.split_late(series)
        arms={}
        for arm in spec["arms"]:
            model=trial.ContextualWindResidual(width=spec["training"]["width"],dropout=spec["training"]["dropout"],bound=spec["training"]["correction_bound_capacity"]).to(device)
            model.load_state_dict(torch.load(directory/arm/"checkpoint.pt",map_location=device,weights_only=False)["model"],strict=True)
            arms[arm]=model
        for name,dataset,ready in (("memory_fit",series.subset("memory"),False),("early_fit",series.subset("calibration_early"),True),("selection",selection,True),("fresh_calibration",calibration,True)):
            pack=trial.pack_data(dataset,trial.predict(base,dataset,spec["seed"],device),expert,contract,p["rules"],True,memory_ready=ready)
            if pack["turbines"].max()>5:
                raise ValueError("Knowledge-held-out labels in train audit")
            row=dict(fold=fold,partition=name,windows=len(pack["truth"]),earliest_target=str(pack["times"].min()),latest_target=str(pack["times"].max()),
                unique_issue_times=len(np.unique(pack["times"][:,0])),reference=trial.model_metrics(pack["reference"],pack["truth"],pack["anchor"],p["rated_power_kw"]),models={})
            for arm,model in arms.items():
                prediction=trial.infer(model,pack,p["rated_power_kw"],device,arm=="WikiContext")
                row["models"][arm]=summarize(pack,prediction,p["rated_power_kw"])
                if arm=="WikiContext":
                    no_memory=trial.infer(model,pack,p["rated_power_kw"],device,False)
                    row["models"]["WikiContext_no_memory_same_weights"]=summarize(pack,no_memory,p["rated_power_kw"])
                    row["wiki_input_mae_gain_kw"]=float((np.abs(no_memory-pack["truth"])-np.abs(prediction-pack["truth"])).mean())
            rows.append(row)
            print(f"[TRAIN_AUDIT] fold={fold} partition={name} numeric_gain={row['models']['NumericContext']['mae_gain_kw']:.4f} wiki_gain={row['models']['WikiContext']['mae_gain_kw']:.4f} memory_gain={row['wiki_input_mae_gain_kw']:.4f}",flush=True)
        del base,arms;gc.collect();torch.cuda.empty_cache()
    trial.write(output,dict(scope="Only original outer training partitions; no outer validation or sealed test inference",source_suite=str(suite.relative_to(ROOT)),protocol_sha256=trial.sha(suite/"protocol.json"),audit_source_sha256=trial.sha(Path(__file__)),rows=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.suite.resolve(),args.output.resolve(),torch.device("cuda:0"))
