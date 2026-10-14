"""Wiki-only observed sensor consensus and operating-state case matching.

Nothing here edits a numerical Dataset, target, scaler, or evaluation cohort.
Low power and stopped rotation are regimes, not automatically bad labels.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from utils import kelmarsh_episode_wiki as episode
from utils import kelmarsh_hourly_wiki as legacy


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sensor_consensus(blades, spread_limit=5., pair_tolerance=1.):
    blades=np.asarray(blades,float)
    if blades.shape[-1]!=3:
        raise ValueError("Three contemporaneous pitch sensors required")
    finite=np.isfinite(blades).all(-1)
    spread=blades.max(-1)-blades.min(-1)
    pair=np.minimum.reduce([np.abs(blades[...,0]-blades[...,1]),np.abs(blades[...,0]-blades[...,2]),np.abs(blades[...,1]-blades[...,2])])
    repaired=finite&(spread>spread_limit)&(pair<=pair_tolerance)
    unresolved=(~finite)|((spread>spread_limit)&~repaired)
    pitch=np.where(repaired,np.median(blades,axis=-1),blades.mean(-1))
    return pitch,repaired,unresolved


class ObservedQuality:
    def __init__(self, frame, series, settings):
        frame=frame.copy()
        frame["date"]=pd.to_datetime(frame.timestamp_hour_end_utc,utc=True).dt.tz_localize(None)
        frame=frame[frame.date<=series.dates.max()]
        if frame.duplicated(["turbine_id","date"]).any():
            raise ValueError("Duplicate quality ledger identity")
        index=pd.MultiIndex.from_arrays([series.turbines,series.dates],names=["turbine_id","date"])
        rows=frame.set_index(["turbine_id","date"]).reindex(index)
        self.blades=rows[["pitch_a_deg","pitch_b_deg","pitch_c_deg"]].to_numpy(float)
        self.available=rows.available_capacity_kw.to_numpy(float)
        self.series,self.settings=series,settings

    def context(self,dataset):
        rows=dataset.window_starts[:,None]+dataset.seq_len-12+np.arange(12)
        pitch,repaired,unresolved=sensor_consensus(self.blades[rows],self.settings["pitch_spread_deg"],self.settings["pitch_pair_tolerance_deg"])
        # Invalid contexts are rejected below; keep unused key placeholders finite.
        pitch=np.where(np.isfinite(pitch),pitch,0.)
        context={key:value.copy() for key,value in dataset.physical_context().items()}
        context.update(pitch_mean_div90=pitch.mean(1)/90.,pitch_last_div90=pitch[:,-1]/90.,
            pitch_change_div90=(pitch[:,6:].mean(1)-pitch[:,:6].mean(1))/90.)
        last=rows[:,-1]
        available,rotor=self.available[last],self.series.raw[last,5]
        good=np.isfinite(available)&(available>=0)&np.isfinite(rotor)&(rotor>=0)&~unresolved.any(1)
        good&=(self.series.raw[rows,0]>=0).all(1)
        # Current observed capacity is categorical; it is never future capacity.
        full=available>=self.settings["capacity_kw"]-1e-6
        availability=np.where(available<=0,0,np.where(full,2,1))
        rotation=(rotor>=self.settings["rotor_low_rpm"]).astype(int)
        state=np.where(good,availability*2+rotation,-1).astype(int)
        return dict(context=context,state=state,good=good,repaired_hours=repaired.sum(1),
            unresolved_hours=unresolved.sum(1),last_observed_ns=self.series.dates[last].astype("datetime64[ns]").astype(np.int64))


def fit_memory(block,quality,contract,settings,quality_settings):
    ids=legacy.history._source_ids(block["turbines"],contract)
    x=legacy.keys(quality["context"],block["anchor"],contract["rated_power_kw"],settings["use_controls"])
    labels=(block["truth"]-block["anchor"])/contract["rated_power_kw"]
    clock=np.asarray(block["times"]).astype("datetime64[ns]")
    if not np.isfinite(labels).all() or clock.shape!=labels.shape or np.isnat(clock).any():
        raise ValueError("Invalid case outcome or target clock")
    times=clock.astype(np.int64)
    if np.any(quality["last_observed_ns"]>times[:,0]-3600*10**9):
        raise ValueError("Future source operating metadata")
    groups=[]
    for k,event in enumerate(legacy.EVENTS):
        for state in range(6):
            mask=block["events"][:,k]&(quality["state"]==state)
            xx,unit=x[mask],ids[mask]
            unique,counts=np.unique(unit,return_counts=True)
            active=len(xx)>=settings["min_fit_windows"] and len(unique)>=settings["min_neighbor_turbines"]+1 and len(xx)-counts.max(initial=0)>=settings["neighbors"]
            row=dict(event=k,state=state,active=bool(active),windows=int(mask.sum()),turbines=int(len(unique)))
            if active:
                mean,scale=xx.mean(0),xx.std(0);scale=np.where(scale>1e-6,scale,1.)
                z=(xx-mean)/scale
                _,distance,diversity=legacy.neighbors._neighbors(z,z,unit,settings["neighbors"],excluded_turbines=unit)
                supported=np.isfinite(distance)&(diversity>=settings["min_neighbor_turbines"])
                row.update(keys=xx.tolist(),values=labels[mask].tolist(),units=unit.tolist(),mean=mean.tolist(),scale=scale.tolist(),
                    radius=float(np.quantile(distance[supported],settings["support_quantile"])) if supported.any() else 0.,
                    available_ns=int(times[mask].max()),weather_blocks=(times[mask,0]//(24*3600*10**9)).tolist())
            groups.append(row)
    model=dict(groups=groups,contract=contract,settings=settings,quality_settings=quality_settings,
        implementation_sha256=implementation_digest(),readout_sha256=episode.implementation_digest(),
        coordinate_sha256=legacy.implementation_digest(),neighbor_sha256=legacy.neighbors.implementation_digest(),
        labels_modified=False,uncertain_case_handling="Reject ambiguous sensor context; match observed capacity and rotation; retain evaluation windows")
    model["expert_id"]=legacy.history.digest(model)
    return model


def candidates(model,query,quality,contract):
    expected={k:v for k,v in model.items() if k!="expert_id"}
    if (model["expert_id"]!=legacy.history.digest(expected) or model["contract"]!=contract or
        model["implementation_sha256"]!=implementation_digest() or model["readout_sha256"]!=episode.implementation_digest() or
        model["coordinate_sha256"]!=legacy.implementation_digest() or model["neighbor_sha256"]!=legacy.neighbors.implementation_digest()):
        raise ValueError("Quality expert identity changed; old benefit evidence is invalid")
    reference=np.asarray(query["reference"],float)
    clock=np.asarray(query["times"]).astype("datetime64[ns]")
    if clock.shape!=reference.shape or np.isnat(clock).any():
        raise ValueError("Invalid query target clock")
    origins=clock[:,0].astype(np.int64)-3600*10**9
    if np.any(quality["last_observed_ns"]>origins):
        raise ValueError("Future operating metadata at issue time")
    cfg,capacity=model["settings"],contract["rated_power_kw"]
    x=legacy.keys(quality["context"],query["anchor"],capacity,cfg["use_controls"])
    shape=(*reference.shape,len(legacy.EVENTS))
    prediction=np.repeat(reference[...,None],shape[-1],axis=-1)
    alpha,variance=np.zeros(shape),np.zeros(shape)
    physical,overlap=np.zeros(shape,bool),np.zeros(shape,bool)
    for group in model["groups"]:
        k=group["event"]
        mask=query["events"][:,k]&(quality["state"]==group["state"])
        if not group["active"] or not mask.any():
            continue
        if np.any(origins[mask]<group["available_ns"]):
            raise ValueError("Case outcomes not yet mature")
        mean,scale=np.asarray(group["mean"]),np.asarray(group["scale"])
        unit,values=np.asarray(group["units"]),np.asarray(group["values"])
        idx,distance,diversity=legacy.neighbors._neighbors((x[mask]-mean)/scale,(np.asarray(group["keys"])-mean)/scale,unit,cfg["neighbors"])
        physical[mask,:,k]=True
        overlap[mask,:,k]=((distance<=group["radius"])&(diversity>=cfg["min_neighbor_turbines"]))[:,None]
        blocks=np.asarray(group["weather_blocks"])[idx]
        for h in range(reference.shape[1]):
            out=episode.readout(values[idx,h],unit[idx],blocks,(reference[mask,h]-query["anchor"][mask,h])/capacity,cfg["max_scale"],2/3)
            prediction[mask,h,k]=query["anchor"][mask,h]+capacity*out["prediction"]
            alpha[mask,h,k],variance[mask,h,k]=out["alpha"],capacity**2*out["variance"]
    return dict(prediction=prediction,alpha=alpha,variance_kw2=variance,physical=physical,overlap=overlap)
