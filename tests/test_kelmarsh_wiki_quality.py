import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from tests import test_kelmarsh_episode_wiki as episode_tests
from utils import kelmarsh_wiki_quality as quality


class WikiQualityTests(unittest.TestCase):
    def fixture(self):
        data,contract,memory=episode_tests.EpisodeWikiTests().fixture()
        n=len(data["truth"])
        info=dict(context=copy.deepcopy(data["context"]),state=np.full(n,5),good=np.ones(n,bool),
            last_observed_ns=data["times"][:,0].astype(np.int64)-3600*10**9)
        settings=dict(capacity_kw=2050.,pitch_spread_deg=5.,pitch_pair_tolerance_deg=1.,rotor_low_rpm=1.)
        model=quality.fit_memory(data,info,contract,memory["settings"],settings)
        query=copy.deepcopy(data);query["times"]+=np.timedelta64(30,"D");query.pop("truth");query.pop("turbines")
        qi=copy.deepcopy(info);qi["last_observed_ns"]+=30*24*3600*10**9
        return data,contract,model,query,qi

    def test_consensus_does_not_delete_low_power_or_mutate_sensors(self):
        blades=np.array([[92.,92.1,260.],[1.,20.,50.],[.1,.2,.3],[np.nan,1.,1.]])
        before=blades.copy()
        pitch,repaired,unresolved=quality.sensor_consensus(blades)
        self.assertAlmostEqual(pitch[0],92.1)
        np.testing.assert_array_equal(repaired,[True,False,False,False])
        np.testing.assert_array_equal(unresolved,[False,True,False,True])
        np.testing.assert_array_equal(blades,before)

    def test_ledger_is_causal_and_numerical_series_is_unchanged(self):
        dates=pd.date_range("2016-07-01",periods=74,freq="h").to_numpy()
        raw=np.zeros((74,7));raw[:,0]=4.;raw[:,4]=.2;raw[:,5]=10.;raw[:,-1]=-2.
        series=SimpleNamespace(dates=dates,turbines=np.ones(74,int),raw=raw)
        def context():
            return dict(pitch_mean_div90=np.array([.2/90]),pitch_last_div90=np.array([.2/90]),pitch_change_div90=np.zeros(1))
        ds=SimpleNamespace(series=series,window_starts=np.array([0]),seq_len=72,physical_context=context)
        frame=pd.DataFrame(dict(timestamp_hour_end_utc=dates,turbine_id=1,pitch_a_deg=.1,pitch_b_deg=.2,pitch_c_deg=.3,available_capacity_kw=2050.))
        settings=dict(capacity_kw=2050.,pitch_spread_deg=5.,pitch_pair_tolerance_deg=1.,rotor_low_rpm=1.)
        before=raw.copy()
        a=quality.ObservedQuality(frame,series,settings).context(ds)
        frame.loc[72:,["pitch_c_deg","available_capacity_kw"]]=np.nan
        b=quality.ObservedQuality(frame,series,settings).context(ds)
        for key in ("state","good","last_observed_ns"):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(raw,before)
        self.assertEqual(a["state"].tolist(),[5])

    def test_state_matching_rejects_cross_regime_not_evaluation_rows(self):
        _,contract,model,query,info=self.fixture()
        original=query["reference"].copy()
        out=quality.candidates(model,query,info,contract)
        self.assertTrue(out["physical"][...,0].all())
        info["state"][:]=4  # Same capacity, low rotation: no such source cases.
        other=quality.candidates(model,query,info,contract)
        self.assertFalse(other["overlap"].any())
        np.testing.assert_array_equal(other["prediction"],np.repeat(original[...,None],4,-1))
        np.testing.assert_array_equal(query["reference"],original)
        self.assertEqual(other["prediction"].shape[:2],original.shape)

    def test_label_independent_inference_and_exact_identity_maturity(self):
        data,contract,model,query,info=self.fixture()
        a=quality.candidates(model,query,info,contract)
        query["truth"]=np.full_like(query["reference"],1e9)
        np.testing.assert_array_equal(a["prediction"],quality.candidates(model,query,info,contract)["prediction"])
        invalid=copy.deepcopy(query);invalid["times"][0,0]=np.datetime64("NaT")
        with self.assertRaises(ValueError):quality.candidates(model,invalid,info,contract)
        info["last_observed_ns"]+=2*3600*10**9
        with self.assertRaises(ValueError):quality.candidates(model,query,info,contract)
        info["last_observed_ns"]-=2*3600*10**9
        query["times"]-=np.timedelta64(30,"D");info["last_observed_ns"]-=30*24*3600*10**9
        with self.assertRaises(ValueError):quality.candidates(model,query,info,contract)
        changed=copy.deepcopy(model);changed["groups"][5]["active"]=False
        with self.assertRaises(ValueError):quality.candidates(changed,query,info,contract)
        data["turbines"][0]=6
        with self.assertRaises(ValueError):quality.fit_memory(data,dict(info,context=data["context"]),contract,model["settings"],model["quality_settings"])

    def test_quality_pack_preserves_numerical_inputs_labels_and_cohort(self):
        from scripts import train_kelmarsh_wiki_quality as runner
        data,contract,model,query,info=self.fixture()
        original=dict(query,truth=data["truth"],turbines=data["turbines"],history=np.zeros((150,7,72),np.float32),
            memory=np.zeros((150,2,20),np.float32),overlap=np.zeros((150,2,4),bool),
            change=np.zeros((150,2),np.float32),anchor_ratio=np.full((150,1),.25,np.float32))
        before=copy.deepcopy(original)
        info.update(repaired_hours=np.zeros(150,int),unresolved_hours=np.zeros(150,int))
        ledger=SimpleNamespace(context=lambda _:info)
        with patch.object(runner.old,"pack_data",return_value=copy.deepcopy(original)):
            result,_=runner.quality_pack(None,query["reference"],model,contract,{},ledger,None,donors=False)
        for key in ("history","truth","anchor","reference","times","turbines","change","anchor_ratio"):
            np.testing.assert_array_equal(result[key],before[key])
            np.testing.assert_array_equal(original[key],before[key])
        self.assertEqual(result["memory"].shape,(150,2,20))
        self.assertTrue(result["memory"].any())


if __name__=="__main__":unittest.main()
