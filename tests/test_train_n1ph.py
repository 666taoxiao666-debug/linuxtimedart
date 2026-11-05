"""Synthetic N1-PH contract tests; no real observations or evaluation access."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scripts import train_n1ph as n


class N1PHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.protocol=n.u.read_json(n.PROTOCOL)
        cls.recipe=n.u.read_json(n.p2.PROTOCOL)
        cls.base,cls.raw,_=n.p2.synthetic_data()
        torch.manual_seed(7)
        cls.model=torch.nn.Sequential(torch.nn.Linear(11,32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,4)).eval()
        cls.stats=(np.arange(11,dtype=float),np.arange(11,dtype=float)+1)
        cls.features=n.ph_features(cls.raw,cls.model,cls.stats)

    def test_n0_statistics_and_missing(self):
        with torch.no_grad():
            expected=self.model(torch.from_numpy(n.n0.transform(self.raw.to_numpy(),self.stats))).numpy()
        np.testing.assert_allclose(self.features.iloc[:,:4],expected,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(self.features.iloc[:,-1],~np.isfinite(self.raw).all(1))

    def test_source_only_statistics(self):
        source=n.ph_series(self.base,self.features)
        target=copy.copy(self.base)
        # Poison all unused target statistics; transfer must neither access nor copy them into the result.
        target.scaler_mean=target.scaler_scale=target.audit=None
        transferred=n.ph_series(target,self.features,source.audit)
        np.testing.assert_array_equal(transferred.scaler_mean,source.scaler_mean)
        shifted=self.features+100
        transferred=n.ph_series(target,shifted,source.audit)
        np.testing.assert_array_equal(transferred.scaler_mean,source.scaler_mean)
        self.assertEqual(transferred.audit['origin'],'source_only')

    def test_channel_count_power_and_queue(self):
        series=n.ph_series(self.base,self.features)
        self.assertEqual(len(series.channels),14)
        self.assertEqual(series.channels[-1],'power')
        np.testing.assert_array_equal(series.raw[:,-1],self.base.raw[:,-1])
        np.testing.assert_array_equal(series.starts,self.base.starts)
        model,count=n.trainer.build_model(n.configuration('K-PH',self.recipe,self.protocol))
        self.assertEqual(count,5872)
        self.assertEqual(tuple(model(torch.zeros(2,24,14)).shape),(2,4,1))

    def test_future_observation_causality(self):
        dates=pd.date_range('2020-01-01',periods=72,freq='h')
        issue=dates[30:34]
        stations=[{'src_id':i,'distance_km':30.+i,'bearing_deg':i*90.} for i in range(4)]
        obs={i:pd.DataFrame({'wind_speed':5.+np.arange(72)/10+i,'wind_direction':float(i*30),
                            'msl_pressure':1000.+np.arange(72)/50+i},index=dates) for i in range(4)}
        a=n.u.hourly_upstream(obs,stations,issue,1)
        altered={i:d.copy() for i,d in obs.items()}
        for d in altered.values(): d.loc[d.index>=issue[0],:]=999.
        b=n.u.hourly_upstream(altered,stations,issue[:1],1)
        np.testing.assert_array_equal(n.ph_features(a,self.model,self.stats).iloc[0],n.ph_features(b,self.model,self.stats).iloc[0])

    def test_cohort_misalignment_rejected(self):
        w=self.base.windows('development_eval')
        ref={'starts':w.window_starts.copy(),'truth':w.truth_and_anchor()[0],
             'times':w.target_times(),'normal_mask':np.ones(len(w),bool)}
        n.assert_cohort(w,ref,ref['normal_mask'])
        for key in ref:
            bad={k:v.copy() for k,v in ref.items()}
            if key=='times': bad[key][0,0]+=np.timedelta64(1,'h')
            elif key=='normal_mask': bad[key][0]=False
            else: bad[key].flat[0]+=1
            with self.assertRaises(ValueError): n.assert_cohort(w,bad,ref['normal_mask'])

    def test_decision_branches(self):
        def case(zp,ft,ni):
            d={k:{'absolute_error':{'ci95_low':1. if good else -1.},'holm_p':.01,
                  'rmse_difference':{'ci95_low_kw':1.}} for k,good in [('ZP',zp),('ZQ',False),('F1P',ft),('F3P',False)]}
            d['IP']={'absolute_error':{'ci95_low':-1. if ni else -2.}}
            return n.verdict(d)['outcome']
        self.assertEqual(case(True,False,True),'continue')
        self.assertEqual(case(True,True,False),'stop')
        self.assertEqual(case(False,True,False),'partial')
        self.assertEqual(case(False,False,True),'stop')

    def test_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ph.pt'; torch.save({'state':self.model.state_dict(),'stats':self.stats},path)
            with self.assertRaises(ValueError): n.load_ph(path,'0'*64,{})
            model,stats=n.load_ph(path,n.u.file_sha256(path),{})
            np.testing.assert_array_equal(stats[0],self.stats[0])
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_synthetic_fit_smoke(self):
        recipe=copy.deepcopy(self.recipe)
        recipe['training'].update(max_epochs=1,scheduler_t_max=1)
        series=n.ph_series(self.base,self.features)
        fit=series.windows('train'); val=series.windows('validation')
        fit.window_starts=fit.window_starts[:16]; val.window_starts=val.window_starts[:16]
        config=n.configuration('K-PH',recipe,self.protocol)
        with tempfile.TemporaryDirectory() as tmp:
            record=n.p2.fit_one(config,recipe,n.trainer.ChannelView(fit,list(range(14))),
                n.trainer.ChannelView(val,list(range(14))),2024,'cpu',Path(tmp),{'synthetic':True})
            n.trainer.verify_trained(Path(tmp),{'synthetic':True})
            model,_=n.trainer.build_model(config)
            model.load_state_dict(torch.load(Path(tmp)/'best.pt',weights_only=False)['model'])
            prediction=n.trainer.v26run.predict(model,n.trainer.ChannelView(val,list(range(14))),2024,'cpu',32)
            self.assertEqual(prediction.shape,(16,4)); self.assertTrue(np.isfinite(prediction).all())


if __name__=='__main__': unittest.main()
