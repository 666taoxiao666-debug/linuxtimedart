"""Synthetic contracts only: no target evaluation or formal-run registry."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import torch
from scripts import train_n1ph2 as n


class N1PH2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base,cls.raw,_=n.p2.synthetic_data()
        cls.recipe=n.u.read_json(n.p2.PROTOCOL)
        torch.manual_seed(7)
        cls.model=torch.nn.Sequential(torch.nn.Linear(11,32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,4)).eval()
        cls.stats=(np.zeros(11),np.ones(11))

    def scores(self,values=(10.,9.7,9.6,9.4)):
        return {v:[{'seed':s,'farm':'Kelmarsh','year':2020,'split':'validation','mae':x} for s in (2024,2025,2026)] for v,x in zip(('V0','V1','V2','V3'),values)}

    def test_source_selection_and_tie(self):
        self.assertEqual(n.choose(self.scores())['selected'],'V1')
        self.assertEqual(n.choose(self.scores((10.,9.5,11.,12.)))['selected'],'V1')
        for key,value in [('farm','Penmanshiel'),('year',2021),('split','test'),('target_labels',[1])]:
            records=self.scores(); records['V0'][0][key]=value
            with self.assertRaises(ValueError): n.choose(records)
        records=self.scores(); records['V0'][1]['seed']=2024
        with self.assertRaises(ValueError): n.choose(records)

    def test_selection_freeze_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with self.assertRaises(ValueError): n.require_selection(out,'0'*64)
            n.save(out/'variant_selection.json',n.choose(self.scores()))
            digest=n.u.file_sha256(out/'variant_selection.json')
            self.assertEqual(n.require_selection(out,digest),'V1')
            d=n.u.read_json(out/'variant_selection.json'); d['selected']='V3'
            n.save(out/'variant_selection.json',d)
            with self.assertRaises(ValueError): n.require_selection(out,digest)
            with self.assertRaises(ValueError): n.require_selection(out,n.u.file_sha256(out/'variant_selection.json'))

    def test_channel_and_label_identity(self):
        for channels in (14,25):
            extra=n.features(self.raw,self.model,self.stats,channels)
            ds=n.series(self.base,extra)
            self.assertEqual(len(ds.channels),channels); self.assertEqual(ds.channels[-1],'power')
            self.assertEqual(ds.channels.count('power'),1)
            np.testing.assert_array_equal(ds.raw[:,-1],self.base.base.raw[:,-1])
            np.testing.assert_array_equal(ds.starts,self.base.starts)
            if channels==25: self.assertEqual(ds.channels[8:19],n.n.up.UPSTREAM_FEATURES)
            model,count=n.trainer.build_model(n.config('K-PH',self.recipe,channels))
            self.assertEqual(count,{14:5872,25:9414}[channels])
            self.assertEqual(tuple(model(torch.zeros(2,24,channels)).shape),(2,4,1))

    def test_v0_exact_and_transfer_statistics(self):
        extra=n.features(self.raw,self.model,self.stats,14)
        old=n.n.ph_series(self.base,extra); new=n.series(self.base,extra)
        np.testing.assert_array_equal(old.values,new.values)
        extra=n.features(self.raw,self.model,self.stats,25)
        source=n.series(self.base,extra)
        target=copy.copy(self.base); target.scaler_mean=target.scaler_scale=target.audit=None
        transfer=n.series(target,extra+100,source.audit)
        np.testing.assert_array_equal(transfer.scaler_mean,source.scaler_mean)
        np.testing.assert_array_equal(transfer.scaler_scale,source.scaler_scale)
        # Alter only validation H: fitted imputation and outer scales must not change.
        changed=extra.copy()
        fit_dates=pd.DatetimeIndex(self.base.dates[np.isin(self.base.label_year,self.base.splits['train'])])
        changed.loc[~changed.index.isin(fit_dates),changed.columns[:11]]=999.
        revised=n.series(self.base,changed)
        for key in ('fill','mean','scale'): np.testing.assert_array_equal(source.audit[key],revised.audit[key])

    def test_ph_structure_and_fit_only_statistics(self):
        rng=np.random.default_rng(6)
        ids=[1,2]
        arrays={s:{'H':rng.normal(size=(24,11)).astype('float32'), 'y':rng.normal(size=(24,4)).astype('float32'),
                   'year':np.array([2019]*16+[2020]*8)} for s in ids}
        expected=n.n0.scaler(np.concatenate([arrays[s]['H'][:16] for s in ids]))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); n.train_mix(arrays,ids,'cpu',out)
            saved=torch.load(out/'selected.pt',weights_only=False)
            model,stats=n.n.load_ph(out/'selected.pt',n.u.file_sha256(out/'selected.pt'),{})
            self.assertEqual(sum(p.numel() for p in model.parameters()),1572)
            for a,b in zip(stats,expected): np.testing.assert_array_equal(a,b)
            record=n.u.read_json(out/'trained.json')
            self.assertEqual(record['train_stations'],ids)
            self.assertEqual(record['fit_years'],[2016,2017,2018,2019])
            self.assertEqual(record['selection_year'],2020)

    def test_no_target_loading_before_selection(self):
        source=inspect.getsource(n.run)
        gate=source.index('selected=require_selection(')
        for call in ('p2.load_data(', 'n.reference(', 'n.other_inputs('):
            self.assertGreater(source.index(call),gate)
        self.assertLess(source.index('p3.begin_evaluation('),source.index('trainer.v26run.predict('))

    def test_incomplete_freeze_and_duplicate_eval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            n.save(out/'selection_freeze.json',{})
            with self.assertRaises(ValueError): n.p3.begin_evaluation(out,{'expected_fits':42,'fits':[]},42)
            self.assertFalse((out/'test_evaluation_started.json').exists())
            n.p3.begin_evaluation(out,{'expected_fits':0,'fits':[]},0)
            with self.assertRaises(FileExistsError): n.p3.begin_evaluation(out,{'expected_fits':0,'fits':[]},0)

    def test_synthetic_25_channel_fit_and_warm_smoke(self):
        recipe=copy.deepcopy(self.recipe); recipe['training'].update(max_epochs=1,scheduler_t_max=1)
        ds=n.series(self.base,n.features(self.raw,self.model,self.stats,25))
        fw,vw=ds.windows('train'),ds.windows('validation')
        fw.window_starts=fw.window_starts[:16]; vw.window_starts=vw.window_starts[:16]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            cfg=n.config('K-PH',recipe,25)
            n.p2.fit_one(cfg,recipe,n.trainer.ChannelView(fw,list(range(25))),n.trainer.ChannelView(vw,list(range(25))),2024,'cpu',out/'K',{'synthetic':True})
            warm=torch.load(out/'K/best.pt',weights_only=False)['model']
            n.p2.fit_one(n.config('FT-1-PH',recipe,25),recipe,n.trainer.ChannelView(fw,list(range(25))),n.trainer.ChannelView(vw,list(range(25))),2024,'cpu',out/'FT',{'synthetic':True},warm)
            n.trainer.verify_trained(out/'FT',{'synthetic':True})
            model,_=n.trainer.build_model(cfg)
            model.load_state_dict(torch.load(out/'FT/best.pt',weights_only=False)['model'])
            pred=n.trainer.v26run.predict(model,n.trainer.ChannelView(vw,list(range(25))),2024,'cpu',32)
            self.assertEqual(pred.shape,(16,4)); self.assertTrue(np.isfinite(pred).all())


if __name__=='__main__': unittest.main()
