import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
from scripts import train_e1_baselines as e


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.protocol=e.u.read_json(e.ROOT/'configs/e1_baselines_protocol.json')

    def test_shapes_gradients_and_external_channel_path(self):
        for family,channels in self.protocol['arms'].items():
            for channel in channels:
                for d in (64,128):
                    cfg=e.configuration(self.protocol,family,channel,d)
                    model,count=e.build_model(cfg);model.eval()
                    x=torch.randn(2,24,cfg['channels'])
                    y=model(x)
                    self.assertEqual(y.shape[:2],(2,4)); self.assertTrue(torch.isfinite(y).all())
                    y[...,-1].sum().backward()
                    if family in ('TimeXer','iTransformer'):
                        changed=x.clone();changed[...,:-1]=torch.randn_like(changed[...,:-1])
                        self.assertGreater((model(changed)[...,-1]-y[...,-1]).abs().max().item(),0.)
                    self.assertEqual(count,sum(p.numel() for p in model.parameters()))

    def test_only_validation_selects_configuration(self):
        records=[{'config':{'args':{'d_model':d}},'record':{'validation_best':{'mae':score}},'test_mae':test}
                 for d,score,test in ((64,10,1000),(128,11,0))]
        self.assertEqual(e.choose(records)['args']['d_model'],64)
        records[1]['record']['validation_best']['mae']=10
        self.assertEqual(e.choose(records)['args']['d_model'],64)

    def test_factory_rejects_parameter_identity_change(self):
        cfg=e.configuration(self.protocol,'TimeXer','upstream',64)
        cfg['expected_parameters']+=1
        with self.assertRaises(ValueError):e.build_model(cfg)

    def test_views_preserve_window_labels(self):
        series,_,_=e.p2.synthetic_data();w=series.windows('validation')
        for channel in ('power','last10','upstream'):
            v=e.view(w,channel)
            self.assertIs(v.window_starts,w.window_starts)
            torch.testing.assert_close(torch.as_tensor(v[0][1][:,-1]),torch.as_tensor(w[0][1][:,-1]))

class RecoveryTests(unittest.TestCase):
    def test_missing_reference_fails_before_evaluation_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); windows=MagicMock()
            with patch.object(e,'reference',side_effect=FileNotFoundError('reference missing')):
                with self.assertRaises(FileNotFoundError):
                    e.evaluate(out,out,{'folds':[{'id':'Kelmarsh'}]},'cpu',{'Kelmarsh':windows},{},[],out,{})
            self.assertFalse((out/'evaluation_started.json').exists())

    def test_changed_frozen_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'freeze.json').write_text('changed')
            with self.assertRaises((ValueError,RuntimeError)):
                e.checked_recovery_records(root,{'source_run':'.','source_metadata':{'freeze.json':'0'*64}})

    def test_recovery_never_calls_fit_or_choose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'original';source.mkdir();out=root/'new';out.mkdir()
            protocol={'folds':[]};e.save(source/'launch.json',{'protocol':protocol});e.save(source/'input_identity.json',{})
            recovery={'output_run':'new','source_metadata':{}}
            with patch.object(e,'checked_recovery_records',return_value=(source,{},[])), patch.object(e,'load_data',return_value=({},{})), patch.object(e,'evaluate') as evaluate, patch.object(e,'fit',side_effect=AssertionError('training forbidden')), patch.object(e,'choose',side_effect=AssertionError('selection forbidden')):
                e.recover_evaluation(root,out,protocol,'cpu',recovery)
                evaluate.assert_called_once()
            with self.assertRaises(ValueError): e.recover_evaluation(root,root/'other',protocol,'cpu',recovery)

if __name__=='__main__':unittest.main()
