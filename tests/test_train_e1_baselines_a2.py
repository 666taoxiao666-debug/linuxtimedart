import unittest,copy
import torch
from scripts import train_e1_baselines_a2 as a

class A2Tests(unittest.TestCase):
    def test_norm_switch_changes_computation_without_parameter_change(self):
        p=a.u.read_json(a.ROOT/'configs/e1_baselines_protocol_a2.json');torch.set_num_threads(2)
        for family in p['arms']:
            cfg=a.e.configuration(p,family,'upstream',64)
            true,n=a.build_model(cfg);cfg=copy.deepcopy(cfg);cfg['args']['use_norm']=False;false,m=a.build_model(cfg)
            self.assertEqual(n,m);false.load_state_dict(true.state_dict(),strict=True);true.eval();false.eval()
            x=torch.randn(2,24,21)+3
            self.assertGreater((true(x)-false(x)).abs().max().item(),1e-4)
            self.assertEqual(false(x).shape,true(x).shape)
            false(x).sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in false.parameters() if p.grad is not None))
    def test_selection_uses_mae_then_true_then_width(self):
        rs=[{'config':{'args':{'use_norm':n,'d_model':d}},'record':{'validation_best':{'mae':1.}}} for n,d in [(False,64),(True,128),(True,64)]]
        self.assertEqual(a.choose(rs)['args'],{'use_norm':True,'d_model':64})
        rs[0]['record']['validation_best']['mae']=.9;self.assertFalse(a.choose(rs)['args']['use_norm'])
    def test_true_factory_is_original(self):
        p=a.u.read_json(a.ROOT/'configs/e1_baselines_protocol_a2.json');cfg=a.e.configuration(p,'iTransformer','last10',64)
        self.assertIs(type(a.build_model(cfg)[0]),type(a.e.build_model(cfg)[0]))
if __name__=='__main__':unittest.main()
