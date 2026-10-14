import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from models.ContextualWindResidual import ContextualWindResidual, CausalBlock
from scripts.train_kelmarsh_contextual_expert import joint_score, train_arm, clip_relative, fit_certificate, route


class ContextualResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self,n=10):
        rng=np.random.default_rng(2024)
        return dict(history=rng.normal(size=(n,7,72)).astype(np.float32),memory=rng.normal(size=(n,2,20)).astype(np.float32),
            change=np.zeros((n,2),np.float32),anchor_ratio=np.full((n,1),.25,np.float32),
            reference=np.full((n,2),500.),truth=np.full((n,2),520.),anchor=np.full((n,2),500.))

    def test_zero_initialization_numeric_ablation_and_batch_independence(self):
        p=self.inputs()
        model=ContextualWindResidual().eval()
        args=[torch.tensor(p[k]) for k in ("history","change","anchor_ratio","memory")]
        np.testing.assert_array_equal(model(*args,use_memory=True).detach(),np.zeros((10,2)))
        with torch.no_grad():
            model.head.weight.fill_(.05)
        result=model(*args,use_memory=False)
        altered=args[:-1]+[args[-1]*100]
        torch.testing.assert_close(result,model(*altered,use_memory=False))
        torch.testing.assert_close(result[:1],model(*(x[:1] for x in args),use_memory=False))
        self.assertTrue((model(*args,use_memory=True).abs()<=.2).all())

    def test_convolution_has_no_future_dependence_and_training_receives_gradients(self):
        layer=CausalBlock(7,32,4,0.).eval()
        x=torch.randn(3,7,72); other=x.clone();other[...,40:]+=20
        torch.testing.assert_close(layer(x)[...,:40],layer(other)[...,:40])
        model=ContextualWindResidual(dropout=0.)
        p=self.inputs(); args=[torch.tensor(p[k]) for k in ("history","change","anchor_ratio","memory")]
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad();(model(*args,use_memory=True)-.01).square().mean().backward();optimizer.step()
        self.assertGreater(model.encoder[0].conv.weight.grad.abs().sum().item(),0.)

    def test_joint_epoch_selection_cannot_hide_rmse_tradeoff(self):
        baseline=dict(mae=10.,rmse=20.)
        self.assertEqual(joint_score(dict(mae=9.,rmse=21.),baseline),float("inf"))
        self.assertLess(joint_score(dict(mae=9.,rmse=19.),baseline),1.)
        np.testing.assert_allclose(clip_relative(np.array([[1000.]]),np.array([[400.]]),2050.,.2),[[810.]])

    def test_actual_training_completion_resume_and_artifact_identity(self):
        p=self.inputs()
        spec=dict(seed=2024,training=dict(epochs=2,width=32,dropout=.1,correction_bound_capacity=.2,
            learning_rate=.001,weight_decay=.0001,batch_size=8,mix_mse_weight=.2,grad_clip=1.))
        with tempfile.TemporaryDirectory() as temp,patch("scripts.train_kelmarsh_contextual_expert.gpu_available_to_this_process"):
            directory=Path(temp)
            model,done=train_arm("NumericContext",p,p,2050.,spec,directory,dict(test="engineering smoke"),torch.device("cpu"))
            self.assertEqual(done["epochs_completed"],2)
            _,again=train_arm("NumericContext",p,p,2050.,spec,directory,done["identity"],torch.device("cpu"))
            self.assertEqual(done,again)
            with self.assertRaises(ValueError):
                train_arm("NumericContext",p,p,2050.,spec,directory,dict(test="changed"),torch.device("cpu"))

    def test_certificate_maturity_identity_and_fallback_without_query_labels(self):
        p=self.inputs(30)
        p.update(turbines=np.tile([1,2,3],10),overlap=np.ones((30,2,4),bool),
            context=dict(power_bin=np.ones(30,int),joint_bin=np.ones(30,int)),
            times=np.tile(np.array(["2016-08-01T00","2016-08-01T01"],dtype="datetime64[h]"),(30,1)))
        settings=dict(min_windows=24,min_turbines=3,positive_fraction=2/3)
        # Digest requires all causal inputs, but query inference never uses truth.
        policy=fit_certificate(p,p["reference"],p["truth"],dict(checkpoint="a"),settings)
        with self.assertRaises(ValueError):
            route(policy,p,p["reference"],p["truth"],dict(checkpoint="a"))
        q=copy.deepcopy(p);q["times"]+=np.timedelta64(1,"D");q.pop("truth")
        out=route(policy,q,p["reference"],np.full((30,2),520.),dict(checkpoint="a"))
        np.testing.assert_array_equal(out["prediction"],np.full((30,2),520.))
        q["overlap"][:]=False
        np.testing.assert_array_equal(route(policy,q,p["reference"],np.full((30,2),520.),dict(checkpoint="a"))["prediction"],p["reference"])
        with self.assertRaises(ValueError):
            route(policy,q,p["reference"],p["reference"],dict(checkpoint="b"))
        p["turbines"][0]=6
        with self.assertRaises(ValueError):
            fit_certificate(p,p["reference"],p["reference"],dict(checkpoint="a"),settings)


if __name__=="__main__":
    unittest.main()
