import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import Dataset_SDWPF
from data_provider.sdwpf_features import yaw_degrees, check_angle_contract, channel_prior_vector
from models.TimeDART import Model
from run import build_parser, configure_args, pretrain_signature, load_finetuned_model
from utils.run_tags import experiment_setting
from utils.tools import transfer_weights
from scripts.train_sdwpf_angle_semantics import selected_window_fingerprint


def model_args(mode):
    args = configure_args(build_parser().parse_args([
        "--task_name", "finetune", "--model_id", "SDWPF", "--data", "SDWPF", "--model", "TimeDART",
        "--sdwpf_angle_mode", mode, "--allow_random_init", "--no-use_gpu", "--input_len", "24",
        "--pred_len", "12", "--d_model", "16", "--n_heads", "4", "--e_layers", "1", "--d_ff", "32",
        "--patch_len", "6", "--stride", "6"]))
    args.device = torch.device("cpu")
    return args


class SDWPFAngleTests(unittest.TestCase):
    def test_relative_yaw_is_independent_of_nacelle_heading(self):
        wdir = np.array([5., 5., 0.])
        ndir = np.array([0., 90., 180.])
        np.testing.assert_array_equal(yaw_degrees(wdir, ndir, "relative_wdir"), wdir)
        np.testing.assert_array_equal(yaw_degrees(wdir, ndir), [5., -85., -180.])
        self.assertEqual(np.cos(np.deg2rad(yaw_degrees(wdir, ndir, "relative_wdir")))[2], 1.)

    def test_actual_dataset_preserves_heading_and_labels_across_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tiny.csv"
            n = 80
            df = pd.DataFrame(dict(date=pd.date_range("2023-01-01", periods=n, freq="10min"),
                TurbID=np.ones(n), Wspd=np.full(n, 6.), Wdir=np.full(n, 5.), Ndir=np.arange(n)*9.,
                Etmp=np.full(n, 20.), Itmp=np.full(n, 25.), Pab1=np.ones(n), Pab2=np.ones(n),
                Pab3=np.ones(n), Prtv=np.zeros(n), power=300.+np.arange(n)))
            # Union is seven points although each individual gap is short.
            df.loc[20:22, "Wdir"] = np.nan
            df.loc[23:26, "Ndir"] = np.nan
            df.to_csv(path,index=False)
            common=dict(root_path=directory,data_path=path.name,flag="train",size=[4,0,2],
                        features="MS",scale=False,train_ratio=.8,val_ratio=.1)
            old=Dataset_SDWPF(**common)
            new=Dataset_SDWPF(**common,angle_mode="relative_wdir")
            np.testing.assert_array_equal(old.dates,new.dates)
            np.testing.assert_array_equal(old.window_starts,new.window_starts)
            np.testing.assert_array_equal(old.data_y[:,-1],new.data_y[:,-1])
            self.assertEqual(len(new.feature_columns),8)
            self.assertIn("Ndir_sin",new.feature_columns)
            self.assertNotIn("Wdir_sin",new.feature_columns)
            np.testing.assert_allclose(new.data_x[:,new.feature_columns.index("yaw_sin")],np.sin(np.deg2rad(5)),atol=1e-6)
            np.testing.assert_allclose(new.data_x[:,new.feature_columns.index("yaw_cos")],np.cos(np.deg2rad(5)),atol=1e-6)
            expected=np.sin(np.deg2rad(np.arange(n)*9.))
            np.testing.assert_allclose(new.data_x[:,new.feature_columns.index("Ndir_sin")],
                                       np.delete(expected,np.arange(20,27)),atol=1e-6)
            np.testing.assert_allclose(channel_prior_vector(new.feature_columns,angle_mode="relative_wdir"),
                                       channel_prior_vector(old.feature_columns))

    def test_window_audit_ignores_sealed_suffix_but_detects_selected_label_changes(self):
        dataset=SimpleNamespace(window_starts=np.array([0,2]), seq_len=2,pred_len=2,
            dates=pd.date_range("2023-01-01",periods=10,freq="10min").to_numpy(),
            turbines=np.ones(10,dtype=np.int16),data_y=np.arange(10,dtype=np.float32).reshape(-1,1),
            available_mask=np.ones(10,dtype=bool),
            scaler=SimpleNamespace(mean_=np.array([0.]),scale_=np.array([1.])))
        before=selected_window_fingerprint(dataset)
        dataset.data_y[8:,0]=1234
        self.assertEqual(before,selected_window_fingerprint(dataset))
        dataset.data_y[3,0]=1234
        self.assertNotEqual(before,selected_window_fingerprint(dataset))

    def test_heading_circular_wrap_and_legacy_defaults(self):
        values=np.array([5.,365.,-355.])
        np.testing.assert_allclose(np.cos(np.deg2rad(values)),np.full(3,np.cos(np.deg2rad(5.))))
        self.assertEqual(build_parser().parse_args(["--task_name","finetune","--model_id","SDWPF",
                         "--model","TimeDART","--data","SDWPF"]).sdwpf_angle_mode,
                         "legacy_difference")

    def test_checkpoint_contract_rejects_shape_compatible_wrong_physics(self):
        args=model_args("relative_wdir")
        with self.assertRaisesRegex(RuntimeError,"angle checkpoint mismatch"):
            check_angle_contract({},args)
        source=dict(sdwpf_angle_mode="relative_wdir",feature_columns=args.feature_columns)
        check_angle_contract(source,args)
        with self.assertRaisesRegex(RuntimeError,"feature order"):
            check_angle_contract(dict(source,feature_columns=list(reversed(args.feature_columns))),args)

    def test_actual_transfer_refuses_old_and_accepts_matching_pretrain(self):
        model=Model(model_args("relative_wdir"))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.pth"
            torch.save({"model_state_dict":model.state_dict()},path)
            with self.assertRaisesRegex(RuntimeError,"angle checkpoint mismatch"):
                transfer_weights(str(path),model)
            torch.save({"model_state_dict":model.state_dict(),"sdwpf_angle_mode":"relative_wdir",
                        "feature_columns":model.feature_columns,"consistent_physics_norm":False},path)
            transfer_weights(str(path),model)

    def test_actual_finetune_loader_refuses_old_manifest(self):
        args=model_args("relative_wdir"); model=Model(args)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.pth"; torch.save(model.state_dict(),path)
            (path.parent/"run_manifest.json").write_text(json.dumps({"args":{
                "sdwpf_angle_mode":"legacy_difference","feature_columns":model_args("legacy_difference").feature_columns}}))
            exp=SimpleNamespace(args=args,model=model,device=torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError,"angle checkpoint mismatch"):
                load_finetuned_model(exp,str(path))

    def test_cli_and_discovery_separate_angle_definitions(self):
        old,new=model_args("legacy_difference"),model_args("relative_wdir")
        self.assertNotEqual(pretrain_signature(old),pretrain_signature(new))
        self.assertNotEqual(experiment_setting(old,0),experiment_setting(new,0))
        with self.assertRaisesRegex(ValueError,"multivariate circular physics"):
            configure_args(build_parser().parse_args(["--task_name","finetune","--model_id","SDWPF",
                "--data","SDWPF","--model","TimeDART","--sdwpf_angle_mode","relative_wdir","--no-sdwpf_physics_features"]))


if __name__=="__main__":
    unittest.main()
