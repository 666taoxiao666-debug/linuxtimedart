from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_Epilepsy,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_M4,
    Dataset_PEMS,
    Dataset_Physio,
    Dataset_SDWPF,
    MSLSegLoader,
    PSMSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    UEAloader,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import numpy as np
import random
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'SDWPF': Dataset_SDWPF,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'HAR': Dataset_Physio,
    'EEG': Dataset_Physio,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Epilepsy': Dataset_Epilepsy,
}

forecast_datasets = {
    'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Traffic',
    'Exchange', 'Weather', 'SDWPF', 'ECL', 'ILI', 'm4', 'PEMS03',
    'PEMS04', 'PEMS07', 'PEMS08',
}
classification_datasets = {'UEA', 'HAR', 'EEG', 'Epilepsy'}
anomaly_detection_datasets = {'PSM', 'MSL', 'SMAP', 'SMD', 'SWAT'}
datasets_by_task = {
    'forecast': forecast_datasets,
    'classification': classification_datasets,
    'anomaly_detection': anomaly_detection_datasets,
}


def _seed_data_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_provider(args, flag):
    supported_datasets = datasets_by_task.get(args.downstream_task)
    if supported_datasets is None:
        raise ValueError(f"Unsupported downstream task: {args.downstream_task}")
    if args.data not in supported_datasets:
        valid = ", ".join(sorted(supported_datasets))
        raise ValueError(
            f"Dataset {args.data!r} is not compatible with downstream task "
            f"{args.downstream_task!r}. Valid datasets: {valid}"
        )

    Data = data_dict[args.data]
    split_offset = {"train": 0, "val": 1, "test": 2}.get(flag, 3)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(getattr(args, "seed", 2024)) + split_offset)
    reproducibility_kwargs = {
        "worker_init_fn": _seed_data_worker,
        "generator": loader_generator,
    }

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = getattr(args, 'eval_batch_size', args.batch_size)
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.downstream_task == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.input_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            **reproducibility_kwargs,
        )
        return data_set, data_loader
    elif args.downstream_task == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            **reproducibility_kwargs,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        data_kwargs = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.input_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        if args.data == 'SDWPF':
            data_kwargs.update(
                train_ratio=args.sdwpf_train_ratio,
                val_ratio=args.sdwpf_val_ratio,
                expected_freq=args.sdwpf_expected_freq,
                window_stride=(
                    args.sdwpf_train_stride if flag == 'train' else args.sdwpf_eval_stride
                ),
                filter_abnormal=args.sdwpf_filter_abnormal,
                rated_power=args.rated_power,
                clip_power=args.sdwpf_clip_power,
                circular_wind=args.sdwpf_circular_wind,
                collapse_pitch=args.sdwpf_collapse_pitch,
                keep_curtailment=args.sdwpf_keep_curtailment,
                causal_fill=args.sdwpf_causal_fill,
                split=args.sdwpf_split,
                fold=args.sdwpf_fold,
                n_folds=args.sdwpf_n_folds,
                physics_features=args.sdwpf_physics_features,
                drop_weak_features=args.sdwpf_drop_weak,
            )
        data_set = Data(**data_kwargs)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=bool(args.use_gpu),
            **reproducibility_kwargs,
        )

        print(flag, len(data_set), len(data_loader))
        return data_set, data_loader
