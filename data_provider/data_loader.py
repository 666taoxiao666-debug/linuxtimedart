import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import pickle
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.utils import load_data
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def _interpolate_short_missing_runs(series, max_gap=6):
    """Interpolate only bounded NaN runs whose total length is <= max_gap.

    pandas ``limit=max_gap, limit_direction='both'`` may fill max_gap values
    from each side, so a 12-point gap can be filled when max_gap is 6.  This
    helper measures the complete run first and therefore enforces the intended
    limit exactly.
    """
    series = series.copy()
    missing = series.isna()
    if not missing.any():
        return series

    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    interpolated = series.interpolate(method="linear", limit_area="inside")
    fillable = missing & (run_length <= int(max_gap)) & interpolated.notna()
    series.loc[fillable] = interpolated.loc[fillable]
    return series


def _stratified_classification_train_val_indices(
    labels, val_ratio=0.2, seed=2024
):
    """Create deterministic, disjoint splits for labelled samples only.

    This helper must not be used for forecasting sequences; forecasting
    datasets split their timelines chronologically before constructing windows.
    """
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
        val_parts.append(indices[:val_count])
        train_parts.append(indices[val_count:])
    train_indices = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val_indices = np.sort(np.concatenate(val_parts)).astype(np.int64)
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Unable to create non-empty stratified train/validation splits")
    return train_indices, val_indices


# UEA datasets
class Dataset_Epilepsy(Dataset):
    def __init__(self, root_path, file_list=None, limit_size=None, flag=None, **kwargs):
        self.kwargs = kwargs
        self.root_path = root_path
        self.feature_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        
        # use all features
        # self.class_names
        # self.min_val, self.max_val
        
        # pre_process
        # normalizer = Normalizer()
        # self.feature_df = normalizer.normalize(self.feature_df)
        # print(self.length)
        
    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        data_p, label_p = None, None
        if flag in {'train', 'val'}:
            data_p = os.path.join(root_path, 'train_d.npy')
            label_p = os.path.join(root_path, 'train_l.npy')
        elif flag == 'test':
            data_p = os.path.join(root_path, 'test_d.npy')
            label_p = os.path.join(root_path, 'test_l.npy')
        else:
            raise Exception("No flag: {}, should be in 'train', 'val' or 'test'".format(flag))
        
        datas, labels = np.load(data_p), np.load(label_p)
        if flag in {'train', 'val'}:
            train_indices, val_indices = (
                _stratified_classification_train_val_indices(labels)
            )
            selected = train_indices if flag == 'train' else val_indices
            datas, labels = datas[selected], labels[selected]
        
        # normalizer = Normalizer(norm_type='minmax', data_type='numpy', axis=(0,1), \
        #             min_val=self.kwargs['min_val'], max_val=self.kwargs['max_val'])
        
        # datas = normalizer.normalize(datas)
        # self.min_val, self.max_val = normalizer.min_val, normalizer.max_val 
        
        # labels = np.expand_dims(labels, axis=1)
        
        self.length = datas.shape[0]
        self.max_seq_len = datas.shape[1]
        self.class_names = np.unique(labels)
        
        print(datas.shape, labels.shape)
        print(self.class_names)

        return datas, labels 
    
    def __getitem__(self, ind):
        data_x = torch.from_numpy(self.feature_df[ind])
        data_y = torch.tensor(self.labels_df[ind], dtype=torch.long)
        x_mark = torch.zeros(data_x.shape)
        y_mark = torch.zeros(data_y.shape)
        return data_x, data_y, x_mark, y_mark

    def __len__(self):
        return self.length


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Physio(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False
                 ):
        
        self.flag = flag
        self.root_path = root_path
        self.use_time_features = False
        
        self.__read_data__()
        
    def __read_data__(self):
        if self.flag in {'train', 'val'}:
            data_path = self.root_path + '/samples_train.pkl'
        else:
            data_path = self.root_path + '/samples_test.pkl'
        
        #加载数据
        with open(data_path, 'rb') as file:
            samples = pickle.load(file)
            
        datas, labels = [], []
        for _, data, label in samples:
            datas.append(data)
            labels.append(label)

        if self.flag in {'train', 'val'}:
            train_indices, val_indices = (
                _stratified_classification_train_val_indices(labels)
            )
            selected = train_indices if self.flag == 'train' else val_indices
            datas = [datas[index] for index in selected]
            labels = [labels[index] for index in selected]
        
        print(len(set(labels)))

        self.data_x = [torch.tensor(data, dtype=torch.float32) for data in datas]
        self.data_y = [torch.tensor(label, dtype=torch.long) for label in labels]
        
        print(self.data_x[0].shape, self.data_y[0].shape)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, item):
        # Just index the pre-converted tensors
        # print(self.data_x[item].shape, self.data_y[item].shape)
        # exit(0)
        x_mark = torch.zeros(self.data_x[item].shape)
        y_mark = torch.zeros(self.data_y[item].shape)
        return self.data_x[item], self.data_y[item], x_mark, y_mark


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        print(type(self.data_x))
        print(self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        s_begin = index
        # if self.set_type == 0:
        #     if index % 2 == 1:
        #         s_begin = index - 1

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        # if self.set_type == 0:
        #     if index % 2 == 1:
        #         s_begin = index - 1

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SDWPF(Dataset):
    """SDWPF windows split by time and kept inside one turbine.

    The public SDWPF table is organised as one block per turbine.  Treating the
    complete CSV as one sequence creates windows that cross turbine boundaries
    and turns a row-wise 70/10/20 split into a turbine split.  This dataset
    sorts each turbine by timestamp, applies global time cut-offs, and builds
    windows only inside gap-free turbine segments.

    ``TurbID`` and ``Day`` are metadata and are never model inputs.  For ``M``
    and ``MS`` the requested target is placed last, matching the framework's
    ``f_dim = -1`` convention for ``MS``.
    """

    _cache = {}

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="MS",
        data_path="sdwpf_fixed.csv",
        target="power",
        scale=True,
        timeenc=0,
        freq="10min",
        seasonal_patterns=None,
        train_ratio=0.7,
        val_ratio=0.1,
        expected_freq="10min",
        window_stride=1,
        filter_abnormal=True,
    ):
        del seasonal_patterns
        if size is None:
            size = [336, 0, 96]
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be one of: train, val, test")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be one of: M, S, MS")
        if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
            raise ValueError("train_ratio and val_ratio must be in (0, 1)")
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be smaller than 1")

        self.seq_len, self.label_len, self.pred_len = map(int, size)
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = bool(scale)
        self.timeenc = timeenc
        self.freq = freq
        self.expected_freq = expected_freq
        self.window_stride = max(1, int(window_stride))

        file_path = os.path.abspath(os.path.join(root_path, data_path))
        cache_key = (
            file_path,
            features,
            target,
            self.scale,
            timeenc,
            freq,
            float(train_ratio),
            float(val_ratio),
            expected_freq,
            bool(filter_abnormal),
        )
        if cache_key not in self._cache:
            self._cache[cache_key] = self._prepare_data(
                file_path=file_path,
                features=features,
                target=target,
                scale=self.scale,
                timeenc=timeenc,
                freq=freq,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                expected_freq=expected_freq,
                filter_abnormal=filter_abnormal,
            )

        prepared = self._cache[cache_key]
        self.data_x = prepared["data"]
        self.data_y = prepared["data"]
        self.data_stamp = prepared["stamp"]
        self.dates = prepared["dates"]
        self.turbines = prepared["turbines"]
        self.segments = prepared["segments"]
        self.train_cutoff = prepared["train_cutoff"]
        self.val_cutoff = prepared["val_cutoff"]
        self.scaler = prepared["scaler"]
        self.feature_columns = prepared["feature_columns"]
        self.window_starts = self._build_window_starts()

        if len(self.window_starts) == 0:
            raise ValueError(
                f"No valid SDWPF windows for {flag=}. Check split ratios, "
                "window lengths, expected frequency, and abnormal-value filtering."
            )

    @staticmethod
    def _prepare_data(
        file_path,
        features,
        target,
        scale,
        timeenc,
        freq,
        train_ratio,
        val_ratio,
        expected_freq,
        filter_abnormal,
    ):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"SDWPF data file not found: {file_path}")

        header = pd.read_csv(file_path, nrows=0).columns.tolist()
        required = {"date", "TurbID", target}
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"SDWPF file is missing required columns: {missing}")

        metadata = {"date", "TurbID", "Day"}
        candidate_features = [c for c in header if c not in metadata and c != target]
        feature_columns = [target] if features == "S" else candidate_features + [target]

        filter_columns = {"Wspd", "Wdir", "Ndir", "Pab1", "Pab2", "Pab3", target}
        usecols = [
            c
            for c in header
            if c in {"date", "TurbID"} or c in feature_columns or c in filter_columns
        ]
        dtype = {c: "float32" for c in usecols if c != "date"}
        # TurbID is read as float first so malformed values can be removed cleanly.
        df = pd.read_csv(file_path, usecols=usecols, dtype=dtype)
        original_rows = len(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        numeric_columns = [c for c in usecols if c != "date"]
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["date", "TurbID"])
        df["TurbID"] = df["TurbID"].astype(np.int16)
        df = (
            df.sort_values(["TurbID", "date"], kind="mergesort")
            .drop_duplicates(["TurbID", "date"], keep="last")
            .reset_index(drop=True)
        )

        repaired_values = 0
        if filter_abnormal:
            invalid_masks = {}
            if {"Wspd", target}.issubset(df.columns):
                wind = df["Wspd"]
                power = df[target]
                invalid_power = (power < 0) & (wind > 2.5)
                invalid_power |= (power == 0) & (wind > 5.0)
                pitch_cols = [c for c in ("Pab1", "Pab2", "Pab3") if c in df]
                if pitch_cols:
                    invalid_power |= (
                        (power > 0)
                        & (wind > 2.5)
                        & (df[pitch_cols].mean(axis=1) > 89)
                    )
                invalid_masks[target] = invalid_power
                invalid_masks["Wspd"] = ~wind.between(0, 40)
            if "Wdir" in df:
                invalid_masks["Wdir"] = ~df["Wdir"].between(-180, 180)
            if "Ndir" in df:
                invalid_masks["Ndir"] = ~df["Ndir"].between(-720, 720)
            for temperature in ("Etmp", "Itmp"):
                if temperature in df:
                    invalid_masks[temperature] = ~df[temperature].between(-50, 80)

            for column, invalid in invalid_masks.items():
                if column in feature_columns:
                    repaired_values += int(invalid.sum())
                    df.loc[invalid, column] = np.nan

            # Repair only short bad runs and never interpolate across a real
            # timestamp gap or a turbine boundary.
            raw_dates = df["date"].to_numpy(dtype="datetime64[ns]")
            raw_turbines = df["TurbID"].to_numpy(dtype=np.int16, copy=False)
            expected_delta = pd.to_timedelta(expected_freq).to_timedelta64()
            raw_boundary = np.empty(len(df), dtype=bool)
            raw_boundary[0] = True
            if len(df) > 1:
                raw_boundary[1:] = (raw_turbines[1:] != raw_turbines[:-1]) | (
                    (raw_dates[1:] - raw_dates[:-1]) != expected_delta
                )
            segment_ids = np.cumsum(raw_boundary) - 1
            df[feature_columns] = df.groupby(segment_ids, sort=False)[
                feature_columns
            ].transform(
                lambda series: _interpolate_short_missing_runs(series, max_gap=6)
            )

        df = df.dropna(subset=feature_columns).reset_index(drop=True)
        if df.empty:
            raise ValueError("No SDWPF rows remain after validation and filtering")

        dates = df["date"].to_numpy(dtype="datetime64[ns]")
        turbines = df["TurbID"].to_numpy(dtype=np.int16, copy=True)
        unique_dates = np.unique(dates)
        if len(unique_dates) < 3:
            raise ValueError("SDWPF data must contain at least three unique timestamps")
        train_pos = min(max(1, int(len(unique_dates) * train_ratio)), len(unique_dates) - 2)
        val_pos = min(
            max(train_pos + 1, int(len(unique_dates) * (train_ratio + val_ratio))),
            len(unique_dates) - 1,
        )
        train_cutoff = unique_dates[train_pos]
        val_cutoff = unique_dates[val_pos]

        values = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        scaler = StandardScaler()
        if scale:
            train_rows = dates < train_cutoff
            if not np.any(train_rows):
                raise ValueError("The SDWPF training split is empty")
            scaler.fit(values[train_rows])
            values = scaler.transform(values).astype(np.float32, copy=False)
        else:
            scaler.fit(np.zeros((1, len(feature_columns)), dtype=np.float32))
            scaler.mean_ = np.zeros(len(feature_columns), dtype=np.float64)
            scaler.scale_ = np.ones(len(feature_columns), dtype=np.float64)
            scaler.var_ = np.ones(len(feature_columns), dtype=np.float64)

        date_index = pd.DatetimeIndex(dates)
        if timeenc == 0:
            stamp = np.column_stack(
                [
                    date_index.month,
                    date_index.day,
                    date_index.dayofweek,
                    date_index.hour,
                    date_index.minute,
                ]
            ).astype(np.float32)
        else:
            stamp = time_features(date_index, freq=freq).transpose(1, 0).astype(np.float32)

        expected_delta = pd.to_timedelta(expected_freq).to_timedelta64()
        boundary = np.empty(len(df), dtype=bool)
        boundary[0] = True
        if len(df) > 1:
            boundary[1:] = (turbines[1:] != turbines[:-1]) | (
                (dates[1:] - dates[:-1]) != expected_delta
            )
        segment_starts = np.flatnonzero(boundary)
        segment_ends = np.r_[segment_starts[1:], len(df)]
        segments = np.column_stack([segment_starts, segment_ends]).astype(np.int64)

        removed = original_rows - len(df)
        print(
            "SDWPF prepared: "
            f"{len(df):,} rows, {len(segments):,} continuous segments, "
            f"{removed:,} unresolved/duplicate rows removed, "
            f"{repaired_values:,} abnormal values marked for short-gap repair, "
            f"features={feature_columns}"
        )
        return {
            "data": values,
            "stamp": stamp,
            "dates": dates,
            "turbines": turbines,
            "segments": segments,
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }

    def _build_window_starts(self):
        total_len = self.seq_len + self.pred_len
        starts = []
        for segment_start, segment_end in self.segments:
            last_start = int(segment_end) - total_len
            if last_start < segment_start:
                continue
            candidates = np.arange(
                int(segment_start), last_start + 1, self.window_stride, dtype=np.int64
            )
            target_start = self.dates[candidates + self.seq_len]
            target_end = self.dates[candidates + total_len - 1]
            if self.flag == "train":
                keep = target_end < self.train_cutoff
            elif self.flag == "val":
                keep = (target_start >= self.train_cutoff) & (target_end < self.val_cutoff)
            else:
                keep = target_start >= self.val_cutoff
            if np.any(keep):
                starts.append(candidates[keep])
        if not starts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(starts)

    def __getitem__(self, index):
        s_begin = int(self.window_starts[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.window_starts)

    def inverse_transform(self, data):
        array = np.asarray(data)
        original_shape = array.shape
        restored = self.scaler.inverse_transform(array.reshape(-1, original_shape[-1]))
        return restored.reshape(original_shape)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array([v[~np.isnan(v)] for v in dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit), high=len(sampled_timeseries), size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(0))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        # df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
        #                                                      replace_missing_vals_with='NaN')
        df, labels = None, None
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)
