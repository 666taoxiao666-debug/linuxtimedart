import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import pickle
from torch.utils.data import Dataset
from data_provider.sdwpf_features import sdwpf_feature_columns
from data_provider.m4 import M4Dataset, M4Meta
from utils.timefeatures import time_features
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

    Linear interpolation uses both endpoints and is not causal.  Prefer
    ``_ffill_short_missing_runs`` for forecasting pipelines.
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


def _ffill_short_missing_runs(series, max_gap=6):
    """Fill short NaN runs using only past values inside one series."""
    series = series.copy()
    missing = series.isna()
    if not missing.any():
        return series

    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    filled = series.ffill()
    fillable = missing & (run_length <= int(max_gap)) & filled.notna()
    series.loc[fillable] = filled.loc[fillable]
    return series


def _segment_ids(turbines, dates, expected_freq):
    expected_delta = pd.to_timedelta(expected_freq).to_timedelta64()
    boundary = np.empty(len(dates), dtype=bool)
    boundary[0] = True
    if len(dates) > 1:
        boundary[1:] = (turbines[1:] != turbines[:-1]) | (
            (dates[1:] - dates[:-1]) != expected_delta
        )
    return np.cumsum(boundary) - 1


def _sdwpf_cutoffs(
    unique_dates,
    split,
    fold,
    n_folds,
    train_ratio,
    val_ratio,
    date_months=None,
):
    unique_dates = np.asarray(unique_dates)
    n = len(unique_dates)
    if n < 3:
        raise ValueError("SDWPF data must contain at least three unique timestamps")
    if split == "time_ratio":
        train_pos = min(max(1, int(n * train_ratio)), n - 2)
        val_pos = min(max(train_pos + 1, int(n * (train_ratio + val_ratio))), n - 1)
        return unique_dates[train_pos], unique_dates[val_pos], None
    if split == "rolling":
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"sdwpf_fold must be in [0, {n_folds}), got {fold}")
        test_ratio = 1.0 - train_ratio - val_ratio
        test_size = max(1, int(round(n * test_ratio / n_folds)))
        val_size = max(1, int(round(n * val_ratio)))
        test_start = n - (n_folds - fold) * test_size
        test_start = max(val_size + 2, test_start)
        val_end = test_start
        val_start = max(2, val_end - val_size)
        train_end = val_start
        test_end = min(n, test_start + test_size)
        if train_end < 2 or val_end <= train_end or test_start >= n:
            raise ValueError(
                f"Rolling fold {fold}/{n_folds} is empty; reduce n_folds or "
                "window length"
            )
        # The upper bound is essential: without it fold 0 evaluates on every
        # later timestamp and therefore overlaps folds 1..N.  ``None`` means
        # that the last fold legitimately runs to the end of the dataset.
        test_cutoff = unique_dates[test_end] if test_end < n else None
        return unique_dates[train_end], unique_dates[val_end], test_cutoff
    if split == "seasonal":
        if date_months is None:
            raise ValueError("seasonal split requires month labels for unique dates")
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"sdwpf_fold must be in [0, {n_folds}), got {fold}")
        val_month = 6 + int(fold)
        if val_month > 11:
            raise ValueError("seasonal fold exceeds the mapped calendar")
        months = np.asarray(date_months)
        val_idx = np.flatnonzero(months == val_month)
        after_idx = np.flatnonzero(months > val_month)
        if len(val_idx) == 0 or len(after_idx) == 0:
            raise ValueError(
                f"Seasonal fold {fold} needs month {val_month} for validation "
                f"and later months for test"
            )
        train_cutoff = unique_dates[val_idx[0]]
        val_cutoff = unique_dates[after_idx[0]]
        if not (unique_dates[0] < train_cutoff < val_cutoff):
            raise ValueError("Seasonal cutoffs are not strictly increasing")
        return train_cutoff, val_cutoff, None
    raise ValueError("sdwpf_split must be one of: time_ratio, rolling, seasonal")


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

    Power is clipped to ``[0, rated_power]`` (consumption as zero).  High-wind
    zeros and feathered pitch are kept as actual power=0 but flagged in
    ``available_mask`` so evaluation can report both actual and available
    generation.  Wind angles are optional sin/cos encodings.  Missing sensor
    values are forward-filled inside a segment so no future observation is used.
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
        rated_power=1500.0,
        clip_power=True,
        circular_wind=True,
        collapse_pitch=True,
        keep_curtailment=True,
        causal_fill=True,
        split="time_ratio",
        fold=0,
        n_folds=3,
        physics_features=True,
        drop_weak_features=True,
    ):
        del seasonal_patterns
        if size is None:
            size = [336, 0, 96]
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be one of: train, val, test")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be one of: M, S, MS")
        if split not in {"time_ratio", "rolling", "seasonal"}:
            raise ValueError("split must be time_ratio, rolling, or seasonal")
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
        self.rated_power = float(rated_power) if rated_power else 0.0
        self.split = split
        self.fold = int(fold)

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
            float(self.rated_power),
            bool(clip_power),
            bool(circular_wind),
            bool(collapse_pitch),
            bool(keep_curtailment),
            bool(causal_fill),
            split,
            int(fold),
            int(n_folds),
            bool(physics_features),
            bool(drop_weak_features),
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
                rated_power=self.rated_power,
                clip_power=clip_power,
                circular_wind=circular_wind,
                collapse_pitch=collapse_pitch,
                keep_curtailment=keep_curtailment,
                causal_fill=causal_fill,
                split=split,
                fold=int(fold),
                n_folds=int(n_folds),
                physics_features=bool(physics_features),
                drop_weak_features=bool(drop_weak_features),
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
        self.test_cutoff = prepared["test_cutoff"]
        self.scaler = prepared["scaler"]
        self.feature_columns = prepared["feature_columns"]
        self.available_mask = prepared["available_mask"]
        self.wspd = prepared["wspd"]
        self.audit_stats = prepared["audit_stats"]
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
        rated_power,
        clip_power,
        circular_wind,
        collapse_pitch,
        keep_curtailment,
        causal_fill,
        split,
        fold,
        n_folds,
        physics_features,
        drop_weak_features,
    ):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"SDWPF data file not found: {file_path}")

        header = pd.read_csv(file_path, nrows=0).columns.tolist()
        required = {"date", "TurbID", target}
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"SDWPF file is missing required columns: {missing}")

        metadata = {"date", "TurbID", "Day"}
        load_columns = [
            c
            for c in header
            if c in {"date", "TurbID"}
            or c not in metadata
        ]
        dtype = {c: "float32" for c in load_columns if c != "date"}
        df = pd.read_csv(file_path, usecols=load_columns, dtype=dtype)
        original_rows = len(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        numeric_columns = [c for c in load_columns if c != "date"]
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["date", "TurbID"])
        df["TurbID"] = df["TurbID"].astype(np.int16)
        df = (
            df.sort_values(["TurbID", "date"], kind="mergesort")
            .drop_duplicates(["TurbID", "date"], keep="last")
            .reset_index(drop=True)
        )
        rows_after_identity_validation = len(df)

        wind = df["Wspd"] if "Wspd" in df else pd.Series(np.nan, index=df.index)
        power = df[target]
        pitch_cols = [c for c in ("Pab1", "Pab2", "Pab3") if c in df]
        pitch_mean = df[pitch_cols].mean(axis=1) if pitch_cols else None
        available = np.ones(len(df), dtype=bool)
        if filter_abnormal:
            # Consumption / standby: keep the row, treat as zero later.
            # Curtailment / outage: keep actual near-zero power, mark unavailable.
            curtailed = (wind > 5.0) & (power.fillna(0) <= 0)
            if pitch_mean is not None:
                curtailed = curtailed | ((wind > 2.5) & (pitch_mean > 89))
            available = ~curtailed.to_numpy()
            if not keep_curtailment:
                df.loc[curtailed, target] = np.nan

        repaired_values = 0
        invalid_counts = {}
        if filter_abnormal:
            invalid_masks = {}
            if "Wspd" in df:
                invalid_masks["Wspd"] = ~df["Wspd"].between(0, 40)
            if "Wdir" in df:
                invalid_masks["Wdir"] = ~df["Wdir"].between(-180, 180)
            if "Ndir" in df:
                invalid_masks["Ndir"] = ~df["Ndir"].between(-720, 720)
            for temperature in ("Etmp", "Itmp"):
                if temperature in df:
                    invalid_masks[temperature] = ~df[temperature].between(-50, 80)
            # Do not NaN the power column for negative/curtailed values.
            # Those are operating regimes, not missing labels.
            for column, invalid in invalid_masks.items():
                invalid_counts[column] = int(invalid.sum())
                repaired_values += invalid_counts[column]
                df.loc[invalid, column] = np.nan

        if clip_power:
            upper = float(rated_power) if rated_power and rated_power > 0 else None
            clipped = df[target].clip(lower=0.0, upper=upper)
            df[target] = clipped.astype(np.float32)

        if physics_features and "Wdir" in df and "Ndir" in df:
            yaw = np.deg2rad(
                (df["Wdir"].to_numpy(dtype=np.float64) - df["Ndir"].to_numpy(dtype=np.float64))
            )
            df["yaw_sin"] = np.sin(yaw).astype(np.float32)
            df["yaw_cos"] = np.cos(yaw).astype(np.float32)

        if circular_wind:
            for angle_col, prefix in (("Wdir", "Wdir"), ("Ndir", "Ndir")):
                if angle_col not in df:
                    continue
                radians = np.deg2rad(df[angle_col].to_numpy(dtype=np.float64))
                df[f"{prefix}_sin"] = np.sin(radians).astype(np.float32)
                df[f"{prefix}_cos"] = np.cos(radians).astype(np.float32)
                df = df.drop(columns=[angle_col])
        if collapse_pitch and pitch_cols:
            df["Pab_mean"] = df[pitch_cols].mean(axis=1).astype(np.float32)
            df = df.drop(columns=pitch_cols)

        feature_columns = sdwpf_feature_columns(
            features=features,
            target=target,
            circular_wind=circular_wind,
            collapse_pitch=collapse_pitch,
            physics_features=physics_features,
            drop_weak_features=drop_weak_features,
        )
        missing_features = [c for c in feature_columns if c not in df]
        if missing_features:
            raise ValueError(f"SDWPF feature columns missing: {missing_features}")

        raw_dates = df["date"].to_numpy(dtype="datetime64[ns]")
        raw_turbines = df["TurbID"].to_numpy(dtype=np.int16, copy=False)
        segment_ids = _segment_ids(raw_turbines, raw_dates, expected_freq)
        fill_fn = _ffill_short_missing_runs if causal_fill else _interpolate_short_missing_runs
        # Input sensors may be repaired over a short gap, but labels must never
        # be imputed: even causal target filling fabricates supervision and
        # makes the validation loss look artificially smooth.
        fill_columns = [column for column in feature_columns if column != target]
        df[fill_columns] = df.groupby(segment_ids, sort=False)[fill_columns].transform(
            lambda series: fill_fn(series, max_gap=6)
        )

        unique_dates = np.unique(raw_dates)
        unique_months = pd.DatetimeIndex(unique_dates).month.to_numpy()
        train_cutoff, val_cutoff, test_cutoff = _sdwpf_cutoffs(
            unique_dates,
            split=split,
            fold=fold,
            n_folds=n_folds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            date_months=unique_months,
        )
        train_rows = raw_dates < train_cutoff
        if not np.any(train_rows):
            raise ValueError("The SDWPF training split is empty")

        # Short gaps have already been filled causally above.  A remaining NaN
        # denotes a long/leading invalid run and must create a hard temporal
        # boundary.  Median filling it would fabricate long stretches of SCADA
        # data and allow windows to bridge turbine outages or sensor failures.
        missing_by_feature_after_fill = {
            column: int(count)
            for column, count in df[feature_columns].isna().sum().items()
            if int(count) > 0
        }
        still_missing = df[feature_columns].isna().any(axis=1)
        unresolved_rows = int(still_missing.sum())
        if still_missing.any():
            keep = ~still_missing.to_numpy()
            df = df.loc[keep].reset_index(drop=True)
            available = available[keep]
            raw_dates = df["date"].to_numpy(dtype="datetime64[ns]")
            raw_turbines = df["TurbID"].to_numpy(dtype=np.int16, copy=False)
        if df.empty:
            raise ValueError("No SDWPF rows remain after validation and filtering")

        dates = df["date"].to_numpy(dtype="datetime64[ns]")
        turbines = df["TurbID"].to_numpy(dtype=np.int16, copy=True)
        wspd = (
            df["Wspd"].to_numpy(dtype=np.float32, copy=True)
            if "Wspd" in df
            else np.full(len(df), np.nan, dtype=np.float32)
        )
        values = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        scaler = StandardScaler()
        if scale:
            train_rows = dates < train_cutoff
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
        audit_stats = {
            "original_rows": int(original_rows),
            "rows_after_identity_validation": int(rows_after_identity_validation),
            "retained_rows": int(len(df)),
            "removed_rows_total": int(removed),
            "removed_invalid_identity_or_duplicate_rows": int(
                original_rows - rows_after_identity_validation
            ),
            "removed_unresolved_feature_rows": unresolved_rows,
            "abnormal_values_total": int(repaired_values),
            "abnormal_values_by_feature": invalid_counts,
            "missing_by_feature_after_short_fill": missing_by_feature_after_fill,
            "available_rows": int(np.asarray(available, dtype=bool).sum()),
            "unavailable_or_curtailed_rows": int((~np.asarray(available, dtype=bool)).sum()),
            "available_fraction": float(np.asarray(available, dtype=bool).mean()),
            "continuous_segments": int(len(segments)),
            "causal_fill": bool(causal_fill),
            "keep_curtailment": bool(keep_curtailment),
            "filter_abnormal": bool(filter_abnormal),
        }
        print(
            "SDWPF prepared: "
            f"{len(df):,} rows, {len(segments):,} continuous segments, "
            f"{removed:,} unresolved/duplicate rows removed, "
            f"{repaired_values:,} abnormal sensor values marked, "
            f"split={split} fold={fold}, "
            f"train_cutoff={pd.Timestamp(train_cutoff)}, "
            f"val_cutoff={pd.Timestamp(val_cutoff)}, "
            f"test_cutoff={pd.Timestamp(test_cutoff) if test_cutoff is not None else 'end'}, "
            f"available_frac={float(available.mean()):.3f}, "
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
            "test_cutoff": test_cutoff,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "available_mask": np.asarray(available, dtype=bool),
            "wspd": wspd,
            "audit_stats": audit_stats,
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
                if self.test_cutoff is not None:
                    keep &= target_end < self.test_cutoff
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

    def _target_row_index(self):
        starts = np.asarray(self.window_starts, dtype=np.int64)
        return starts[:, None] + int(self.seq_len) + np.arange(int(self.pred_len))

    def target_available_mask(self):
        """Boolean mask of uncurtailed target steps, shape [windows, pred_len]."""
        return np.asarray(self.available_mask, dtype=bool)[self._target_row_index()]

    def last_history_wspd(self):
        """Wind speed at the forecast issue time, shape [windows]."""
        starts = np.asarray(self.window_starts, dtype=np.int64)
        return np.asarray(self.wspd)[starts + int(self.seq_len) - 1]


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
