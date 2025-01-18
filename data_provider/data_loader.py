import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

data_path_dict = {
    'ETTh1': 'dataset/ETT-small/ETTh1.csv',
    'ETTh2': 'dataset/ETT-small/ETTh2.csv',
    'ETTm1': 'dataset/ETT-small/ETTm1.csv',
    'ETTm2': 'dataset/ETT-small/ETTm2.csv',
    'ELC': 'dataset/electricity/electricity.csv',
    'Exchange': 'dataset/exchange_rate/exchange_rate.csv',
    'Weather2023': 'dataset/weather/weather2023.csv',
    'Weather': '/home/para/lijinpeng/Time-Series-Library/dataset/weather/weather.csv',
    'Traffic': 'dataset/traffic/traffic.csv',
    'PEMS03': 'dataset/PEMS/PEMS03.npz',
    'PEMS04': 'dataset/PEMS/PEMS04.npz',
    'PEMS07': 'dataset/PEMS/PEMS07.npz',
    'PEMS08': 'dataset/PEMS/PEMS08.npz',
}

flag2type = {
    'train': 0,
    'val': 1,
    'test': 2,
}


class Dataset_Exp(Dataset):
    def __init__(self, dataset_name='ETTh1', flag='train', input_len=None, pred_len=None, scale=True):
        assert flag in ['train', 'test', 'val', 'all']
        self.input_len = input_len
        self.pred_len = pred_len
        self.scale = scale
        self.dataset_name = dataset_name
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = data_path_dict[self.dataset_name]
        
        if dataset_file_path.endswith('.csv'):
            if self.dataset_name == 'Weather2023':
                df_raw = pd.read_csv(dataset_file_path, encoding='iso8859-1')
            else:
                df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))
            

        if self.dataset_name.startswith('ETT') or self.dataset_name.startswith('PEMS'):
            data_len = len(df_raw)
            num_train = int(data_len * 0.6)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]

        if self.flag != 'all':
            border1 = border1s[flag2type[self.flag]]
            border2 = border2s[flag2type[self.flag]]
        
        if self.dataset_name != 'Weather2023':
            df_raw = df_raw[['OT']]
        else:
            df_raw = df_raw[['CO2 (ppm)']]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.flag != 'all':
            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)
            self.data = data[border1:border2]
        else:
            if self.scale:
                self.scaler.fit(data)
                data = self.scaler.transform(data)
            self.data = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data[s_begin:s_end]
        y = self.data[r_begin:r_end]
        return seq_x, y

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

def data_provider(batch_size, dataset_name, flag, input_len, pred_len):
    assert flag in ['train', 'test', 'val', 'all']
    # batch_size = batch_size if flag == 'train' else 1
    data_set = Dataset_Exp(
        dataset_name=dataset_name,
        flag=flag,
        input_len=input_len,
        pred_len=pred_len,
        scale=True,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=128,
        drop_last=False)
    return data_set, data_loader
