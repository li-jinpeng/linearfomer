import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


data_path_dict = {
    'ETTh1': 'dataset/ETT-small/ETTh1.csv',
    'ETTh2': 'dataset/ETT-small/ETTh2.csv',
    'ETTm1': 'dataset/ETT-small/ETTm1.csv',
    'ETTm2': 'dataset/ETT-small/ETTm2.csv',
    'ELC': 'dataset/electricity/electricity.csv',
    'Exchange': 'dataset/exchange_rate/exchange_rate.csv',
    'Weather2023': 'dataset/weather/weather2023.csv',
    'Weather': 'dataset/weather/weather.csv',
    'Traffic': 'dataset/traffic/traffic.csv',
}


class Dataset:
    def __init__(self, dataset_name, lookback, pred_len):
        self.lookback = lookback
        self.pred_len = pred_len
        if dataset_name == 'Weather2023':
            df_raw = pd.read_csv(data_path_dict[dataset_name], encoding='iso8859-1')
            df_raw = df_raw[['CO2 (ppm)']]
        else:
            df_raw = pd.read_csv(data_path_dict[dataset_name])
            df_raw = df_raw[['OT']]
        data = df_raw.values
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.data =  data.reshape(len(data))
    
    def __getitem__(self, index):
        return self.data[index: index + self.lookback], self.data[index + self.lookback: index + self.lookback + self.pred_len]
    
    def __len__(self):
        return len(self.data) - self.lookback - self.pred_len + 1


if __name__ == '__main__':
    dataset = Dataset('ETTh1', 30, 1)
    print(len(dataset))
    print(dataset[0])
