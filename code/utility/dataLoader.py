# -*- coding: UTF-8 -*-
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
try:
    from torch.utils.data import Dataset


    class ModelDataset(Dataset):
        # Initialization
        def __init__(self, data, label, mode='2D'):
            self.data, self.label, self.mode = data, label, mode

        # Get item
        def __getitem__(self, index):
            if self.mode == '2D':
                return self.data[index, :], self.label[index, :]
            elif self.mode == '3D':
                return self.data[:, index, :], self.label[:, index, :]

        # Get length
        def __len__(self):
            if self.mode == '2D':
                return self.data.shape[0]
            elif self.mode == '3D':
                return self.data.shape[1]
except:
    pass


class DataLoaderNormalizer:
    def __init__(self, file_name="pivot_all.xlsx"):
        self.fileName = file_name

        self.XScaler = MinMaxScaler()
        self.YScaler = MinMaxScaler()
        self.XScaler_Stan = StandardScaler()
        self.train_x, self.train_y, self.test_x, self.test_y = None, None, None, None
        self.load_data()

    def load_data(self, sequence_length=48, train_split_ratio=0.8, dim=4, sample=20):
        data = pd.read_excel(io="../dataset/" + self.fileName,
                             usecols=['玉泉山瞬时流量', '麻峪瞬时流量', '杏石口瞬时流量', '石景山南线瞬时流量'])
        data = data.dropna()
        data = data[::sample]  # 每20个点取一个值
        data = data.values

        # data = data.head(16000)  # 取数据的前N个值
        # print(data.info())  # 打印数据统计信息

        # 不知道什么意义的重新存储
        x_temp = data[:, :-1]
        y_temp = data[:, -1]
        x_new = np.zeros([data.shape[0], dim])
        x_new[:, :-1] = x_temp
        x_new[:, -1] = y_temp

        # 训练测试数据集分割
        train_size = int(data.shape[0] * train_split_ratio)
        train_x = x_new[:train_size, :]
        train_y = y_temp[:train_size]
        test_x = x_new[train_size - sequence_length + 1:, :]
        test_y = y_temp[train_size:]

        train_x = self.XScaler.fit_transform(train_x)
        self.train_x = self.XScaler_Stan.fit_transform(train_x)
        test_x = self.XScaler.transform(test_x)
        self.test_x = self.XScaler_Stan.transform(test_x)

        train_y = train_y.reshape(-1, 1)
        self.train_y = self.YScaler.fit_transform(train_y)
        test_y = test_y.reshape(-1, 1)
        self.test_y = self.YScaler.transform(test_y)

    def inverse_trans_y(self, y):
        y = y.reshape(1, -1)
        y_pred_inverse = self.YScaler.inverse_transform(y)
        y_pred_inverse = y_pred_inverse.reshape(-1)
        return y_pred_inverse
