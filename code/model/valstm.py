# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import math
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from torch.utils.data import DataLoader

from utility.dataLoader import ModelDataset


class VALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VALSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 向量化
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        self.Wa = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim * 2, input_dim), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(input_dim), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        # 全连接层(做预测)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X, init_states=None):

        # 参数取得便于后续操作
        batch_size, seq_len, _ = X.size()

        # 参数命名
        x = X

        # 隐藏序列
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h_t, c_t = init_states
        # 序列长度的计算
        HS = self.hidden_dim

        t = 0
        # LSTM的计算
        while t < seq_len:
            # 取出当前的值
            x_t = x[:, t, :]

            # 计算注意力
            a_t = torch.tanh(x_t @ self.Wa + torch.cat((h_t, c_t), dim=1) @ self.Ua + self.ba) @ self.Va

            # softmax归一化
            alpha_t = self.Softmax(a_t)

            # 加权
            x_t = alpha_t * x_t

            # 计算门值
            gates = x_t @ self.W + h_t @ self.U + self.bias

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS * 2])
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[:, HS * 3:])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

            t = t + 1
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        final_feature = hidden_seq[:, seq_len - 1, :].squeeze()

        final_feature = final_feature.view(batch_size, HS)

        # 全连接层做标签预测
        y_pred = self.fc(final_feature)

        return y_pred, hidden_seq, (h_t, c_t), alpha_t


class SLSTMModel(BaseEstimator, RegressorMixin):

    def __init__(self, dim_X, dim_y, dim_H, seq_len=60, n_epoch=200, batch_size=64, lr=0.001,
                 device=torch.device('cuda:0'), seed=1024):
        super(SLSTMModel, self).__init__()

        torch.manual_seed(seed)

        # 分配参数
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_H = dim_H
        self.seq_len = seq_len
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        # 初始化模型
        self.loss_hist = []
        self.model = VALSTM(input_dim=dim_X, hidden_dim=dim_H, output_dim=dim_y).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, X, y):
        y = y.reshape(-1, self.dim_y)
        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_len + 1):
            X_3d.append(X[i: i + self.seq_len, :])
            y_3d.append(y[i + self.seq_len - 1: i + self.seq_len, :])

        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)
        dataset = ModelDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.device),
                               torch.tensor(y_3d, dtype=torch.float32, device=self.device), '3D')

        # 模型训练
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)
                batch_y = batch_y.permute(0, 1, 2)
                batch_y = batch_y.squeeze(1)
                self.optimizer.zero_grad()
                output, _, _, _ = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        return self

    def predict(self, test_x):

        # 转化为三维再预测
        X_3d = []

        for i in range(test_x.shape[0] - self.seq_len + 1):
            X_3d.append(test_x[i: i + self.seq_len, :])
        X_3d = np.stack(X_3d, 1)
        test_x = torch.tensor(X_3d, dtype=torch.float32, device=self.device).permute(1, 0, 2)

        self.model.eval()
        with torch.no_grad():
            y, _, _, alpha = self.model(test_x)

            # 放上cpu转为numpy
            y = y.cpu().numpy()
            alpha = alpha.cpu().numpy()

        return y
