# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import math
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from utility.dataLoader import ModelDataset


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim):
        super(TemporalLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        # 输入序列长度

        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(input_dim, seq_len), requires_grad=True)
        # 用于注意的形状为`(input_dim, seq_len序列长度)`的权重矩阵
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim, seq_len), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(seq_len), requires_grad=True)
        # 用于注意力的长度为`seq_len`的偏差向量。
        self.Va = nn.Parameter(torch.Tensor(seq_len, seq_len), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)
        # todo  沿`dim = 1`（跨序列长度）应用的Softmax激活函数的实例

        # LSTM参数 todo  4个隐层 要改吗？？
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)
        #  LSTM 中的门和单元状态更新。

        # y的权重是在门值的计算上(4个门都有y)
        self.Wy = nn.Parameter(torch.Tensor(output_dim, hidden_dim * 4), requires_grad=True)

        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        # todo  全连接（线性）层，将 LSTM 的隐藏状态作为输入并产生模型的最终输出。它具有hidden_dim输入特征（维度）和output_dim输出特征。

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        # 具有一定方差的值来初始化权重
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, H, y_t_1):
        # todo  两个输入参数：H和y_t_1。这些可能代表输入数据和前一个时间步的输出

        batch_size, seq_len, input_dim = H.size()
        # 序列长度的计算
        HS = self.hidden_dim

        # 参数命名
        h = H
        # h被分配输入数据的值H。

        # 隐藏序列
        hidden_seq = []
        # 空列表，将用于收集每个时间步的隐藏状态。

        # 初始状态
        s_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_h_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_c_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        # todo  s_t、LSTM_h_t、 和 均LSTM_c_t用零初始化，表示 LSTM 的初始隐藏状态和单元状态。
        # 打循环开始
        t = 0
        # 注意力机制的计算
        save_beta_t = []
        while t < seq_len:
            # todo t小于序列步长
            # 取出当前的值
            h_t = h[:, t, :]

            # 计算注意力(第二个维度对应了是时间序列长度)
            beta_t = torch.tanh(h_t @ self.Wa + s_t @ self.Ua + self.ba) @ self.Va
            # todo  隐藏，时间，序列偏差
            # todo `beta_t` 的计算结果是对 `h_t`（隐藏状态的当前时间步长）的线性组合和`s_t`（之前的隐藏状态）、偏差项   应用 tanh 激活的结果、。
            # softmax过一次
            beta_t = self.Softmax(beta_t)
            save_beta_t.append(beta_t)
            # todo  沿（时间维度）应用softmax 函数进行标准化，确保它们的总和为 1
            # 扩充对齐inpupt_dim维度(重复之后直接做哈达玛积运算)
            beta_t = beta_t.unsqueeze(2)
            beta_t = beta_t.repeat(1, 1, input_dim)
            # beta_t被扩展为与输入数据具有相同的维度h。这样做是为了准备逐元素乘法 ( beta_t * h)。

            # 合并掉时间序列的维度(全序列)
            h_t = torch.sum(input=beta_t * h, dim=1)
            # todo 注意力分数用于计算输入数据的加权和h，从而有效地生成上下文向量h_t。该上下文向量基于注意力机制从输入数据中捕获相关信息。
            # LSTM门值的计算(y加进去算)
            gates = h_t @ self.W + LSTM_h_t @ self.U + y_t_1 @ self.Wy + self.bias
            # todo 使用上下文向量 h_t、先前的隐藏状态LSTM_h_t、先前的单元状态LSTM_c_t和来计算y_t_1。这些门通过线性变换和激活函数（Sigmoid 和 tanh）进行计算。

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS * 2])
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[:, HS * 3:])
            # todo 根据门值i_t、f_t、g_t并  o_t使用 LSTM 方程进行更新。
            # 隐藏层状态的计算
            LSTM_c_t = f_t * LSTM_c_t + i_t * g_t
            LSTM_h_t = o_t * torch.tanh(LSTM_c_t)
            hidden_seq.append(LSTM_h_t.unsqueeze(0))
            #  todo 更新后的隐藏状态LSTM_h_t将附加到hidden_seq列表中，该列表收集每个时间步的隐藏状态。

            y_t_1 = self.fc(LSTM_h_t)
            # todo 用线性层 ( )根据当前 LSTM 隐藏状态进行预测self.fc。这是模型在当前时间步的输出。

            # 时刻加一
            t = t + 1
        # 隐藏状态的计算
        # 连接隐藏状态：
        #
        # 处理完所有时间步后，hidden_seq将列表连接并转置以获得所有时间步的隐藏状态张量。
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # todo 返回y_t_1，它是最后一个时间步的输出预测，并且hidden_seq包含所有时间步的隐藏状态。
        return y_t_1, hidden_seq, save_beta_t


class SpatialLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialLSTM, self).__init__()

        # 超参数继承,将超参数input_dim和  hidden_dim为类的属性。
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 向量化
        # todo LSTM 部分中使用的可学习权重参数。它们用于计算 LSTM 中的门和单元状态更新。
        # self.W：输入的权重矩阵。
        # self.U：隐藏状态的权重矩阵。
        # self.bias：偏置向量。
        # todo hidden_dim * 4总之，在权重维度中使用 是 LSTM 实现中的常见做法，用于解释 LSTM 的四个门，每个门都有自己的一组输入权重、先前的隐藏状态和偏差
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim * 2, input_dim), requires_grad=True)
        # todo  隐藏状态的权重矩阵。
        self.ba = nn.Parameter(torch.Tensor(input_dim), requires_grad=True)
        # todo 注意力机制中的偏差向量。
        self.Va = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)

        # todo Softmax 激活函数 ( self.Softmax) 沿维度 1 使用来标准化注意力分数。
        self.Softmax = nn.Softmax(dim=1)

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X):

        # 参数取得便于后续操作
        batch_size, seq_len, _ = X.size()

        # 参数命名
        x = X

        # 隐藏序列
        hidden_seq = []

        # 初始值计算
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 序列长度的计算
        HS = self.hidden_dim

        # 打循环开始
        t = 0
        save_alpha_t = []
        # LSTM的计算
        while t < seq_len:
            # 取出当前的值
            x_t = x[:, t, :]

            # 计算注意力
            a_t = torch.tanh(x_t @ self.Wa + torch.cat((h_t, c_t), dim=1) @ self.Ua + self.ba) @ self.Va

            # softmax归一化
            alpha_t = self.Softmax(a_t)
            save_alpha_t.append(alpha_t)

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

        return hidden_seq, (h_t, c_t), save_alpha_t


class STA_LSTM(nn.Module):
    def __init__(self, input_dim, sa_hidden, ta_hidden, seq_length, output_dim):
        super(STA_LSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        # todo  gaile
        # self.sa_hidden = sa_hidden
        self.ta_hidden = ta_hidden
        self.seq_length = seq_length
        self.output_dim = output_dim

        # 预测模型
        # self.SA：类的实例SpatialLSTM，负责处理输入数据的空间方面。
        # self.TA：类的实例TemporalLSTM，负责处理输入数据的时间（时间相关）方面。
        # self.SA = SpatialLSTM(input_dim=input_dim, hidden_dim=sa_hidden)
        # SA输出sa  hidden 当ta的输入
        self.SA = SpatialLSTM(input_dim=input_dim, hidden_dim=sa_hidden)
        self.TA = TemporalLSTM(input_dim=sa_hidden, hidden_dim=ta_hidden, seq_len=seq_length, output_dim=output_dim)

    #         # SA输出sa  hidden 当ta的输入
    #         self.TA = TemporalLSTM(input_dim=sa_hidden, hidden_dim=ta_hidden, seq_len=seq_length, output_dim=output_dim)
    #
    # todo

    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        # 切分输入与输出关系
        x = X[:, :, 0: self.input_dim]
        #  todo 它选择self.input_dim每个时间步长的第一个维度。???? why
        y = X[:, :, self.input_dim - 1:]
        # todo  它选择每个时间步的最后一个维度并将其重塑以(batch_size, seq_len, self.output_dim)匹配预期的形状。

        # 重新对齐tensor维度
        y = y.view(batch_size, seq_len, self.output_dim)

        # 参数的预测
        hidden_seq, (_, _), save_alpha_t = self.SA(X=x)
        # todo  X通过SpatialLSTM模型(self.SA)。该模型处理输入数据的空间方面并生成一个hidden_seq张量，其中包含每个时间步的隐藏状态。

        y_pred, _, save_beta_t = self.TA(H=hidden_seq, y_t_1=y[:, 0, :])
        # todo hidden_seq将空间处理获得的张量传递给模型TemporalLSTM( self.TA)。该模型处理数据的时间方面。
        #  它还将y[:, 0, :]表示初始目标值（通常是第一个时间步的目标）作为输入来启动序列预测。
        return y_pred, save_alpha_t, save_beta_t


class STALSTMModel(BaseEstimator, RegressorMixin):
    # 继承自2个基类 BaseEstimator, RegressorMixin；用作回归问题
    def __init__(self, input_dim, sa_hidden, ta_hidden, seq_length, output_dim, n_epoch=60, batch_size=64, lr=0.001,
                 device=torch.device('cuda:0'), seed=1024):
        super(STALSTMModel, self).__init__()
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.sa_hidden = sa_hidden
        self.ta_hidden = ta_hidden
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        # Initialize Scaler

        # todo  为什么没自动初始化？？  ：这会初始化一个StandardScaler对象。它表明在预处理期间输入数据将被标准化（以均值为中心并缩放至单位方差）。

        # Model Instantiation
        self.loss_hist = []
        sa_hidden = sa_hidden
        self.model = STA_LSTM(input_dim=input_dim, sa_hidden=sa_hidden, ta_hidden=ta_hidden, seq_length=seq_length,
                              output_dim=output_dim).to(device)
        #  创建模型的实例 STA_LSTM
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # 初始化模型参数的 Adam 优化器
        self.criterion = nn.MSELoss(reduction='mean')

    # 置用于训练的损失函数。在本例中，它是均方误差损失。reduction='mean'参数指定损失应在批次上取平均值。
    # todo 不知道预测了多少步，是多步
    def fit(self, X, y):
        X_3d = []
        y_3d = []

        # todo  不太懂这个
        # todo 它通过循环数据并创建长度为重叠的序列来实现这一点self.seq_length。对于每个序列，它收集序列的最后一个元素作为该序列的目标标签。
        for i in range(X.shape[0] - self.seq_length + 1):
            X_3d.append(X[i: i + self.seq_length, :])
            y_3d.append(y[i + self.seq_length - 1: i + self.seq_length, :])
        # todo  X_3d和y_3d分别是序列和目标的列表。为了准备训练，它们被转换为 3D 张量：
        # X_3d沿着第二个维度（时间维度）堆叠以创建形状为 的张量(batch_size, seq_length, input_dim)。
        # y_3d类似地堆叠以创建形状张量(batch_size, seq_length, output_dim)。
        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)

        dataset = ModelDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.device),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.device), '3D')

        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            # 初始化一个列表self.loss_hist来跟踪每个时期的损失。初始值 0 被附加到该列表中。
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            # todo ???
            # 第一个维度（维度0）变成了新张量的第二个维度（维度1）。
            # 第二个维度（维度1）变成了新张量的第一个维度（维度0）。
            # 第三个维度（维度2）保持不变。
            # 它排列 的尺寸batch_X以匹配模型的预期输入格式。为了确保尺寸与模型的输入形状正确对齐，可能需要执行此步骤。
            alphas = []
            betas = []

            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)

                batch_y = batch_y.permute(0, 1, 2)

                batch_y = batch_y.squeeze(1)
                # 此行删除中间尺寸batch_y。这可能是确保尺寸正确对齐
                self.optimizer.zero_grad()
                # 用于output对输入批次进行预测 ( )。
                output, alpha, beta = self.model(batch_X)
                alphas.append(torch.stack(alpha))
                betas.append(torch.stack(beta))

                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()

            alpha_res = torch.concat(alphas, dim=1)
            beta_res = torch.concat(betas, dim=1)

            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')
        return self

    def predict(self, test_x):
        X_3d = []

        for i in range(test_x.shape[0] - self.seq_length + 1):
            # todo 从输入数据X中选择从索引开始的时间步序列i。
            X_3d.append(test_x[i: i + self.seq_length, :])
        X_3d = np.stack(X_3d, 1)
        # 转换X_3d为具有数据类型的 PyTorch 张量torch.float32并将其发送到指定设备 ( self.device)。
        # 然后，它会使用排列张量的维度.permute(1, 0, 2)，以确保维度与模型的输入形状正确对齐。
        test_x = torch.tensor(X_3d, dtype=torch.float32, device=self.device).permute(1, 0, 2)

        self.model.eval()
        with torch.no_grad():
            y, save_alpha_t, save_beta_t = self.model(test_x)
            # 放上cpu转为numpy
            y = y.cpu().numpy()
            # TODO 保存save_alpha_t, save_beta_t
        return y
