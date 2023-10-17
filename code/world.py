# -*- coding: UTF-8 -*-
try:
    import torch
    device = torch.device('cuda:0')
except:
    pass

sequence_length = 48
train_split_ratio = 0.8
dim = 4
sample_number = 20
epoch = 3
batch_size = 64
lr = 0.01
seed = 1024

model = "CNNLSTM"  # STALSTM、VALSTM、SLSTM、CNNLSTM
