# -*- coding: UTF-8 -*-
try:
    import torch
    device = torch.device('cuda:0')
except:
    pass

sequence_length = 32
train_split_ratio = 0.8
dim = 12
sample_number = 1
epoch = 1000
batch_size = 128
lr = 0.001
seed = 1024

model = "SEQ2SEQ"  # STALSTM、VALSTM、SLSTM、CNNLSTM、SEQ2SEQ

early_stop_cnt = 5
