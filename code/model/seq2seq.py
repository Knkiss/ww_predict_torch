# -*- coding: UTF-8 -*-
import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset

import world

batch_size = 32
in_seq_len = 24  # How far to look back
out_seq_len = 12  # How far to look forward

dec_in_size = 1  # Number of known future features + target feature
output_size = 1  # Number of target features
hidden_size = 64  # Dimensions in hidden layers
num_layers = 1  # Number of hidden layers
teaching_forcing_prob = 0.75  # Probability of teaching forcing

num_epochs = 3
learning_rate = 5e-3
es_patience = 15
lr_patience = 5


# split a multivariate sequence past, future samples (X and y)
def sequence_generator(arr, past_step=in_seq_len, future_step=out_seq_len, y_features=1):
    # instantiate X and y
    X, y = [], []
    for i in range(len(arr)):
        # find the end of the input, output sequence
        input_end = i + past_step
        output_end = input_end + future_step
        # check if we are beyond the dataset
        if output_end > len(arr):
            break
        else:
            # gather input and output of the pattern
            seq_x, seq_y = arr[i: input_end], arr[input_end: output_end, -y_features:]
            X.append(seq_x), y.append(seq_y)

    return np.array(X), np.array(y)


def data_reshape(dataset):
    train = np.concatenate((dataset.train_x, dataset.train_y), axis=1)
    test_x = dataset.test_x[-dataset.test_y.shape[0]:]
    test = np.concatenate((test_x, dataset.test_y), axis=1)

    train_x, train_y = sequence_generator(train)
    test_x, test_y = sequence_generator(test)

    dataset.train_x = train_x
    dataset.train_y = train_y
    dataset.test_x = test_x
    dataset.test_y = test_y
    return dataset


def data_process_final(dataset, y_pred):
    y_pred = y_pred.cpu().numpy()
    forecast_length = out_seq_len
    truth = dataset.test_y[:, forecast_length-1, -1].ravel()
    forecast = y_pred[:, forecast_length-1].ravel()
    return truth, forecast


class Encoder(nn.Module):
    def __init__(self, encoder_input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()

        # Number of input features
        self.encoder_input_size = encoder_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.encoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, encoder_inputs):
        """
        encoder_inputs: Shape (batch_size x look_back x input_size)
          [(X_t-N, feature_t-N), ..., (X_t-1, feature_t-1), (X_t, feature_t)]
        """
        # Feed in the whole sequence
        # PyTorch will handle the rollout
        _, hidden = self.lstm(encoder_inputs)
        # Return final hidden state (context vector)
        return hidden


class Decoder(nn.Module):
    def __init__(self, decoder_input_size, target_size, hidden_size, num_layers):
        super(Decoder, self).__init__()

        # Predection size
        # i.e univariate forecasting has output size of 1
        self.target_size = target_size
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.decoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Transform the latent representation
        # back to a single prediction
        self.fc = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, input_init, encoder_hidden, targets, teaching_forcing_prob):
        """
        input_init:  Shape (batch_size x 1 x input_size)
          (X_t, feature_t)
        encoder_hidden: Shape (num_layers*num_directions x batch_size x hidden_size)
          Context vector from the encoder
        targets: Shape (batch_size x pred_len x features_size)
          Our targets [(X_t+1, feature_t+1), (X_t+2, feature_t+2), .. (X_t+N, feature_t+N)]
          Note: features are used as future covariates for prediction only
        teaching_forcing_prob: float
          The chance of using truth value instead of predicted output for next time step
        """
        # Forecasting steps
        decoder_seq_len = targets.shape[1]
        # Store decoder outputs
        outputs = torch.tensor([]).to(world.device)
        # Input at time step t
        input_t = input_init[:, -1, -self.decoder_input_size:].unsqueeze(1)

        # Rollout the sequence
        for t in range(decoder_seq_len):
            # Change input dim from (B, C) into (B, 1, C)
            out, hidden = self.lstm(input_t, encoder_hidden)
            # Shape (batch size, 1, target size)
            out = self.fc(out)
            outputs = torch.cat((outputs, out), 1)
            # Setup for next time step
            # Use actual output
            if random.random() < teaching_forcing_prob:
                input_t = targets[:, t, :].unsqueeze(1)
            # Use predicted output with known future features (DateTime...)
            else:
                future_features = targets[:, t, -self.decoder_input_size:-1].unsqueeze(1)
                input_t = torch.concat((future_features, out), 2)

        # Shape (batch size, pred step, target features)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, enc_in_size, dec_in_size, output_size, hidden_size, num_layers):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(enc_in_size, hidden_size, num_layers)
        self.decoder = Decoder(dec_in_size, output_size, hidden_size, num_layers)

    def forward(self, encoder_inputs, targets, teaching_forcing_prob):
        encoder_hidden = self.encoder(encoder_inputs)
        # Feed last time step value from encoder
        outputs = self.decoder(encoder_inputs, encoder_hidden, targets, teaching_forcing_prob)
        return outputs


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target


class SEQ2SEQModel:
    def __init__(self, enc_in_size, seed=world.seed):
        super(SEQ2SEQModel, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.model = Seq2Seq(enc_in_size, dec_in_size, output_size, hidden_size, num_layers).to(world.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=world.lr)
        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, X, y):
        train_dataset = TimeSeriesDataset(X, y)
        data_loader = DataLoader(train_dataset, batch_size=world.batch_size, shuffle=True)

        total_loss = 0
        self.model.train()
        for i in range(1):
            for X, y in data_loader:
                X, y = X.to(world.device), y.to(world.device)
                # Forward pass
                output = self.model(X, y, teaching_forcing_prob)
                loss = self.criterion(output, y[:, :, -1].unsqueeze(2))
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            train_avg_loss = total_loss / len(data_loader)
            print('Loss:{}'.format(train_avg_loss))
        print('Optimization finished')
        return self

    def predict(self, test_x, test_y):
        test_dataset = TimeSeriesDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=world.batch_size, shuffle=False)

        output = torch.tensor([]).to(world.device)
        self.model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                y_pred = self.model(X.to(world.device), y.to(world.device), 0)
                output = torch.cat((output, y_pred.to(world.device)), 0)

        return output
