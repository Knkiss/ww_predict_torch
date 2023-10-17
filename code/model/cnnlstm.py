import pandas as pd
import numpy as np
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten, RepeatVector, LSTM
from keras.models import Sequential


def CNNLSTM(train_X):
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=1, padding='same', strides=1, activation='relu',
                     input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(RepeatVector(30))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def data_reshape(dataset):
    train = np.concatenate((dataset.train_x, dataset.train_y), axis=1)
    test_x = dataset.test_x[-dataset.test_y.shape[0]:]
    test = np.concatenate((test_x, dataset.test_y), axis=1)

    train = series_to_supervised(train).values
    test = series_to_supervised(test).values

    dataset.train_x = train[:, :-1]
    dataset.train_y = train[:, -1]
    dataset.test_x = test[:, :-1]
    dataset.test_y = test[:, -1]

    dataset.train_x = dataset.train_x.reshape((dataset.train_x.shape[0], 1, dataset.train_x.shape[1]))
    dataset.test_x = dataset.test_x.reshape((dataset.test_x.shape[0], 1, dataset.test_x.shape[1]))
    return dataset


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg
