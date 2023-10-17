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
    dataset.train_x = dataset.train_x.reshape((dataset.train_x.shape[0], 1, dataset.train_x.shape[1]))
    dataset.test_x = dataset.test_x.reshape((dataset.test_x.shape[0], 1, dataset.test_x.shape[1]))
    return dataset
