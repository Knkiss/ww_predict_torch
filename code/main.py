# -*- coding: UTF-8 -*-
import world
import math

from sklearn.metrics import mean_squared_error, r2_score
from utility.dataLoader import DataLoaderNormalizer
from utility import plot


if __name__ == '__main__':
    dataset = DataLoaderNormalizer()

    if world.model == "STALSTM":
        from model.stalstm import STALSTMModel
        model = STALSTMModel(input_dim=world.dim, sa_hidden=60, ta_hidden=60, seq_length=world.sequence_length,output_dim=1,
                             n_epoch=world.epoch, batch_size=world.batch_size, lr=world.lr, device=world.device, seed=world.seed)
        model = model.fit(X=dataset.train_x, y=dataset.train_y)

    elif world.model == "VALSTM":
        from model.valstm import SLSTMModel
        model = SLSTMModel(dim_X=dataset.train_x.shape[1], dim_y=1, dim_H=60, seq_len=world.sequence_length, n_epoch=world.epoch,
                     batch_size=world.batch_size, lr=world.lr, device=world.device, seed=world.seed)
        model = model.fit(X=dataset.train_x, y=dataset.train_y)

    elif world.model == "SLSTM":
        from model.slstm import SLSTMModel
        model = SLSTMModel(dim_X=dataset.train_x.shape[1], dim_y=1, dim_H=60, seq_len=world.sequence_length, n_epoch=world.epoch,
                     batch_size=world.batch_size, lr=world.lr, device=world.device, seed=world.seed)
        model = model.fit(X=dataset.train_x, y=dataset.train_y)

    elif world.model == "CNNLSTM":
        from model.cnnlstm import CNNLSTM, data_reshape
        dataset = data_reshape(dataset)
        model = CNNLSTM(dataset.train_x)
        model = model.fit(dataset.train_x, dataset.train_y, epochs=world.epoch, batch_size=world.batch_size,
                            validation_data=(dataset.test_x, dataset.test_y), verbose=2, shuffle=False)
        pred_y = model.predict(dataset.test_x)

    else:
        raise NotImplementedError()

    # 测试结果
    pred_y = model.predict(test_x=dataset.test_x)
    RMSE = math.sqrt(mean_squared_error(dataset.test_y, pred_y))
    print('\n测试集的MSE：', mean_squared_error(dataset.test_y, pred_y))
    print('\n测试集的RMSE:', RMSE)
    print('\n测试集的相关系数：', r2_score(dataset.test_y, pred_y))

    # 画图
    plot.plot_result_scatter(dataset, pred_y)
