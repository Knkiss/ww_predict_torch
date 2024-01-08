# -*- coding: UTF-8 -*-
import world
import math

from sklearn.metrics import mean_squared_error, r2_score
from utility.dataLoader import DataLoaderNormalizer
from utility import plot


def get_result(test_y, pred_y):
    # 测试结果
    MSE = mean_squared_error(test_y, pred_y)
    RMSE = math.sqrt(mean_squared_error(test_y, pred_y))
    R2 = r2_score(test_y, pred_y)
    print('测试集的MSE：' + str(MSE))
    print('测试集的RMSE:' + str(RMSE))
    print('测试集的相关系数：' + str(R2))
    return MSE, RMSE, R2


def early_stop(early_stop_cnt, early_stop_flag, best_r2, res):
    if res[2] > best_r2:
        best_r2 = res[2]
        early_stop_flag = 0
    else:
        early_stop_flag += 1
        if early_stop_flag >= early_stop_cnt:
            return True, best_r2, early_stop_flag
    return False, best_r2, early_stop_flag


if __name__ == '__main__':
    dataset = DataLoaderNormalizer()
    best_pred_y = None
    best_r2 = 0.
    early_stop_cnt = world.early_stop_cnt
    early_stop_flag = 0

    if world.model == "STALSTM":
        from model.stalstm import STALSTMModel

        model = STALSTMModel(input_dim=world.dim, sa_hidden=60, ta_hidden=60, seq_length=world.sequence_length,
                             output_dim=1,
                             n_epoch=1, batch_size=world.batch_size, lr=world.lr, device=world.device, seed=world.seed)

        for i in range(world.epoch):
            print("\nEpoch:", i, "   Early Stop Flag:", early_stop_flag)
            model = model.fit(X=dataset.train_x, y=dataset.train_y)
            pred_y = model.predict(dataset.test_x)
            res = get_result(dataset.test_y, pred_y)
            finish, best_r2, early_stop_flag = early_stop(early_stop_cnt, early_stop_flag, best_r2, res)
            if best_r2 == res[2]:
                best_pred_y = pred_y
            if finish:
                break

    elif world.model == "VALSTM":
        from model.valstm import SLSTMModel

        model = SLSTMModel(dim_X=dataset.train_x.shape[1], dim_y=1, dim_H=60, seq_len=world.sequence_length, n_epoch=1,
                           batch_size=world.batch_size, lr=world.lr, device=world.device, seed=world.seed)

        for i in range(world.epoch):
            print("\nEpoch:", i, "   Early Stop Flag:", early_stop_flag)
            model = model.fit(X=dataset.train_x, y=dataset.train_y)
            pred_y = model.predict(dataset.test_x)
            res = get_result(dataset.test_y, pred_y)
            finish, best_r2, early_stop_flag = early_stop(early_stop_cnt, early_stop_flag, best_r2, res)
            if best_r2 == res[2]:
                best_pred_y = pred_y
            if finish:
                break

    elif world.model == "SLSTM":
        from model.slstm import SLSTMModel

        model = SLSTMModel(dim_X=dataset.train_x.shape[1], dim_y=1, dim_H=60, seq_len=world.sequence_length, n_epoch=1,
                           batch_size=world.batch_size, lr=world.lr, device=world.device, seed=world.seed)

        for i in range(world.epoch):
            print("\nEpoch:", i, "   Early Stop Flag:", early_stop_flag)
            model = model.fit(X=dataset.train_x, y=dataset.train_y)
            pred_y = model.predict(dataset.test_x)
            res = get_result(dataset.test_y, pred_y)
            finish, best_r2, early_stop_flag = early_stop(early_stop_cnt, early_stop_flag, best_r2, res)
            if best_r2 == res[2]:
                best_pred_y = pred_y
            if finish:
                break


    elif world.model == "CNNLSTM":
        from model.cnnlstm import CNNLSTM, data_reshape

        dataset = data_reshape(dataset)
        model = CNNLSTM(dataset.train_x)

        for i in range(world.epoch):
            print("\nEpoch:", i, "   Early Stop Flag:", early_stop_flag)
            history = model.fit(dataset.train_x, dataset.train_y, epochs=1, batch_size=world.batch_size,
                                validation_data=(dataset.test_x, dataset.test_y), verbose=2, shuffle=False)
            pred_y = model.predict(dataset.test_x)
            res = get_result(dataset.test_y, pred_y)
            finish, best_r2, early_stop_flag = early_stop(early_stop_cnt, early_stop_flag, best_r2, res)
            if best_r2 == res[2]:
                best_pred_y = pred_y
            if finish:
                break

    elif world.model == 'SEQ2SEQ':
        from model.seq2seq import SEQ2SEQModel, data_reshape, data_process_final

        dataset = data_reshape(dataset)
        model = SEQ2SEQModel(dataset.train_x.shape[2])

        for i in range(world.epoch):
            print("\nEpoch:", i, "   Early Stop Flag:", early_stop_flag)
            model.fit(X=dataset.train_x, y=dataset.train_y)
            pred_y = model.predict(dataset.test_x, dataset.test_y)
            truth, pred_y = data_process_final(dataset, pred_y)
            res = get_result(truth, pred_y)
            finish, best_r2, early_stop_flag = early_stop(early_stop_cnt, early_stop_flag, best_r2, res)
            if best_r2 == res[2]:
                best_pred_y = pred_y
            if finish:
                dataset.test_y = truth
                break

    else:
        raise NotImplementedError()

    print("\nThet Best Result is:")
    get_result(dataset, best_pred_y)

    # 画图 这里会反归一化
    plot.plot_result_scatter(dataset, best_pred_y)
