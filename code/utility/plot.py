# -*- coding: UTF-8 -*-
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_result_scatter(dataset, y_pred):
    acc = dataset.inverse_trans_y(dataset.test_y)  # 实际值数据
    pre = dataset.inverse_trans_y(y_pred)  # 预测值数据
    time_points = range(len(acc))  # 时间点

    # 用圆点表示实际值和预测值
    plt.scatter(time_points, acc, color="r", label="Actual", marker="o", s=10)
    plt.scatter(time_points, pre, color="y", label="Predicted", marker="o", s=10)

    plt.xlabel("时间")
    plt.ylabel("水量")
    plt.title("实际值与预测值散点图")
    plt.legend()
    plt.show()
    # plt.savefig('stalstmpivot.png')

    print("每次预测的点数:", len(acc))
    print("测试集数据形状:", dataset.test_x.shape)
    print("训练集数据形状:", dataset.train_x.shape)

    # todo 存储预测结果与注意力
    y_pred = y_pred.flatten()  # 将y_pred转换为一维数组

    print(dataset.test_y.shape)
    print(y_pred.shape)

    # results = np.hstack((test_y, y_pred_inverse))
    # np.savetxt('STA_LSTM_DC.csv', results, delimiter=',')
    # np.savetxt('Attention_Value.csv', alpha_t, delimiter=',')
