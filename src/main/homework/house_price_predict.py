# 通过面向对象的方式实现线性回归，支持多维特征向量
# 实现对数据的训练和预测linear regression

import numpy as np
import matplotlib.pyplot as plt

class HousePricePredict(object):
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.theta = None

    def fit(self, x_train:np.ndarray, y_train:np.ndarray):
        """
        用本地房价数据进行训练
        :param x_train: 特征矩阵
        :param y_train: 标签向量
        :return:
        """
        x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        self.x_train = x_train
        self.y_train = y_train.flatten()
        # 参数计算公式
        # self.theta = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
        # print(self.theta)
        self.theta = np.linalg.pinv(x_train) @ y_train
        print(self.theta)

    def predict(self, x_test):
        """
        预测测试数据
        :param x_test: 预测特征向量
        :return: 预测值
        """
        if self.theta is None:
            raise Exception("请先调用fit方法进行训练")

        # 自动加 bias
        x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
        return x_test @ self.theta

    def metrics(self, y_test, y_predict):
        """
        计算准确率
        :param y_test: 测试数据对应的真实标签
        :param y_predict: 测试结果
        :return:
        """

        pass

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x1 = self.x_train[:, 1]
        x2 = self.x_train[:, 2]

        ax.scatter(x1, x2, self.y_train, c='r')

        x1 = np.arange(0, 100)
        x2 = np.arange(0, 4)
        x1, x2 = np.meshgrid(x1, x2)

        z = self.theta[0] + x1 * self.theta[1] + x2 * self.theta[2]
        ax.plot_surface(x1, x2, z)

        plt.show()