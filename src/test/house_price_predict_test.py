import unittest
import numpy as np
import pandas as pd
from homework.house_price_predict import HousePricePredict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

class HousePricePredictTest(unittest.TestCase):
    def test_predict(self):
        self.assertEqual(1, 1)

        x = np.array([
            [10, 1],
            [15, 1],
            [20, 1],
            [30, 1],
            [50, 2],
            [60, 1],
            [60, 2],
            [70, 2]]).reshape((-1, 2))
        # x1: 面积
        # x2: 房间数量

        y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))
        # y: 价格
        house_price_predict = HousePricePredict()
        house_price_predict.fit(x, y)

        xx = np.array([[55.0, 2.0]])
        yy = house_price_predict.predict(xx)
        print(yy)
        house_price_predict.show()

    def test_boston_housing_data1(self):
        """
        使用原始数据进行预测
        :return:
        """
        data_path = Path(__file__).parent.parent.parent / 'src/resources/assets/boston_housing.csv'
        data = pd.read_csv(data_path, sep='\\s+', header=None)
        self.assertEqual(data.shape[1], 14)

        # 分离特征和标签
        x = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]

        # 划分测试集和训练集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

        # 标准化，去除量纲影响
        # ‼️fit_transform = 用训练数据“制定规则并应用”，transform = 用同一规则处理新数据
        standard_x_train, standard_x_test = self.standard(x_train, x_test)

        # 构建模型
        # fit_intercept是否需要截距项
        standard_x_train = np.hstack((np.ones((standard_x_train.shape[0], 1)), standard_x_train))
        standard_x_test = np.hstack((np.ones((standard_x_test.shape[0], 1)), standard_x_test))
        linear = LinearRegression(fit_intercept=True)
        linear.fit(standard_x_train, y_train)

        # 模型预测
        y_hat_train = linear.predict(standard_x_train)
        y_hat_test = linear.predict(standard_x_test)

        # 模型评估
        print(linear.coef_)  # 每个特征的权重theta参数列表
        print(linear.intercept_)  # 截距theta0
        print(linear.score(standard_x_test, y_test)) # R^2值，1:表示模型很好，0:表示模型很差，-1:表示模型没有预测能力
        print(linear.score(standard_x_train, y_train))

        # 数据可视化
        self.show_data(standard_x_test, standard_x_train, y_hat_test, y_hat_train, y_test, y_train)

    @staticmethod
    def show_data(standard_x_test, standard_x_train, y_hat_test, y_hat_train, y_test, y_train):
        # 训练集
        plt.plot(range(len(standard_x_train)), y_train, 'r', label=u'true')
        plt.plot(range(len(standard_x_train)), y_hat_train, 'g', label=u'predict')
        plt.title(u'train')
        plt.legend(loc='upper right')
        plt.show(block=True)
        # 测试集
        plt.plot(range(len(standard_x_test)), y_test, 'r', label=u'true')
        plt.plot(range(len(standard_x_test)), y_hat_test, 'g', label=u'predict')
        plt.title(u'test')
        plt.legend(loc='upper right')
        plt.show(block=True)

    def test_boston_housing_data2(self):
        """
        使用多项式扩展进行训练
        :return:
        """
        data_path = Path(__file__).parent.parent.parent / 'src/resources/assets/boston_housing.csv'
        data = pd.read_csv(data_path, sep='\\s+', header=None)
        self.assertEqual(data.shape[1], 14)

        # 分离特征和标签
        x = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]

        # 划分测试集和训练集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

        # 标准化，去除量纲影响
        # ‼️fit_transform = 用训练数据“制定规则并应用”，transform = 用同一规则处理新数据
        standard_x_train, standard_x_test = self.standard(x_train, x_test)

        # ‼️多项式扩展
        polynomial = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        standard_x_train = polynomial.fit_transform(standard_x_train)
        standard_x_test = polynomial.fit_transform(standard_x_test)

        # 填充theta0
        standard_x_train = np.hstack((np.ones((standard_x_train.shape[0], 1)), standard_x_train))
        standard_x_test = np.hstack((np.ones((standard_x_test.shape[0], 1)), standard_x_test))

        # 构建模型
        # fit_intercept是否需要截距项
        linear = LinearRegression(fit_intercept=True)
        linear.fit(standard_x_train, y_train)

        # 模型预测
        y_hat_train = linear.predict(standard_x_train)
        y_hat_test = linear.predict(standard_x_test)

        # 模型评估
        print(linear.coef_)  # 每个特征的权重theta参数列表
        print(linear.intercept_)  # 截距theta0
        print(linear.score(standard_x_test, y_test))  # R^2值，1:表示模型很好，0:表示模型很差，-1:表示模型没有预测能力
        print(linear.score(standard_x_train, y_train))

        # 数据可视化
        self.show_data(standard_x_test, standard_x_train, y_hat_test, y_hat_train, y_test, y_train)

    @staticmethod
    def standard(x1, x2):
        """
        为什么要去除量纲差异？
            ✔ 避免某些特征“数值大就占优势”
            ✔ 让模型公平学习所有特征
            ✔ 提高优化稳定性
            ✔ 加快收敛速度
        | 方法             | 名称      | 范围     | 是否抗异常值 | 推荐程度  |
        | -------------- | ------- | ------ | ------ | ----- |
        | StandardScaler | Z-score | 无固定范围  | ❌      | ⭐⭐⭐⭐⭐ |
        | MinMaxScaler   | 归一化     | [0,1]  | ❌      | ⭐⭐⭐   |
        | RobustScaler   | 稳健标准化   | 无固定范围  | ✅      | ⭐⭐⭐⭐  |
        | MaxAbsScaler   | 最大绝对值   | [-1,1] | ❌      | ⭐⭐⭐   |
        | Normalizer     | 向量归一化   | 长度=1   | ❌      | ⭐⭐    |
        :return:
        """
        z_score = StandardScaler()
        # ‼️fit_transform = 用训练数据“制定规则并应用”，transform = 用同一规则处理新数据
        return z_score.fit_transform(x1), z_score.transform(x2)
if __name__ == '__main__':
    unittest.main()