# 梯度下降
import numpy as np
import matplotlib.pyplot as plt
import unittest
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

class GradientDescent(unittest.TestCase):

    def test_two_dim_gradient_descent(self):
        """
        一元二次曲线梯度下降
        """
        def fx(x):
            return x**2 + 2*x + 1

        def df(x):
            return 2*x + 2
        GD_X = []
        GD_Y = []
        alpha = 0.1
        f_change = 10
        x_current = 4
        f_current = fx(x_current)
        GD_X.append(x_current)
        GD_Y.append(f_current)

        iter_num = 0
        while f_change > 0.00000000001:
            iter_num += 1
            # X 下降
            """
            每一步更新公式是： 𝑥（𝑡+1）=𝑥𝑡−𝛼⋅𝑓′(𝑥𝑡)
            α：学习率（步长）
            f′(xt)：当前点的斜率
            本质就是：沿着下降最快方向走一小步
            """
            x_current = x_current - alpha * df(x_current)

            tmp = fx(x_current)
            f_change = np.abs(f_current - tmp)
            f_current = tmp
            GD_X.append(x_current)
            GD_Y.append(f_current)
        self.assertTrue(iter_num < 1000)
        print("梯度下降结果：(%.5f, %.5f)"% (x_current, f_current))
        print("迭代次数：%d"% iter_num)
        print("X:", GD_X)
        print("Y:", GD_Y)

        X = np.arange(-6, 4, 0.05)
        Y = np.array(list(map(lambda t: fx(t), X)))
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        fig.suptitle('一元二次曲线梯度下降')
        axes = fig.add_subplot()
        axes.set_xlabel('X')
        axes.set_ylabel('f(x)')
        axes.plot(X, Y, 'r-', linewidth=2)
        axes.plot(GD_X, GD_Y, 'bo--', linewidth=2)
        plt.show()
        # plt.figure(facecolor='w')
        # plt.plot(X, Y, 'r-', linewidth=2)
        # plt.plot(GD_X, GD_Y, 'bo--', linewidth=2)
        # plt.show()

    def test_three_dim_gradient_descent(self):
        self.assertTrue(True)
        """
        二维二次曲线梯度下降
        """
        def fx(x, y):
            """
            二维二次曲线函数
            """
            return 0.6 * (x + y) ** 2 - x * y

        def dfx(x, y):
            """
            关于X的导数
            """
            return 0.6 * 2 * (x + y) - y

        def dfy(x, y):
            """
            关于Y的导数
            """
            return 0.6 * 2 * (x + y) - x

        grad_x = []
        grad_y = []
        grad_fx = []


        # 初始值(超参)
        x_current = 4
        y_current = 4
        alpha = 0.1
        fx_change = 10

        fx_current = fx(x_current, y_current)
        grad_x.append(x_current)
        grad_y.append(y_current)
        grad_fx.append(fx_current)

        iter_num = 0
        while fx_change > 0.00000000001:
            iter_num += 1
            x_current = x_current - alpha * dfx(x_current, y_current)
            y_current = y_current - alpha * dfy(x_current, y_current)
            fx_change = np.abs(fx_current - fx(x_current, y_current))

            # 梯度更新
            fx_current = fx(x_current, y_current)
            grad_x.append(x_current)
            grad_y.append(y_current)
            grad_fx.append(fx_current)

        print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x_current, y_current, fx_current))
        print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
        print(grad_x)
        print(grad_y)
        print(grad_fx)


        # 构建数据
        x1 = np.arange(-4, 4.5, 0.2)
        x2 = np.arange(-4, 4.5, 0.2)
        xx1, xx2 = np.meshgrid(x1, x2)
        np.array(list)
        fig = plt.figure(facecolor='w')
        fig.suptitle('二维二次曲线梯度下降')
        axes = Axes3D(fig)
        axes.plot_surface(grad_x, grad_y, grad_fx, rstride=1, cstride=1, cmap=plt.cm.jet)
        axes.plot(grad_x, grad_y, grad_fx, 'ro--', linewidth=2)
        axes.set_title('二维二次曲线梯度下降')
        plt.show()



