import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# # 生成数据
# X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.2)
#
# # 可视化
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=50)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Generated Data")
# plt.show()
#
#

#
# # 创建 SVM 模型（线性核）
# model = SVC(kernel='linear')
# model.fit(X, y)
#
#
# # 获取分界线系数
# w = model.coef_[0]
# b = model.intercept_[0]
#
# # 分界线函数
# def plot_hyperplane(X, y, w, b):
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=50)
#
#     # 画决策边界
#     x_plot = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
#     y_plot = -(w[0]/w[1])*x_plot - b/w[1]
#     plt.plot(x_plot, y_plot, 'k-', label="Decision Boundary")
#
#     # 画支持向量
#     plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
#                 s=100, facecolors='none', edgecolors='k', label="Support Vectors")
#
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title("SVM Decision Boundary & Support Vectors")
#     plt.legend()
#     plt.show()
#
# plot_hyperplane(X, y, w, b)


# 用 RBF 核处理非线性可分数据
# from sklearn.datasets import make_moons
# X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
#
# model_rbf = SVC(kernel='rbf', C=1)
# model_rbf.fit(X, y)
#
# # 绘制非线性边界
# def plot_decision_boundary(model, X, y):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
#                          np.linspace(y_min, y_max, 500))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=50)
#     plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
#                 s=100, facecolors='none', edgecolors='k')
#     plt.title("SVM with RBF Kernel")
#     plt.show()
#
# plot_decision_boundary(model_rbf, X, y)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from matplotlib.animation import FuncAnimation

# ----------------------
# 1️⃣ 生成数据
# ----------------------
X, y = make_blobs(n_samples=50, centers=2, random_state=42, cluster_std=1.2)

# ----------------------
# 2️⃣ 创建 SVM 模型
# ----------------------
# 我们用 linear kernel
C_values = np.linspace(0.1, 10, 50)  # 模拟训练过程中不同的惩罚参数变化

# ----------------------
# 3️⃣ 绘制函数
# ----------------------
fig, ax = plt.subplots(figsize=(6, 6))


def update(frame):
    ax.clear()
    C = C_values[frame]
    model = SVC(kernel='linear', C=C)
    model.fit(X, y)

    # 决策边界
    w = model.coef_[0]
    b = model.intercept_[0]
    x_plot = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_plot = -(w[0] / w[1]) * x_plot - b / w[1]

    # 绘制点
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=50)

    # 支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # 决策边界
    ax.plot(x_plot, y_plot, 'k-', label='Decision Boundary')

    # 标题
    ax.set_title(f"SVM Decision Boundary (C={C:.2f})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)


# ----------------------
# 4️⃣ 创建动画
# ----------------------
ani = FuncAnimation(fig, update, frames=len(C_values), interval=200)
plt.show()