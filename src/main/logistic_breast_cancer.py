import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

breast_cancer_data_path = (
        Path(__file__).parent.parent / r"resources/assets/breast-cancer-wisconsin.csv")

csv = pd.read_csv(breast_cancer_data_path)

# 处理缺失值
csv.replace('?', np.nan, inplace=True)
csv.dropna(inplace=True)

X = csv.iloc[:, :-1].astype(np.uint64)
Y = csv.iloc[:, -1]

print(Y.value_counts())

# 划分训练集和测试集
x_train, x_test, y_train, y_test = (
    train_test_split(X, Y, test_size=0.25, random_state=5))

logistic = LogisticRegression(max_iter=1140)
logistic.fit(x_train, y_train)

# 准确率统计
train_score = logistic.score(x_train, y_train)
test_score = logistic.score(x_test, y_test)
y_test_hat = logistic.predict(x_test)
print(train_score, test_score)


fig = plt.figure(figsize=(10, 6))
fig.suptitle("逻辑回归预测乳腺癌患病概率")
ax = fig.add_subplot()
ax.plot(range(len(x_test)), y_test, "ro", markersize=4, zorder=3, label=u"真实值")
ax.plot(range(len(x_test)), y_test_hat, "go", markersize=10, zorder=2, label=u"预测值")

ax.legend()
plt.show()
