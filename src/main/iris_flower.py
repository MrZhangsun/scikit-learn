from matplotlib.pyplot import title
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# 加载数据集

iris_data_path = Path(__file__).parent.parent / r"resources/assets/iris.csv"
labels = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label'] # 定义列名
iris_data = pd.read_csv(iris_data_path, header=None, names=labels, sep=',')  # DataFrame
print(iris_data['label'].value_counts())

# 数据预览
plt.figure(figsize=(12, 8)) # 绘制画布
# 绘制散点图
# plt.scatter(iris_data[0:50]['petal length'], iris_data[:50]['petal width'], label='Iris-setosa')
# plt.scatter(iris_data[50:100]['petal length'], iris_data[50:100]['petal width'], label='Iris-versicolor')
# plt.scatter(iris_data[100:150]['petal length'], iris_data[100:150]['petal width'], label='Iris-virginica')
# plt.xlabel('sepal length', fontsize=18)
# plt.ylabel('sepal width', fontsize=18)
# plt.title('Iris flower', fontsize=18)
# plt.legend() # 图例
# plt.show()

X = iris_data[labels[:-1]]
print(X.shape)
Y = iris_data['label']
print(Y.value_counts())
#
# 数据分割，分成测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, random_state=28)
print(x_train.shape)
print("训练数据X的格式:{}, 以及类型：{}".format(x_train.shape, type(x_train)))
print("测试集数据X的格式：{}".format(x_test.shape))
print("训练集数据Y的类型：{}".format(type(y_train)))


# 特征工程的操作
# 这里暂时不做特征工程

# 构建模型对象
k = 5
# weights权重类型，uniform为距离权重（等权），distance为距离倒数（加权）
KNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='kd_tree')
KNN.fit(x_train, y_train) # 用训练集数据x_train和label y_train进行训练
print("训练集准确率：{}".format(KNN.score(x_train, y_train)))
print("测试集准确率：{}".format(KNN.score(x_test, y_test)))

# 模型效果评估(交叉验证)
train_predict = KNN.predict(x_train)
test_predict = KNN.predict(x_test)

# 计算召回率、精准度
recall_score = recall_score(y_test, test_predict, average='weighted')
precision_score = precision_score(y_test, test_predict, average='weighted')
accuracy_score = accuracy_score(y_test, test_predict)
print("召回率：{}".format(recall_score))
print("精准度：{}".format(precision_score))
print("准确率：{}".format(accuracy_score))

# # 保存模型
# joblib.dump(KNN, './models/iris_knn.m')
#
# # 加载回复模型
# KNN = joblib.load('./models/iris_knn.m')
# x = [[5.1, 3.5, 1.4, 0.2]]
# y_hat = KNN.predict(x)
# print(y_hat)