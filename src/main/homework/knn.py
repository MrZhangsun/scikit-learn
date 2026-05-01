# 通过面向对象的方式实现KNN等权、加权方式计算分类、回归问题
from tkinter.font import names

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from enum import Enum
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ProblemType(Enum):
    CLASSIFIER = 1
    REGRESSION = 2

class KNN(object):
    def __init__(self, k, weight_type, problem_type: ProblemType, algorithm):
        self.k = k
        self.weight_type = weight_type
        self.problem_type = problem_type
        self.model = None
        self.score = None
        self.algorithm = algorithm
        self.metrics = {
            'recall_score': 0.,
            'precision_score': 0.,
            'accuracy_score': 0.
        }


    def fit(self, x, y):
        if self.problem_type == ProblemType.CLASSIFIER:
            self.model = KNeighborsClassifier(n_neighbors=self.k, weights=self.weight_type, algorithm=self.algorithm)
        elif self.problem_type == ProblemType.REGRESSION:
            self.model = KNeighborsRegressor(n_neighbors=self.k, weights=self.weight_type, algorithm=self.algorithm)
        # 训练模型
        self.model.fit(x, y)
        self.score = self.model.score(x, y)

    def predict(self, x, y):
        # 预测
        y_hat = self.model.predict(x)

        # 统计
        self._metrics(y, y_hat)
        return y_hat

    def score(self):
        return self.score()

    def metrics(self):
        return self.metrics

    def _metrics(self, y_hat, y):
        self.metrics['recall_score'] = recall_score(y, y_hat, average='weighted')
        self.metrics['precision_score'] = precision_score(y, y_hat, average='weighted')
        self.metrics['accuracy_score'] = accuracy_score(y, y_hat)

    def save(self, path):
        joblib.dump(self.model, path)
        print("save model to " + path)

    def load(self, path):
        self.model = joblib.load(path)
        print("loaded model from " + path)

if __name__ == '__main__':
    # 读取原始数据
    data_path = Path(__file__).parent.parent.parent / "resources/assets/iris.csv"
    labels = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']  # 定义列名
    data = pd.read_csv(data_path, header=None, names=labels, sep=',')
    print(data.value_counts('label'))

    # 划分数据集
    x = data[labels[:-1]]
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=28)
    print("训练数据X的格式:{}, 以及类型：{}".format(x_train.shape, type(x_train)))
    print("测试集数据X的格式：{}".format(x_test.shape))
    print("训练集数据Y的类型：{}".format(type(y_train)))

    # 创建模型
    knn = KNN(k=5, weight_type='distance', problem_type=ProblemType.CLASSIFIER, algorithm='auto')
    knn.fit(x_train, y_train) # 训练模型
    y_hat = knn.predict(x_test, y_test) # 预测
    print(knn.metrics) # 评估
