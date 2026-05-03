import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, auc, roc_curve

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

# 一、数据加载
headers = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '标签',]
iris_path = Path(__file__).parent.parent.joinpath("resources/assets/iris.csv")
df = pd.read_csv(iris_path, names=headers)

# 二、数据预处理
# class_name_2_label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# df['cla'] = list(map(lambda cla: class_name_2_label[cla], df['标签'].values))
# print(df['cla'].values)
print(df.info())
# 三、特征工程
X = df.iloc[:, :4] # 特征
X = np.asarray(X).astype(np.float64)
Y = df['标签'] # 标签
# 对标签进行编码，将字符串的数据转换为从0开始的int值
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
print(encoder.classes_)
print(encoder.transform(['Iris-setosa'])) # 编码器可以逆向进行解码
print(encoder.inverse_transform([0, 1, 2]))

# 数据分割(将数据分割为训练数据和测试数据)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("训练数据X的格式:{}, 以及数据类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的数据类型:{}".format(type(y_train)))
print("Y的取值范围:{}".format(np.unique(Y)))

# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
x_test = scaler.transform(x_test)

# 四、模型训练
"""
def __init__(self,
             criterion="gini",
             splitter="best",
             max_depth=None,
             min_samples_split=2,
             min_samples_leaf=1,
             min_weight_fraction_leaf=0.,
             max_features=None,
             random_state=None,
             max_leaf_nodes=None,
             min_impurity_decrease=0.,
             min_impurity_split=None,
             class_weight=None,
             presort=False)
    criterion: 给定决策树构建过程中的纯度的衡量指标，可选值: gini、entropy， 默认gini
    splitter：给定选择特征属性的方式，best指最优选择，random指随机选择(局部最优)
    max_features：当splitter参数设置为random的有效，是给定随机选择的局部区域有多大。
    max_depth：剪枝参数，用于限制最终的决策树的深度，默认为None，表示不限制
    min_samples_split=2：剪枝参数，给定当数据集中的样本数目大于等于该值的时候，允许对当前数据集进行分裂；如果低于该值，那么不允许继续分裂。
    min_samples_leaf=1, 剪枝参数，要求叶子节点中的样本数目至少为该值。
    class_weight：给定目标属性中各个类别的权重系数。
"""
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train, y_train)

# 五、模型评估
print("各个特征属性的重要性权重系数(值越大，对应的特征属性就越重要):{}"
      .format(model.feature_importances_))
print("训练数据上的分类报告:")
print(classification_report(y_train, model.predict(x_train)))
print("测试数据上的分类报告:")
print(classification_report(y_test, model.predict(x_test)))
print("训练数据上的准确率:{}".format(model.score(x_train, y_train)))
print("测试数据上的准确率:{}".format(model.score(x_test, y_test)))

index = 19
result4 = model.predict([x_test[index]])
print("标签值：{}，预测值：{}".format(y_test[index], result4))
print("预测特征属于不同标签类别的概率:", model.predict_proba([x_test[index]]))

y_predict_proba = model.predict_proba(x_train)
print("预测特征属于不同标签类别的概率:", y_predict_proba) # 本质就是预测值

class_dict = {0: ['r'], 1: ['g'], 2: ['b'],}

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
for key in class_dict:
    y_true = [int(y == key) for y in y_train]
    print("预测值中的thresholds:", np.unique(y_predict_proba[:, key]))
    print(f"key: {key}", y_true)
    fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba[:, key])
    print("fpr:", fpr, "tpr:", tpr, "thresholds:", thresholds)
    auc_area = auc(fpr, tpr)
    print("auc:", auc_area)
    ax.plot(fpr, tpr, f'{class_dict[key][0]}-o')

plt.show()
# 六、模型保存


