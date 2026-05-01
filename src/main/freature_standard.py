from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, KBinsDiscretizer, OneHotEncoder
import numpy as np
"""
数据离散化方式：
1.分箱：把连续数据离散成多个区间，比如：
年龄：23 → [20,30) → 类别2
收入：8000 → [5000,10000) → 类别3
作用：
    1）降低噪声：原始：7999 vs 8001（差异很小），分箱后：同一档 → 更稳定
    2）处理非线性关系（核心🔥），分箱后：连续变量 → 分段函数，可以拟合非线性关系
    3）提升模型鲁棒性，对异常值不敏感，减少过拟合
    4）提高可解释性（业务非常爱），收入 < 5k → 风险高，收入 > 20k → 风险低

2.二值化：把连续数据，根据阈值划分成正反（1/0）两个类别，作用：
    1）简化模型，复杂数值 → 是否满足条件
    2）突出“阈值效应”，很多业务是这样的：是否逾期，是否购买，是否点击
    
"""
X = np.array([[10], [20], [30], [40], [90]])
"""
n_bins: 把数据分成几个区间
strategy：可选值，
    uniform：等宽分箱，区间宽度一样
    quantile：等频分箱，每个箱子样本数量一样；
    kmeans：基于聚类，用 KMeans 自动找分箱边界（更智能）
encode：输出结果编码方式，输出数值表示所属具区间索引，可选值：
    ordinal：默认，表示第几个箱
    onehot：变成 one-hot（适合线性模型）
    onehot-dense：one-hot + dense矩阵
注意数值不在区间范围内时：
小于最小值 → 放到第一个箱
大于最大值 → 放到最后一个箱
"""
kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
kb.fit(X)
result = kb.transform([[50], [120]])

print(result)
print(kb.bin_edges_)


b = Binarizer(threshold=101)
b.fit(X)
print(b.transform([[100]]))


# iris = load_iris()
# print(iris.keys())
#
# x = iris.data
# y = iris.target
#
# min_max_scale = MinMaxScaler() # 归一化(0,1)
# z_score = StandardScaler() # 标准化 z-score
# barbarize = Binarizer(threshold=5) # 二值化
# kBinsDiscretizer = KBinsDiscretizer() # 分箱
#
# # 二值化参数学习
# barbarize.fit(x)
# # 归一化参数学习
# min_max_scale.fit(x)
# # 分箱参数学习
# kBinsDiscretizer.fit(x)
#
# # 原始特征属性的最大值
# print("data_max: ", min_max_scale.data_max_)
# # 原始特征属性的最小值
# print("data_min: ", min_max_scale.data_min_)
# # 原始特征属性的取值范围大小(最大值-最小值)
# print("data_range", min_max_scale.data_range_)
#
# # z-score参数学习
# # z_score.fit(x)
#
# X_train_bin = barbarize.transform(x)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
# x_train = min_max_scale.transform(x_train)
# # x_train = z_score.transform(x_train)
#
# x_test = min_max_scale.transform(x_test)


