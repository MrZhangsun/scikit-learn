"""
集成学习
集成学习是组合多个学习器来完成学习任务的策略，通常能获得比单一学习器更优越的泛化性能。
常见的继承学习思想：
装袋（Bagging）：通过多个弱模型的预测结果，通过计算得到一个强预测结果，并行生成多个树
提升（Boosting）：一步一步训练模型，每一步都专门修正前一步的错误，串行生成树，每一轮都在“补充信息”
堆叠（Stacking）：


Boosting三大经典算法：
1.AdaBoost（自适应提升）
    思想：让错的样本权重变大
    流程：
        初始化样本权重（一样）
        训练一个弱分类器
        错的样本 → 权重增加
        再训练下一个模型
2. GBDT（Gradient Boosting Decision Tree）
    思想：用梯度下降来“拟合误差”
    核心：每一轮拟合的是“残差（误差）”
    举例：
        真实值：y
        预测值：ŷ
        残差：r = y - ŷ
    下一棵树学的是：r
    本质：一步一步逼近真实值
3. XGBoost / LightGBM / CatBoost（工业级）
    这些是GBDT的增强版：
    XGBoost：正则化 + 并行优化
    LightGBM：更快（基于直方图）
    CatBoost：对类别特征友好


"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeRegressor


"""
自定义集成学习实现
"""
#
# df = pd.DataFrame([[1, 10.56],
#                    [2, 27],
#                    [3, 39.1],
#                    [4, 40.4],
#                    [5, 58],
#                    [6, 60.5],
#                    [7, 79],
#                    [8, 87],
#                    [9, 90],
#                    [10, 95]], columns=['X', 'Y'])
#
# models = [] # 保存决策树集合
# n_trees = 200 # 需要生成的决策树的数量
#
# # 循环采样生成决策树
# for i in range(n_trees):
#     ###对样本进行有放回的抽样m次
#     '''
#     sample() 抽样
#     n=None,  抽样数据的条数
#     frac=None, 抽样的比例
#     replace=False, 是否有放回抽样
#     weights=None, 权重
#     random_state=None, 随机数种子
#     axis=None 维度
#     '''
#     samples = df.sample(frac=1.0, replace=True) # 不要设置随机数种子，从而达到随机采样的目的
#     x_train = samples.iloc[:, :-1]
#     y_train = samples.iloc[:, -1]
#
#     # 生成决策树
#     tree = DecisionTreeRegressor(max_depth=1)
#     tree.fit(x_train, y_train)
#
#     # 保存决策树
#     models.append(tree)
#
# # 生成强决策树
# # 从特征矩阵中随机抽取一部分特征用于测试
# samples = df.sample(frac=0.6, replace=True, random_state=1)
# x_test = samples.iloc[:, :-1]
# y_test = samples.iloc[:, -1]
#
# # 回归问题，预测结果为均值
# # 分类问题，预测结果为众数
# y_predict = 0
# for model in models:
#     y_predict += model.predict(x_test)
#
# # 预测结果
# y_predict = y_predict / len(models)
# print(y_predict)
#
# # 评估
# r2 = r2_score(y_test, y_predict)
# print(r2)
#

"""
Bagging（袋装）分类
经典算法：随机森林（Random Forest, RF）
RF推广算法：
Extra Tree:极端随机树
Totally Random Trees Embedding, TRTE：完全随机树编码
Isolation Forest：孤立森林
"""
df = pd.DataFrame([[0, 1],
                   [1, 1],
                   [2, 1],
                   [3, -1],
                   [4, -1],
                   [5, -1],
                   [6, 1],
                   [7, 1],
                   [8, 1],
                   [9, -1]])

models = []
n_trees =5

for i in range(n_trees):
    samples = df.sample(frac=1.0, replace=True)
    x_train = samples.iloc[:, :-1]
    y_train = samples.iloc[:, -1]
    # splitter='random': 表示随机切分，默认是按特征值进行切分
    tree = DecisionTreeRegressor(max_depth=1, splitter='random', max_features=1)
    tree.fit(x_train, y_train)
    models.append(tree)


samples = df.sample(frac=0.9, replace=True, random_state=1)
x_test = samples.iloc[:, :-1]
y_test = samples.iloc[:, -1]

y_predict = np.zeros(y_test.shape)
for model in models:
    y_predict += model.predict(x_test)
y_predict = np.sign(y_predict)
print(y_predict)
print(accuracy_score(y_test, y_predict))
print(f1_score(y_test, y_predict))
tpr, fpr, thresholds = roc_curve(y_test, y_predict)
auc_area = auc(tpr, fpr)
print(auc_area)



