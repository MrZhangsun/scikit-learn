from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np

"""
特征选择：
当特征向量中的特征比较多的时候，为了让模型训练数据更好，需要
从一堆特征里挑出“对模型最有用”的那一部分
特征选择的方法有以下几种：
Filter（过滤法）——先筛一遍再建模：
    1、方差过滤：去掉“几乎不变化”的特征，适合：去掉无信息特征
    2、相关系数：看特征和目标的关系强不强
    3、卡方检验（Chi-Square）：判断“词”和“类别”是否相关
Wrapper（包裹法）——用模型来选，试出来哪些特征好用：
Embedded（嵌入法）——模型自带选择：
    1、L1正则（Lasso）：
    2、树模型特征重要性：
"""

X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0.1, 1, 1, 3],
    [1, 2, 3, 1],
    [2, 3, 4, 3]
], dtype=np.float32)

Y = np.array([1,2,1,2,1])

"""
Filter - 方差过滤，去掉“几乎不变化”的特征
threshold: 方差阈值，小于这个阈值的特征会被过滤掉
"""
# 基于方差选择最优的特征属性
variance = VarianceThreshold(threshold=0.6)
variance.fit(X)
print("各个特征属性的方差为:", variance.variances_)
print("特征属性为:", variance.get_support())
print("特征属性为:", variance.transform(X))

"""
Filter - 相关系数：看特征和目标的关系强不强
score_func：使用线性回归来"检验显著性"，
    f_regression：回归问题
    f_classif: 分类问题
    
k: 表示保留关系最强的k个特征
F值：通过线性回归来"检验显著性"，反映的是 “特征能解释的目标变量变异” 与 “无法解释的随机变异” 之间的比例。
    F = （模型解释的变异 / 模型自由度）/ （残差变异 / 残差自由度）
    简单理解：
    F值越大 → 特征对目标变量的影响越显著 → 特征越重要
    F值接近0 → 特征几乎与目标无关

"""
variance = SelectKBest(score_func=f_regression, k=3)
variance.fit(X, Y) # 目标是：Y是标签，X是特征，看X和Y的相关性
print("各个特征属性的相关系数为:", variance.scores_) # 特征的F值
print("特征属性为:", variance.get_support())
print("特征属性为:", variance.transform(X))

"""
Filter - 卡方检验,判断“词”和“类别”是否相关

重要：chi2 的使用条件
chi2（卡方检验）只能用于分类问题，并且要求：
Y（标签）：必须是离散的类别（整数编码）
X（特征）：必须非负（通常建议二值化或离散化后的计数数据）

k: 表示保留关系最强的k个特征
值越大 → 特征与标签越相关
"""
sk2 = SelectKBest(score_func=chi2, k=2)
sk2.fit(X, Y)
print("各个特征属性的卡方检验值为:", sk2.scores_)
print("特征属性为:", sk2.get_support())
print("特征属性为:", sk2.transform(X))


"""
Wrapper-递归特征消除法
工作原理
    用逻辑回归训练模型
    根据特征系数的绝对值（或重要性）排序
    移除最不重要的特征
    重复1-3步，直到剩下指定数量的特征
"""
# RFE特征选择
logistic = LogisticRegression()
selector = RFE(estimator=logistic, n_features_to_select=2)
selector.fit(X, Y)

# 查看结果
print("被选中的特征:", selector.get_support())
print("特征排名（1=最好）:", selector.ranking_)
print("被选中的特征索引:", np.where(selector.get_support())[0])
print("特征重要性（系数）:", selector.estimator_.coef_)

# 转换数据
X_selected = selector.transform(X)
print("原始数据形状:", X.shape)
print("降维后形状:", X_selected.shape)

"""
Embedded【嵌入法】-基于惩罚项的特征选择法
SelectFromModel 是一种基于模型的特征选择方法，它使用任何带有 
coef_ 或 feature_importances_ 属性的模型来评估特征重要性，并选择重要性超过阈值的特征。

核心原理
1. 训练模型（逻辑回归、SVM、随机森林等）
2. 获取特征重要性：
   - 线性模型：使用 |系数|
   - 树模型：使用 feature_importances_
3. 应用阈值筛选：
   - 重要性 > threshold → 保留
   - 重要性 ≤ threshold → 移除
"""
X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2],
    [ 4.9,  3. ,  1.4,  0.2],
    [ -6.2,  0.4,  5.4,  2.3],
    [ -5.9,  0. ,  5.1,  1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
estimator = LogisticRegression(l1_ratio=0, C=1.0)
"""
SelectFromModel(
    estimator,           # 带 coef_ 或 feature_importances_ 的模型
    threshold=None,      # 阈值：'mean', 'median', 数值, 或 callable
    prefit=False,        # 是否使用已训练好的模型
    norm_order=1,        # 多分类时使用的范数（1=L1, 2=L2）
    max_features=None,   # 最多选择几个特征
    importance_getter='auto'  # 自定义获取重要性的方法
)
"""
sfm = SelectFromModel(estimator= estimator, threshold=0.09)
sfm.fit(X2, Y2)
X_selected = sfm.transform(X2)
# 查看结果
print("被选中的特征:", sfm.get_support())
print("被选中的特征索引:", np.where(sfm.get_support())[0])
print("特征重要性（系数）:", sfm.estimator_.coef_)

# 转换数据
print("原始数据形状:", X2.shape)
print("降维后形状:", X_selected.shape)

