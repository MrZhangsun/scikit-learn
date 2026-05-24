"""
# 项目背景及介绍
## 任务描述：
    随着电信行业的不断发展，运营商们越来越重视如何扩大其客户群体。据研究，获取新客户所需的成本远高于保留现有客户的成本，因此为了满足在激烈竞争中的优势，保留现有客户成为一大挑战。对电信行业而言，可以通过数据挖掘等方式来分析可能影响客户决策的各种因素，以预测他们是否会产生流失（停用服务、转投其他运营商等）。

## 数据集：
    数据集一共提供了7043条用户样本，每条样本包含21列属性，由多个维度的客户信息以及用户是否最终流失的标签组成，客户信息具体如下：
    基本信息：包括性别、年龄、经济情况、入网时间等；
    开通业务信息：包括是否开通电话业务、互联网业务、网络电视业务、技术支持业务等；
    签署的合约信息：包括合同年限、付款方式、每月费用、总费用等。

## 评测：
    电信用户流失预测中，运营商最为关心的是客户的召回率，即在真正流失的样本中，我们预测到多少条样本。其策略是宁可把未流失的客户预测为流失客户而进行多余的留客行为，也不漏掉任何一名真正流失的客户。

## 思路
    1. 数据预处理：数据缺失
    2. 可视化分析：
    3. 特征工程：
    4. 模型预测：
    5. 模型评估：
    6. 分析与决策：



"""

import numpy as np
import pandas as pd


"""
对pandas进行设置：
"""
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
### 设置不使用科学计数法  #为了直观的显示数字，不采用科学计数法
pd.set_option('display.float_format', lambda x: '%.2f' % x)
np.set_printoptions(precision=3, suppress=True)

"""
加载原始数据
"""
raw_data_path = 'datas/Telco-Customer-Churn.csv'
raw_data = pd.read_csv(raw_data_path)

# print(raw_data.head(10))
# print(raw_data.info())
# print(raw_data.describe())

"""
数据预处理
"""
# 检查数据集中是否存在缺失值
# isnull().sum() 返回每一列的缺失值数量
missing_values = raw_data.isnull().sum()
print("各列缺失值统计：")
print(missing_values)

# 判断是否所有列的缺失值都为0
if missing_values.sum() == 0:
    print("\n结论：数据集中没有缺失值。")
else:
    print("\n结论：数据集中存在缺失值，请进一步处理。")

# 检查每一列是否存在缺失值（返回布尔序列，True表示该列存在至少一个缺失值）
has_missing = raw_data.isnull().any()
print("\n各列是否存在缺失值：")
print(has_missing)

"""
对TotalCharges这列原本为字符串类型的特征，由于其特征值含有数值意义，
应该首先将其特征值转换为数值形式（浮点数）。此外，对其中不可转换的空格字符，
可以用convert_objects()函数转换成标准的数值型缺失值NaN。
"""
selected_columns = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
# 先筛选出 TotalCharges 为空格或空字符串的行，再选择指定的列
# 注意：TotalCharges 中的缺失值可能以空格 ' ' 形式存在，需先处理或直接比较
# empty_val_rows = raw_data[raw_data['TotalCharges'].astype(str).str.strip() == ''][selected_columns]
# print("空值行：\n", empty_val_rows)
# print(empty_val_rows.describe())

# 已知存在空值，且该列为字符串类型，因此：将字符串转换为数值型，如果错误 occurred，则返回NaN
raw_data['TotalCharges'] = pd.to_numeric(raw_data['TotalCharges'], errors='coerce')

nan_val_rows = raw_data[raw_data['TotalCharges'].isnull()][selected_columns]
print(nan_val_rows)
print(nan_val_rows.describe())

"""
按照业务逻辑处理缺失值
一般缺失值填充方式为均值填充、众数填充、中值填充、0、或者特殊值填充

"""
# 方式一：填充0
# raw_data['TotalCharges'] = raw_data['TotalCharges'].fillna(0)
# print(raw_data['TotalCharges'].isnull().sum())

# 方式二：在这里我们根据实际业务场景的字段描述可以发现，MonthlyCharges【每月费用】 和TotalCharges【总费用】之间应该存在一定的关系，
# 同时我们发现缺省值对应的数据tenure【入网月数】全部是0，且在整个数据集中tenure为0与TotalCharges为缺失值是一一对应的。
# 结合实际业务分析，这些样本对应的客户可能入网当月就流失了，但仍然要收取当月的费用，因此总费用即为该用户的每月费用（MonthlyCharges）。
# 因此本案例我们最终采用MonthlyCharges的数值对TotalCharges进行填充。
# 用MonthlyCharges的数值填充TotalCharges的缺失值
raw_data["TotalCharges"] = raw_data["TotalCharges"].fillna(raw_data["MonthlyCharges"])
# 查看填充后的结果
print(raw_data[raw_data['tenure'] == 0][selected_columns])
print(raw_data[selected_columns].describe())
print(raw_data.describe())

"""
数据可视化
"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(311)


# 绘制箱线图
# vert: 是否垂直绘制
# whis：设置箱线图上下限
# showmeans：是否显示均值(绿色三角)
# flierprops：设置异常值属性
tenure = raw_data['tenure'].tolist()
print(raw_data['tenure'].describe())
ax1.set_title('tenure特征分布箱线图')
ax1.boxplot(tenure, vert=False, showmeans=True,
            flierprops={'marker': 'o', 'markerfacecolor': 'r',
                        'markeredgecolor': 'r', 'markersize': 5},)
# MonthlyCharges特征
ax2 = fig.add_subplot(312)
ax2.set_title('MonthlyCharges特征分布箱线图')
monthly_charges = raw_data["MonthlyCharges"].tolist()
ax2.boxplot(monthly_charges, vert=False, showmeans=True,)
print(raw_data["MonthlyCharges"].describe())

# TotalCharges特征
ax3 = fig.add_subplot(313)
ax3.set_title('TotalCharges特征分布箱线图')
total_charges = raw_data["TotalCharges"].tolist()
ax3.boxplot(total_charges, vert=False, showmeans=True,)
print(raw_data["TotalCharges"].describe())

plt.subplots_adjust(hspace=0.5)
plt.show()






