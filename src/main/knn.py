import numpy as np
import pandas as pd
# KNN训练数据集
T = [[3, 104, -1],
     [2, 100, -1],
     [1, 81, -1],
     [101, 10, 1],
     [99, 5, 1],
     [98, 2, 1], ]

# 测试数据
x = [18, 90]

# KNN参数
k = 5

# 初始化距离列表
listDistance = []
for i in T:
    # 计算欧式距离
    # distance = ((i[0] - x[0]) ** 2 + (i[1] - x[1]) ** 2) ** 0.5
    distance = np.sum((np.array(i[0:-1]) - np.array(x)) ** 2) ** 0.5
    listDistance.append([distance, i[2]])

# 按照距离进行排序
listDistance.sort()

# 获取前K个最接近的点
tags = np.array(listDistance[:k])[:,-1] # [:, -1] 是 NumPy 的二维切片语法，array[行选择, 列选择]逗号前面控制“行”，逗号后面控制“列”
print(pd.Series(tags).value_counts())
print("*")
# 投票(等权)
print(pd.Series(tags).value_counts().idxmax())

# 投票(加权)
# 用距离的倒数来作为权重
weights = 1 / np.array(listDistance[:k])[:,0] # 获取距离倒数
weights = weights / np.sum(weights) # 归一化，计算权重
print( weights)
print(np.sum(weights * tags))
# 返回最大值所在的位置（索引）
print(pd.Series(tags).value_counts().idxmax())









