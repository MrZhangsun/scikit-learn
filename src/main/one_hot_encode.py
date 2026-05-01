from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X = [
    ["Male", 1],
    ["Female", 2],
    ["Male", 3]
]
encoder.fit(X) # fit 本质是在“记录每一列的所有可能取值（类别）
print(encoder.categories_)

print(encoder.transform(X)) # 返回的是一个坐标，描述了被编码的值所在位置
print(encoder.transform(X).toarray()) # 返回的是一个稀疏矩阵，1的位置表示被编码的值

# 将转换后的特征向量反向恢复
print(encoder.inverse_transform([[0.,1.,1.,0.,0.]]))

# 获取特征向量的名字，并添加前缀
print(encoder.get_feature_names_out(["gender", "age"]))

# drop 参数的作用
'''
first: 所有列在编码的时候，如果都转成0/1两个分类是，会把第一列删除掉，形成一个0，1的编码，但是这种方式会把不是0/1的列也编码成0/1，导致错误，因此只适用于所有
列都是0/1的列
if_binary: 只会把能转成0/1列进行转换成0/1.
'''
encoder2 = OneHotEncoder(drop="first", handle_unknown='ignore')
encoder2.fit(X)
print(encoder2.categories_)
print(encoder2.transform(X).toarray())

encoder3 = OneHotEncoder(drop="if_binary", handle_unknown='ignore')
encoder3.fit(X)
print(encoder3.categories_)
print(encoder3.transform(X).toarray())


# 使用pandas的API进行哑编码转换
import pandas as pd
a = pd.DataFrame([
    ['a', 1, 2],
    ['b', 1, 1],
    ['a', 2, 1],
    ['b', 1, 2],
    ['b', 1, 2]
], columns=['c1', 'c2', 'c3'])
a = pd.get_dummies(a)
print(a)


