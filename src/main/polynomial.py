# 多项式扩展，用于特征向量扩展
from sklearn.preprocessing import PolynomialFeatures
orign_features = [[1, 2], [4, 5]]

polynomial = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
new_features = polynomial.fit_transform(orign_features)
print(new_features)

# print(polynomial.transform(orign_features))