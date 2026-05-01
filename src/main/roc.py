from sklearn.metrics import roc_curve, auc, accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsTransformer, KNeighborsClassifier
from sklearn.datasets import load_iris
y_true = [0, 0, 1, 1] # 真实标签
y_score = [0.1, 0.4, 0.35, 0.8] # 模型预测概率

# 计算 FPR 和 TPR
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# 计算 AUC
roc_auc = auc(fpr, tpr)

knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='kd_tree')
knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='ball_tree')
knn_transformer = KNeighborsTransformer(n_neighbors=5)

print(knn_classifier)