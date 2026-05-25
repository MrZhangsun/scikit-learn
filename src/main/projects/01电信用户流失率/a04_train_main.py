import time

import joblib
import lightgbm
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

from a03_feature_project import feature_engineering, feature_balance
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from lightgbm import LGBMClassifier as LGB
import matplotlib.pyplot as plt

def load_data():
    output_path = feature_engineering(rebuild=True)
    features = pd.read_csv(output_path)
    X = features.iloc[:, :-1]
    y = features.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # print(X_train.shape)
    # print(y_train.shape)

    categorical_features = ['Partner', 'Dependents', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport',
                            'PaperlessBilling',]
    # 'InternetService', 'Contract', 'PaymentMethod'
    indexes = [X_train.columns.get_loc(i) for i in categorical_features]
    the_balanced = feature_balance(X_train, y_train, indexes)
    X_train = the_balanced.iloc[:, :-1]
    y_train = the_balanced.iloc[:, -1]
    # print(X_train.shape)
    # print(y_train.shape)

    return X_train, X_test, y_train, y_test

def k_fold(X_train, y_train, classifier, **kwargs):
    kf = KFold(n_splits=5, shuffle=True)
    y_predicts = np.zeros(len(y_train))
    y = np.zeros(len(y_train))
    start = time.time()

    # 测试数据会被等分成5份，1份作为测试集，4份作为训练集，循环5次，直到每一份数据都参与了训练和测试
    for train_idx, test_idx in kf.split(X_train):
        # 训练
        current_x_train = X_train.iloc[train_idx]
        current_y_train = y_train.iloc[train_idx]

        clf = classifier(**kwargs)
        clf.fit(current_x_train, current_y_train)

        # 验证
        current_x_test = X_train.iloc[test_idx]
        current_y_test = y_train.iloc[test_idx]
        y_predicts[test_idx] = clf.predict(current_x_test)
        y[test_idx] = current_y_test

    print("used time : {}".format(time.time()-start))
    return y_predicts, y

def select_model(x_train, y_train):
    lr_predicts, lr_y = k_fold(x_train, y_train, LR, l1_ratio=0, C=0.1)
    svm_predicts, svc_y = k_fold(x_train, y_train, SVC, C=1.0)
    rf_predicts, rf_y = k_fold(x_train, y_train, RF, n_estimators=100, max_depth=10)
    lgb_predicts, lgb_y = k_fold(x_train, y_train, LGB, learning_rate=0.1, n_estimators=1000, max_depth=10)

    algorithms = ["LR", "SVC", "RF", "LGB"]
    score_df = pd.DataFrame(columns=algorithms)
    results = [(lr_predicts, lr_y), (svm_predicts, svc_y), (rf_predicts, rf_y), (lgb_predicts, lgb_y)]
    for i, (y_hat, y) in enumerate(results):
        recall = recall_score(y, y_hat)
        precision = precision_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        score_df.iloc[:, i] = pd.Series([recall, precision, f1])

    score_df.index = ["recall", "precision", "f1"]
    print(score_df)

def train_main():
    the_x_train, the_x_test, the_y_train, the_y_test = load_data()

    # select_model(the_x_train, the_y_train)

    # 由上表可知，LGB模型效果最好，我们选择LGB模型单模型进行演示，并且输出其特征重要性
    lgb_clf = LGB(learning_rate=0.1, n_estimators=1000, max_depth=10)
    lgb_clf.fit(the_x_train, the_y_train)

    the_train_y_hat = lgb_clf.predict(the_x_train)
    the_test_y_hat = lgb_clf.predict(the_x_test)

    the_train_reporter = classification_report(the_y_train, the_train_y_hat)
    the_test_reporter = classification_report(the_y_test, the_test_y_hat)
    print(the_train_reporter)
    print(the_test_reporter)
    print(lgb_clf.feature_importances_)

    feature_importance_df = pd.DataFrame(the_x_train.columns, columns=["features"])
    feature_importance_df["importance"] = lgb_clf.feature_importances_
    print(feature_importance_df.sort_values(by="importance", ascending=False))

    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    lightgbm.plot_importance(lgb_clf, ax=ax1, height=0.5, grid=False)
    ax1.set_title("LightGBM Feature Importance")
    plt.show()

    # 保存模型
    joblib.dump(lgb_clf, "./models/lgb_model.m")

if __name__ == '__main__':
    train_main()