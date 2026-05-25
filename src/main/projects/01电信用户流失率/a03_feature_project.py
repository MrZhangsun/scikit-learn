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
import pandas as pd
from pathlib import Path

"""
特征工程
1.标准化：对于数值特征进行标准化，去除量纲对模型的影响
    StandardScaler：适合正态分布的特征
2.特征编码：对离散特征进行编码
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from a01_load_data import process_nan

def feature_selection(raw_data):
    # 特征选择
    # 1. 删除对于分类没有影响的特征，可视化过程中发现有四列特征对结果影响可以忽略，后续直接删除
    drop_cols = ['customerID', 'gender', 'PhoneService', 'StreamingTV', 'StreamingMovies']
    raw_data.drop(drop_cols, axis=1, inplace=True)

    # 2. 删除重复特征：通过相关系数法，卡方检验等衡量相关性的方法，来判断是否需要删除
    raw_data.drop(['TotalCharges'], axis=1, inplace=True)

    # print(raw_data.info())
    # print(raw_data.describe())
    return raw_data

def standardize(raw_data):
    # 标准化
    scaler = StandardScaler()
    raw_data['tenure'] = scaler.fit_transform(raw_data[['tenure']])
    raw_data['MonthlyCharges'] = scaler.fit_transform(raw_data[['MonthlyCharges']])
    raw_data['TotalCharges'] = scaler.fit_transform(raw_data[['TotalCharges']])
    # print(raw_data[["tenure", "MonthlyCharges", "TotalCharges"]].describe())
    return raw_data

def feature_encoding(raw_data):
    # 处理离散特征，将No phone service/No internet service 映射为No
    raw_data.loc[raw_data['MultipleLines'] == 'No phone service', 'MultipleLines'] = 'No'
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport']
    for i in internet_cols:
        raw_data.loc[raw_data[i]=='No internet service', i] = 'No'

    assert raw_data[raw_data['MultipleLines']=='No phone service'].shape[0] == 0
    assert raw_data[raw_data['OnlineSecurity']=='No internet service'].shape[0] == 0

    # 部分类别特征只有两类取值，可以直接用0、1代替, 选择特征值为‘Yes’和 'No' 的列名
    encode_cols = ['Partner','Dependents','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','PaperlessBilling','Churn']
    for i in encode_cols:
        raw_data[i] = raw_data[i].map({'Yes':1, 'No':0})

    # 其他无序的类别特征采用独热编码
    # One-Hot
    one_hot_cols = [
        'InternetService',
        'Contract',
        'PaymentMethod'
    ]

    # label 分离
    y = raw_data['Churn']
    X = raw_data.drop(columns=['Churn'])

    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    # 编码
    encoded = encoder.fit_transform(X[one_hot_cols])

    # 转 DataFrame
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(one_hot_cols),
        index=X.index
    )

    # 删除原列
    X = X.drop(columns=one_hot_cols)

    # 拼接
    X = pd.concat(
        [X, encoded_df],
        axis=1
    )

    # 拼接标签
    return pd.concat([X, y], axis=1)

def feature_balance(X, y, categorical_features=None):
    # pip install imbalanced-learn
    from imblearn.over_sampling import SMOTENC
    """
    SMOTE 是机器学习里非常经典的：少数类过采样（Oversampling）技术。全称：
    Synthetic Minority Over-sampling Technique 它专门解决：类别不平衡 问题。
    什么是类别不平衡，例如：银行欺诈检测：
        类别	数量
        正常交易	9900
        欺诈交易	100
    此时：欺诈样本太少，模型会倾向：全部预测正常，甚至：准确率99%，但：毫无意义
    因为：少数类完全学不到，最简单的解决办法
        方法1：欠采样（Undersampling）
            减少多数类：9900 → 100
            问题：浪费大量数据
        方法2：过采样（Oversampling）
            增加少数类。最简单：直接复制少数类，例如：100 → 1000
            问题：容易过拟合，模型会：死记硬背
        SMOTE 的核心思想：SMOTE 不直接复制。而是：“人工合成新的少数类样本”。
        所以：Synthetic，就是：合成
    """
    # print(y.value_counts())
    # smote = SMOTE(random_state=43, sampling_strategy='minority', k_neighbors=5)
    smote_nc = SMOTENC(random_state=43, sampling_strategy='minority', k_neighbors=5,
                       categorical_features=categorical_features)
    X_smote, y_smote = smote_nc.fit_resample(X, y)
    data_smote = pd.concat(
        [
            X_smote,
            y_smote.to_frame()
        ],
        axis=1
    )

    # print(y_smote.value_counts())
    return data_smote

def feature_engineering(rebuild=False):
    # 输出路径
    output_path = Path(__file__).parent.joinpath('./datas/processed_data.csv')

    if output_path.exists() and not rebuild:
        return output_path
    # 数据空值处理
    the_raw_data = process_nan()
    # 标准化
    the_standardized_data = standardize(the_raw_data)
    # 特征编码
    the_encoded_data = feature_encoding(the_standardized_data)
    # 特征选择
    the_selected_features = feature_selection(the_encoded_data)
    # # 数据平衡
    # the_balanced_data = feature_balance(the_selected_features.drop(columns=['Churn']),
    #                                     the_selected_features['Churn'])
    # 数据保存
    the_selected_features.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    feature_engineering(rebuild=True)



