
"""
数据预处理
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

def process_nan():
    """
    加载原始数据
    """
    raw_data_path = 'datas/Telco-Customer-Churn.csv'
    csv_data = pd.read_csv(raw_data_path)

    # print(raw_data.head(10))
    # print(raw_data.info())
    # print(raw_data.describe())
    # 检查数据集中是否存在缺失值
    # isnull().sum() 返回每一列的缺失值数量
    missing_values = csv_data.isnull().sum()
    print("各列缺失值统计：")
    print(missing_values)

    # 判断是否所有列的缺失值都为0
    assert missing_values.sum() == 0

    # 检查每一列是否存在缺失值（返回布尔序列，True表示该列存在至少一个缺失值）
    has_missing = csv_data.isnull().any()
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
    csv_data['TotalCharges'] = pd.to_numeric(csv_data['TotalCharges'], errors='coerce')

    nan_val_rows = csv_data[csv_data['TotalCharges'].isnull()][selected_columns]
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
    csv_data["TotalCharges"] = csv_data["TotalCharges"].fillna(csv_data["MonthlyCharges"])
    # 查看填充后的结果
    # print(csv_data[csv_data['tenure'] == 0][selected_columns])
    # print(csv_data[selected_columns].describe())
    # print(csv_data.describe())

    return csv_data