"""
决策树

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from collections import Counter

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题
"""
计算信息熵 H(X) = - sum(p * log(p, 2)
"""
def calc_entropy(x):
    # 计算信息熵 H(X) = - sum(p * log(p, 2)
    return -np.sum([p * np.log2(p) for p in x])

def H(x):
    return calc_entropy(x)
#
# print(calc_entropy((0.5, 0.5)))
# print(calc_entropy((0.25, 0.25, 0.25, 0.25)))
#
# print(calc_entropy([0.25, 0.25, 0.25, 0.25]))
# print(calc_entropy([0.65, 0.25, 0.05, 0.05]))
# print(calc_entropy([0.97, 0.01, 0.01, 0.01]))

"""
计算信息增益
Gain = H(D) - H(D|A)
"""
def gain(d, a):
    return calc_entropy(d) - calc_entropy(a)

"""
计算基尼系数
gini = 1 - sum(p^2)
"""
def gini(p):
    return 1 - np.sum([i * i for i in p])

"""
计算错误率
error = 1 - max(p)
"""
def error(p):
    return 1 - np.max(p)


"""
不纯度指标图像
"""
def show_purity():
    plt.figure(num="不纯度")
    P0 = np.linspace(0.0, 1.0, 1000)  # 这个表示在0到1之间生成1000个p值(另外的就是1-p)
    P1 = 1 - P0
    y1 = [H([p0, 1 - p0]) * 0.5 for p0 in P0]  ##熵之半
    y2 = [gini([p0, 1 - p0]) for p0 in P0]
    y3 = [error([p0, 1 - p0]) for p0 in P0]
    plt.plot(P0, y1, label=u'熵之半')
    plt.plot(P0, y2, label=u'gini')
    plt.plot(P0, y3, label=u'error')
    plt.legend(loc='upper left')
    plt.xlabel(u'p', fontsize=18)
    plt.ylabel(u'不纯度', fontsize=18)
    plt.title(u'不纯度指标', fontsize=20)
    plt.show()  # 绘制图像


def calc_entropy_from_labels(labels):
    """
    根据标签列表自动计算熵

    参数:
    labels: list - 标签列表，例如 ["是", "否", "是", "是", ...]

    返回:
    entropy: float - 熵值
    """
    total = len(labels)
    if total == 0:
        return 0

    # 自动统计各类别数量
    label_counts = Counter(labels)

    # 自动计算概率并累加熵
    entropy = 0
    for count in label_counts.values():
        p = count / total
        entropy -= p * np.log2(p)

    return entropy


def calc_conditional_entropy_and_gain(feature_values, labels, base_entropy=None):
    """
    计算特征的条件熵和信息增益（完全自动化）

    参数:
    feature_values: list - 特征值列表，例如 ["晴", "晴", "多云", "雨", ...]
    labels: list - 对应的标签列表，例如 ["是", "否", ...]
    base_entropy: float - 父节点的熵，如果不提供则自动计算

    返回:
    conditional_entropy: float - 条件熵 H(D|A)
    info_gain: float - 信息增益 Gain(D, A)
    feature_stats: dict - 每个特征值的统计信息
    """
    # 自动计算根节点熵（如果未提供）
    if base_entropy is None:
        base_entropy = calc_entropy_from_labels(labels)
        print(f"自动计算根节点熵: {base_entropy:.4f}")

    # 统计每个特征值对应的标签分布
    feature_stats = {}

    # 遍历每个特征值及其对应的标签
    for feat_val, label in zip(feature_values, labels):
        if feat_val not in feature_stats:
            feature_stats[feat_val] = {"total": 0, "labels": []}

        feature_stats[feat_val]["total"] += 1
        feature_stats[feat_val]["labels"].append(label)

    # 计算条件熵
    total_samples = len(labels)
    conditional_entropy = 0

    print(f"\n特征值统计：")
    for feat_val, stats in feature_stats.items():
        prob = stats["total"] / total_samples  # P(特征值)

        # 自动计算该特征值下的熵
        entropy = calc_entropy_from_labels(stats["labels"])

        # 累加条件熵
        conditional_entropy += prob * entropy

        # 打印详细信息
        label_counts = Counter(stats["labels"])
        labels_info = ", ".join([f"{label}:{count}" for label, count in label_counts.items()])
        print(f"  {feat_val}: {stats['total']}个样本 ({labels_info}), 熵: {entropy:.4f}")

    # 计算信息增益
    info_gain = base_entropy - conditional_entropy

    return conditional_entropy, info_gain, feature_stats

def build_decision_tree():
    """
    构建决策树实现经典“是否打网球”问题，数据：
        | 编号 | 天气 | 温度 | 湿度 | 风 | 是否打球 |
        | -- | -- | -- | -- | - | ---- |
        | 1  | 晴  | 高  | 高  | 弱 | 否    |
        | 2  | 晴  | 高  | 高  | 强 | 否    |
        | 3  | 多云 | 高  | 高  | 弱 | 是    |
        | 4  | 雨  | 中  | 高  | 弱 | 是    |
        | 5  | 雨  | 低  | 正常 | 弱 | 是    |
        | 6  | 雨  | 低  | 正常 | 强 | 否    |
        | 7  | 多云 | 低  | 正常 | 强 | 是    |
        | 8  | 晴  | 中  | 高  | 弱 | 否    |
        | 9  | 晴  | 低  | 正常 | 弱 | 是    |
        | 10 | 雨  | 中  | 正常 | 弱 | 是    |
        | 11 | 晴  | 中  | 正常 | 强 | 是    |
        | 12 | 多云 | 中  | 高  | 强 | 是    |
        | 13 | 多云 | 高  | 正常 | 弱 | 是    |
        | 14 | 雨  | 中  | 高  | 强 | 否    |

    :return:
    """
    # 特征数据
    X = [
        ["晴", "高", "高", "弱"],
        ["晴", "高", "高", "强"],
        ["多云", "高", "高", "弱"],
        ["雨", "中", "高", "弱"],
        ["雨", "低", "正常", "弱"],
        ["雨", "低", "正常", "强"],
        ["多云", "低", "正常", "强"],
        ["晴", "中", "高", "弱"],
        ["晴", "低", "正常", "弱"],
        ["雨", "中", "正常", "弱"],
        ["晴", "中", "正常", "强"],
        ["多云", "中", "高", "强"],
        ["多云", "高", "正常", "弱"],
        ["雨", "中", "高", "强"]
    ]

    # 标签
    Y = [
        "否", "否", "是", "是", "是", "否", "是",
        "否", "是", "是", "是", "是", "是", "否"
    ]

    # 自动计算根节点熵
    hd = calc_entropy_from_labels(Y)
    print(f"根节点熵 H(D): {hd:.4f}\n")

    # 计算每个特征的信息增益
    features_names = ["天气", "温度", "湿度", "风力"]

    print("=" * 60)
    for i, feature_name in enumerate(features_names):
        print(f"\n特征: {feature_name}")
        print("-" * 40)

        # 提取特征列
        feature_values = [row[i] for row in X]

        # 自动计算条件熵和信息增益
        conditional_entropy, gain, stats = calc_conditional_entropy_and_gain(
            feature_values, Y, hd
        )

        print(f"\n条件熵 H(D|{feature_name}): {conditional_entropy:.4f}")
        print(f"信息增益 Gain(D, {feature_name}): {gain:.4f}")

    # 简化版：直接计算天气特征
    print("\n" + "=" * 60)
    print("\n简化版 - 天气特征计算：")
    print("-" * 40)

    weather_values = [row[0] for row in X]
    hd_weather, gain_weather, _ = calc_conditional_entropy_and_gain(weather_values, Y)
    print(f"\n天气的条件熵: {hd_weather:.4f}")
    print(f"天气的信息增益: {gain_weather:.4f}")


if __name__ == '__main__':
    build_decision_tree()
    pass
