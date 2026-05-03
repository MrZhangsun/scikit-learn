import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import numpy as np
from collections import Counter

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题
# 熵的计算
def entropy(labels):
    """计算熵"""
    total = len(labels)
    if total == 0:
        return 0

    label_counts = Counter(labels)
    ent = 0
    for count in label_counts.values():
        p = count / total
        ent -= p * np.log2(p)
    return ent


# 信息增益
def info_gain(X_feature, Y, base_entropy):
    """
    信息增益 = 父节点熵 - 条件熵

    ID3选择信息增益最大的特征
    """
    # 按特征值分组
    groups = {}
    for feat_val, label in zip(X_feature, Y):
        if feat_val not in groups:
            groups[feat_val] = []
        groups[feat_val].append(label)

    # 计算条件熵
    total = len(Y)
    conditional_entropy = 0
    for group_labels in groups.values():
        prob = len(group_labels) / total
        conditional_entropy += prob * entropy(group_labels)

    # 信息增益
    gain = base_entropy - conditional_entropy
    return gain


def split_info(X_feature):
    """
    分裂信息：衡量特征的分支数量和信息
    """
    total = len(X_feature)
    feat_counts = Counter(X_feature)

    split = 0
    for count in feat_counts.values():
        p = count / total
        split -= p * np.log2(p)

    return split


def gain_ratio(X_feature, Y, base_entropy):
    """
    信息增益率 = 信息增益 / 分裂信息

    C4.5选择信息增益率最大的特征
    """
    gain = info_gain(X_feature, Y, base_entropy)
    split = split_info(X_feature)

    # 避免除以0
    if split == 0:
        return 0

    return gain / split


def gini(labels):
    """
    基尼系数：衡量数据集的不纯度
    值越小，纯度越高
    """
    total = len(labels)
    if total == 0:
        return 0

    label_counts = Counter(labels)
    gini_val = 1
    for count in label_counts.values():
        p = count / total
        gini_val -= p ** 2

    return gini_val


def weighted_gini(X_feature, Y):
    """
    加权基尼系数（用于特征选择）

    CART选择加权基尼系数最小的特征
    """
    # 按特征值分组
    groups = {}
    for feat_val, label in zip(X_feature, Y):
        if feat_val not in groups:
            groups[feat_val] = []
        groups[feat_val].append(label)

    # 计算加权基尼系数
    total = len(Y)
    weighted_gini_val = 0
    for group_labels in groups.values():
        prob = len(group_labels) / total
        weighted_gini_val += prob * gini(group_labels)

    return weighted_gini_val

class DecisionTreeCriteria:
    """三种分裂准则的完整实现"""

    def __init__(self, criterion='id3'):
        """
        criterion: 'id3', 'c45', 'cart'
        """
        self.criterion = criterion

    def entropy(self, labels):
        """计算熵"""
        if len(labels) == 0:
            return 0
        counts = Counter(labels)
        probs = [c / len(labels) for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def gini(self, labels):
        """计算基尼系数"""
        if len(labels) == 0:
            return 0
        counts = Counter(labels)
        probs = [c / len(labels) for c in counts.values()]
        return 1 - sum(p ** 2 for p in probs)

    def split_info(self, feature_values):
        """计算分裂信息"""
        counts = Counter(feature_values)
        probs = [c / len(feature_values) for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def info_gain(self, feature_values, labels, base_entropy):
        """信息增益 (ID3)"""
        # 分组
        groups = {}
        for f, l in zip(feature_values, labels):
            groups.setdefault(f, []).append(l)

        # 条件熵
        cond_entropy = sum(
            (len(g) / len(labels)) * self.entropy(g)
            for g in groups.values()
        )

        return base_entropy - cond_entropy

    def gain_ratio(self, feature_values, labels, base_entropy):
        """信息增益率 (C4.5)"""
        gain = self.info_gain(feature_values, labels, base_entropy)
        split = self.split_info(feature_values)

        return gain / split if split > 0 else 0

    def weighted_gini(self, feature_values, labels):
        """加权基尼系数 (CART)"""
        groups = {}
        for f, l in zip(feature_values, labels):
            groups.setdefault(f, []).append(l)

        weighted_gini = sum(
            (len(g) / len(labels)) * self.gini(g)
            for g in groups.values()
        )

        return weighted_gini

    def best_feature(self, X, Y, feature_names=None):
        """选择最佳分裂特征"""
        base_entropy = self.entropy(Y)
        best_idx = -1
        best_score = -1 if self.criterion != 'cart' else float('inf')

        for i in range(len(X[0])):
            feature_values = [row[i] for row in X]

            if self.criterion == 'id3':
                score = self.info_gain(feature_values, Y, base_entropy)
                # ID3: 最大化信息增益
                if score > best_score:
                    best_score = score
                    best_idx = i

            elif self.criterion == 'c45':
                score = self.gain_ratio(feature_values, Y, base_entropy)
                # C4.5: 最大化信息增益率
                if score > best_score:
                    best_score = score
                    best_idx = i

            elif self.criterion == 'cart':
                score = self.weighted_gini(feature_values, Y)
                # CART: 最小化基尼系数
                if score < best_score:
                    best_score = score
                    best_idx = i

        return best_idx, best_score


# 测试数据（打球数据集）
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

Y = ["否", "否", "是", "是", "是", "否", "是",
     "否", "是", "是", "是", "是", "是", "否"]

feature_names = ["天气", "温度", "湿度", "风力"]


# 比较三种准则
def compare_criteria():
    """比较三种分裂准则"""

    results = {}

    for criterion in ['id3', 'c45', 'cart']:
        dt = DecisionTreeCriteria(criterion=criterion)

        print(f"\n{'=' * 60}")
        print(f"使用 {criterion.upper()} 准则")
        print('=' * 60)

        base_entropy = dt.entropy(Y)
        print(f"根节点熵: {base_entropy:.4f}\n")

        for i, feature_name in enumerate(feature_names):
            feature_values = [row[i] for row in X]

            if criterion == 'id3':
                score = dt.info_gain(feature_values, Y, base_entropy)
                print(f"{feature_name}: 信息增益 = {score:.4f}")

            elif criterion == 'c45':
                gain = dt.info_gain(feature_values, Y, base_entropy)
                ratio = dt.gain_ratio(feature_values, Y, base_entropy)
                split = dt.split_info(feature_values)
                print(f"{feature_name}: 增益率 = {ratio:.4f} (信息增益={gain:.4f}, 分裂信息={split:.4f})")

            elif criterion == 'cart':
                score = dt.weighted_gini(feature_values, Y)
                print(f"{feature_name}: 加权基尼系数 = {score:.4f}")

        # 选择最佳特征
        best_idx, best_score = dt.best_feature(X, Y, feature_names)
        print(f"\n最佳分裂特征: {feature_names[best_idx]} (得分: {best_score:.4f})")
        results[criterion] = feature_names[best_idx]

    return results


compare_criteria()


def comparison_table():
    """三种算法的详细对比"""

    comparison = pd.DataFrame({
        '特性': [
            '全称',
            '分裂准则',
            '计算公式',
            '选择策略',
            '处理连续值',
            '处理缺失值',
            '剪枝方法',
            '分支方式',
            '适用场景',
            '主要缺点'
        ],
        'ID3': [
            'Iterative Dichotomiser 3',
            '信息增益',
            'Gain(D,A) = Ent(D) - Σ|Dv|/|D|·Ent(Dv)',
            '最大化信息增益',
            '不支持',
            '不支持',
            '预剪枝',
            '多叉树',
            '离散特征分类',
            '偏向多值特征'
        ],
        'C4.5': [
            'ID3的改进版本',
            '信息增益率',
            'Gain_ratio = Gain(D,A) / SplitInfo(A)',
            '最大化信息增益率',
            '支持（二分法）',
            '支持（概率加权）',
            '后剪枝',
            '多叉树',
            '通用分类问题',
            '计算复杂度高'
        ],
        'CART': [
            'Classification and Regression Tree',
            '基尼系数 / 均方误差',
            'Gini(D) = 1 - Σp²',
            '最小化基尼系数',
            '支持',
            '支持（代理分裂）',
            '后剪枝（CCP）',
            '二叉树',
            '分类+回归',
            '二叉树可能不自然'
        ]
    })

    print("=" * 80)
    print("ID3, C4.5, CART 详细对比")
    print("=" * 80)
    print(comparison.to_string(index=False))

    return comparison


comparison_table()


def visualize_criteria():
    """可视化三种准则的计算过程"""

    # 创建不同纯度的数据集
    datasets = {
        '纯数据集': [1, 1, 1, 1, 1],
        '较纯数据集': [1, 1, 1, 1, 0],
        '中等纯度': [1, 1, 1, 0, 0],
        '较不纯': [1, 1, 0, 0, 0],
        '不纯数据集': [1, 0, 1, 0, 1]
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 1. 熵 vs 基尼系数对比
    ax = axes[0]
    purities = np.linspace(0, 1, 100)
    entropies = []
    ginis = []

    for p in purities:
        # 二分类，正例概率为p
        if p == 0 or p == 1:
            entropies.append(0)
            ginis.append(0)
        else:
            entropies.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
            ginis.append(1 - p ** 2 - (1 - p) ** 2)

    ax.plot(purities, entropies, 'b-', linewidth=2, label='熵 (Entropy)')
    ax.plot(purities, ginis, 'r-', linewidth=2, label='基尼系数 (Gini)')
    ax.set_xlabel('正例概率')
    ax.set_ylabel('不纯度')
    ax.set_title('熵 vs 基尼系数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 不同数据集的比较
    ax = axes[1]
    names = list(datasets.keys())
    ent_vals = [entropy(d) for d in datasets.values()]
    gini_vals = [gini(d) for d in datasets.values()]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, ent_vals, width, label='熵', color='blue', alpha=0.7)
    ax.bar(x + width / 2, gini_vals, width, label='基尼系数', color='red', alpha=0.7)
    ax.set_xlabel('数据集')
    ax.set_ylabel('不纯度值')
    ax.set_title('不同数据集上的不纯度比较')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 信息增益示例
    ax = axes[2]
    # 模拟分裂前后的熵
    parent_entropy = 0.94
    child_entropies = [0.5, 0.2, 0.8]
    weights = [0.3, 0.4, 0.3]

    cond_entropy = sum(w * e for w, e in zip(weights, child_entropies))
    info_gain = parent_entropy - cond_entropy

    ax.bar(['父节点', '子节点1', '子节点2', '子节点3'],
           [parent_entropy] + child_entropies,
           color=['gray', 'lightblue', 'lightblue', 'lightblue'])
    ax.axhline(y=cond_entropy, color='red', linestyle='--', label=f'加权平均={cond_entropy:.3f}')
    ax.set_ylabel('熵值')
    ax.set_title(f'信息增益示例\n信息增益 = {info_gain:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 信息增益率示例
    ax = axes[3]
    # 模拟不同分支数的特征
    features = {
        '特征A (2分支)': [0.5, 0.5],
        '特征B (4分支)': [0.25, 0.25, 0.25, 0.25],
        '特征C (10分支)': [0.1] * 10
    }

    x = np.arange(len(features))
    gains = [0.3, 0.32, 0.35]  # 信息增益
    split_infos = [-sum(p * np.log2(p) for p in probs) for probs in features.values()]
    gain_ratios = [g / s if s > 0 else 0 for g, s in zip(gains, split_infos)]

    width = 0.25
    ax.bar(x - width, gains, width, label='信息增益', color='blue', alpha=0.7)
    ax.bar(x, split_infos, width, label='分裂信息', color='green', alpha=0.7)
    ax.bar(x + width, gain_ratios, width, label='信息增益率', color='red', alpha=0.7)
    ax.set_xlabel('特征')
    ax.set_ylabel('值')
    ax.set_title('ID3 vs C4.5: 多值特征问题')
    ax.set_xticks(x)
    ax.set_xticklabels(features.keys())
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 三种算法的决策边界对比
    ax = axes[4]
    # 简化的2D示例
    np.random.seed(42)
    X_data = np.random.randn(200, 2)
    y_data = (X_data[:, 0] ** 2 + X_data[:, 1] ** 2 > 1).astype(int)

    from sklearn.tree import DecisionTreeClassifier

    models = {
        'ID3 (熵)': DecisionTreeClassifier(criterion='entropy', max_depth=3),
        'CART (基尼)': DecisionTreeClassifier(criterion='gini', max_depth=3)
    }

    # 绘制决策边界
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_data, y_data)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 只在第一个子图显示
        if i == 0:
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
            ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
                       c='blue', label='类别0', alpha=0.5)
            ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
                       c='red', label='类别1', alpha=0.5)
            ax.set_xlabel('特征1')
            ax.set_ylabel('特征2')
            ax.set_title('决策边界对比 (ID3/CART)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 6. 总结对比
    ax = axes[5]
    ax.axis('off')
    summary_text = """
    核心区别总结:

    ID3 (信息增益):
    • 选择信息增益最大的特征
    • 偏向选择取值多的特征
    • 只能处理离散特征

    C4.5 (信息增益率):
    • 选择信息增益率最大的特征
    • 克服了ID3的偏向问题
    • 支持连续值和缺失值

    CART (基尼系数):
    • 选择基尼系数最小的特征
    • 构建二叉树
    • 支持分类和回归
    • 计算速度快
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


visualize_criteria()