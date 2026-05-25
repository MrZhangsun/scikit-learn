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
from a01_load_data import process_nan

raw_data = process_nan()

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
# print(raw_data["TotalCharges"].describe())

plt.subplots_adjust(hspace=0.5)
plt.close(fig)


fig2 = plt.figure(figsize=(6, 15), constrained_layout=True)
ax21 = fig2.add_subplot(511)
churn_group = raw_data['Churn'].value_counts()
# l_text是饼图对着文字大小，p_text是饼图内文字大小
patches, l_text, p_text = ax21.pie(churn_group,
                                   labels=['NO', 'YES'],
                                   autopct='%1.2f%%',
                                   startangle=90, # 设置饼图起始角度
                                   explode=(0, 0.1), # 设置饼图分离
                                   radius=2.0,) # 设置饼图半径
ax21.set_title('客户流失正负样本分布图', pad=70)
# for t in l_text:
#     t.set_size(15)
# for t in p_text:
#     t.set_size(15)

"""
由图可知：
- - 性别对客户流失基本没有影响；
- - 年龄对客户流失有影响，老年人流失占比高于年轻人；
- - 是否有配偶对客户流失有影响，无配偶客户流失占比高于有配偶客户；
- - 是否有家属对客户流失有影响，无家属客户流失占比高于有家属客户.
"""
basic_cols = ["gender", "SeniorCitizen", 'Partner', 'Dependents']
for i, col in enumerate(basic_cols):
    ax22 = fig2.add_subplot(5, 1, i + 2)
    ax22.set_title(f"{col}特征正负样本分布图")

    # 交叉统计：每个特征值对应的正负标签分布
    cross = pd.crosstab(raw_data[col], raw_data['Churn'])
    cross.plot.bar(ax=ax22, stacked=True)
    ax22.set_ylabel('样本数量')
    ax22.legend(loc='upper right')

plt.close(fig2)

### 观察流失率与入网月数的关系
tenure_churn = raw_data[["tenure", "Churn"]]
tenure_churn["Churn"] = tenure_churn["Churn"].map({"Yes": 1, "No": 0})
tenure_group = tenure_churn.groupby("tenure")
# 统计所有的流失客户数据量
tenure_yes_rate = (tenure_group.sum() # 每个网龄流失的客户数量
                   / tenure_group.count() # 每个网龄的总客户数量
                   )
print("*" * 90)
print(tenure_yes_rate)

fig3 = plt.figure(figsize=(10, 6))
ax31 = fig3.add_subplot(111)
ax31.plot(tenure_group.groups.keys(), tenure_yes_rate, 'r-', label='网龄流失客户占比/月')
plt.legend()
"""
由图可知：除了刚入网（tenure=0）的客户之外，流失率随着入网时间的延长呈下降趋势；当入网超过两个月时，流失率小于留存率，这段时间可以看做客户的适应期。
"""
plt.close(fig3)

"""
业务特征对客户流失影响
"""

crosses = pd.crosstab(raw_data["Churn"], raw_data["PhoneService"])
# print(crosses)
ps_no = crosses["No"]
ps_yes = crosses["Yes"]
# print(ps_no)
# print(ps_yes)
fig4 = plt.figure(figsize=(10, 6))
ax421 = fig4.add_subplot(211)
ax422 = fig4.add_subplot(212)
ax421.pie(ps_no, labels=['NO', 'YES'],
          autopct='%1.2f%%',
            radius=1.5,
            explode=(0, 0.1),
          )
ax421.set_title('没有订阅PhoneService的正负样本分布', pad=30)

ax422.pie(ps_yes, labels=['NO', 'YES'],
          autopct='%1.2f%%',
            radius=1.5,
            explode=(0, 0.1),
          )
ax422.set_title('订阅PhoneService的正负样本分布', pad=30)
plt.subplots_adjust(hspace=0.5)
plt.close(fig4)

ps = pd.crosstab(raw_data["Churn"], raw_data["MultipleLines"])
ps_no = ps["No"]
ps_yes = ps["Yes"]
ps_no_service = ps["No phone service"]

fig5 = plt.figure(figsize=(10, 6))
ax531 = fig5.add_subplot(131)
ax532 = fig5.add_subplot(132)
ax533 = fig5.add_subplot(133)
ax531.pie(ps_yes, labels=['NO', 'YES'],
          autopct='%1.2f%%',
            radius=1.5,
            explode=(0, 0.1),
          )
ax531.set_title('订阅MultipleLines的正负样本分布', pad=30)
ax532.pie(ps_no, labels=['NO', 'YES'],
          autopct='%1.2f%%',
            radius=1.5,
            explode=(0, 0.1),
          )
ax532.set_title('没有订阅MultipleLines的正负样本分布', pad=30)
ax533.pie(ps_no_service, labels=['NO', 'YES'],
          autopct='%1.2f%%',
            radius=1.5,
            explode=(0, 0.1),
          )
ax533.set_title('没有订阅PhoneService的正负样本分布', pad=30)
plt.subplots_adjust(wspace=0.5)
plt.close(fig5)

"""
由图可知，是否开通多线业务对客户流失影响很小。此外 MultipleLines 取值为 'No'和 'No phone service' 的两种情况基本一致，后续可以合并在一起。
"""

# 互联网业务
cnt = pd.crosstab(raw_data["InternetService"], raw_data["Churn"],)
ax611 = cnt.plot.barh(stacked=True, figsize=(15, 6))
ax611.set_title('互联网业务对客户流失的影响')
fig6 = ax611.get_figure()
plt.close(fig6)
"""
由图可知，未开通互联网的客户总数最少，而流失比例最低（7.40%）；开通光纤网络的客户总数最多，流失比例也最高（41.89%）；
开通数字网络的客户则均居中（18.96%）。可以推测应该有更深层次的因素导致光纤用户流失更多客户，下一步观察与互联网相关的各项业务。
"""

# 与互联网相关的业务
internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
fig7 = plt.figure(figsize=(15, 10))
for i, col in enumerate(internetCols):
    cnt = pd.crosstab(raw_data['Churn'], raw_data[col],)
    print(cnt)
    print("*" * 90)

    ax711 = fig7.add_subplot(len(internetCols), 3, i * 3 + 1)
    ax712 = fig7.add_subplot(len(internetCols), 3, i * 3 + 2)
    ax713 = fig7.add_subplot(len(internetCols), 3, i * 3 + 3)

    ax711.pie(cnt["No"], labels=['NO', 'YES'],
              autopct='%1.2f%%',
              radius=1.5,
              explode=(0, 0.1),
              )
    ax711.set_title(f'没有开通{col}的业务正负样本分布', pad=30)
    ax712.pie(cnt["Yes"], labels=['NO', 'YES'],
              autopct='%1.2f%%',
              radius=1.5,
              explode=(0, 0.1),
              )
    ax712.set_title(f'开通{col}的业务正负样本分布', pad=30)
    ax713.pie(cnt["No internet service"], labels=['NO', 'YES'],
              autopct='%1.2f%%',
              radius=1.5,
              explode=(0, 0.1),
              )
    ax713.set_title(f'没有开通InternetService的业务正负样本分布', pad=30)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.close(fig7)

# 合约期限
contract_types = raw_data['Contract'].value_counts()
cnt = pd.crosstab(raw_data['Contract'], raw_data['Churn'],)
print(cnt)
ax811 = cnt.plot.barh(stacked=True, figsize=(15, 6))
fig8 = ax811.get_figure()
plt.close(fig8)

# 是否采用电子结算
pb_types = raw_data['PaperlessBilling'].value_counts()
cnt = pd.crosstab(raw_data['PaperlessBilling'], raw_data['Churn'],)
print(cnt)
ax911 = cnt.plot.barh(stacked=True, figsize=(15, 6))
fig9 = ax911.get_figure()
plt.close(fig9)

pm_types = raw_data['PaymentMethod'].value_counts()
# 付款方式
cnt = pd.crosstab(raw_data['PaymentMethod'], raw_data['Churn'],)
ax1011 = cnt.plot.barh(stacked=True, figsize=(15, 6))
fig10 = ax1011.get_figure()
plt.close(fig10)

# 每月费用核密度估计图
no_df = raw_data[raw_data['Churn'] == 'No']
no_yes = raw_data[raw_data['Churn'] == 'Yes']

fig11 = plt.figure(figsize=(10, 6))
ax111 = fig11.add_subplot(111)
sns.kdeplot(no_df['MonthlyCharges'], label="No", ax=ax111)
sns.kdeplot(no_yes['MonthlyCharges'], label="Yes", ax=ax111)
ax111.legend()
plt.close(fig11)

# 总费用核密度估计图
fig12 = plt.figure(figsize=(10, 6))
ax121 = fig12.add_subplot(111)
sns.kdeplot(no_df['TotalCharges'], label="No", ax=ax121)
sns.kdeplot(no_yes['TotalCharges'], label="Yes", ax=ax121)
ax121.legend()
plt.close(fig12)

# 特征相关性
fig13 = plt.figure(figsize=(10, 6))
ax131 = fig13.add_subplot(111)

nu_fea = raw_data[['tenure', 'MonthlyCharges', 'TotalCharges']]
list_fea = list(nu_fea)
pearson_mat = raw_data[list_fea].corr(method='spearman')  # 计算皮尔逊相关系数矩阵
sns.heatmap(pearson_mat, square=True, annot=True, cmap="YlGnBu", ax=ax131)  # 用热度图表示相关系数矩阵

plt.show()

