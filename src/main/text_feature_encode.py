from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
"""
文本特征提取方式：
1. Bag of Words（词袋模型）：只看词出现次数，不考虑顺序，例如：
    文本1：I love AI
    文本2：I love NLP
    词表：[I, love, AI, NLP]
    编码后的向量：
    文本1：[1, 1, 1, 0]
    文本2：[1, 1, 0, 1]
优点：简单
缺点：丢失语序信息，对词表的完整性有较高要求


2. TF-IDF:常见词权重低，稀有词更重要，公式（核心）：TF-IDF=TF * log(N/DF)
    TF：词频
    DF：包含该词的文档数
    N：总文档数
    ✔ 优点：比词袋更有区分度
    ❌ 缺点：仍然不理解语义
    
3. 嵌入法：
"""
corpus = [
    "this is spark spark sql",
    "spark hadoop hbase",
    "this is sample",
    "this is another example another example",
    "spark hbase hadoop spark hive hbase hue oozie",
    "hue oozie spark"
]


"""
# 1️⃣ 词袋模型（CountVectorizer）
输出含义：
每列 = 一个词
每个值 = 词频
"""
cv = CountVectorizer()
X_bow = cv.fit_transform(corpus)

print("词表：", cv.get_feature_names_out())
print("向量：\n", X_bow.toarray())

"""
2️⃣ TF-IDF（最常用🔥）
和词袋区别：
不再是“次数”
而是“重要程度”
"""
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

print("词表：", tfidf.get_feature_names_out())
print("TF-IDF向量：\n", X_tfidf.toarray())

"""
3️⃣ N-gram（考虑词序）
"""
cv_ngram = CountVectorizer(ngram_range=(1, 2))  # unigram + bigram
X_ngram = cv_ngram.fit_transform(corpus)

print("Ngram词表：", cv_ngram.get_feature_names_out())
print("Ngram向量：\n", X_ngram.toarray())