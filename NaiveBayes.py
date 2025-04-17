#贝叶斯分类器
from scipy.io import loadmat
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer

# 加载 mat 文件
data = loadmat('20Newsgroups.mat')

# 获取训练数据、测试数据以及对应的标签
train_vectors = data['traindata']
test_vectors = data['testdata']
train_target = data['traingnd'].flatten()
test_target = data['testgnd'].flatten()

# 应用 Tfidf 特征工程
tfidf_transformer = TfidfTransformer()
train_vectors_tfidf = tfidf_transformer.fit_transform(train_vectors)
test_vectors_tfidf = tfidf_transformer.transform(test_vectors)

# 将稀疏矩阵转换为稠密数组（供高斯和伯努利朴素贝叶斯使用）
train_vectors_dense = train_vectors_tfidf.toarray()
test_vectors_dense = test_vectors_tfidf.toarray()

# GaussianNB分类器
gaussian_clf = GaussianNB()
gaussian_clf.fit(train_vectors_dense, train_target)
gaussian_pred = gaussian_clf.predict(test_vectors_dense)

print("\nGaussianNB 评估结果:")
print("F1 分数 (macro):", f1_score(test_target, gaussian_pred, average='macro'))
print("准确率:", accuracy_score(test_target, gaussian_pred))

# BernoulliNB 分类器
bernoulli_clf = BernoulliNB(alpha=0.1)
bernoulli_clf.fit(train_vectors_dense, train_target)
bernoulli_pred = bernoulli_clf.predict(test_vectors_dense)

print("\nBernoulliNB 评估结果:")
print("F1 分数 (macro):", f1_score(test_target, bernoulli_pred, average='macro'))
print("准确率:", accuracy_score(test_target, bernoulli_pred))

# MultinomialNB 分类器
multinomial_clf = MultinomialNB(alpha=0.1)
multinomial_clf.fit(train_vectors, train_target) # MultinomialNB分类器不采用TF-IDF特征化后的数据
multinomial_pred = multinomial_clf.predict(test_vectors)

print("\nMultinomialNB 评估结果:")
print("F1 分数 (macro):", f1_score(test_target, multinomial_pred, average='macro'))
print("准确率:", accuracy_score(test_target, multinomial_pred))