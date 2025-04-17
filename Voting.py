from scipy.io import loadmat
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

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

# 初始化分类器
multinomial_nb = MultinomialNB()
sgd_clf = SGDClassifier()
logreg_clf = LogisticRegression(max_iter=1000)
linear_svc = LinearSVC()
random_forest_clf = RandomForestClassifier()

# 创建投票分类器
voting_clf = VotingClassifier(
    estimators=[
        ('mnb', multinomial_nb),
        ('sgd', sgd_clf),
        ('logreg', logreg_clf),
        ('linear_svc', linear_svc),
        ('rf', random_forest_clf)
    ],
    voting='hard' 
)

# 训练投票分类器
voting_clf.fit(train_vectors_tfidf, train_target)

# 预测并评估
pred = voting_clf.predict(test_vectors_tfidf)
print("\n投票分类器评估结果:")
print("F1 分数 (macro):", f1_score(test_target, pred, average='macro'))
print("准确率:", accuracy_score(test_target, pred))