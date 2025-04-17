from scipy.io import loadmat
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score

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
sgd_clf = SGDClassifier()
logreg_clf = LogisticRegression(max_iter=1000)
linear_svc = LinearSVC()
random_forest_clf = RandomForestClassifier()

# 训练和评估模型
def train_and_evaluate(model, train_X, train_y, test_X, test_y, model_name):
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print(f"\n{model_name} 评估结果:")
    print("F1 分数 (macro):", f1_score(test_y, pred, average='macro'))
    print("准确率:", accuracy_score(test_y, pred))

# 训练并评估 SGDClassifier
train_and_evaluate(sgd_clf, train_vectors_tfidf, train_target, test_vectors_tfidf, test_target, "SGDClassifier")

# 训练并评估 LogisticRegression
train_and_evaluate(logreg_clf, train_vectors_tfidf, train_target, test_vectors_tfidf, test_target, "LogisticRegression")

# 训练并评估 LinearSVC
train_and_evaluate(linear_svc, train_vectors_tfidf, train_target, test_vectors_tfidf, test_target, "LinearSVC")

# 训练并评估 RandomForestClassifier
train_and_evaluate(random_forest_clf, train_vectors_tfidf, train_target, test_vectors_tfidf, test_target, "RandomForestClassifier")
