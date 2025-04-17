代码逻辑顺序参考classification.ipynb，包括各种尝试、优化及运行结果。

使用的算法包括：

KNN(accuracy: 0.732608)

GaussianNB(accuracy: 0.711630)

BernoulliNB(accuracy: 0.790361)

MultinomialNB(accuracy: 0.842538)

SGDClassifier(accuracy: 0.863913)

LogisticRegression(accuracy: 0.860063)

LinearSVC(accuracy: 0.859930)

RandomForestClassifier(accuracy: 0.772968)

集成学习模型: 使用MultinomialNB、SGDClassifier、LogisticRegression、LinearSVC、RandomForestClassifier进行硬投票(accuracy: 0.866171)

也可以直接通过kaggle notebook运行ipynb文件(避免本地部署): https://www.kaggle.com/code/byerscrip/20newsgroups-classification