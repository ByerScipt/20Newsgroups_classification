import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

# 检测设备并打印信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {'CUDA: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

# PyTorch实现PCA类
class TorchPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X_centered = X - X.mean(dim=0)
        _, _, V = torch.pca_lowrank(X_centered, q=self.n_components)
        self.components_ = V[:, :self.n_components]
        return self

    def transform(self, X):
        return torch.mm(X, self.components_)

# 加载并处理数据
data = loadmat('20Newsgroups.mat')
traindata = torch.FloatTensor(data['traindata'].toarray()).to(device)
traingnd = torch.LongTensor(data['traingnd'].flatten())
testdata = torch.FloatTensor(data['testdata'].toarray()).to(device)
testgnd = data['testgnd'].flatten()

# PCA降维并计算解释方差比例
num_components = 80
pca = TorchPCA(n_components=num_components).fit(traindata)
traindata_pca = pca.transform(traindata)
testdata_pca = pca.transform(testdata)
explained_variance_ratio = traindata_pca.var(dim=0).sum() / traindata.var(dim=0).sum()
print(f'PCA降维后，前{num_components}个主成分解释的方差比例为: {explained_variance_ratio*100:.2f}%')

# PyTorch实现KNN类
class TorchKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y.to(X.device)

    def predict(self, X_test):
        X_test_norm = X_test / X_test.norm(dim=1, keepdim=True)
        X_norm = self.X / self.X.norm(dim=1, keepdim=True)
        sims = torch.mm(X_test_norm, X_norm.T)
        _, indices = torch.topk(sims, self.k, largest=True)
        votes = self.y[indices]
        return torch.mode(votes, dim=1).values.cpu().numpy()

class TorchKNNEnsemble:
    def __init__(self, k_list):
        self.models = [TorchKNN(k) for k in k_list]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X_test):
        preds = torch.stack([torch.from_numpy(model.predict(X_test)) for model in self.models])
        return torch.mode(preds, dim=0).values.numpy()

# 训练、预测并评估
knn = TorchKNNEnsemble(k_list=[1, 3, 5, 7])
knn.fit(traindata_pca, traingnd)
test_pred = knn.predict(testdata_pca)
accuracy = accuracy_score(testgnd, test_pred)
print(f'模型准确率: {accuracy:.6f}')
