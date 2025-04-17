import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse
import matplotlib.pyplot as plt

# 加载mat文件
data = loadmat('20Newsgroups.mat')

# 打印所有键
print("Mat文件中的所有键:")
for key in data.keys():
    print(key)

# 分析每个键对应的数据
for key in data.keys():
    if key.startswith('__'):  # 跳过MATLAB的元数据
        continue
    value = data[key]
    print(f"\n键: {key}")
    print(f"数据类型: {type(value)}")
    if isinstance(value, np.ndarray):
        print(f"形状: {value.shape}")
        print(f"数据类型: {value.dtype}")
    elif issparse(value):
        print(f"稀疏矩阵形状: {value.shape}")
        print(f"稀疏矩阵数据类型: {value.dtype}")
        print(f"非零元素数量: {value.nnz}")
        sparsity = 1 - (value.nnz / (value.shape[0] * value.shape[1]))
        print(f"稀疏度: {sparsity * 100:.2f}%")
    else:
        print(f"值: {value}")

# 分析标签分布
if 'traingnd' in data and 'testgnd' in data:
    train_labels = data['traingnd'].flatten()
    test_labels = data['testgnd'].flatten()

    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)

    print("\n训练集标签分布:")
    for label, count in zip(unique_train_labels, train_counts):
        print(f"标签 {label}: {count} 个样本")

    print("\n测试集标签分布:")
    for label, count in zip(unique_test_labels, test_counts):
        print(f"标签 {label}: {count} 个样本")

    # 绘制训练集标签分布柱状图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(unique_train_labels, train_counts)
    plt.title('Training Set Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Number of Samples')

    # 绘制测试集标签分布柱状图
    plt.subplot(1, 2, 2)
    plt.bar(unique_test_labels, test_counts)
    plt.title('Test Set Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Number of Samples')

    plt.tight_layout()
    plt.show()