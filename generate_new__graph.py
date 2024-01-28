import numpy as np
import scipy
import torch
import dgl
import torch.nn.functional as F
import scipy.sparse as sp
import time
from scipy.sparse import csr_matrix, save_npz
prefix = r'dataset/ACM'
features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').A  # 节点类型0的特征，4019行4000列  p-a-p 只加载p的

labels = np.load(prefix + '/labels.npy')  # 加载标签，4019

train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引

adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()

num_target = 4019

# 初始化存储一阶邻居的列表
neighbors = []

# 遍历目标节点
for i in range(num_target):
    # 找到对应行的索引为1的列号,即一阶邻居
    temp = np.where(adjM[i, i:] > 0)[0]
    # 将索引保存
    neighbors.append(temp)


# neighbors 中每一行存储一个目标节点的一阶邻居索引
print(neighbors)




# element = [[0, 1, 0, 0],
#            [1, 0, 1, 1],
#            [0, 1, 0, 1],
#            [0, 1, 1, 0]]
#
# matrix = np.array(element)
# copy_matrix = matrix.copy()
# for i in range(4):
#     for j in range(2):
#         if matrix[i][j] == 1 and i >= j:
#             copy_matrix[i][j] = 0
#             copy_matrix[j][i] = 0  # 邻接矩阵是对称的
#             # first_neigh.append(j)
#             for k in range(4):
#                 if k != j and matrix[i][k] == 1:
#                     copy_matrix[j][k] = 1
#                     copy_matrix[k][j] = 1
#
# print(copy_matrix)
