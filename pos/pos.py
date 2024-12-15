import numpy as np
import numpy as np
import scipy.sparse as sp
import scipy
import pickle
import torch.nn as nn
import torch
import dgl
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as cos

prefix=r'F:\Desktop\stACM\duibi (1) (2)\新建文件夹\ACM'
label = np.load(prefix + '/labels.npy')

#元路径
p = 4019
pap = np.load(prefix + '/pap.npy')
pap = pap / pap.sum(axis=-1).reshape(-1,1)
psp = np.load(prefix + '/psp.npy')

psp = psp / psp.sum(axis=-1).reshape(-1,1)
mp = (pap + psp)
mp = mp/mp.sum(-1)

#特征余弦相似度
features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
fsim = cos(features_0)
fsim= fsim/fsim.sum(-1).reshape(-1,1)


#特征转置
features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
fT=np.dot(features_0,features_0.T)
fT= fT/fT.sum(-1).reshape(-1,1)

#拓扑余弦相似度
emb = np.load(prefix + '/metapath2vec_emb.npy')
tsim = cos(emb[:4019])
tsim= tsim/tsim .sum(-1).reshape(-1,1)

#拓扑转置
emb = np.load(prefix + '/metapath2vec_emb.npy')
tT= np.dot(emb[:4019],emb[:4019].T)
tT= tT/tT.sum(-1).reshape(-1,1)

#ppr
pap = np.load(prefix + '/pap.npy')
psp = np.load(prefix + '/psp.npy')
alpha=0.85
I1 = np.eye(pap.shape[0])
sim_matrix1 = alpha*np.linalg.inv((I1-(1-alpha)*(pap/np.sum(pap, axis=1).reshape(1, -1)))).astype(np.float32)
I2 = np.eye(psp.shape[0])
sim_matrix2 = alpha*np.linalg.inv((I2-(1-alpha)*(psp/np.sum(pap, axis=1).reshape(1, -1)))).astype(np.float32)
sim_matrix=sim_matrix1+sim_matrix2



pos_num =150#选取正样本数
all =mp+fsim#任意组合
# print(all_.max(),all_.min(),all_.mean())
########
#正样本选取前K个
pos = np.zeros((p,p))
k=0
for i in range(len(all)):
  one = all[i].nonzero()[0]
  if len(one) > pos_num:
    oo = np.argsort(-all[i, one])
    sele = one[oo[:pos_num]]
    pos[i, sele] = 1
    k+=1
  else:
    pos[i, one] = 1
#pos = sp.coo_matrix(pos)
sp.save_npz("F:\Desktop\stACM\duibi (1) (2)\新建文件夹\ACM\pos_15.npz", pos)
#########
#########
#计算噪声率，正确率
c=0
d=0
for i in range(4019):
  for j in range(4019):
    if pos[i][j]==1:
      if label[i]==label[j]:
        c=c+1
      if label[i] != label[j]:
        d=d+1
print("zhengque",c/(c+d))
print("cuowu",d/(c+d))


