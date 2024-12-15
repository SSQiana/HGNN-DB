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


p = 4019
pap = np.load(prefix + '/pap.npy')
pap = pap / pap.sum(axis=-1).reshape(-1, 1)
psp = np.load(prefix + '/psp.npy')

psp = psp / psp.sum(axis=-1).reshape(-1,1)
mp = (pap + psp)
onemp=np.where(mp!=0,1,0)
mp = mp/mp.sum(-1)


features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
print(features_0.shape)
fsim = cos(features_0)
fsim=fsim*onemp
fsim= fsim/fsim.sum(-1).reshape(-1,1)


emb = np.load(prefix + '/metapath2vec_emb.npy')
tsim = cos(emb[:4019])
tsim=tsim*onemp
tsim= tsim/tsim .sum(-1).reshape(-1,1)
pos_num =150
all =mp+fsim
# print(all_.max(),all_.min(),all_.mean())
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
c=0
d=0
for i in range(4019):
  for j in range(4019):
    if pos[i][j]==1:
      if label[i]==label[j]:
        c=c+1
      if label[i] != label[j]:
        d=d+1
print(pos.sum(1).sum(0))
print("zhengque",c/(c+d))
print("cuowu",d/(c+d))

print(pos.shape)
pos = sp.coo_matrix(pos)
sp.save_npz("F:\Desktop\stACM\duibi (1) (2)\新建文件夹\ACM\pos_15.npz", pos)
