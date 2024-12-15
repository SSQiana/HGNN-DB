import scipy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
prefix=r'F:\Desktop\HGCML_acm\HGCML-main\data\acm\raw'


pap = np.load(prefix + '/pap.npy')
psp = np.load(prefix + '/psp.npy')
alpha=0.85
I1 = np.eye(pap.shape[0])
sim_matrix1 = alpha*np.linalg.inv((I1-(1-alpha)*(pap/np.sum(pap, axis=1).reshape(1, -1)))).astype(np.float32)
I2 = np.eye(psp.shape[0])
sim_matrix2 = alpha*np.linalg.inv((I2-(1-alpha)*(psp/np.sum(pap, axis=1).reshape(1, -1)))).astype(np.float32)
sim_matrix=sim_matrix1+sim_matrix2
pos = np.zeros((4019,128))
for i in range(4019):
    b = np.argsort(-(sim_matrix[i]))
    pos[i][0:128]=b[0:128]
np.save(r"F:\Desktop\HGCML_acm\HGCML-main\data\acm\raw\pos_topo.npy",pos)
wad



features = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
dist = cos(features)
pos = np.zeros((4019,128))
for i in range(4019):
    b = np.argsort(-dist[i])
    pos[i][0:128]=b[0:128]
np.save(r"F:\Desktop\HGCML_acm\HGCML-main\data\acm\raw\pos_sem.npy",pos)








