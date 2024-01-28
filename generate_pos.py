import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos

prefix = r'D:\yangchun\model\new_model\heco_hgcml\pos'

p = 6564
pap= sp.load_npz(prefix + '\\aminer_pap.npz')
pap = pap / pap.sum(axis=-1).reshape(-1, 1)  # 归一化
prp = sp.load_npz(prefix + '\\aminer_prp.npz')
prp = prp / prp.sum(axis=-1).reshape(-1, 1)
# apvpa = sp.load_npz(prefix + '\\dblp_apvpa.npz')
# apvpa = apvpa / apvpa.sum(axis=-1).reshape(-1, 1)  # 归一化
mp = (pap + prp)
mp = mp/mp.sum(-1)

# features_0 = sp.load_npz(prefix + '/dblp_features.npz').toarray()
# fsim = cos(features_0)
# fsim = fsim / fsim.sum(-1).reshape(-1, 1)

alpha = 0.85
I1 = np.eye(pap.shape[0])
sim_matrix1 = alpha*np.linalg.inv((I1-(1-alpha)*(pap/np.sum(pap, axis=1).reshape(1, -1)))).astype(np.float32)
I2 = np.eye(prp.shape[0])
sim_matrix2 = alpha*np.linalg.inv((I2-(1-alpha)*(prp/np.sum(prp, axis=1).reshape(1, -1)))).astype(np.float32)
# I2 = np.eye(aptpa.shape[0])
# sim_matrix3 = alpha*np.linalg.inv((I2-(1-alpha)*(aptpa/np.sum(aptpa, axis=1).reshape(1, -1)))).astype(np.float32)
sim = sim_matrix1+sim_matrix2

all = mp + sim

pos_num = 80
array = []
pos = np.zeros((p, p))  # all中保存了两个节点之间元路径的数量
k = 0
for i in range(len(all)):
    one = all[i].nonzero()[1]
    # print(one)
    if len(one) > pos_num:
        oo = np.argsort(-all[i, one])
        sele = one[oo[0, :pos_num]]
        pos[i, sele] = 1
        k += 1
    else:
        pos[i, one] = 1
pos_mp = sp.coo_matrix(pos)
sp.save_npz("new_pos.npz", pos_mp)
