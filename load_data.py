import numpy as np
import scipy
import torch
import dgl
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos


def load_ACM_data(prefix=r'dataset/ACM'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').A

    labels = np.load(prefix + '/labels.npy')

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    PP = scipy.sparse.csr_matrix(adjM[:4019, :4019])
    PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    PSP = scipy.sparse.load_npz(prefix + '/psp.npz')

    g1 = dgl.DGLGraph(PAP)
    g2 = dgl.DGLGraph(PSP)
    g3 = dgl.DGLGraph(PP)
    g = [g1, g2, g3]
    features = torch.FloatTensor(features_0)
    features = F.normalize(features, dim=1, p=2)

    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = train_val_test_idx['train_idx']

    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    pos = scipy.sparse.load_npz(prefix + "/new_pos.npz")
    pos = torch.FloatTensor(pos.todense())

    # features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    # fsim = cos(features_0)
    # fsim = fsim / fsim.sum(-1).reshape(-1, 1)

    return g, features, labels, num_classes, train_idx, val_idx, test_idx, pos


def load_DBLP_data(prefix=r'dataset/DBLP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').A

    labels = np.load(prefix + '/labels.npy')

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    APA = scipy.sparse.load_npz(prefix + '/apa.npz')
    APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz')
    APVPA = scipy.sparse.load_npz(prefix + '/apvpa.npz')

    g1 = dgl.DGLGraph(APA)
    g2 = dgl.DGLGraph(APTPA)
    g3 = dgl.DGLGraph(APVPA)
    g = [g1, g2, g3]
    features = torch.FloatTensor(features_0)

    labels = torch.LongTensor(labels)
    num_classes = 4
    train_idx = train_val_test_idx['train_idx']

    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    pos = scipy.sparse.load_npz(prefix + "/new_pos.npz")
    pos = torch.FloatTensor(pos.todense())

    return g, features, labels, num_classes, train_idx, val_idx, test_idx, pos


def load_YELP_data(prefix=r'dataset/YELP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').A

    labels = np.load(prefix + '/labels.npy')

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy', allow_pickle=True)
    train_val_test_idx = train_val_test_idx.item()
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    BLB = scipy.sparse.load_npz(prefix + '/adj_blb.npz')
    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb.npz')
    BUB = scipy.sparse.load_npz(prefix + '/adj_bub.npz')
    g1 = dgl.DGLGraph(BLB)
    g2 = dgl.DGLGraph(BSB)
    g3 = dgl.DGLGraph(BUB)
    g = [g1, g2, g3]
    features = torch.FloatTensor(features_0)

    labels = torch.LongTensor(labels)
    num_classes = 3

    pos = scipy.sparse.load_npz(prefix + "/new_pos.npz")
    pos = torch.FloatTensor(pos.todense())

    return g, features, labels, num_classes, train_idx, val_idx, test_idx, pos

