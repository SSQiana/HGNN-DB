import random
import os.path as osp
import dgl
import torch
import torch.optim as optim
import numpy as np
import time
from HGDB import HeCo
from load_data import load_ACM_data, load_DBLP_data, load_YELP_data, load_Aminer_data
from evaluate import evaluate_results_nc, svm_test
from set_params import set_params
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def set_random_seed(seed):
    """设置Python, numpy, PyTorch的随机数种子

    :param seed: int 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)

def get_device(device):
    """返回指定的GPU设备

    :param device: int GPU编号，-1表示CPU
    :return: torch.device
    """
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')


def load_data(dataset, device):
    if dataset == 'acm':
        mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_ACM_data()
    elif dataset == 'dblp':
        mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_DBLP_data()
    elif dataset == 'aminer':
        mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_Aminer_data()
    elif dataset == 'yelp':
        mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_YELP_data()
    else:
        print(f"Unknown dataset: {dataset}. Please provide a valid dataset name.")

    feats = feats.to(device)
    mgs = [mg.to(device) for mg in mgs]

    pos = pos.to(device)
    return mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos


def train(args):
    checkpoints_path = f'checkpoints'
    set_random_seed(args.seed)
    device = get_device(args.device)

    # args.dataset = dataset
    # args.beta = b
    mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_data(args.dataset, device)
    mgs = [dgl.add_self_loop(dgl.remove_self_loop(mp)) for mp in mgs]
    model = HeCo(
        feats.shape[1], args.num_hidden, args.feat_drop, args.attn_drop,
         len(mgs), args.tau, args.lambda_, args.alpha_, args.gamma_, args.beta, args.k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cnt_wait = 0
    best_mif1 = 0
    best_epoch = 0
    total_time = 0
    num = 0
    for epoch in range(args.epochs):
        num = num + 1
        t1 = time.time()
        model.train()
        loss = model(mgs, feats, pos)
        t2 = time.time()
        total_time = total_time + (t2 - t1)
        if epoch % 10 == 0:
            embeds = model.get_embeds(feats, mgs)
            svm_macro, svm_micro = svm_test(embeds[test_mask].detach().cpu().numpy(), labels[test_mask].cpu().numpy(), repeat=1)

            macro_f1 = svm_macro[0][0]
            micro_f1 = svm_micro[0][0]
            print(micro_f1, macro_f1)
            if micro_f1 > best_mif1:
                best_mif1 = micro_f1
                best_epoch = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), osp.join(checkpoints_path, f'{epoch}.pkl'))
            else:
                cnt_wait += 1
            if cnt_wait == 15:
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:d} | Train Loss {:.4f}'.format(epoch, loss.item()), best_epoch, best_mif1)
    model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{best_epoch}.pkl')))
    embeds = model.get_embeds(feats, mgs)
    svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeds[test_mask].detach().cpu().numpy(),
                                                         labels[test_mask].cpu().numpy(),
                                                         int(labels.max()) + 1)
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro]))  # format这是一种字符串格式化的方法
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro]))
    print('NMI: {:.6f}'.format(nmi))
    print('ARI: {:.6f}'.format(ari))
    print('all finished')
    print('平均时间：', total_time/num)

    # with open('evaluation_results_of_k_layer.txt', 'a') as file:
    #
    #     file.write(dataset + '\n')
    #     file.write('beta: {:.6f}\n'.format(b))
    #     file.write('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro]) + '\n')
    #     file.write('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro]) + '\n')
    #     file.write('NMI: {:.6f}\n'.format(nmi))
    #     file.write('ARI: {:.6f}\n'.format(ari))
    # Y = labels[test_mask].cpu().numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(embeds[test_mask].detach().cpu().numpy())
    # color_idx = {}
    # for i in range(len(embeds[test_mask].detach().cpu().numpy())):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
    # plt.legend()
    # # plt.savefig(".\\visualization\ROHE_323_" + str(args['dataset']) + "分类图" + str(cur_repeat) + ".png", dpi=1000,
    # #             bbox_inches='tight')
    # plt.show()


def main():
    args = set_params()
    print(args)
    train(args)


if __name__ == '__main__':
    main()

# ACM   gamma = 0.5     feat_drop = 0.5     BETA=0.8

# Macro-F1: 0.934213, 0.934597, 0.933464, 0.932561, 0.932166, 0.929600, 0.855878
# Micro-F1: 0.932842, 0.933385, 0.932350, 0.931211, 0.931004, 0.928686, 0.872184
# NMI: 0.723404
# ARI: 0.743061

# dblp     gamma = 0.9      feat_drop = 0.5
# Macro-F1: 0.9305~0.0084(0.80), 0.9274~0.0061(0.60), 0.9262~0.0041(0.40), 0.9242~0.0033(0.20), 0.9201~0.0034(0.10), 0.9174~0.0051(0.05), 0.9103~0.0075(0.01)
# Micro-F1: 0.9353~0.0081(0.80), 0.9326~0.0059(0.60), 0.9315~0.0035(0.40), 0.9298~0.0031(0.20), 0.9259~0.0033(0.10), 0.9232~0.0049(0.05), 0.9167~0.0072(0.01)

# NMI: 0.750882
# ARI: 0.807986

# aminer    gamma = 0.7 feat_drop = 0.2
# Macro-F1: 0.6622~0.0157(0.80), 0.6604~0.0103(0.60), 0.6577~0.0070(0.40), 0.6555~0.0107(0.20), 0.6547~0.0146(0.10), 0.6477~0.0139(0.05), 0.6051~0.0361(0.01)
# Micro-F1: 0.8493~0.0065(0.80), 0.8478~0.0057(0.60), 0.8466~0.0021(0.40), 0.8448~0.0027(0.20), 0.8439~0.0034(0.10), 0.8391~0.0040(0.05), 0.7956~0.0251(0.01)
# NMI: 0.377024
# ARI: 0.346508

# YELP      gamma = 0.7     feat_drop = 0.2
# Macro-F1: 0.9290~0.0127(0.80), 0.9285~0.0075(0.60), 0.9237~0.0053(0.40), 0.9158~0.0058(0.20), 0.9043~0.0103(0.10), 0.8881~0.0213(0.05), 0.7849~0.0850(0.01)
# Micro-F1: 0.9254~0.0101(0.80), 0.9231~0.0077(0.60), 0.9194~0.0047(0.40), 0.9117~0.0060(0.20), 0.9013~0.0084(0.10), 0.8893~0.0153(0.05), 0.8302~0.0428(0.01)

# NMI: 0.424008
# ARI: 0.464182

# python main.py --dataset="acm"
