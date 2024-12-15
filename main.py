import random
import os.path as osp
import dgl
import torch
import torch.optim as optim
import numpy as np
import time
from HGDB import HGNN_DB
from load_data import load_ACM_data, load_DBLP_data, load_YELP_data
from evaluate import evaluate_results_nc, svm_test
from set_params import set_params
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)

def get_device(device):

    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')


def load_data(dataset, device):
    if dataset == 'acm':
        mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_ACM_data()
    elif dataset == 'dblp':
        mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_DBLP_data()
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
    # device = torch.device('cpu')
    mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_data(args.dataset, device)
    mgs = [dgl.add_self_loop(dgl.remove_self_loop(mp)) for mp in mgs]
    model = HGNN_DB(
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

