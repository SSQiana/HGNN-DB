import random
import os.path as osp
import dgl
import torch
import torch.optim as optim
import numpy as np
import time
from HGNN_DB_link_prediction import HGNN_DB, MergeLayer
from load_data import load_ACM_data, load_DBLP_data, load_YELP_data, load_Aminer_data
from evaluate import evaluate_results_nc, svm_test
from set_params import set_params
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.sparse import csr_matrix
import torch.nn as nn
from sklearn.model_selection import train_test_split

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)

def get_device(device):
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')

def generate_samples(pp, strategy):
    rows, cols = pp.nonzero()
    if len(rows) > 2e+5:
        rows = rows[:100000]
        cols = cols[:100000]
    positive_pairs = np.array(list(zip(rows, cols)))

    num_nodes = pp.shape[0]
    if strategy == 'random':
        total_possible_pairs = num_nodes * num_nodes
        sampled_negatives = set()
        while len(sampled_negatives) < len(positive_pairs):
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)
            if i != j and pp[i, j] == 0:
                sampled_negatives.add((i, j))
        negative_pairs = np.array(list(sampled_negatives))
    else:
        positive_pairs = set(zip(rows, cols))
        all_pairs = {(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j}
        candidate_negatives = all_pairs - positive_pairs
        candidate_negatives = list(candidate_negatives)
        sampled_negatives = np.random.choice(len(candidate_negatives), len(positive_pairs), replace=False)
        negative_pairs = np.array([candidate_negatives[i] for i in sampled_negatives])
        positive_pairs = np.array(list(zip(rows, cols)))


    pos_src_node_id = positive_pairs[:, 0]
    pos_dst_node_id = positive_pairs[:, 1]
    neg_src_node_id = negative_pairs[:, 0]
    neg_dst_node_id = negative_pairs[:, 1]
    train_pos_src = pos_src_node_id[int(len(pos_src_node_id)*0.7):]
    train_pos_dst = pos_dst_node_id[int(len(pos_dst_node_id)*0.7):]
    test_pos_src = pos_src_node_id[:int(len(pos_src_node_id)*0.3)]
    test_pos_dst = pos_dst_node_id[:int(len(pos_dst_node_id)*0.3)]


    train_neg_src = neg_src_node_id[int(len(neg_src_node_id)*0.7):]
    train_neg_dst = neg_dst_node_id[int(len(neg_dst_node_id)*0.7):]
    test_neg_src = neg_src_node_id[:int(len(pos_src_node_id)*0.3)]
    test_neg_dst = neg_dst_node_id[:int(len(neg_dst_node_id)*0.3)]
    return [train_pos_src, train_pos_dst, test_pos_src, test_pos_dst], [train_neg_src, train_neg_dst, test_neg_src, test_neg_dst]


def load_data(dataset, device):
    if dataset == 'acm':
        adj, mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_ACM_data()
    elif dataset == 'dblp':
        adj, mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_DBLP_data()
    elif dataset == 'aminer':
        adj, mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_Aminer_data()
    elif dataset == 'yelp':
        adj,mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_YELP_data()
    else:
        print(f"Unknown dataset: {dataset}. Please provide a valid dataset name.")

    feats = feats.to(device)
    mgs = [mg.to(device) for mg in mgs]

    pos = pos.to(device)
    return adj, mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos


def train(args):
    checkpoints_path = f'checkpoints'
    strategy = args.strategy # random excluded
    if args.dataset == 'acm':
        patience = 3
    else:
        patience = 20
    set_random_seed(args.seed)
    device = get_device(args.device)
    # device = torch.device('cpu')
    pp, mgs, feats, labels, num_classes, train_mask, val_mask, test_mask, pos = load_data(args.dataset, device)
    mgs = [dgl.add_self_loop(dgl.remove_self_loop(mp)) for mp in mgs]
    model = HGNN_DB(
        feats.shape[1], args.num_hidden, args.feat_drop, args.attn_drop,
         len(mgs), args.tau, args.lambda_, args.alpha_, args.gamma_, args.beta, args.k)
    link_predictor = MergeLayer(input_dim1=args.num_hidden, input_dim2=args.num_hidden,
                                hidden_dim=args.num_hidden, output_dim=1)
    backbone = nn.Sequential(model, link_predictor).to(device)
    optimizer = optim.Adam(backbone.parameters(), lr=args.lr)

    # [train_pos_src, train_pos_dst, test_pos_src, test_pos_dst], [train_neg_src, train_neg_dst, test_neg_src, test_neg_dst]
    pos_pair, neg_pair = generate_samples(pp, strategy)


    cnt_wait = 0
    best_mif1 = 0
    best_epoch = 0
    total_time = 0
    num = 0
    loss_func = nn.BCELoss()
    patience_counter = 0
    best_val_loss = 0
    best_loss = float('inf')
    pos_train_idx, pos_val_idx = train_test_split(range(len(pos_pair[0])), test_size=0.1, random_state=42)
    neg_train_idx, neg_val_idx = train_test_split(range(len(neg_pair[0])), test_size=0.1, random_state=42)
    for epoch in range(args.epochs):
        num = num + 1

        t1 = time.time()
        backbone.train()
        z = backbone[0](mgs, feats, pos)

        train_pos_src_emb = z[pos_pair[0][pos_train_idx]]
        train_pos_dst_emb = z[pos_pair[1][pos_train_idx]]
        train_neg_src_emb = z[neg_pair[0][neg_train_idx]]
        train_neg_dst_emb = z[neg_pair[1][neg_train_idx]]

        pos_probability = torch.sigmoid(backbone[1](train_pos_src_emb, train_pos_dst_emb))
        neg_probability = torch.sigmoid(backbone[1](train_neg_src_emb, train_neg_dst_emb))

        predicts = torch.cat([pos_probability, neg_probability], dim=0)
        labels = torch.cat([torch.ones_like(pos_probability), torch.zeros_like(neg_probability)], dim=0)
        loss = loss_func(input=predicts, target=labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pos_src_emb = z[pos_pair[0][pos_val_idx]]
            val_pos_dst_emb = z[pos_pair[1][pos_val_idx]]
            val_neg_src_emb = z[neg_pair[0][neg_val_idx]]
            val_neg_dst_emb = z[neg_pair[1][neg_val_idx]]

            val_pos_probability = torch.sigmoid(backbone[1](val_pos_src_emb, val_pos_dst_emb))
            val_neg_probability = torch.sigmoid(backbone[1](val_neg_src_emb, val_neg_dst_emb))

            val_predicts = torch.cat([val_pos_probability, val_neg_probability], dim=0)
            val_labels = torch.cat([torch.ones_like(val_pos_probability), torch.zeros_like(val_neg_probability)], dim=0)

            val_ap = average_precision_score(val_labels.detach().cpu().numpy(), val_predicts.detach().cpu().numpy())

        with torch.no_grad():
            test_pos_src_emb = z[pos_pair[2]]
            test_pos_dst_emb = z[pos_pair[3]]
            test_neg_src_emb = z[neg_pair[2]]
            test_neg_dst_emb = z[neg_pair[3]]

            test_pos_probability = torch.sigmoid(backbone[1](test_pos_src_emb, test_pos_dst_emb))
            test_neg_probability = torch.sigmoid(backbone[1](test_neg_src_emb, test_neg_dst_emb))

            test_predicts = torch.cat([test_pos_probability, test_neg_probability], dim=0)
            test_labels = torch.cat([torch.ones_like(test_pos_probability), torch.zeros_like(test_neg_probability)],
                                    dim=0)

            test_predicts_np = test_predicts.cpu().numpy()
            test_labels_np = test_labels.cpu().numpy()
            test_roc_auc = roc_auc_score(test_labels_np, test_predicts_np)
            test_ap = average_precision_score(test_labels_np, test_predicts_np)

        print(f"Test ROCAUC: {test_roc_auc:.4f}, Test AP: {test_ap:.4f}")

        if val_ap > best_val_loss:
            best_val_loss = val_ap
            patience_counter = 0
            print(f"Epoch {epoch + 1}, Validation Ap Improved: {val_ap:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch + 1}, No Improvement in Validation Loss for {patience_counter} Epochs")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        predicts_np = predicts.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        train_roc_auc = roc_auc_score(labels_np, predicts_np)
        train_ap = average_precision_score(labels_np, predicts_np)

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}, ROCAUC: {train_roc_auc:.4f}, AP: {train_ap:.4f}, "
            f"Val Ap: {val_ap:.4f}")

    print(f"Training stopped at epoch {epoch + 1} with best loss: {best_loss:.4f}")
    model.eval()
    with torch.no_grad():
        test_pos_src_emb = z[pos_pair[2]]
        test_pos_dst_emb = z[pos_pair[3]]
        test_neg_src_emb = z[neg_pair[2]]
        test_neg_dst_emb = z[neg_pair[3]]

        test_pos_probability = torch.sigmoid(backbone[1](test_pos_src_emb, test_pos_dst_emb))
        test_neg_probability = torch.sigmoid(backbone[1](test_neg_src_emb, test_neg_dst_emb))

        test_predicts = torch.cat([test_pos_probability, test_neg_probability], dim=0)
        test_labels = torch.cat([torch.ones_like(test_pos_probability), torch.zeros_like(test_neg_probability)], dim=0)

        test_predicts_np = test_predicts.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        test_roc_auc = roc_auc_score(test_labels_np, test_predicts_np)
        test_ap = average_precision_score(test_labels_np, test_predicts_np)

    print(f"Test ROCAUC: {test_roc_auc:.4f}, Test AP: {test_ap:.4f}")

def main():
    args = set_params()
    print(args)
    train(args)


if __name__ == '__main__':
    main()

