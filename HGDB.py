"""Self-Supervised Heterogeneous Graph Neural Network with Co-Contrastive Learning (HeCo)

论文链接：https://arxiv.org/pdf/2105.09111
"""
import dgl.function as fn
import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from appnp import APPNPConv
from attention import Attention
import time

EPS = 1e-15


class MetapathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, attn_drop):
        super().__init__()

        self.attn = Attention(hidden_dim, attn_drop)

    def forward(self, z_mp1, z_mp2, p):
        z_mp1 = self.attn(z_mp1)
        z_mp2 = self.attn(z_mp2)
        return z_mp1, z_mp2


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lambda_ = lambda_
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

    def sim(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        numerator = torch.mm(x, y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(numerator / denominator / self.tau)

    def forward(self, z_sc, z_mp, pos):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(N, N) 0-1张量，每个顶点的正样本
        :return: float 对比损失
        """
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)
        sim_inter1 = self.sim(z_sc_proj, z_mp_proj)
        intra_mp = self.sim(z_mp_proj, z_mp_proj)
        intra_sc = self.sim(z_sc_proj, z_sc_proj)
        loss1 = -torch.log(
            (sim_inter1 * pos).sum(1) /
            (intra_mp.sum(1) + sim_inter1.sum(1) - (intra_mp * pos).sum(1))
        )
        loss2 = -torch.log(
            (sim_inter1 * pos).sum(1) /
            (intra_sc.sum(1) + sim_inter1.sum(1) - (intra_sc * pos).sum(1))
        )
        return (loss1.mean() + loss2.mean()) / 2


class HeCo(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, feat_drop, attn_drop,
            num_metapaths, tau, lambda_, alpha, gamma, beta, k):
        super().__init__()
        self.fcs = nn.Linear(in_dims, hidden_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.global_projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
                                              nn.Linear(hidden_dim, hidden_dim))
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim), requires_grad=True)
        self.feat_drop = nn.Dropout(feat_drop)
        self.mp_encoder = MetapathEncoder(num_metapaths, hidden_dim, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lambda_)
        self.conv1 = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim, norm='right', activation=nn.PReLU())
            for _ in range(num_metapaths)
        ])
        self.conv2 = nn.ModuleList([
            APPNPConv(hidden_dim, attn_drop, k=self.k, alpha=0.1, beta=self.beta)
            for _ in range(num_metapaths)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.xavier_normal_(self.fcs.weight, gain)
        for model in self.global_projector:
            if isinstance(model, nn.Linear):
                # 均匀分布初始化
                nn.init.kaiming_uniform_(model.weight, a=1.414)

    def forward(self, mgs, feats, pos):
        h = F.elu(self.feat_drop(self.fcs(feats)))
        h1 = [conv(mg, h) for conv, mg in zip(self.conv1, mgs)]
        h2 = [conv(mg, h) for conv, mg in zip(self.conv2, mgs)]

        p = len(h1)
        h1 = torch.stack(h1, dim=1)  # (N, M, d)
        h2 = torch.stack(h2, dim=1)  # (N, M, d)
        h2 = torch.squeeze(h2, dim=2)
        z_mp1, z_mp2 = self.mp_encoder(h1, h2, p)  # (N_tgt, d_hid)
        z = [z_mp1, z_mp2]
        # z_neg = self.corrupt(z_mp1, z_mp2)
        loss2 = 0.
        num_contrasts = 0
        for i in range(len(z)):
            for j in range(i, len(z)):
                loss2 += self.global_loss(z[i], z[j]) + self.global_loss(z[j], z[i])
                num_contrasts += 1
        loss2 = loss2 / num_contrasts
        loss1 = self.contrast(z_mp1, z_mp2, pos)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss

    def discriminate(self, z, summary, sigmoid=True):

        summary = torch.matmul(self.weight, summary)
        value = torch.matmul(z, summary)
        return torch.sigmoid(value) if sigmoid == True else value

    def global_loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor):
        s = pos_z.mean(dim=0)
        # h = self.global_projector(s)
        h = s
        EPS = 1e-15
        pos_loss = -torch.log(self.discriminate(pos_z, h, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, h, sigmoid=True) + EPS).mean()
        loss = (pos_loss + neg_loss) * 0.5
        return loss

    @torch.no_grad()
    def get_embeds(self, feats, mgs):
        """计算目标顶点的最终嵌入(z_mp)

        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feats: tensor(N_tgt, d_in) 目标顶点的输入特征
        :return: tensor(N_tgt, d_hid) 目标顶点的最终嵌入
        """
        h = F.elu(self.feat_drop(self.fcs(feats)))
        h1 = [conv(mg, h) for conv, mg in zip(self.conv1, mgs)]

        h2 = [conv(mg, h) for conv, mg in zip(self.conv2, mgs)]

        p = len(h1)
        h1 = torch.stack(h1, dim=1)
        h2 = torch.stack(h2, dim=1)
        z_mp1, z_mp2 = self.mp_encoder(h1, h2, p)
        z = self.gamma * z_mp1 + (1 - self.gamma) * z_mp2
        return z
