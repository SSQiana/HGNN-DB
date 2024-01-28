"""Torch Module for APPNPConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import numpy as np
# from .... import function as fn
# from .graphconv import EdgeWeightNorm
from dgl.nn.pytorch.conv import EdgeWeightNorm
from dgl import function as fn
from attention import Attention
import math


class APPNPConv(nn.Module):
    def __init__(self, hidden_dim, dropout, k, alpha, beta, edge_drop=0.0):
        super(APPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)
        self.feats = hidden_dim
        self.weights = nn.Parameter(th.randn(self._k + 1))
        self.softmax = nn.Softmax()
        self.attn = Attention(hidden_dim, dropout)
        self.beta = beta
        # self.linear = nn.Linear(hidden_dim*(k+1), hidden_dim)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is None:
                src_norm = th.pow(
                    graph.out_degrees().to(feat).clamp(min=1), -0.5
                )
                shp = src_norm.shape + (1,) * (feat.dim() - 1)
                src_norm = th.reshape(src_norm, shp).to(feat.device)
                dst_norm = th.pow(
                    graph.in_degrees().to(feat).clamp(min=1), -0.5
                )
                shp = dst_norm.shape + (1,) * (feat.dim() - 1)
                dst_norm = th.reshape(dst_norm, shp).to(feat.device)
            else:
                edge_weight = EdgeWeightNorm("both")(graph, edge_weight)
            z = []
            feat_0 = feat
            z.append(feat_0)
            for _ in range(self._k):
                # normalization by src node
                if edge_weight is None:
                    feat = feat * src_norm
                graph.ndata["h"] = feat
                w = (
                    th.ones(graph.number_of_edges(), 1)
                    if edge_weight is None
                    else edge_weight
                )
                graph.edata["w"] = self.edge_drop(w).to(feat.device)
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))  # 第一个为消息传递函数 第二个为聚合函数
                feat = graph.ndata.pop("h")
                # normalization by dst node
                if edge_weight is None:
                    feat = feat * dst_norm
                # feat = (1 - self._alpha) * feat + self._alpha * feat_0
                # return feat
                # if _ != 0:
                z.append(feat)

            # 注意力
            # z = th.stack(z, dim=1)
            # print('不同层之间的权重：')
            # feat = self.attn(z)
            # return feat

            # # 自适应权重
            # weights = self.softmax(self.weights)
            # weighted_matrices = []
            # for i in range(len(z)):
            #     weighted_matrix = weights[i] * z[i]
            #     weighted_matrices.append(weighted_matrix)
            #
            # feat = th.stack(weighted_matrices).sum(dim=0)
            # print(weights.cpu().detach())
            # return feat/len(z)

            # 拼接
            # z_concat = th.cat(z, dim=1)  # 拼接
            # # z_concat = z_concat.view(z_concat.size(0), -1)
            # z_out = self.linear(z_concat)  # 线性层映射
            #
            # return z_out

            # 指数衰减
            # beta = 0.7
            # denominator = sum([beta ** i for i in range(1, len(z) + 1)])
            # normalized_matrices = []
            # for k in range(len(z)):
            #     normalization_factor = beta ** k / denominator
            #     normalized_matrix = normalization_factor * z[k]
            #     normalized_matrices.append(normalized_matrix)
            # feat1 = th.stack(normalized_matrices)
            # return th.sum(feat1, dim=0)

            # 对数分布
            beta = self.beta
            denominator = sum([math.log(beta + i) for i in range(1, len(z) + 1)])
            normalized_matrices = []

            for k in range(len(z)):
                normalization_factor = math.log(beta + (k+1)) / denominator
                normalized_matrix = normalization_factor * z[k]
                normalized_matrices.append(normalized_matrix)

            feat1 = th.stack(normalized_matrices)
            return th.sum(feat1, dim=0)
