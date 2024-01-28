import torch.nn as nn
import torch


class Attention(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        """语义层次的注意力

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn, gain)

    def forward(self, h):
        """
        :param h: tensor(N, M, d) 顶点基于不同元路径/类型的嵌入，N为顶点数，M为元路径/类型数
        :return: tensor(N, d) 顶点的最终嵌入
        """
        attn = self.attn_drop(self.attn)
        # (N, M, d) -> (M, d) -> (M, 1)
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)  # (M, 1)
        # print(beta.detach().cpu().numpy())
        beta = beta.expand((h.shape[0],) + beta.shape)  # (N, M, 1)
        z = (beta * h).sum(dim=1)  # (N, d)
        return z
