import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dim, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(dim, dropout), 2)
        self.dim = dim

    def forward(self, q, k, v):
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])
        return self.sublayer[1](q, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, masksize=0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.masksize = masksize

    def attention(self, query, key, value, masksize, dropout=None):
        # [bs, nhead=8, T=10, d_k=512/8=64]
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    # [bs, 8, 10, 10]
        
        if masksize != 0:
            masksize = int(masksize / 2)
            mask = torch.ones(scores.size()).to(scores.device)         # [bs, 8, 10, 10]
            for i in range(mask.shape[2]):
                # 对每一行进行mask操作，将对角线i前后窗口大小d以外的置0.
                if i - masksize > 0:
                    mask[:, :, i, :i - masksize] = 0
                if i + masksize + 1 < mask.shape[3]:
                    mask[:, :, i, masksize + i + 1:] = 0
            # print(mask[0][0])
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)

        out = torch.matmul(attn, value)
        return out, attn

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # [B, T=10, d_model=512] -> [B, h=8, T=10, d_k=64]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, self.masksize, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)   # [B, 10, 512]
        return out, self.attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.W_2(self.dropout(F.relu(self.W_1(x))))
        return output

