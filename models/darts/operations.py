import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.central.transformer import *


SMMT_CANDIDATES = [
    'WeightedSum',
    'ConcatFC',
    'LinearGLU',
    'GuidedAttn',
    'HybridAttn',
    'JointCoAttn',
]

CMMT_CANDIDATES = [
    'GuidedAttn',
    'HybridAttn',
]

OPERATIONS = { 
    #! SMT
    'Sum': lambda args: Sum(args),
    'WeightedSum': lambda args: WeightedSum(args),
    'ConcatFC': lambda args: ConcatFC(args),
    'LinearGLU': lambda args: LinearGLU(args),
    'GuidedAttn': lambda args: GuidedAttn(args),
    'HybridAttn': lambda args: HybridAttn(args),
    'JointCoAttn': lambda args: JointCoAttn(args),


    #! CMT
    # 'SelfAttn_1': lambda args: SelfAttn(args, masksize=1),
    # 'SelfAttn_3': lambda args: SelfAttn(args, masksize=3),
    # 'SelfAttn_5': lambda args: SelfAttn(args, masksize=5),
    # 'SelfAttn_7': lambda args: SelfAttn(args, masksize=7),
    # 'SelfAttn_9': lambda args: SelfAttn(args, masksize=9),

    'GuidedAttn_1': lambda args: GuidedAttn(args, masksize=1),
    'GuidedAttn_2': lambda args: GuidedAttn(args, masksize=2),
    'GuidedAttn_3': lambda args: GuidedAttn(args, masksize=3),
    'GuidedAttn_4': lambda args: GuidedAttn(args, masksize=4),
    'GuidedAttn_5': lambda args: GuidedAttn(args, masksize=5),
    'GuidedAttn_6': lambda args: GuidedAttn(args, masksize=6),
    'GuidedAttn_7': lambda args: GuidedAttn(args, masksize=7),
    'GuidedAttn_8': lambda args: GuidedAttn(args, masksize=8),
    'GuidedAttn_9': lambda args: GuidedAttn(args, masksize=9),
    'GuidedAttn_10': lambda args: GuidedAttn(args, masksize=10),

    'HybridAttn_1': lambda args: HybridAttn(args, masksize=1),
    'HybridAttn_2': lambda args: HybridAttn(args, masksize=2),
    'HybridAttn_3': lambda args: HybridAttn(args, masksize=3),
    'HybridAttn_4': lambda args: HybridAttn(args, masksize=4),
    'HybridAttn_5': lambda args: HybridAttn(args, masksize=5),
    'HybridAttn_6': lambda args: HybridAttn(args, masksize=6),
    'HybridAttn_7': lambda args: HybridAttn(args, masksize=7),
    'HybridAttn_8': lambda args: HybridAttn(args, masksize=8),
    'HybridAttn_9': lambda args: HybridAttn(args, masksize=9),
    'HybridAttn_10': lambda args: HybridAttn(args, masksize=10),
}


class Sum(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x, y):
        return x + y


class WeightedSum(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x, y):
        return self.alpha * x + (1 - self.alpha) * y


class ConcatFC(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        D = args.hid_dim
        self.conv = nn.Conv1d(2*D, D, 1, 1)
        self.bn = nn.BatchNorm1d(D)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x.permute(0, 2, 1), y.permute(0, 2, 1)], dim=1)    # [B, 2D, T]
        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out.permute(0, 2, 1)


class LinearGLU(nn.Module):
    def __init__(self, args):
        super().__init__()

        D = args.hid_dim
        self.conv = nn.Conv1d(2*D, 2*D, 1, 1)
        self.bn = nn.BatchNorm1d(2*D)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x.permute(0, 2, 1), y.permute(0, 2, 1)], dim=1)
        out = self.conv(out)
        out = self.bn(out)

        # apply glu on channel dim
        out = F.glu(out, dim=1)
        out = self.dropout(out)

        return out.permute(0, 2, 1)


class SelfAttn(nn.Module):
    ''' Self Attention '''

    def __init__(self, args, masksize=0):
        super().__init__()

        dim, ffn_dim, nhead, dropout = 2 * args.hid_dim, 2 * args.ffn_dim, args.nhead, args.drpt
        attn = MultiHeadAttention(dim, nhead, dropout, masksize=masksize)
        feed_forward = PositionwiseFeedForward(dim, ffn_dim, dropout)
        self.layer = TransformerLayer(attn, feed_forward, dim, dropout)
        self.fc = nn.Linear(dim, dim // 2)

    def forward(self, audio, video):
        x = torch.cat([audio, video], dim=-1)   # [B, T, 1024]
        z_out = self.layer(x, x, x)
        z_out = self.fc(z_out)                  # [B, T, 512]
        return z_out


class GuidedAttn(nn.Module):
    ''' Cross-Modal Attention '''

    def __init__(self, args, masksize=0):
        super().__init__()

        dim, ffn_dim, nhead, dropout = args.hid_dim, args.ffn_dim, args.nhead, args.drpt
        attn = MultiHeadAttention(dim, nhead, dropout, masksize=masksize)
        feed_forward = PositionwiseFeedForward(dim, ffn_dim, dropout)
        self.layer = TransformerLayer(attn, feed_forward, dim, dropout)

    def forward(self, audio, video):
        a_out = self.layer(audio, video, video)
        v_out = self.layer(video, audio, audio)
        return a_out + v_out 


class HybridAttn(nn.Module):

    def __init__(self, args, masksize=0):
        super(HybridAttn, self).__init__()

        dim, ffn_dim, nhead, dropout = args.hid_dim, args.ffn_dim, args.nhead, args.drpt
        self.self_attn = MultiHeadAttention(dim, nhead, dropout, masksize=masksize)
        self.cross_attn = MultiHeadAttention(dim, nhead, dropout, masksize=masksize)
        self.ffn = PositionwiseFeedForward(dim, ffn_dim, dropout)
        self.sublayer = SublayerConnection(dim, dropout)

        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)

    def forward_feature(self, src_q, src_v):
        # Hybrid Attn
        out_ca = self.cross_attn(src_q, src_v, src_v)[0]
        out_sa = self.self_attn(src_q, src_q, src_q)[0]
        out_ha = src_q + self.dropout11(out_ca) + self.dropout12(out_sa)    
        # FFN
        out = self.sublayer(out_ha, self.ffn)

        return out

    def forward(self, audio, video):
        a_out = self.forward_feature(audio, video)
        v_out = self.forward_feature(video, audio)

        return a_out + v_out 


class JointCoAttn(nn.Module):

    def __init__(self, args, masksize=0):
        super(JointCoAttn, self).__init__()
        
        dim1 = dim2 = args.hid_dim
        drpt = args.drpt
        # Joint co-attention
        dim = dim1 + dim2
        self.masksize = masksize
        self.dropouts = nn.ModuleList([nn.Dropout(p=drpt) for _ in range(2)])
        self.query_linear = nn.Linear(dim, dim)
        self.key1_linear = nn.Linear(10, 10)
        self.key2_linear = nn.Linear(10, 10)
        self.value1_linear = nn.Linear(dim1, dim1)
        self.value2_linear = nn.Linear(dim2, dim2)
        self.relu = nn.ReLU()
        
        # fusion
        self.fc_fusion = nn.Sequential(
                            nn.Linear(dim1 + dim2, dim1),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=drpt),
                        )
        
        
    def coattn(self, src1, src2):
        joint = torch.cat((src1, src2), dim=-1)
        joint = self.query_linear(joint)
        key1 = self.key1_linear(src1.transpose(1, 2))
        key2 = self.key2_linear(src2.transpose(1, 2))
        value1 = self.value1_linear(src1)
        value2 = self.value2_linear(src2)

        out1, attn1 = self.qkv_attention(joint, key1, value1, self.masksize, dropout=self.dropouts[0])
        out2, attn2 = self.qkv_attention(joint, key2, value2, self.masksize, dropout=self.dropouts[1])

        return out1, out2

    def qkv_attention(self, query, key, value, masksize, dropout=None):
        # query=[B, 10, 1024], key=[B, 512, 10], value=[B, 10, 512]
        d_k = query.size(-1)
        scores = torch.bmm(key, query) / math.sqrt(d_k)         # [B, 512, 1024]
        scores = torch.tanh(scores)

        if dropout:
            scores = dropout(scores)
        weighted = torch.tanh(torch.bmm(value, scores))         # [B, 10, 1024]

        return self.relu(weighted), scores

    def forward(self, audio, video):
        out1, out2 = self.coattn(audio, video)  # [B, T, 2D]
        a_out = audio + self.fc_fusion(out1)
        v_out = video + self.fc_fusion(out2)
        return a_out + v_out 

