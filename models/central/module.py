import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.central.transformer import *

#! Preprocessor
class AVVPreprocessor(nn.Module):
    def __init__(self, args):
        super(AVVPreprocessor, self).__init__()

        self.device = args.device

        self.fc_a = nn.Linear(128, args.hid_dim)
        self.fc_v = nn.Linear(2048, args.hid_dim)
        self.fc_st = nn.Linear(512, args.hid_dim)
        self.fc_fusion = nn.Linear(1024, args.hid_dim)

    def forward(self, inputs):
        audio, video_s, video_st = inputs['audio'].to(self.device), inputs['video_s'].to(self.device), inputs['video_st'].to(self.device)

        # audio feature: [B, 10, 128]
        f_a = self.fc_a(audio)

        # 2d and 3d visual feature: [B, 80, 2048], [B, 10, 512]
        vid_s = self.fc_v(video_s).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(video_st)
        f_v = torch.cat((vid_s, vid_st), dim=-1)
        f_v = self.fc_fusion(f_v)

        return f_a, f_v


#! Multi Scale Encoder
class MultiScaleEncoderModule(nn.Module):
    def __init__(self, args):
        super(MultiScaleEncoderModule, self).__init__()

        dim = args.hid_dim
        ffn_dim = args.ffn_dim
        nhead = args.nhead
        dropout = args.drpt
        self.scale = args.scale       
        self.num_scale = args.num_scale       
        
        self_attn = MultiHeadAttention(dim, nhead, dropout)
        feed_forward = PositionwiseFeedForward(dim, ffn_dim, dropout)
        
        conv_layers = []
        transformer_layers = []
        for i in range(self.num_scale):
            kernal_size = self.scale[i]
            conv_layers.append(nn.Conv1d(dim, dim, kernal_size))

            # share parameters of SA and FFN, except LN and Drpt
            transformer_layers.append(TransformerLayer(self_attn, feed_forward, dim, dropout))  
            
        self.conv_layer = nn.ModuleList(conv_layers)
        self.encoder_layer = nn.ModuleList(transformer_layers)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 10, dim)) 
        self.pos_drop = nn.Dropout(p=dropout)


    def forward(self, x):
        # x: [B, 10, 512]
        x_stage_list = []

        # positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        #! parallel structure
        upsampler = nn.Upsample(size=(10, 1), mode='bilinear', align_corners=False)
        for i in range(self.num_scale):
            x_stage = self.conv_layer[i](x.permute(0, 2, 1)).permute(0, 2, 1)   # [B, 10, 512] -> [B, scale[i], 512]
            x_stage = self.encoder_layer[i](x_stage, x_stage, x_stage)          # [B, scale[i], 512]
            
            # reshape: [B, scale[i], 512] -> [B, 512, scale[i], 1] -> [B, 512, 10, 1] -> [B, 10, 512]
            x_stage = upsampler(x_stage.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)    
            x_stage_list.append(x_stage)

        return x_stage_list


#! Task-specific Head
class AVVPHead(nn.Module):
    def __init__(self, args):
        super(AVVPHead, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)

    def forward(self, a_out, v_out):
        # projection
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2)   # [B, 10, 2, 512]
        frame_prob = torch.sigmoid(self.fc_prob(x))                         # [B, 10, 2, 25]

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = (frame_att * frame_prob)                    
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)

        outputs = {
            'global_prob': global_prob, 
            'a_prob': a_prob, 
            'v_prob': v_prob, 
            'frame_prob': frame_prob
        }
        return outputs
