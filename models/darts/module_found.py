import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.darts.operations import OPERATIONS, SMMT_CANDIDATES, CMMT_CANDIDATES
from models.darts.genotypes import Genotype, FusionGenotype


#! Found Fusion Cell
class FoundSingleModalFusionCell(nn.Module):

    def __init__(self, args, op):
        super(FoundSingleModalFusionCell, self).__init__()
        
        self.op = OPERATIONS[op[1]](args)

    def forward(self, x, y):
        out = self.op(x, y)
        return out

class FoundMultiModalFusionCell(nn.Module):

    def __init__(self, args, pos, win, op):
        super(FoundMultiModalFusionCell, self).__init__()
        
        self.pos = pos 
        self.m = win
        self.op = OPERATIONS[op[1]](args)

    def forward(self, x, y):
        px, py = self.pos
        m = self.m

        z_out_g = self.op(x, y)

        x_in = x[:, px-m:px+m+1, :]
        y_in = y[:, py-m:py+m+1, :]
        z_out_m = self.op(x_in, y_in)

        upsampler = nn.Upsample(size=(10, 1), mode='bilinear', align_corners=False)
        z_out_m = upsampler(z_out_m.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        z_out = z_out_g + z_out_m
  
        return z_out


#! Found Fusion Module
class FoundFusionModule(nn.Module):

    def __init__(self, args, genotype):
        super(FoundFusionModule, self).__init__()

        self.args = args 
        self.geno = genotype
        self.T = 10
        self.D = args.hid_dim
        self.num_cells = args.num_cells
        self.num_input_nodes = args.num_scale

        self.edges = genotype.edges
        self.fusion_cells = nn.ModuleList()

        # aggregate outputs of all cells
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.D * self.num_cells, self.D),
            nn.LayerNorm([self.T, self.D]),
            nn.ReLU()
        )

    def build_cells(self, FoundFusionCell):
        # build fusion cells and corresponding alpha params
        for i in range(self.num_cells):
            cell = FoundFusionCell(self.args, self.geno.ops[i])
            self.fusion_cells.append(cell)

    def genotype(self):
        return self.geno



class FoundSingleModalFusionModule(FoundFusionModule):

    def __init__(self, args, genotype):
        super(FoundSingleModalFusionModule, self).__init__(args, genotype)
        
        self.build_cells(FoundSingleModalFusionCell)

    def forward(self, feats):
        stage_list = []
        for input_feature in feats:
            stage_list.append(input_feature)
        
        for i in range(self.num_cells):
            input_x, input_y = self.edges[i]
            input_x, input_y = stage_list[input_x], stage_list[input_y]
            out_z = self.fusion_cells[i](input_x, input_y)  # [B, T, D]

            stage_list.append(out_z)

        out = torch.cat(stage_list[-self.num_cells:], dim=-1)   # [B, T, D*num_cells]    
        out = self.fusion_layer(out)       # [B, T, D]
        
        return out, stage_list[-self.num_cells:]
    

class FoundMultiModalFusionModule(FoundFusionModule):

    def __init__(self, args, genotype):
        super(FoundMultiModalFusionModule, self).__init__(args, genotype)
        
        self.build_cells(FoundMultiModalFusionCell)
        
    def build_cells(self, FoundFusionCell):
        # build fusion cells and corresponding alpha params
        for i in range(self.num_cells):
            cell = FoundFusionCell(self.args, self.geno.poses[i], self.geno.wins[i], self.geno.ops[i])
            self.fusion_cells.append(cell)

    def forward(self, aud_feats, vid_feats):
        a_stage_list, v_stage_list = [], []
        for a, v in zip(aud_feats, vid_feats):
            a_stage_list.append(a)
            v_stage_list.append(v)

        for i in range(self.num_cells):
            input_x, input_y = self.edges[i]
            input_x, input_y = a_stage_list[input_x], v_stage_list[input_y]
            z_out = self.fusion_cells[i](input_x, input_y)  # [B, T, D]

            a_stage_list.append(z_out)
            v_stage_list.append(z_out)
        
        z_out = torch.cat(a_stage_list[-self.num_cells:], dim=-1)   # [B, T, D*num_cells]   
        z_out = self.fusion_layer(z_out)       # [B, T, D] 
        return z_out



#! Found M3T Module
class FoundMultiModalMultiTemporalModule(nn.Module):

    def __init__(self, args, genotype):
        super(FoundMultiModalMultiTemporalModule, self).__init__()
        
        self.args = args 
        self.geno = genotype

        self.amt = FoundSingleModalFusionModule(args, genotype.amt)
        self.vmt = FoundSingleModalFusionModule(args, genotype.vmt)
        self.avmt = FoundMultiModalFusionModule(args, genotype.avmt)

    def genotype(self):
        return self.geno

    def forward(self, aud_feats, vid_feats):

        a_out, a_out_list = self.amt(aud_feats)
        v_out, v_out_list = self.vmt(vid_feats)
        z_out = self.avmt(aud_feats, vid_feats)
        # z_out = self.avmt(a_out_list, v_out_list)

        return a_out, v_out, z_out
