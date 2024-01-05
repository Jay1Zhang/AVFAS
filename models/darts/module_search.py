import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.darts.operations import OPERATIONS, SMMT_CANDIDATES, CMMT_CANDIDATES
from models.darts.genotypes import Genotype, FusionGenotype, CrossFusionGenotype


#! Searchable Fusion Cell
class SearchableSingleModalFusionCell(nn.Module):

    def __init__(self, args):
        super(SearchableSingleModalFusionCell, self).__init__()
        
        # assert num_ops == len(CANDIDATE_OPERATIONS), 'Error, num_ops must be equal to the length of CANDIDATE_OPERATIONS.'
        num_ops = len(SMMT_CANDIDATES)
        self.betas = Variable(1e-3*torch.randn(num_ops), requires_grad=True) 
        self.ops = nn.ModuleList()
        for cand in SMMT_CANDIDATES:
            op = OPERATIONS[cand](args)
            self.ops.append(op)

    def arch_params(self):
        """
            betas: [num_ops]
        """
        return self.betas

    def forward(self, x, y):
        weights = F.softmax(self.betas, dim=-1)     # [num_ops]
        out = sum(w * op(x, y) for w, op in zip(weights, self.ops))
        return out

#! position
POSITION = [3, 5, 7]
POSITION_CANDIDATES = [(px, py) for py in POSITION for px in POSITION]
WINDOW_CANDIDATES = [1, 2]
class SearchableMultiModalFusionCell(nn.Module):

    def __init__(self, args):
        super(SearchableMultiModalFusionCell, self).__init__()
        
        #! position
        self.positions = POSITION_CANDIDATES
        self.gammas_p = Variable(1e-3*torch.randn(len(self.positions)), requires_grad=True)
        
        #! window
        self.windows = WINDOW_CANDIDATES
        self.gammas_w = Variable(1e-3*torch.randn(len(self.windows)), requires_grad=True)
        
        #! operation
        self.ops = nn.ModuleList()
        for cand in CMMT_CANDIDATES:
            op = OPERATIONS[cand](args)
            self.ops.append(op)
        self.gammas_o = Variable(1e-3*torch.randn(len(CMMT_CANDIDATES)), requires_grad=True) 


    def arch_params(self):
        """
            gammas: [num_position]
        """
        return [self.gammas_p, self.gammas_w, self.gammas_o]

    def forward(self, x, y):
        # input: x,y=[B,T,D]

        weights_p = F.softmax(self.gammas_p, dim=-1)     # [num_position]
        weights_w = F.softmax(self.gammas_w, dim=-1)     # [num_window]
        weights_o = F.softmax(self.gammas_o, dim=-1)     # [num_operation]

        upsampler = nn.Upsample(size=(10, 1), mode='bilinear', align_corners=False)

        z_out = 0
        for w_p, (px, py) in zip(weights_p, self.positions):
            
            z_out_p = 0
            for w_m, m in zip(weights_w, self.windows):      
                
                #! 1. global
                z_out_global = sum(w_o * op(x, y) for w_o, op in zip(weights_o, self.ops))

                #! 2. window 
                # m = int(m / 2)      # [1, 2, 3]
                x_in = x[:, px-m:px+m+1, :]
                y_in = y[:, py-m:py+m+1, :]
            
                z_out_m = sum(w_o * op(x_in, y_in) for w_o, op in zip(weights_o, self.ops))     # [B, m, D]
                # reshape: [B, m, 512] -> [B, 512, m, 1] -> [B, 512, 10, 1] -> [B, 10, 512]
                z_out_m = upsampler(z_out_m.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)    
                
                z_out_p += w_m * (z_out_global + z_out_m)

            z_out += w_p * z_out_p

        return z_out
        
#! position end


#! Searchable Fusion Module
class SearchableFusionModule(nn.Module):

    def __init__(self, args):
        super(SearchableFusionModule, self).__init__()
        
        self.args = args 
        self.T = 10
        self.D = args.hid_dim
        self.num_cells = args.num_cells
        self.num_input_nodes = args.num_scale

        self.alphas = []
        self.fusion_cells = nn.ModuleList()

        # aggregate outputs of all cells
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.D * self.num_cells, self.D),
            nn.LayerNorm([self.T, self.D]),
            nn.ReLU()
        )

    def arch_params(self):
        """
            alphas: [[2, 4], [2, 5], ...]
            betas: [[6], [6], ...]
        """
        return self.alphas + [cell.arch_params() for cell in self.fusion_cells]

    def build_cells(self, SearchableFusionCell):
        # build fusion cells and corresponding alpha params
        for i in range(self.num_cells):
            num_input = self.num_input_nodes + i  # {5, 6, ...}
            alpha = Variable(1e-3*torch.randn(2, num_input), requires_grad=True)    # [2, 5], [2, 6]
            cell = SearchableFusionCell(self.args)     

            self.alphas.append(alpha)
            self.fusion_cells.append(cell)


    def genotype(self):
        """
            e.g. if num_cells=2, num_scales=5
                    arch params is: [2, 5], [2, 6], [4], [4]
        """
        arch_params = self.arch_params()
        num_cells = self.args.num_cells
        
        arch_weights = []
        edges, ops = [], []
        for i in range(num_cells):
            edge, edge_weight = self.parse_edge(arch_params[i])
            op, op_weight = self.parse_op(arch_params[i + num_cells])
            edges.append(edge)
            ops.append(op)

            arch_weights.append({
                'Cell': i+1,
                'Input_X': edge_weight.tolist()[0],
                'Input_Y': edge_weight.tolist()[1],
                'OP': op_weight.tolist()
            })

        geno = FusionGenotype(edges=edges, ops=ops)
        return geno, arch_weights


class SearchableSingleModalFusionModule(SearchableFusionModule):

    def __init__(self, args):
        super(SearchableSingleModalFusionModule, self).__init__(args)
        
        self.build_cells(SearchableSingleModalFusionCell)

    def forward(self, feats):
        stage_list = []
        for input_feature in feats:
            stage_list.append(input_feature)
        
        for i in range(self.num_cells):
            stages = torch.stack(stage_list, dim=-1)        # [B, T, D, num_input]
            weights = F.softmax(self.alphas[i], dim=-1).to(stages.device)     # [2, num_input]
 
            input_x = stages @ weights[0]
            input_y = stages @ weights[1]
            out_z = self.fusion_cells[i](input_x, input_y)  # [B, T, D]

            stage_list.append(out_z)

        out = torch.cat(stage_list[-self.num_cells:], dim=-1)   # [B, T, D*num_cells]    
        out = self.fusion_layer(out)       # [B, T, D]
        # return out 

        # out = sum(stage_list[-self.num_cells:])
        return out, stage_list[-self.num_cells:]

    def parse_edge(self, alpha):
        # alpha: [2, num_input]
        weight = F.softmax(alpha, dim=-1)
        input_x, input_y = torch.max(weight, dim=1)[1].data.cpu().numpy()
        # 以免两个输入取到同一个结点
        if input_x == input_y:
            if weight[0][input_x] >= weight[1][input_y]:
                # keep input_x, select another edge for input_y
                weight[1][input_y] = 0.
                input_x, input_y = torch.max(weight, dim=1)[1].data.cpu().numpy()
            else:
                # keep input_y, select another edge for input_x
                weight[0][input_x] = 0.
                input_x, input_y = torch.max(weight, dim=1)[1].data.cpu().numpy()
            
        return (input_x, input_y), weight.data.cpu().numpy()

    def parse_op(self, beta):
        # beta: [num_ops]
        weight = F.softmax(beta, dim=-1)
        idx = torch.max(weight, dim=-1)[1].data.cpu().numpy()
        op = SMMT_CANDIDATES[idx]
        return (int(idx), op), weight.data.cpu().numpy()
    

class SearchableMultiModalFusionModule(SearchableFusionModule):

    def __init__(self, args):
        super(SearchableMultiModalFusionModule, self).__init__(args)
        
        self.build_cells(SearchableMultiModalFusionCell)
        
    def arch_params(self):
        """
            alphas: [[2, 4], [2, 5], ...]
            betas: [[6], [6], ...]
        """
        arch_params = []
        arch_params += self.alphas
        for cell in self.fusion_cells:
            arch_params += cell.arch_params()
        return arch_params


    def forward(self, aud_feats, vid_feats):
        a_stage_list, v_stage_list = [], []
        for a, v in zip(aud_feats, vid_feats):
            a_stage_list.append(a)
            v_stage_list.append(v)

        for i in range(self.num_cells):
            a_stages = torch.stack(a_stage_list, dim=-1)        # [B, T, D, num_input]
            v_stages = torch.stack(v_stage_list, dim=-1)        # [B, T, D, num_input]

            weights = F.softmax(self.alphas[i], dim=-1).to(a_stages.device)     # [2, num_input]
            input_x = a_stages @ weights[0]
            input_y = v_stages @ weights[1]

            z_out = self.fusion_cells[i](input_x, input_y)  # [B, T, D]

            a_stage_list.append(z_out)
            v_stage_list.append(z_out)
        
        z_out = torch.cat(a_stage_list[-self.num_cells:], dim=-1)   # [B, T, D*num_cells]   
        z_out = self.fusion_layer(z_out)       # [B, T, D] 
        return z_out

        # z_out = sum(a_stage_list[-self.num_cells:])
        # return z_out


    def parse_edge(self, alpha):
        # alpha: [2, num_input]
        weight = F.softmax(alpha, dim=-1)
        input_x, input_y = torch.max(weight, dim=1)[1].data.cpu().numpy()
        return (input_x, input_y), weight.data.cpu().numpy()

    def parse_pos(self, gamma):
        # alpha: [num_positions]
        weight = F.softmax(gamma, dim=-1)
        idx = torch.max(weight, dim=-1)[1].data.cpu().numpy()
        px, py = POSITION_CANDIDATES[idx]
        return (px, py), weight.data.cpu().numpy()

    def parse_win(self, gamma):
        # alpha: [num_window]
        weight = F.softmax(gamma, dim=-1)
        idx = torch.max(weight, dim=-1)[1].data.cpu().numpy()
        win = WINDOW_CANDIDATES[idx]
        return win, weight.data.cpu().numpy()

    def parse_op(self, gamma):
        # beta: [num_ops]
        weight = F.softmax(gamma, dim=-1)
        idx = torch.max(weight, dim=-1)[1].data.cpu().numpy()
        op = CMMT_CANDIDATES[idx]
        return (int(idx), op), weight.data.cpu().numpy()

    def genotype(self):
        """
            e.g. if num_cells=2, num_scales=5
                    arch params is: [2, 5], [2, 6], [num_position], [num_window], [num_op], [num_position], [num_window], [num_op]
        """
        num_cells = self.args.num_cells
        arch_params = self.arch_params()
        alpha_params = arch_params[:num_cells]
        gamma_params = arch_params[num_cells:]
        
        arch_weights = []
        edges, poses, wins, ops = [], [], [], []
        for i in range(num_cells):
            edge, edge_weight = self.parse_edge(alpha_params[i])
            pos, pos_weight = self.parse_pos(gamma_params[3*i])
            win, win_weight = self.parse_win(gamma_params[3*i + 1])
            op, op_weight = self.parse_op(gamma_params[3*i + 2])
            edges.append(edge)
            poses.append(pos)
            wins.append(win)
            ops.append(op)

            arch_weights.append({
                'Cell': i+1,
                'Input_X': edge_weight.tolist()[0],
                'Input_Y': edge_weight.tolist()[1],
                'Position': pos_weight.tolist(),
                'Window': win_weight.tolist(),
                'OP': op_weight.tolist()
            })

        geno = CrossFusionGenotype(edges=edges, poses=poses, wins=wins, ops=ops)
        return geno, arch_weights


#! Searchable M3T Module
class SearchableMultiModalMultiTemporalModule(nn.Module):

    def __init__(self, args):
        super(SearchableMultiModalMultiTemporalModule, self).__init__()
        
        self.args = args 
        self.mode = args.mode 

        self.amt = SearchableSingleModalFusionModule(args)
        self.vmt = SearchableSingleModalFusionModule(args)
        self.avmt = SearchableMultiModalFusionModule(args)

    def forward(self, aud_feats, vid_feats):

        a_out, a_out_list = self.amt(aud_feats)
        v_out, v_out_list = self.vmt(vid_feats)
        z_out = self.avmt(aud_feats, vid_feats)
        # z_out = self.avmt(a_out_list, v_out_list)

        return a_out, v_out, z_out

    def arch_params(self):
        assert 'search' in self.mode, 'Error, architecture params are only available in search mode.'
        return self.amt.arch_params() + self.vmt.arch_params() + self.avmt.arch_params()

    def genotype(self):
        """
            e.g. num_cells=4, num_scales=4
                Genotype(
                    
                )
        """
        geno = Genotype(
            amt=self.amt.genotype()[0],
            vmt=self.vmt.genotype()[0],
            avmt=self.avmt.genotype()[0]
        )
        arch_params = {
            'amt': self.amt.genotype()[1],
            'vmt': self.vmt.genotype()[1],
            'avmt': self.avmt.genotype()[1],
        }

        return geno, arch_params

