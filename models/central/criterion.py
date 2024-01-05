import torch
import torch.nn as nn
import torch.nn.functional as F


class AVVPCriterion(nn.Module):
    def __init__(self, args):
        super(AVVPCriterion, self).__init__()

        self.device = args.device
        self.margin = 1.0
        self.criter_cls = nn.BCELoss()
        self.criter_mse = nn.MSELoss()
        
    def forward(self, inputs, outputs):
        target = inputs['label'].type(torch.FloatTensor).to(self.device)
        Pa = inputs['pa'].type(torch.FloatTensor).to(self.device)
        Pv = inputs['pv'].type(torch.FloatTensor).to(self.device)

        global_prob, a_prob, v_prob = outputs['global_prob'], outputs['a_prob'], outputs['v_prob']
        global_prob.clamp_(min=1e-7, max=1-1e-7)
        a_prob.clamp_(min=1e-7, max=1-1e-7)
        v_prob.clamp_(min=1e-7, max=1-1e-7)

        loss_avmt = self.criter_cls(global_prob, target)
        loss_amt = self.criter_cls(a_prob, Pa) + self.margin - self.criter_mse(a_prob, v_prob.detach())
        loss_vmt = self.criter_cls(v_prob, Pv) + self.margin - self.criter_mse(v_prob, a_prob.detach())
        
        loss = loss_avmt + loss_amt + loss_vmt 
        return loss


class ContrastiveCriterion(nn.Module):
    def __init__(self, args):
        super(ContrastiveCriterion, self).__init__()

        self.device = args.device
        self.W_a = nn.Linear(args.hid_dim, args.hid_dim)
        self.W_v = nn.Linear(args.hid_dim, args.hid_dim)

        self.smoothing = 0.1
        self.confidence = 1.0 - self.smoothing
        self.cls = 10
        self.dim = -1

    def label_smoothing_loss(self, pred, target):
        # pred: [B*10, 10]
        # target: [B*10]
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)    # scatter_(dim, index, src)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


    def forward(self, a_out, v_out, z_out):
        a_feat = F.normalize(a_out, p=2, dim=-1)    
        v_feat = F.normalize(v_out, p=2, dim=-1)   
        z_feat_a = F.normalize(self.W_a(z_out), p=2, dim=-1)   
        z_feat_v = F.normalize(self.W_v(z_out), p=2, dim=-1)   
        logits1 = torch.bmm(z_feat_a, a_feat.permute(0, 2, 1)) / 0.07      # [B, 10, 10]
        logits2 = torch.bmm(z_feat_v, v_feat.permute(0, 2, 1)) / 0.07      # [B, 10, 10]
        logits1 = logits1.reshape(-1, 10)
        logits2 = logits2.reshape(-1, 10)

        labels = torch.zeros(a_out.size(0), 10).long()
        for i in range(10):
            labels[:, i] = i
        labels = labels.to(self.device).reshape(-1)

        loss = self.label_smoothing_loss(logits1, labels) + self.label_smoothing_loss(logits2, labels)

        outputs = {
            'z_feat_a': z_feat_a,
            'z_feat_v': z_feat_v,
            'a_feat': a_feat,
            'v_feat': v_feat,

            'a_out': a_feat + z_feat_a,
            'v_out': v_feat + z_feat_v
        }
        return loss, outputs 

