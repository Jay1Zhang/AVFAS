import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.central.module import *
from models.central.criterion import *

from models.darts.module_search import SearchableMultiModalMultiTemporalModule as SearchableM3T
from models.darts.module_found import FoundMultiModalMultiTemporalModule as FoundM3T
from models.darts.genotypes import Genotype


class AVVPMultiModalMultiTemporalModel(nn.Module):

    def __init__(self, args):
        super(AVVPMultiModalMultiTemporalModel, self).__init__() 

        self.args = args
        #! preprocess module
        self.preprocessor = AVVPreprocessor(args)

        #! multi-scale encoder module
        args.num_scale = len(args.scale)
        self.mse_a = MultiScaleEncoderModule(args)
        self.mse_v = MultiScaleEncoderModule(args)

        #! task-specific head
        self.classifier = AVVPHead(args)

        #! criterion
        self.criterion = AVVPCriterion(args)
        self.criter_cpc = ContrastiveCriterion(args)

        #! searchable multi-temporal fusion module
        # self.m3t = MultiTemporalFusionModule(args)
        if 'search' in args.mode:
            self.m3t = SearchableM3T(args)
        else:
            assert args.genotype is not None, 'Error, genotype must be provided if not in search mode.'
            self.m3t = FoundM3T(args, args.genotype)


    def arch_params(self):
        return self.m3t.arch_params()

    def genotype(self):
        return self.m3t.genotype()

    def forward(self, inputs, with_ca=True):
        
        #! 0. preprocess raw features
        audio, video = self.preprocessor(inputs)    # [B, 10, 512], [B, 10, 512]

        #! 1. generate multi-scale temporal features
        a_feats = self.mse_a(audio)
        v_feats = self.mse_v(video)

        #! 2. search multi-temporal fusion module
        a_out, v_out, z_out = self.m3t(a_feats, v_feats)   # [B, 10, 512] x3
        # representative learning 
        loss_cpc, outputs_cpc = self.criter_cpc(a_out, v_out, z_out)

        #! 3. classify
        if with_ca:
            outputs = self.classifier(a_out + z_out, v_out + outputs_cpc['z_feat_v'])
        else:
            outputs = self.classifier(a_out, v_out)
        
        #! 4. loss
        loss = self.criterion(inputs, outputs)
        loss = loss + 0.2 * loss_cpc

        return loss, outputs
