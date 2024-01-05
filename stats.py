import os
import sys
import pickle
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from ptflops import get_model_complexity_info
from thop import profile


from utils.args import *
from models.model import AVVPMultiModalMultiTemporalModel as M3T


def build_args():
    # parse args
    parser = argparse.ArgumentParser(description='AV-NAS Configuration')
    parser = add_meta_args(parser)
    parser = add_avvp_args(parser)
    args = parser.parse_args()

    # config args
    args = config_args(args)
    
    return args

def main():
    args = build_args()
    logger = args.logger

    #! load genotype
    if not 'search' in args.mode:
        args.genotype = None
        geno_path = os.path.join(args.search_dir, 'best/genotype.pkl')
        with open(geno_path, 'rb') as geno_file:
            args.genotype = pickle.load(geno_file)
        args.num_cells = len(args.genotype.avmt.ops)
        if args.log:
            logger.info('Loaded genotype: {}'.format(args.genotype))

    #! load model
    model = M3T(args).to(args.device)

    #! statistic params
    inputs = {
        'audio': torch.rand(1, 10, 128),
        'video_s': torch.rand(1, 80, 2048),
        'video_st': torch.rand(1, 10, 512),
        'label': torch.rand(1, 25),
        'pa': torch.rand(1, 25),
        'pv': torch.rand(1, 25),
    }
    # flops, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)
    
    flops, params = profile(model, (inputs,))
    # logger.info('flops: ', flops, 'params: ', params)
    logger.info('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info('model.params(): %.2f M' % (pytorch_total_params / 1000000.0))

if __name__ == '__main__':
    main()

