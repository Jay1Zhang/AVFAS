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


from utils.args import *
from searcher import AVVPSearcher
from models.darts.architect import Architect
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


def search(args, searcher, model, optimizer, scheduler, logger):

    #! load architecture
    arch_params = model.module.arch_params() if args.parallel else model.arch_params()
    arch_optimizer = optim.Adam(arch_params,
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    architect = Architect(args, model, arch_optimizer)

    #! search
    if args.log:
        logger.info("-" * 50)
        logger.info('Start searching for AVVP task with {} GPUs.'.format(torch.cuda.device_count()))

    start_time = time.time()
    best_epoch, best_f1_seg, best_f1_eve, best_genotype = searcher.search(model, architect, optimizer, scheduler, logger)
    time_elapsed = time.time() - start_time

    if args.log:
        logger.info("-" * 50)
        logger.info('Search completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best epoch is {}'.format(best_epoch))
        logger.info('Best segment-level type@avg f1 score is {:.4f}'.format(best_f1_seg))
        logger.info('Best event-level type@avg f1 score is {:.4f}'.format(best_f1_eve))
        logger.info('Best genotype is {}'.format(best_genotype))


def train(args, searcher, model, optimizer, scheduler, logger):
    
    #! train
    if args.log:
        logger.info("-" * 50)
        logger.info('Start training for AVVP task with {} GPUs.'.format(torch.cuda.device_count()))

    start_time = time.time()
    best_epoch, best_f1_seg, best_f1_eve = searcher.train(model, optimizer, scheduler, logger)
    time_elapsed = time.time() - start_time

    if args.log:
        logger.info("-" * 50)
        logger.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best epoch is {}'.format(best_epoch))
        logger.info('Best segment-level type@avg f1 score is {:.4f}'.format(best_f1_seg))
        logger.info('Best event-Level type@Avg f1 Score is {:.4f}'.format(best_f1_eve))
 

def eval(args, searcher, model, logger):

    #! load checkpoint
    checkpoint_path = f'{args.result_dir}/best/model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): checkpoint[k] for k in checkpoint}, strict=True) 

    #! evaluate
    searcher.eval(model, logger)


def estimate_ma(args, searcher, model, logger):

    #! load checkpoint
    checkpoint_path = f'{args.result_dir}/best/model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): checkpoint[k] for k in checkpoint}, strict=True) 

    #! evaluate
    searcher.estimate_ma(model, logger)


def estimate_noise(args, searcher, model, logger):

    #! load checkpoint
    checkpoint_path = f'{args.result_dir}/best/model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): checkpoint[k] for k in checkpoint}, strict=True) 

    #! evaluate
    searcher.estimate_noise(model, logger)


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
    if args.parallel:
        model = DDP(model,
                    device_ids=[args.device],
                    output_device=args.device,
                    find_unused_parameters=True)

    #! load optimizer and architecture
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min, last_epoch=-1, verbose=False)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    #! main procedure
    searcher = AVVPSearcher(args)
    
    if 'eval' in args.mode:
        eval(args, searcher, model, logger)
    elif 'search' in args.mode:
        search(args, searcher, model, optimizer, scheduler, logger)
    elif 'train' in args.mode:
        train(args, searcher, model, optimizer, scheduler, logger)
    elif 'estimate_ma' in args.mode:
        estimate_ma(args, searcher, model, logger)
    elif 'estimate_noise' in args.mode:
        estimate_noise(args, searcher, model, logger)


if __name__ == '__main__':
    main()

