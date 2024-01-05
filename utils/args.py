import os
import sys
import ast
import time
import logging
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def add_meta_args(parser: argparse.ArgumentParser):
    """
    Add meta arguments to parser.
    """
    # meta
    parser.add_argument('--task', type=str, default='avvp', help="task")
    parser.add_argument('--mode', type=str, default='search', help="mode")
    
    # exp
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--port', type=str, default='123456', help='random seed')
    parser.add_argument('--search_dir', type=str, default=None, help='search experiment directory')
    parser.add_argument('--result_dir', type=str, default=None, help='where to save the experiment')
    parser.add_argument('--parallel', help='use several GPUs', action='store_true', default=False) 
    parser.add_argument('--tensorboard', help='use tensorboard', action='store_true', default=False)

    # model
    parser.add_argument('--model', type=str, default='M3T', help='which model to use')
    parser.add_argument('--T', type=int, default=10, help='number of temporal dimension')
    parser.add_argument('--hid_dim', type=int, default=512, help='number of hidden units per layer')
    parser.add_argument('--ffn_dim', type=int, default=512, help='dimension of feed forward layer')
    parser.add_argument('--nhead', type=int, default=8, help='dimension of feed forward layer')
    parser.add_argument('--num_cells', type=int, help='number of fusion cells', default=4)    
    parser.add_argument('--scale', type=ast.literal_eval, default='[10, 5, 2, 1]')               

    # learning settings
    parser.add_argument('--epochs', type=int, default=40, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--num_workers', type=int, help='dataloader CPUs', default=32)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--drpt', action="store", default=0.1, dest="drpt", type=float, help="dropout")

    # network optimizer and scheduler
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--stepsize', type=int, default=20, help='step size of #StepLR')
    parser.add_argument('--gamma', type=float, default=0.8, help='gamma of #StepLR')
    parser.add_argument('--T_max', type=int, help='T_max of #CosineAnnealingLR.', default=40)
    parser.add_argument('--eta_min', type=float, help='min laerning rate of #CosineAnnealingLR', default=1e-7)

    # archtecture optimizer
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')
    
    return parser


def add_avvp_args(parser: argparse.ArgumentParser):
    """
    Add avvp arguments to parser.
    """
    # dataset
    data_dir = '/home/xxx/research/datasets/llp/'
    parser.add_argument('--audio_dir', type=str, default=data_dir + 'feats/vggish/', help="audio dir")
    parser.add_argument('--video_dir', type=str, default=data_dir + 'feats/res152/', help="video dir")
    parser.add_argument('--st_dir', type=str, default=data_dir + 'feats/r2plus1d_18/', help="video dir")
    parser.add_argument('--label_train', type=str, default=data_dir + "AVVP_train.csv", help="weak train csv file")
    parser.add_argument('--label_val', type=str, default=data_dir + "AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument('--label_test', type=str, default=data_dir + "AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--eval_audio', type=str, default=data_dir + "AVVP_eval_audio.csv", help="eval audio annotation csv file")
    parser.add_argument('--eval_visual', type=str, default=data_dir + "AVVP_eval_visual.csv", help="eval visual annotation csv file")
    parser.add_argument('--label_ma', type=str, default=None, help='modality aware label path')
    parser.add_argument('--label_denoise', type=str, default=None, help='label denoise path')
    parser.add_argument('--warm_up_epoch', type=float, default=0.9, help='warm-up epochs')
    
    return parser

def add_avel_args(parser: argparse.ArgumentParser):
    """
    Add avel arguments to parser.
    """
    parser.add_argument('--weak', help='use weakly supervised setting', action='store_true', default=False) 
    # dataset 
    data_dir = '/home/xxx/research/datasets/ave/feats/'
    parser.add_argument('--audio_dir', type=str, default=data_dir + 'audio_feature.h5', help="audio dir")
    parser.add_argument('--video_dir', type=str, default=data_dir + 'visual_feature.h5', help="video dir")
    parser.add_argument('--label_dir', type=str, default=data_dir + 'labels.h5', help="label dir")
    parser.add_argument('--label_train', type=str, default=data_dir + "train_order.h5", help="weak train csv file")
    parser.add_argument('--label_val', type=str, default=data_dir + "val_order.h5", help="weak val csv file")
    parser.add_argument('--label_test', type=str, default=data_dir + "test_order.h5", help="weak test csv file")
    parser.add_argument('--audio_bg_dir', type=str, default=data_dir + 'audio_feature_noisy.h5', help="audio dir")
    parser.add_argument('--video_bg_dir', type=str, default=data_dir + 'visual_feature_noisy.h5', help="video dir")
    parser.add_argument('--label_bg_dir', type=str, default=data_dir + "labels_noisy.h5", help="weak train csv file")
    parser.add_argument('--label_mil_dir', type=str, default=data_dir + "mil_labels.h5", help="weak train csv file")
    
    return parser


def config_args(args):

    ### config seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.enabled=True

    ### config parallel
    if args.parallel:
        init_distributed_mode(args)
    else:
        args.dist_rank = 0
        args.world_size = 1
        args.device = torch.device('cuda:0')

    print('Distributed init (local rank {}/{})'.format(args.dist_rank, args.world_size))

    ### config result dir
    if args.result_dir is None:
        args.result_dir = '{}-{}-{}'.format(args.mode, args.model, time.strftime("%Y%m%d-%H%M%S"))
        if args.search_dir is None:
            args.result_dir = os.path.join(f'/home/xxx/research/mm-nas/results/{args.task}/', args.result_dir)
        else:
            args.result_dir = os.path.join(args.search_dir, args.result_dir)
        os.makedirs(os.path.join(args.result_dir, 'best'), exist_ok=True)

    ### config logger and tensorboard    
    args.log = (args.dist_rank == 0)
    logger = logging.getLogger()
    if args.log:
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.result_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        # logger = logging.getLogger()
        logger.addHandler(fh)
        logging.info("args = %s", args)
    args.logger = logger

    if args.log:
        args.tb_writer = None
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_path = os.path.join(args.result_dir, 'tb_logs/')
            args.tb_writer = SummaryWriter(log_dir=tb_log_path, purge_step=1)

    return args


def init_distributed_mode(args):

    if 'SLURM_PROCID' in os.environ:
        local_rank = int(os.environ['SLURM_PROCID'])
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        addr = node_list[8:].replace('-', '.')
        
        os.environ['MASTER_PORT'] = args.port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(local_rank)

    args.dist_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.device = torch.device('cuda', args.dist_rank)

    torch.cuda.set_device(args.dist_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AVFAS Configuration')
    parser = add_meta_args(parser)
    args = parser.parse_args()

    print(type(args.scale))
    print(args.scale)

