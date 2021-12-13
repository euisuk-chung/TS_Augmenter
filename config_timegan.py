import argparse
import multiprocessing
import torch

"""
Set configuration for the module
"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parser_setting(parser):
    '''
    Set arguments
    '''
    # for loading data
    parser.add_argument(
        '--scale_type',
        choices=['Standard', 'MinMax', 'Robust'],
        default='MinMax',
        type=str)
    parser.add_argument(
        '--file_name',
        default='netis',
        type=str)
    parser.add_argument(
        '--cols_to_remove',
        default='Time',
        type=str,
        nargs='*',
        help = 'Columns to Remove')
    parser.add_argument(
        "--split",
        type=str2bool,
        default=False,
        help = 'Argument for Train/Test split')
    parser.add_argument(
        '--time_gap',
        default=100,
        type=int)
    
    # train/generate argument
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        "--is_generate",
        type=str2bool,
        default=True)
    parser.add_argument(
        "--undo",
        type=str2bool,
        default=False)
    parser.add_argument(
        '--num_generation',
        default=1000,
        type=int)
    
    # etc
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--window_size',
        default=30,
        type=int)

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=1000,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=1000,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=1000,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=256,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=200,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)
    parser.add_argument(
        '--dload',
        default="save_model",
        type=str)
    return parser

def get_config():
    parser = argparse.ArgumentParser()
    default_parser = parser_setting(parser)
    args, _ = default_parser.parse_known_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    return args




