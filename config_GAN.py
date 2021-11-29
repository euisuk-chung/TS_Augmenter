"""Set configuration for the module
"""
import argparse
import multiprocessing
import torch

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
    """Set arguments
    """
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
        "--is_train",
        type=str2bool,
        default=True)
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
        '--max_seq_len',
        default=100,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)

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
        default=512,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
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

    return parser

def get_config():
    parser = argparse.ArgumentParser()
    default_parser = parser_setting(parser)
    args, _ = default_parser.parse_known_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    return args




