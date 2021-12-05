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
    '''
    Set arguments
    '''
    
    parser.add_argument(
        '--scale_type',
        choices=['Standard', 'MinMax', 'Robust'],
        default='MinMax',
        type=str)
    
    parser.add_argument(
        "--undo",
        type=str2bool,
        default=False)
    
    # Model Arguments
    parser.add_argument(
        '--sequence_length',
        default=30,
        type=int)
    
    parser.add_argument(
        '--number_of_features',
        default=92,
        type=int)

    parser.add_argument(
        '--hidden_size',
        default=90,
        type=int)
    
    parser.add_argument(
        '--hidden_layer_depth',
        default=1,
        type=int)
    
    parser.add_argument(
        '--latent_length',
        default=30,
        type=int)
    
    parser.add_argument(
        '--n_epochs',
        default=1000,
        type=int)
    
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int)
    
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float)
    
    parser.add_argument(
        '--dropout_rate',
        default=0.2,
        type=float)
    
    parser.add_argument(
        '--optimizer',
        choices=['Adam'],
        default='Adam',
        type=str)
    
    parser.add_argument(
        "--cuda",
        type=str2bool,
        default=True)
    
    parser.add_argument(
        '--print_every',
        default=50,
        type=int)
    
    parser.add_argument(
        "--clip",
        type=str2bool,
        default=True)
    
    parser.add_argument(
        '--max_grad_norm',
        default=5,
        type=int)
    
    parser.add_argument(
        '--loss',
        choices=['SmoothL1Loss', 'MSELoss'],
        default='MSELoss',
        type=str)
    
    parser.add_argument(
        '--block',
        choices=['LSTM', 'GRU'],
        default='LSTM',
        type=str)
    
    parser.add_argument(
        '--dload',
        default='./saved_model',
        type=str)
    
    return parser

def get_config():
    parser = argparse.ArgumentParser()
    default_parser = parser_setting(parser)
    args = default_parser.parse_args()
    return args




