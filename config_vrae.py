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
        "--is_generate_train",
        type=str2bool,
        default=True)
    
    # TODO : For further testing
    parser.add_argument(
        "--is_generate_test",
        type=str2bool,
        default=False)
    
    # rescaling argument
    parser.add_argument(
        "--undo",
        type=str2bool,
        default=False)
    
    # etc
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    
    # Data Arguments
    parser.add_argument(
        '--window_size',
        default=30,
        type=int)
    
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
        default=184,
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




