import os
import sys
import pickle

import dateutil
from datetime import timedelta

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import config_vrae as config

from TaPR_pkg import etapr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from scipy.stats import mode
from tqdm.auto import trange
from scipy.stats import norm

from models.vrae import VRAE
from utils.custom_dataset import *
from utils.utils_vrae import *

# argument 호출
args = config.get_config() 

# seed 고정
fix_seed(args.seed)

file_name = args.file_name # 데이터 파일명
WINDOW_SIZE = args.window_size # Window size
scale_type = args.scale_type # Scaler Type 설정 ('Standard' or 'MinMax' or 'Robust')
undo = args.undo # reconstruction 시 unscale 수행여부
cols_to_remove = args.cols_to_remove # 제거할 변수
split = args.split # TRAIN/TEST Split 여부
time_gap = args.time_gap # 데이터수집단위


# Load & Scale data
TRAIN_DF, TEST_DF, TRAIN_SCALED, TEST_SCALED, TRAIN_Time, TEST_Time, cols, scaler = load_gen_data(file_name = file_name, \
                                                                                                  scale_type = scale_type,\
                                                                                                  cols_to_remove = cols_to_remove,\
                                                                                                  split = split)

# under custom_dataset.py
## Train dataset with stride
train_dataset = TimeSeriesDataset(data = TRAIN_SCALED, timestamps = TRAIN_Time, window_size = WINDOW_SIZE, stride =1, time_gap = time_gap)

# SET ARGUMENTS
args.dload = "./save_model"
args.sequence_length = WINDOW_SIZE
args.number_of_features = train_dataset[0].shape[1]

# TRAIN
if args.is_train:
    # DEFINE MODEl
    vrae = VRAE(args.sequence_length, args.number_of_features, args.hidden_size,\
            args.hidden_layer_depth, args.latent_length,\
            args.batch_size, args.learning_rate, args.block,\
            args.n_epochs, args.dropout_rate, args.optimizer, args.loss,\
            args.cuda, args.print_every, args.clip, args.max_grad_norm, args.dload)
    
    # STARTING TRAINING
    print('INITIATING TRAINING PROCESS...')
    history = vrae.fit(train_dataset)
    print('>>>> TRAINING COMPLETE!')

    print('SAVING TRAINED MODEL...')
    
    # save model
    vrae.save(f'VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}.pth')
    print(f'>>>> model saved as "VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}"')

# TRAIN dataset reconstruction
if args.is_generate_train:
    
    # define train generation data
    train_gen_dataset = TimeSeriesDataset(data = TRAIN_SCALED, timestamps = TRAIN_Time, window_size = WINDOW_SIZE, stride = WINDOW_SIZE, time_gap = time_gap)

    # FOR GENERATION MUST HAVE batch_size 1
    args.batch_size = 1

    # DEFINE MODEl FOR GENERATION
    vrae = VRAE(args.sequence_length, args.number_of_features, args.hidden_size,\
            args.hidden_layer_depth, args.latent_length,\
            args.batch_size, args.learning_rate, args.block,\
            args.n_epochs, args.dropout_rate, args.optimizer, args.loss,\
            args.cuda, args.print_every, args.clip, args.max_grad_norm, args.dload)
    
    # load trained model
    vrae.load(f'VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}.pth')
    
    # reconstruct train data
    train_recon = vrae.reconstruct(train_gen_dataset)
    train_recon = concat_recon(train_recon)
    
    # original train data
    for window_num in range(len(train_gen_dataset)):
        if window_num == 0:
            # intialize train_org
            train_org = train_gen_dataset[window_num]
        else:
            # get curr window
            tmp_window = train_gen_dataset[window_num]
            # concat next window
            train_org = np.concatenate((train_org, tmp_window), axis=0)
    
    # get loss
    # TODO : update unscaled version
    train_loss = eval_recon(recon = train_recon, real = train_org, scaler = scaler, undo = args.undo)
    print(f'>> TRAIN RECONSTRUCTION LOSS : {train_loss}')
    
    # save original data
    train_org = pd.DataFrame(train_org, columns= cols)
    # train_org = pd.DataFrame(TRAIN_DF if args.undo == True else TRAIN_SCALED, columns= cols)
    train_org.to_csv(f'./gen_data_vae/train/original_{args.scale_type}_un_{args.undo}.csv', index=False)
    print('>> SAVED TRAIN ORIGINAL Data!! (Loc: gen_data_vae)')
    
    # save reconstructed data
    train_gen = pd.DataFrame(train_recon, columns= cols)
    train_gen.to_csv(f'./gen_data_vae/train/VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}.csv', index=False)
    print('>> SAVED TRAIN RECONSTRUCTED Data!! (Loc: gen_data_vae)')
    
#     print(f'TRAIN ORG SHAPE : {train_org.shape}')
#     print(f'TRAIN GEN SHAPE : {train_gen.shape}')
#     print(f'SHAPE COMPARE : {train_org.shape==train_gen.shape}')
    
# TEST dataset reconstruction
if args.is_generate_test:
    
    # define test generation data
    test_gen_dataset = TimeSeriesDataset(data = TEST_SCALED, timestamps = TEST_Time, window_size = WINDOW_SIZE, stride = WINDOW_SIZE, time_gap = time_gap)

    # FOR GENERATION MUST HAVE batch_size 1
    args.batch_size = 1

    # DEFINE MODEl FOR GENERATION
    vrae = VRAE(args.sequence_length, args.number_of_features, args.hidden_size,\
            args.hidden_layer_depth, args.latent_length,\
            args.batch_size, args.learning_rate, args.block,\
            args.n_epochs, args.dropout_rate, args.optimizer, args.loss,\
            args.cuda, args.print_every, args.clip, args.max_grad_norm, args.dload)
    
    # load trained model
    vrae.load(f'VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}.pth')

    # reconstruct test data
    test_recon = vrae.reconstruct(test_gen_dataset)
    test_recon = concat_recon(test_recon)

    # original test data
    for window_num in range(len(test_gen_dataset)):
        if window_num == 0:
            # intialize train_org
            test_org = test_gen_dataset[window_num]
        else:
            # get curr window
            tmp_window = test_gen_dataset[window_num]
            # concat next window
            test_org = np.concatenate((test_org, tmp_window), axis=0)
            
    # get loss
    # TODO : update unscaled version
    test_loss = eval_recon(recon = test_recon, real = test_org, scaler = scaler, undo = args.undo)
    print(f'>> TEST RECONSTRUCTION LOSS : {test_loss}')
    
    # save original data
    test_org = pd.DataFrame(test_org, columns= cols)
    # test_org = pd.DataFrame(TRAIN_DF if args.undo == True else TRAIN_SCALED, columns= cols)
    test_org.to_csv(f'./gen_data_vae/test/original_{args.scale_type}_un_{args.undo}.csv', index=False)
    print('>> SAVED TEST ORIGINAL Data!! (Loc: gen_data_vae)')
    
    # save reconstructed data
    test_gen = pd.DataFrame(test_recon, columns= cols)
    test_gen.to_csv(f'./gen_data_vae/test/VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}.csv', index=False)
    print('>> SAVED TEST RECONSTRUCTED Data!! (Loc: gen_data_vae)')
    
#     print(f'TEST ORG SHAPE : {test_org.shape}')
#     print(f'TEST GEN SHAPE : {test_gen.shape}')
#     print(f'SHAPE COMPARE : {test_org.shape==test_gen.shape}')

# If we train and test at the same time
# to get train/test diff and loss history
if args.is_train and args.is_generate_train and args.is_generate_test:

    # train recon diff
    train_diff = pd.DataFrame(get_diff(recon = train_recon, real = TRAIN_DF if args.undo == True else TRAIN_SCALED, scaler = scaler, undo = args.undo), columns= cols)
                     
    # test recon diff
    test_diff = pd.DataFrame(get_diff(recon = test_recon, real = TEST_DF if args.undo == True else TEST_SCALED, scaler = scaler, undo = args.undo), columns= cols)

    # save & export result
    result = dict()
    result['history'] = history
    result['train_loss'] = train_loss
    result['test_loss'] = test_loss
    result['train_diff'] = train_diff
    result['test_diff'] = test_diff

    with open(f'./result/VRAE_{args.scale_type}_un_{args.undo}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}_ep_{args.n_epochs}.pkl', 'wb') as f:
         pickle.dump(result, f)

    print('COMPLETE!')