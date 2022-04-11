import sys
import os
import pickle

import dateutil
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import config_timegan as config

from TaPR_pkg import etapr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from scipy.stats import mode
from tqdm.auto import trange
from scipy.stats import norm

from models.TimeGAN import TimeGAN
from utils.custom_dataset import *
from utils.utils_timegan import *

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

# SET DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SET ARGUMENTS
args.feature_dim = train_dataset[0].size(1)
args.Z_dim = train_dataset[0].size(1)
args.dload = "./save_model"

# model_path
# DEFINE MODEl
model = TimeGAN(args)
model = model.to(device)

if args.is_train:
    print('INITIATING TRAINING PROCESS...')
    timegan_trainer(model, train_dataset, args)
    print('>>>> TRAINING COMPLETE!')
if args.is_generate:
    gen_data=timegan_generator(model, args.num_generation, args)
    np.save(f'./gen_data_gan/gen_data',gen_data)
    print('>>>> GENERATION COMPLETE!')



