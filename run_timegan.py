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

from model.TimeGAN import TimeGAN
from model.custom_dataset import *
from model.utils_timegan import *

args = config.get_config() # argument 호출
fix_seed(args.seed) # seed 고정
file_name = args.file_name # 데이터 파일명
WINDOW_SIZE = args.window_size # Window size
scale_type = args.scale_type # Scaler Type 설정 ('Standard' or 'MinMax' or 'Robust')
undo = args.undo # reconstruction 시 unscale 수행여부

# Load & Scale data
TRAIN_DF, TEST_DF, TRAIN_SCALED, TEST_SCALED, TRAIN_Time, TEST_Time, cols, scaler = load_gen_data(file_name = file_name, scale_type = scale_type)

# under custom_dataset.py
## Train dataset with stride
train_dataset = NetisDataset(data = TRAIN_SCALED, timestamps = TRAIN_Time, window_size = WINDOW_SIZE, stride =1)

## Test dataset with no window collapse (for generation window size must be WINDOW_SIZE)
test_dataset = NetisDataset(data = TEST_SCALED, timestamps = TEST_Time, window_size = WINDOW_SIZE, stride = WINDOW_SIZE)

# SET DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SET ARGUMENTS
args.feature_dim = train_dataset[0].size(1)
args.Z_dim = train_dataset[0].size(1)
args.model_path = "save_model"

# DEFINE MODEl
model = TimeGAN(args)
model = model.to(device)

if args.is_train:
    print('INITIATING TRAINING PROCESS...')
    timegan_trainer(model, train_dataset, args)
    print('>>>> TRAINING COMPLETE!')
if args.is_generate:
    gen_data=timegan_generator(model, args.num_generation, args)
    np.save(os.getcwd()+"\\save_data\\",gen_data)




