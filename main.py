import os
import sys

import dateutil
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# import config_GAN as config
import config_VAE as config
import pickle

from TaPR_pkg import etapr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy.stats import mode
from tqdm.auto import trange
from scipy.stats import norm

from model.vrae import VRAE
# from model.TimeGAN import TimeGAN
from model.custom_dataset import GenerationDataset

from model.utils import *
# from model.utils_data import timegan_trainer, timegan_generator

fix_seed(555)

args = config.get_config()

file_name = 'netis'
scale_type = 'MinMax' # 'Standard' or 'MinMax'
undo = False

TRAIN_DF, TEST_DF, TRAIN_SCALED, TEST_SCALED, cols, scaler = load_data(file_name = file_name, scale_type = scale_type)

WINDOW_SIZE = 30

# Define Train/Test
train_dataset = GenerationDataset(TRAIN_SCALED, WINDOW_SIZE)
test_dataset = GenerationDataset(TEST_SCALED, WINDOW_SIZE)

# SET ARGUMENTS
args.dload = "./save_model"
args.sequence_length = WINDOW_SIZE
args.number_of_features = train_dataset[0].shape[1]
# args.hidden_layer_depth = 1
args.hidden_layer_depth = 2
args.batch_size = 64
args.n_epochs = 1000

# FOR VRAE
vrae = VRAE(args.sequence_length, args.number_of_features, args.hidden_size,\
            args.hidden_layer_depth, args.latent_length,\
            args.batch_size, args.learning_rate, args.block,\
            args.n_epochs, args.dropout_rate, args.optimizer, args.loss,\
            args.cuda, args.print_every, args.clip, args.max_grad_norm, args.dload)

# TRAIN
print('>> INITIATING TRAINING')
loss_arr = vrae.fit(train_dataset)
print('>>>> TRAINING COMPLETE!')

# SAVE MODEL
vrae.save(f'VRAE_{scale_type}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}.pth')
print(f'>>>> SAVED VRAE_{scale_type}_hidden_{args.hidden_layer_depth}_win_{args.sequence_length}')

# TRAIN reconstruct
train_recon = vrae.reconstruct(train_dataset)
train_recon = concat_recon(train_recon)
train_loss = eval_recon(recon = train_recon, real = TRAIN_DF if undo == True else TRAIN_SCALED, scaler = scaler, undo = undo)

print('>>TRAIN RECONSTRUCTION LOSS')
print(f'>>>> {train_loss}')

# TEST reconstruct
test_recon = vrae.reconstruct(test_dataset)
test_recon = concat_recon(test_recon)
test_loss = eval_recon(recon = test_recon, real = TEST_DF if undo == True else TEST_SCALED, scaler = scaler, undo = undo)

print('>>TEST RECONSTRUCTION LOSS')
print(f'>>>> {test_loss}')

# For Visualization
train_diff = pd.DataFrame(get_diff(recon = train_recon, real = TRAIN_DF, scaler = scaler, undo = undo), columns= cols)
test_diff = pd.DataFrame(get_diff(recon = test_recon, real = TEST_DF, scaler = scaler, undo = undo), columns= cols)

# save & export result
result = dict()
result['loss_arr'] = loss_arr
result['train_loss'] = train_loss
result['test_loss'] = test_loss
result['train_diff'] = train_diff
result['test_diff'] = test_diff

with open(f'./result/{scale_type}_un_{undo}_hl_{args.hidden_layer_depth}_win_{args.sequence_length}.pkl', 'wb') as f:
     pickle.dump(result, f)

print('COMPLETE!')