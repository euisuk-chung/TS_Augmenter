import os
import numpy as np
import pandas as pd
import pickle

import random
import torch
import numpy as np

from einops import rearrange
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(file_name, scale_type = 'Standard', cols_to_remove = None):
    """
    folder: folder where data is located
    """
    
    # define path(must be in pkl file)
    data_loc = f'./data/netis/{file_name}.pkl'    
    
    # get data
    with open(data_loc, 'rb') as f:
        df = pickle.load(f)
    
    # if needed remove columns that is not necessary
    if cols_to_remove != None:
        df = df_total.drop(cols_to_remove, axis=1)
    
    df = df.dropna()
    
    # TRAIN TEST SPLIT
    # TRAIN
    TRAIN_DF = df.query('Time < 20211103184400 or Time > 20211106084400 and label==0')
    
    # TEST(GET ONLY 정상)
    TEST_DF = df.query('Time >= 20211103184400 and Time <= 20211106084400 and label==0')

    TOTAL_DF = df.to_numpy()
    
    # REMOVE TIME & LABEL
    TRAIN_DF = TRAIN_DF.iloc[:,1:-1]
    cols = TRAIN_DF.columns
    TRAIN_DF = TRAIN_DF.to_numpy()
    TEST_DF = TEST_DF.iloc[:,1:-1].to_numpy()
    
    if scale_type == 'MinMax':
        scaler = MinMaxScaler()
    elif scale_type == 'Standard':
        scaler = StandardScaler()
    elif scale_type == 'Robust':
        scaler = RobustScaler()    
    else:
        pass
    
    TRAIN_SCALED = scaler.fit(TRAIN_DF).transform(TRAIN_DF)
    TEST_SCALED = scaler.transform(TEST_DF)
    
    return TRAIN_DF, TEST_DF, TRAIN_SCALED, TEST_SCALED, cols, scaler

def concat_recon(recon_output):
    
    w,b,f = recon_output.shape
    tmp = rearrange(recon_output, 'w b f -> b w f')
    output = tmp.reshape(w*b,f)

    return output

def eval_recon(recon, real, scaler = None, undo = False):
    criterion = nn.MSELoss()
    
    if undo == True:
        assert scaler != None, 'Scaler should be defined!!'
        
        # reverse scaling
        recon = scaler.inverse_transform(recon)
    
    r = recon.shape[0]
    real = real[:r,:]

    # compute loss
    eval_loss = criterion(torch.tensor(recon), torch.tensor(real))
    
    return eval_loss

def get_diff(recon, real, scaler = None, undo = False):
    
    if undo == True:
        assert scaler != None, 'Scaler should be defined!!'
        
        # reverse scaling
        recon = scaler.inverse_transform(recon)
    
    r = recon.shape[0]
    real = real[:r,:]
    
    return np.abs(recon-real)