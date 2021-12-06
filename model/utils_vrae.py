import os
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn

from einops import rearrange
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# concat reconstruction data if needed
# CAUTION : Windows should not collapse
def concat_recon(recon_output):
    
    w,b,f = recon_output.shape
    tmp = rearrange(recon_output, 'w b f -> b w f')
    output = tmp.reshape(w*b,f)

    return output

# reconstruction evaluation
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

# get difference between reconstruction data and real data
def get_diff(recon, real, scaler = None, undo = False):
    
    if undo == True:
        assert scaler != None, 'Scaler should be defined!!'
        
        # reverse scaling
        recon = scaler.inverse_transform(recon)
    
    r = recon.shape[0]
    real = real[:r,:]
    
    return np.abs(recon-real)