import os
import sys
import random
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# fix seed
def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# load generation data
def load_gen_data(file_name, scale_type = 'MinMax',\
                  cols_to_remove = ['Time'], split = False):
    """
    file_name: file_name in data location
    scale_type : choose scaling type
    """
    
    # define path(must be uder data folder and in pkl file)
    data_loc = f'./data/{file_name}.pkl'    
    
    # get data
    with open(data_loc, 'rb') as f:
        df = pickle.load(f)
    
    # 결측치 제거
    df = df.dropna()
    
    # TRAIN/TEST SPLIT 없으면 전체 데이터 사용
    if split == True :
        # Netis 데이터 기준으로 작성되어 있기때문에 도메인에 맞도록 Train/Test Split을 진행함
        TOTAL_DF = df
        TRAIN_DF = df.query('Time < 20211103184400 or Time > 20211106084400 and label==0')
        TEST_DF = df.query('Time >= 20211103184400 and Time <= 20211106084400 and label==0')
    else:
        TOTAL_DF = df
        TRAIN_DF = df
        TEST_DF = df
        
    # 시간 정보 저장
    TRAIN_Time = TRAIN_DF['Time']
    TEST_Time = TEST_DF['Time']
    
    # if needed remove columns that is not necessary
    # 지정 변수 제거
    if cols_to_remove != None:
        TOTAL_DF = TOTAL_DF.drop(cols_to_remove, axis=1)
        TRAIN_DF = TRAIN_DF.drop(cols_to_remove, axis=1)
        TEST_DF = TEST_DF.drop(cols_to_remove, axis=1)
    
    # Get column Info
    cols = TOTAL_DF.columns
    
    # To numpy
    TOTAL_DF = TOTAL_DF.to_numpy()
    TRAIN_DF = TRAIN_DF.to_numpy()
    TEST_DF = TEST_DF.to_numpy()
    
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
    
    return TRAIN_DF, TEST_DF, TRAIN_SCALED, TEST_SCALED, TRAIN_Time, TEST_Time, cols, scaler
    
# loader with stride
class TimeSeriesDataset(Dataset):
    def __init__(self, data, timestamps, window_size, stride=1, time_gap=100):
        self.data = torch.from_numpy(np.array(data))
        self.ts = np.array(timestamps)
        self.valid_idxs = []
        self.window_size = window_size
        
        for L in trange(len(self.ts) - self.window_size + 1):
            # define Right
            R = L + self.window_size - 1
            
            # append val indexs
            if self.ts[R]-self.ts[L] == (self.window_size-1)*time_gap:
                self.valid_idxs.append(L)
        
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, index):
        i = self.valid_idxs[index]
        x = self.data[i: i + self.window_size]
        return x.float()


# with no window collapsing
class GenerationDataset(Dataset):
    def __init__(self, data, window):
        self.data = torch.Tensor(data)
        self.window = window
 
    def __len__(self):
        return len(self.data) // self.window # -1
    
    def __getitem__(self, index):
        x = self.data[index*self.window:(index+1)*(self.window)]
        return x