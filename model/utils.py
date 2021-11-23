import os
import numpy as np
import glob
import pandas as pd


def load_data(folder, cols_to_remove = None):
    """
    folder: folder where data is located
    """
    # define path
    data_path = f'./{folder}/*.csv'
    
    # get data
    file_list = glob.glob(data_path)
    file_list.sort()
    
    # load dataset
    df_total = pd.DataFrame()

    for i in file_list:
        data = pd.read_csv(i)
        df_total = pd.concat([df_total, data])

    # Sort by date
    df_total = df_total.reset_index(drop = True)
    df_total = df_total.drop(cols_to_remove, axis=1)
    df_total = df_total.to_numpy()
    
    return df_total

# TODO : Delete function
def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]

