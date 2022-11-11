# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:52:26 2021
    ##----------------------------## 
    # Split train set and test set
    ##----------------------------## 
@author: Cai Yanan
"""

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

# A->W
# datapath = 'datasets/office31.mat'
# data = sio.loadmat(datapath)
# data_source = np.transpose(data['amazon_amazon'])
# data_target = np.transpose(data['amazon_webcam'])
# mat_path = 'A_W.mat'
# Label = np.transpose(data['Label'])

# Ar->Cl
# datapath = 'datasets/Office-Home.mat'
# data = sio.loadmat(datapath)
# data_source = np.transpose(data['Art_Art'])
# data_target = np.transpose(data['Art_Clipart'])
# mat_path = 'Ar_Cl.mat'
# Label = np.transpose(data['Label'])

# C->I
# datapath = 'datasets/imageCLEF.mat'
# data = sio.loadmat(datapath)
data_source = np.transpose(data['c_c'])
data_target = np.transpose(data['c_i'])
mat_path = 'C_I.mat'
Label = np.transpose(data['Label'])

# data:需要进行分割的数据集
# random_state:设置随机种子，保证每次运行生成相同的随机数
# test_size:将数据分割成训练集的比例
S_train_x,S_test_x,S_train_y,S_test_y = train_test_split(data_source, Label, test_size = 0.4, random_state = 32, stratify=Label)
T_train_x,T_test_x,T_train_y,T_test_y = train_test_split(data_target, Label, test_size = 0.4, random_state = 32, stratify=Label)

if (S_train_y==T_train_y).all():
    sio.savemat(mat_path, {'S_train': np.transpose(S_train_x), 
                           'S_test': np.transpose(S_test_x), 
                           'T_train': np.transpose(T_train_x), 
                           'T_test': np.transpose(T_test_x), 
                           'Label_train': np.transpose(S_train_y), 
                           'Label_test': np.transpose(S_test_y)})