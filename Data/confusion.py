# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:41:18 2022
    ##----------------## 
    # Confusion matrix
    ##----------------## 
@author: Cai Yanan
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io as sio
from sklearn.metrics import confusion_matrix

# C->I 
model='C_I'
datapath_test = model + '.mat'
data_test = sio.loadmat(datapath_test)
datapath_pred = '../models_DSC/results_' + model + '.mat'
data_pred = sio.loadmat(datapath_pred)
y_pred = np.transpose(data_pred['pre'])
y_test = np.transpose(data_test['Label_test'])

# imageCLEF(C_I)
classes=['aeroplane', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'dog', 'horse', 'monitor', 'motorbike', 'people']

# office31(A_W)
# classes=['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
#           'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor', 
#           'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',  'punchers', 'ring_binder',  
#           'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash can']

# Office-Home(Ar_Cl)
# classes=['Alarm Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 
#           'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk Lamp', 
#           'Drill',' Eraser', 'Exit Sign', 'Fan', 'File Cabinet', 'Flipflops', 'Flowers', 'Folder',
#           'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp Shade',
#           'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 
#           'Pan', 'Paper Clip', 'Pen', 'Pencil', 'Postit Notes', 'Printer', 'Push Pin', 'Radio', 
#           'Refrigerator', 'ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 
#           'Speaker', 'Spoon', 'Table', 'Telephone', 'Toothbrush', 'Toys', 'Trash Can', 'TV', 'Webcam']

con_mat = confusion_matrix(y_test, y_pred)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)

#-- plot --#
fig, ax = plt.subplots(figsize= (12,10))
plt.subplots_adjust(left=0.2, right=1, top=0.76, bottom=0.02)
# heatmap = ax.pcolor(con_mat_norm, cmap=plt.cm.Greens)
sns.heatmap(con_mat_norm,annot = True, cmap='Greens', annot_kws={"fontsize":8.5})
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(con_mat_norm.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(con_mat_norm.shape[0]) + 0.5, minor=False)

# want a more natural, table-like display
# ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(classes, minor=False, fontsize=18,fontfamily="Times New Roman", rotation=90) #20, rotation=45 
ax.set_yticklabels(classes, minor=False, fontsize=18,fontfamily="Times New Roman", rotation=0)  #20

plt.show()


