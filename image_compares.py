# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:22:27 2020

@author: hamze
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Input Parameters
instances = 3
bmode = True

with open('label_names', 'rb') as f: 
            pickleList = pickle.load(f)
faList = []
faCounter = 0

cyList = []
cyCounter = 0

idcList = []
idcCounter = 0

for i in range(len(pickleList)):
    data = pickleList[i]
    img = data[1]
    
    
    if bmode :
        rows = img.shape[1] 
        for row in range(rows):
            img[:,row] = 20 * np.log(img[:,row])
        
    if data[2] == 'FA' and faCounter < instances:
        faCounter += 1
        faList.append(img)
    elif data[2] == 'CYST' and cyCounter < instances:
        cyCounter += 1
        cyList.append(img)
    elif data[2] == 'IDC' and idcCounter < instances:
        idcCounter += 1
        idcList.append(img)  

plt.rcParams["figure.figsize"] = (50,50)
for i in range(instances):
    plt.subplot(3,instances,i+1)
    plt.imshow(faList[i] ,cmap='gray',aspect='auto')
    plt.subplot(3,instances,instances+i+1)
    plt.imshow(cyList[i] ,cmap='gray',aspect='auto')
    plt.subplot(3,instances,2*instances+i+1)
    plt.imshow(idcList[i] ,cmap='gray',aspect='auto')

print('END')