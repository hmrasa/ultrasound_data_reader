# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:22:27 2020

@author: hamze
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Input Parameters
instances = 30
bmode = True

with open('label_names', 'rb') as f: 
            pickleList = pickle.load(f)
faName = []
faList = []
faCounter = 0

cyName = []
cyList = []
cyCounter = 0

idcName = []
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
        faName.append(data[0])
    elif data[2] == 'CYST' and cyCounter < instances:
        cyCounter += 1
        cyList.append(img)
        cyName.append(data[0])
    elif data[2] == 'IDC' and idcCounter < instances:
        idcCounter += 1
        idcList.append(img)  
        idcName.append(data[0])

print('Total No. of FA: ',len(faList))
print('Total No. of CYST: ',len(cyList))
print('Total No. of IDC: ',len(idcList))

plt.rcParams["figure.figsize"] = (50,10)
plt.rcParams["font.size"] = 30
plt.rcParams["figure.max_open_warning"] = 100
for i in range(instances):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(faList[i],cmap='gray',aspect='auto')
    axs[1].imshow(cyList[i],cmap='gray',aspect='auto')
    axs[2].imshow(idcList[i],cmap='gray',aspect='auto')
    axs[0].set_title('FA---'+faName[i])
    axs[1].set_title('CYST---'+cyName[i])
    axs[2].set_title('IDC---'+idcName[i])
    


print('END')