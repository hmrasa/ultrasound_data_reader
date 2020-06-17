# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:37:36 2020

@author: hamze
"""

from glob import glob
import numpy as np
from enum import Enum

class FileType(Enum):
    KU = 1
    CC_M = 2 
    NA = 3

'''
   @ filename
   @return KU,CC_M
'''
def getFileType(filename,frame):
    data_type = np.dtype ('uint32').newbyteorder ('>')
    alldata = np.fromfile (filename, dtype=data_type)
    frms = alldata[4]
    if alldata[1] == 72:
        return (FileType.CC_M,frms > frame)
    
    data_type = np.dtype ('uint16').newbyteorder ('>')
    alldata = np.fromfile (filename, dtype=data_type)
    header = alldata[0:8]
        
    alines = header[1]
    datalen = header[3]
    
    alinelen = 16 + datalen
    startFrame = 9 + (9 + alines * alinelen) * frame 
    endFrame =  startFrame + alines * alinelen

    return (FileType.KU,endFrame<=alldata.shape[0])

filenames = glob('large_files\Timothy Hall - KU_Breast_data\*\*.DAT') 
filenames += glob('large_files\Timothy Hall - KU_Breast_data\*\*.dat') 
filenames += glob('large_files\Timothy Hall - CC_Breast_data\*\*.DAT')  
filenames += glob('large_files\Timothy Hall - CC_Breast_data\*\*.dat')  
filenames += glob('large_files\Timothy Hall - M_Breast_data\*\*.DAT') 
filenames += glob('large_files\Timothy Hall - M_Breast_data\*\*.dat')  

goodfilenames = []
for i in range(len(filenames)):
    fn = filenames[i]
    goodfile = {'name':fn}
    frames = []
    for j in range(1,100):
        ft,valid = getFileType(fn,j)   
        print(str(ft)+"{"+str(j)+"} is "+ str(valid) +" ...> "+ fn)
        if valid:           
            frames.append(j)
            
        
    goodfile = {'frames':frames}        
    goodfilenames.append(goodfile)       

    
    
print("*****END****")

  

