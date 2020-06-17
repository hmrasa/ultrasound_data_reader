# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:42:06 2020

@author: hamze
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert

from enum import Enum

class DataType(Enum):
    RF = 1
    ENV = 2 
    IQ = 3
    IMG = 4
    ALL = 5

class FileType(Enum):
    KU = 1
    CC_M = 2 
    NA = 3

'''
   @ filename
   @return KU,CC_M
'''
def getFileType(filename):
    data_type = np.dtype ('uint32').newbyteorder ('>')
    alldata = np.fromfile (filename, dtype=data_type)
    if alldata[1] == 72:
        return FileType.CC_M
    return FileType.KU
        
'''
    @ filename,
    @ frame
    @ return I,Q
'''
def ku_iq_reader(filename,frame):
    try:

        data_type = np.dtype ('uint16').newbyteorder ('>')
        alldata = np.fromfile (filename, dtype=data_type)
        header = alldata[0:8]
        
        alines = header[1]
        datalen = header[3]
        
        alinelen = 16 + datalen
        startFrame = 9 + (9 + alines * alinelen) * frame 
        endFrame =  startFrame + alines * alinelen
        
        vect = alldata[startFrame :endFrame]  
        if vect.shape[0]==0:
            return ([],[],FileType.NA)
        
        rawdata = np.reshape(vect, (alines,alinelen))
        rawdata = rawdata[:,16:]
        I = rawdata[:,::3]
        Q = rawdata[:,1::3]
        
        It = I > 2047;
        It = 4096*It;
        I = I-It;
        
        Qt = Q > 2047;
        Qt = 4096*Qt;
        Q = Q-Qt; 
    
        return (I,Q,FileType.KU)
    except:
        return ([],[],FileType.NA) 
'''

'''    
def cc_m_rf_reader(filename, frame):

    try:
        data_type = np.dtype ('uint32').newbyteorder ('>')
        alldata = np.fromfile (filename, dtype=data_type)
        head = alldata[0:8]
        
        w = head[6]
        h = head[7]
        frms = head[4]
        offset = head[1]
        if frame > frms:
          return ("Invalid Frame", [],frms)
                
        
        data_type = np.dtype ('int16').newbyteorder ('>')
        alldata = np.fromfile (filename, dtype=data_type)
        
        start =  offset + (frame-1) * w * h * 2 
        start = int(start/2)
        
        end = start + w * h
        
        vect = alldata[start: end]
        
        
        rf = np.reshape(vect, (w,h))
        return ('',rf,frms) 
    except:
        return ("ERR", [],[])

'''
'''
def interpft(x,ny):
   # siz = x.shape
    m = x.shape[0]
    
    #If necessary, increase ny by an integer multiple to make ny > m.
    if ny > m:
        incr = 1
    else:
        incr = np.floor(m/ny) + 1
        ny = incr*ny
    
    a = np.fft.fft(x)
    nyqst = int(np.ceil((m+1)/2))
    p1 = a[0:nyqst]
    p2 = np.zeros(ny-m)
    p3 = a[nyqst:m]
    b = np.concatenate((p1 , p2 , p3))
    if np.remainder(m,2) == 0 :
        b[nyqst] = b[nyqst]/2
        b[nyqst+ny-m] = b[nyqst]
    y = np.fft.ifft(b);
    y = y * ny / m;
    y = y[0:ny:incr] # Skip over extra points when oldny <= m.
    
    return y.real

'''
    @ I,Q
    @ f: center frequency, 
    @ decim:
    @ intf:
'''
def ku_rf(I,Q,f,decim,intf):
    [n,m] = I.shape
    rf = np.zeros((n,intf * m));
    sample_time_interval = decim/(36*intf);
    t = np.arange(0,(intf*m)*sample_time_interval,sample_time_interval)
    theta = 2 * np.pi * f * t
    c = np.cos(theta); 
    c = c[:]
    s = np.sin(theta)
    s = s[:]
    
    for aline in range(n):
        (intf1) = interpft(I[aline,:], m * intf)
        intf1 = intf1 * c
        
        (intf2) = interpft(Q[aline,:], m * intf)
        intf2 = intf2 * s
        
        rf[aline,:] = intf1 - intf2;

    return rf
    
'''
    @ file_name: DAT file
    @ frame: # of Frame in DAT file
    @ data_type: RF,ENV,IQ,ALL
'''
def rf_extractor(file_name,frame,data_type):
    file_type = getFileType(file_name)
    print(file_type)
    if file_type == FileType.CC_M:
        (err,rf,frms) = cc_m_rf_reader(file_name, frame)
        if data_type == DataType.RF:
            return rf
        rf2 = np.swapaxes(rf,0,1)
        analytical_signal = hilbert(rf2)
        env = np.abs(analytical_signal)
        if data_type == DataType.ENV:         
            return env
        if data_type == DataType.IQ:
             I = np.imag(analytical_signal)
             Q = np.real(analytical_signal)
             return (I,Q)
        if data_type == DataType.IMG:
            img = np.zeros(env.shape)
            rows = env.shape[0] 
            for row in range(rows):
                img[row,:] = 20 * np.log(env[row,:])  
            return img    
        if data_type == DataType.ALL:
            I = np.imag(analytical_signal)
            Q = np.real(analytical_signal)
            img = np.zeros(env.shape)
            rows = env.shape[0] 
            for row in range(rows):
                img[row,:] = 20 * np.log(env[row,:])
            return(rf,env,I,Q,img) 

    if file_type == FileType.KU:
        (I,Q,err) = ku_iq_reader(file_name,frame)
        if err == FileType.NA:
             return ([])
         
        rf = ku_rf(I,Q,7.5,1,2)
        if data_type == DataType.IQ:
            return (I,Q)
        if data_type == DataType.RF:           
            return rf
        if data_type == DataType.ENV:
            rf2 = np.swapaxes(rf,0,1)
            analytical_signal = hilbert(rf2)
            env = np.abs(analytical_signal)
            return env
        if data_type == DataType.IMG:
            rf2 = np.swapaxes(rf,0,1)
            analytical_signal = hilbert(rf2)
            env = np.abs(analytical_signal)           
            img = np.zeros(env.shape)
            rows = env.shape[0] 
            for row in range(rows):
                img[row,:] = 20 * np.log(env[row,:])  
            return img 
        if data_type == DataType.ALL:
            rf2 = np.swapaxes(rf,0,1)
            analytical_signal = hilbert(rf2)
            env = np.abs(analytical_signal)           
            img = np.zeros(env.shape)
            rows = env.shape[0] 
            for row in range(rows):
                img[row,:] = 20 * np.log(env[row,:]) 
            return(rf,env,I,Q,img) 
    

#TEST
#fileName = 'large_files\Timothy Hall - CC_Breast_data\CC003/CC003_EI_RF.DAT'
#fileName = 'large_files\Timothy Hall - M_Breast_data\B140-2/B140-2_EI_RF.DAT'
#fileName = 'large_files\Timothy Hall - KU_Breast_data\KU_02OCT00PC1\KU_02OCT00PC1_EI_RF.DAT'

fileName = 'large_files\Timothy Hall - KU_Breast_data\KU019a\KU019a_EI_RF.DAT'



#(rf,env,I,Q,img) = 
img = rf_extractor(fileName,1,DataType.IMG)

plt.imshow(img,aspect='auto',cmap='gray')
print( 'Done')

        
             
            
            
            

            
            
    
        
        
     
    

