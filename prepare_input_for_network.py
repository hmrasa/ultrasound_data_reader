# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:09:58 2020

@author: hamze
"""

import numpy as np
from PIL import Image
from scipy.signal import hilbert
import pickle, gzip, lzma, bz2

def iqread(filename,frame_number):
    try:

        data_type = np.dtype ('uint16').newbyteorder ('>')
        alldata = np.fromfile (filename, dtype=data_type)
        header = alldata[0:8]
        
        alines = header[1]
        datalen = header[3]
        
        alinelen = 16 + datalen
        startFrame = 9 + (9 + alines * alinelen) * frame_number 
        endFrame =  startFrame + alines * alinelen
        
        #vect = alldata[9+startFrame :(alines*alinelen+9)]
        vect = alldata[startFrame :endFrame]  
        if vect.shape[0]==0:
            return ([],[],False)
        
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
    
        return (I,Q,True)
    except:
        return ([],[],False)

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


def iq2rf_jj(I,Q,f,decim,intf):
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

"""
amplitude_envelope,
Lesion Type
"""
def bmode(rf,label_dataset,filename):
    rf2 = np.swapaxes(rf,0,1)
    img = np.zeros(rf2.shape)
    analytical_signal = hilbert(rf2)
    amplitude_envelope = np.abs(analytical_signal)
    rows = rf2.shape[0] 
    for row in range(rows):
        img[row,:] = 20 * np.log(amplitude_envelope[row,:])
    
    img = Image.fromarray(img).convert('L') 
    label = filename.split('\\')[-2] 
    lesion_type = label_dataset[label_dataset.eq(label).any(1)]['Lesion Type']
    isValid = False
    if lesion_type.empty:
        lesion_type = 'UNKNOWN'
    else:
        lesion_type = lesion_type.values[0]
        if lesion_type.upper() == 'CYST' or lesion_type.upper() == 'FA' or lesion_type.upper() == 'IDC':
            isValid = True
    if rf.shape[1] < 1000 or rf.shape[0] < 100 :
        isValid = False
        
    lesion_type = lesion_type.replace('/', '_SLASH_').replace('&', '_AND_').replace(',', '_COMA_')
    return img,amplitude_envelope,label,lesion_type,isValid

### Start Program

from glob import glob
import pandas as pd

excel_path ='OldDataInfo.xls'
label_file = pd.read_excel (excel_path, sheet_name='Query1')
labels = pd.DataFrame(label_file, columns= ['No','Lesion Type'])

lesions = []
lesionList = labels['Lesion Type']

for label in lesionList:
    label = label.replace('/', '_SLASH_').replace('-', '_').replace(' ', '').replace('&', '_AND_').replace(',', '_COMA_')
    if not( label in lesions):
        lesions.append(label)

filenames = glob('large_files\Timothy Hall - KU_Breast_data\*\*.DAT') 
#filenames += glob('large_files\Timothy Hall - CC_Breast_data\*\*.DAT')    

cc_bmp_folder = 'large_files\cc_bmp_folder'
length = len(filenames)

imgs_labels_env = []
imgs_labels_env_RF = []
imgs_labels_env_I_Q = []

file_count = 0;
numberOfFrame = 10
for counter in range(length):
    filename = filenames[counter]
    
    for frame_number in range(numberOfFrame):

        (I,Q,isValid) = iqread(filename,frame_number)
        
        if isValid:
            (rf) = iq2rf_jj(I,Q,7.5,1,2)
            (img,amplitude_envelope,file_name,lesion_type,isValid) = bmode(rf,labels,filename)
            out_file_name = cc_bmp_folder 
                    
            if isValid:
                imgs_labels_env_Row = {}
                imgs_labels_env_Row[0] = file_name 
                imgs_labels_env_Row[1] = lesion_type.upper()
                imgs_labels_env_Row[2] = amplitude_envelope 
                imgs_labels_env.append(imgs_labels_env_Row)
                
                imgs_labels_env_RF_Row = {}
                imgs_labels_env_RF_Row[0] = file_name 
                imgs_labels_env_RF_Row[1] = lesion_type.upper()
                imgs_labels_env_RF_Row[2] = amplitude_envelope
                imgs_labels_env_RF_Row[3] = rf
                imgs_labels_env_RF.append(imgs_labels_env_RF_Row)
                
                imgs_labels_env_I_Q_Row = {}
                imgs_labels_env_I_Q_Row[0] = file_name 
                imgs_labels_env_I_Q_Row[1] = lesion_type.upper()
                imgs_labels_env_I_Q_Row[2] = amplitude_envelope
                imgs_labels_env_I_Q_Row[3] = I
                imgs_labels_env_I_Q_Row[4] = Q
                imgs_labels_env_I_Q.append(imgs_labels_env_I_Q_Row)
                
                file_count += 1
            
        print(f'{counter+1}/{length} - {frame_number} - {isValid} - {file_count} - File:{file_name}')

   
pickle.dump(imgs_labels_env, gzip.open( 'imgs_labels_env.pkl.gz', 'wb' ) )
pickle.dump(imgs_labels_env_RF, gzip.open( 'imgs_labels_env_RF.pkl.gz', 'wb' ) )
pickle.dump(imgs_labels_env_I_Q, gzip.open( 'imgs_labels_env_I_Q.pkl.gz', 'wb' ) )  
         
print('End')

