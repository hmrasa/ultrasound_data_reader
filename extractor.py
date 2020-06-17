# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:13:00 2020

@author: hamze
"""

import pickle, gzip
gz = gzip.open( 'all_imgs_labels_env.pkl.gz', 'rb' );
all_imgs_labels_env = pickle.load(gz)
gz.close()


print('File loded')
imgs_labels_env = []
imgs_labels_env_RF = []
imgs_labels_env_I_Q = []
length = len(all_imgs_labels_env)
for i in range(length):
    row = all_imgs_labels_env[i]
    
    newrow = {}
    newrow[0] = row[0]
    newrow[1] = row[2]
    newrow[2] = row[1]
    imgs_labels_env.append(newrow)
    
    newrow1 = {}
    newrow1[0] = row[0]
    newrow1[1] = row[2]
    newrow1[2] = row[3]
    newrow1[3] = row[1]
    imgs_labels_env_RF.append(newrow1)
    
    newrow2 = {}
    newrow2[0] = row[0]
    newrow2[1] = row[2]
    newrow2[3] = row[4]
    newrow2[4] = row[5]
    newrow2[5] = row[1]
    imgs_labels_env_I_Q.append(newrow2)
    
    print(str(i))
    
pickle.dump(imgs_labels_env, gzip.open( 'imgs_labels_env.pkl.gz', 'wb' ) )
pickle.dump(imgs_labels_env_RF, gzip.open( 'imgs_labels_env.pkl.gz', 'wb' ) )
pickle.dump(imgs_labels_env_I_Q, gzip.open( 'imgs_labels_env.pkl.gz', 'wb' ) )

print('End')