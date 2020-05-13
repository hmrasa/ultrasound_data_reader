# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:45:48 2020

@author: hamze
"""

import numpy as np
import matplotlib.pyplot as plt
import rf_reader as rf
### Start Program
filename = 'large_files\Timothy Hall - KU_Breast_data\KU_12SEP00LR3\KU_12SEP00LR3_EI_RF.DAT'

(I,Q) = rf.iqread(filename)
(rf) = rf.iq2rf_jj(I,Q,7.5,1,2)

rf2 = np.swapaxes(rf,0,1)
plt.plot(rf2)
plt.show()