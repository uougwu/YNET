# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:59:08 2023

@author: uugwu01
"""
import numpy as np
def trans(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:,:] = x[i,:,:].T
    return y