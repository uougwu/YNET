# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:56:14 2024

@author: uugwu01
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from transposition import trans

def trans(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:,:] = x[i,:,:].T
    return y

def calcRedoxMap(NADH, FAD, mask):

    rr_map = FAD / (FAD + NADH)
    rr_map[np.isnan(rr_map)] = 0
    rr_map[np.isinf(rr_map)] = 0
    nadh_masked = NADH * mask
    fad_masked = FAD * mask
    
    num_depths = 1
    mean_rr = np.zeros((1,num_depths))

    for i in range(num_depths):
        nadh_image = nadh_masked
        fad_image = fad_masked
        mean_rr[i] = np.sum(fad_image) / (np.sum(fad_image) + np.sum(nadh_image))
    return mean_rr

def imtile(S):
    A = S[0,:,:]
    B = S[1,:,:]
    C = S[2,:,:]
    D = np.zeros_like(C)
    return np.block([[A,C],[B,D]])

def computeRedoxRatio(A,M):
    Ksi4 = 0.71      
    result_mean_rr = []

    for i in range(0, M.shape[0]):
        cell_m_group = M[i,:,:, :,2]
      
        tile_roi_mask = imtile(cell_m_group)
        
        I75 = imtile(A[i,:,:, :,1])
        I84 = imtile(A[i,:,:, :,2])
        I85 = imtile(A[i,:,:, :,3])
           
        Lipo_cytoMask = (tile_roi_mask * ((I85 / I75) > 0.8))#.astype(np.uint8)
        tile_nadh_image_array = (1 - Lipo_cytoMask).astype(np.uint8) * (2 * (6967669713504718261986677655339008 * I84 
                                                            + 639531254069473395025846086926336 * I75 * Ksi4 
                                                            - 4746566188796147214139252047110791 * I85 * Ksi4) 
                                                       / (640155554689971829081431822952503 * Ksi4))
        tile_fad_image_array = (1 - Lipo_cytoMask) * I85 #.astype(np.uint8)
    
        tile_fad_image_array = np.array(tile_fad_image_array, dtype=float)
        tile_fad_image_array[np.isnan(tile_fad_image_array)] = 0
        tile_fad_image_array[np.isinf(tile_fad_image_array)] = 0
        tile_fad_image_array[tile_fad_image_array < 0] = 0
    
        tile_nadh_image_array = np.array(tile_nadh_image_array, dtype=float)
        tile_nadh_image_array[np.isnan(tile_nadh_image_array)] = 0
        tile_nadh_image_array[np.isinf(tile_nadh_image_array)] = 0
        tile_nadh_image_array[tile_nadh_image_array < 0] = 0
    
        mean_rr = calcRedoxMap(tile_nadh_image_array, tile_fad_image_array, tile_roi_mask)
    
        result_mean_rr.append(mean_rr)
    redox_ratio = [item[0,0] for item in result_mean_rr]
    return redox_ratio


def compute_rmse(y_true, y_pred):
    squared_diff = (y_true - y_pred) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)
    return rmse

def plot_redox(A,M,UM,var):
    redox_ratio_main = computeRedoxRatio(A,M)      
    redox_ratio = computeRedoxRatio(A,UM)   
   #print(redox_ratio,redox_ratio_main)
    rmse = compute_rmse(M,UM)
    #plt.figure(figsize=(16, 8))
    plt.plot(redox_ratio_main,'*--b',label='Manual Seg',lw=3.5, ms=10)
    plt.plot(redox_ratio,'*--r',label='Unet Seg',lw=3.5, ms=10)
    plt.xlabel('ROI',fontsize=15)
    plt.ylabel('Mean Redox Ratio',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.title(f'{var}, RMSE: {rmse:.4f}',fontsize=15)
    plt.show()

    
def plot_redoxy(A,UM,var):     
    redox_ratio = computeRedoxRatio(A,UM)   
    #print(redox_ratio)
    #plt.figure(figsize=(16, 8))
    plt.plot(redox_ratio,'*--r',label='Unet Seg',lw=3.5, ms=10)
    plt.xlabel('ROI',fontsize=15)
    plt.ylabel('Mean Redox Ratio',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.title(var,fontsize=15)
    plt.show()


# import numpy as np
# import h5py
# from transposition import trans
# from combination import matching_slices, matching_masks
# from tensorflow.keras.utils import to_categorical

# img11 = h5py.File('Week11_Tensor.mat','a')   
# e = 'EX745_845' 
# f = 'Shg_Neu_Silk'  
# im11 = matching_slices(trans(np.array(img11.get(e))))
# mk11 = matching_masks(trans(np.array(img11.get(f))))
# plot_redoxy(im11,to_categorical(mk11,4),'me')



# import scipy.io
# scipy.io.savemat('data_test.mat', {'data_test': test_code})