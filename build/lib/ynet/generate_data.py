# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 03:05:58 2023

@author: uugwu01
"""
import numpy as np
import h5py
from transposition import trans
from tensorflow.keras.utils import normalize
import scipy.io
import matplotlib.pyplot as plt

def enams_generator_mar_16(var):
    img1 = h5py.File('Z:/DeyPlay/W1_ENAMS_Tensor_Mar16.mat','a')
    img3 = h5py.File('Z:/DeyPlay/W3_ENAMS_Tensor_Mar16.mat','a')
    img4 = h5py.File('Z:/DeyPlay/W4_ENAMS_Tensor_Mar16.mat','a')
    img5 = h5py.File('Z:/DeyPlay/W5_ENAMS_Tensor_Mar16.mat','a')
    img6 = h5py.File('Z:/DeyPlay/W6_ENAMS_Tensor_Mar16.mat','a')
    img7 = h5py.File('Z:/DeyPlay/W7_ENAMS_Tensor_Mar16.mat','a')
    img8 = h5py.File('Z:/DeyPlay/W8_ENAMS_Tensor_Mar16.mat','a')
    img9 = h5py.File('Z:/DeyPlay/W9_ENAMS_Tensor_Mar16.mat','a')
    img11 = h5py.File('Z:/DeyPlay/W11_ENAMS_Tensor_Mar16.mat','a')
    e = 'EX745_845' 
    f = 'nam_mask' 
    im1 = trans(np.array(img1.get(e)))
    mk1 = trans(np.array(img1.get(f)))
    im3 = trans(np.array(img3.get(e)))
    mk3 = trans(np.array(img3.get(f)))
    im4 = trans(np.array(img4.get(e)))
    mk4 = trans(np.array(img4.get(f)))
    im5 = trans(np.array(img5.get(e)))
    mk5 = trans(np.array(img5.get(f))) 
    im6 = trans(np.array(img6.get(e)))
    mk6 = trans(np.array(img6.get(f)))
    im7 = trans(np.array(img7.get(e)))  
    mk7 = trans(np.array(img7.get(f)))  
    im8 = trans(np.array(img8.get(e)))
    mk8 = trans(np.array(img8.get(f)))
    im9 = trans(np.array(img9.get(e)))
    mk9 = trans(np.array(img9.get(f)))
    im11 = trans(np.array(img11.get(e)))
    mk11 = trans(np.array(img11.get(f)))
    
    if var == 'tri_3_5_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im3,im5,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk3,mk5,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_1_3_5_6_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im1,im3,im5,im6,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk1,mk3,mk5,mk6,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_1_3_4_5_6_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im1,im3,im4,im5,im6,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk1,mk3,mk4,mk5,mk6,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_1_3_5_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im1,im3,im5,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk1,mk3,mk5,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_6':
        cat_images = np.nan_to_num(im6,nan=0.0)
        cat_nams = mk6
          
    return cat_images, cat_nams

def nams_generator(var):
    img3 = h5py.File('Z:/DeyPlay/Week3_NAMS_Tensor.mat','a')
    img4 = h5py.File('Z:/DeyPlay/Week4_NAMS_Tensor.mat','a')
    img5 = h5py.File('Z:/DeyPlay/Week5_NAMS_Tensor.mat','a')
    img6 = h5py.File('Z:/DeyPlay/Week6_NAMS_Tensor.mat','a')
    img7 = h5py.File('Z:/DeyPlay/Week7_NAMS_Tensor.mat','a')
    img8 = h5py.File('Z:/DeyPlay/Week8_NAMS_Tensor.mat','a')
    img9 = h5py.File('Z:/DeyPlay/Week9_NAMS_Tensor.mat','a')
    img11 = h5py.File('Z:/DeyPlay/Week11_NAMS_Tensor.mat','a')
    e = 'EX745_845' 
    f = 'nam_mask' 
    im3 = trans(np.array(img3.get(e)))
    mk3 = trans(np.array(img3.get(f)))
    im4 = trans(np.array(img4.get(e)))
    mk4 = trans(np.array(img4.get(f)))
    im5 = trans(np.array(img5.get(e)))
    mk5 = trans(np.array(img5.get(f))) 
    im6 = trans(np.array(img6.get(e)))
    mk6 = trans(np.array(img6.get(f)))
    im7 = trans(np.array(img7.get(e)))  
    mk7 = trans(np.array(img7.get(f)))  
    im8 = trans(np.array(img8.get(e)))
    mk8 = trans(np.array(img8.get(f)))
    im9 = trans(np.array(img9.get(e)))
    mk9 = trans(np.array(img9.get(f)))
    im11 = trans(np.array(img11.get(e)))
    mk11 = trans(np.array(img11.get(f)))
    
    if var == 'tri_3_5_7_8_9_11':
        cat_images = np.concatenate([im3,im5,im6,im7,im8,im9,im11],axis=0)
        cat_nams = np.concatenate([mk3,mk5,mk6,mk7,mk8,mk9,mk11],axis=0) 

    elif var == 'tri_3_4_5_6_7_8_9_11':
        cat_images = np.concatenate([im3,im4,im5,im6,im7,im8,im9,im11],axis=0)
        cat_nams = np.concatenate([mk3,mk4,mk5,mk6,mk7,mk8,mk9,mk11],axis=0) 
          
    return cat_images, cat_nams

def mono_generatory(var):
    img1 = h5py.File('Z:/DeyPlay/Week1_Tensor.mat','a')
    img2 = h5py.File('Z:/DeyPlay/Week2_Tensor.mat','a')
    img3 = h5py.File('Z:/DeyPlay/Week3_Tensor.mat','a')
    img5 = h5py.File('Z:/DeyPlay/Week5_Tensor.mat','a')
    img7 = h5py.File('Z:/DeyPlay/Week7_Tensor.mat','a')
    img8 = h5py.File('Z:/DeyPlay/Week8_Tensor.mat','a')
    img9 = h5py.File('Z:/DeyPlay/Week9_Tensor.mat','a')
    img11 = h5py.File('Z:/DeyPlay/Week11_Tensor.mat','a')
    img14 = h5py.File('Z:/DeyPlay/Week14_Tensor.mat','a')
    e = 'EX745_845'
    f = 'Shg_Neu_Silk'
    im1 = trans(np.array(img1.get(e)))
    mk1 = trans(np.array(img1.get(f)))
    im2 = trans(np.array(img2.get(e)))
    mk2 = trans(np.array(img2.get(f)))      
    im3 = trans(np.array(img3.get(e)))
    mk3 = trans(np.array(img3.get(f)))
    im5 = trans(np.array(img5.get(e)))
    mk5 = trans(np.array(img5.get(f)))
    im7 = trans(np.array(img7.get(e)))
    mk7 = trans(np.array(img7.get(f)))
    im8 = trans(np.array(img8.get(e)))
    mk8 = trans(np.array(img8.get(f)))
    im9 = trans(np.array(img9.get(e)))
    mk9 = trans(np.array(img9.get(f)))
    im11 = trans(np.array(img11.get(e)))
    mk11 = trans(np.array(img11.get(f)))
    im14 = trans(np.array(img14.get(e)))
    mk14 = trans(np.array(img14.get(f)))
    if var == 'mono_3_5_7_8_9_11_14':
        cat_images = np.concatenate([im3,im5,im7,im8,im9,im11,im14],axis=0)
        cat_nams = np.concatenate([mk3,mk5,mk7,mk8,mk9,mk11,mk14],axis=0)  
        return cat_images, cat_nams    
    elif var =='mono_1_2':
        cat_images = np.concatenate([im1,im2],axis=0)
        cat_nams = np.concatenate([mk1,mk2],axis=0) 
        return cat_images, cat_nams 
    elif var == 'mono_1_2_3_5_7_8_9_11_14':
        cat_images = np.concatenate([im1,im2,im3,im5,im7,im8,im9,im11,im14],axis=0)
        cat_nams = np.concatenate([mk1,mk2,mk3,mk5,mk7,mk8,mk9,mk11,mk14],axis=0)  
        return cat_images, cat_nams  

def coculture_tri(var):
    e = 'EX745_845'
    f = 'Cell_M'
    g = 'index'
    if var == 'ind':
        img1 = h5py.File('Z:/DeyPlay/CCI_Con_Tri_2ndInj_Full.mat','a')
        imk1 = trans(np.array(img1.get(f)))
        im1 = trans(np.array(img1.get(e)))
        i = np.array(img1.get(g))
        cat_images = im1
        cat_masks = imk1
    elif var == 'id':
        img2 = h5py.File('Z:/DeyPlay/CCI_P110_Tri_2ndInj_Full.mat','a')
        im2 = trans(np.array(img2.get(e)))    
        imk2 = trans(np.array(img2.get(f)))    
        i = np.array(img2.get(g))
        cat_images = im2
        cat_masks = imk2
    elif var == 'nind':
        img3 = h5py.File('Z:/DeyPlay/Sham_Con_tri_2ndInj_Full.mat','a')
        im3 = trans(np.array(img3.get(e)))
        imk3 = trans(np.array(img3.get(f)))
        i = np.array(img3.get(g))
        cat_images = im3
        cat_masks = imk3
    elif var == 'nid':
        img4 = h5py.File('Z:/DeyPlay/Sham_P110_tri_2ndInj_Full.mat','a')
        im4 = trans(np.array(img4.get(e)))
        imk4 = trans(np.array(img4.get(f)))
        i = np.array(img4.get(g))
        cat_images = im4
        cat_masks = imk4
    return cat_images, cat_masks , i

def coculture_na(var):
    e = 'EX745_845'
    f = 'Cell_M'
    g = 'index'
    if var == 'ind':
        img1 = h5py.File('Z:/DeyPlay/CCI_Con_NA_2ndInj_Full.mat','a')
        imk1 = trans(np.array(img1.get(f)))
        im1 = trans(np.array(img1.get(e)))
        i = np.array(img1.get(g))
        cat_images = im1
        cat_masks = imk1
    elif var == 'id':
        img2 = h5py.File('Z:/DeyPlay/CCI_P110_NA_2ndInj_Full.mat','a')
        im2 = trans(np.array(img2.get(e)))    
        imk2 = trans(np.array(img2.get(f)))    
        i = np.array(img2.get(g))
        cat_images = im2
        cat_masks = imk2
    elif var == 'nind':
        img3 = h5py.File('Z:/DeyPlay/Sham_Con_NA_2ndInj_Full.mat','a')
        im3 = trans(np.array(img3.get(e)))
        imk3 = trans(np.array(img3.get(f)))
        i = np.array(img3.get(g))
        cat_images = im3
        cat_masks = imk3
    elif var == 'nid':
        img4 = h5py.File('Z:/DeyPlay/Sham_P110_NA_2ndInj_Full.mat','a')
        im4 = trans(np.array(img4.get(e)))
        imk4 = trans(np.array(img4.get(f)))
        i = np.array(img4.get(g))
        cat_images = im4
        cat_masks = imk4
    return cat_images, cat_masks , i

def coculture_nm(var):
    e = 'EX745_845'
    f = 'Cell_M'
    g = 'index'
    if var == 'ind':
        img1 = h5py.File('Z:/DeyPlay/CCI_Con_NM_2ndInj_Full.mat','a')
        imk1 = trans(np.array(img1.get(f)))
        im1 = trans(np.array(img1.get(e)))
        i = np.array(img1.get(g))
        cat_images = im1
        cat_masks = imk1
    elif var == 'id':
        img2 = h5py.File('Z:/DeyPlay/CCI_P110_NM_2ndInj_Full.mat','a')
        im2 = trans(np.array(img2.get(e)))    
        imk2 = trans(np.array(img2.get(f)))    
        i = np.array(img2.get(g))
        cat_images = im2
        cat_masks = imk2
    elif var == 'nind':
        img3 = h5py.File('Z:/DeyPlay/Sham_Con_NM_2ndInj_Full.mat','a')
        im3 = trans(np.array(img3.get(e)))
        imk3 = trans(np.array(img3.get(f)))
        i = np.array(img3.get(g))
        cat_images = im3
        cat_masks = imk3
    elif var == 'nid':
        img4 = h5py.File('Z:/DeyPlay/Sham_P110_NM_2ndInj_Full.mat','a')
        im4 = trans(np.array(img4.get(e)))
        imk4 = trans(np.array(img4.get(f)))
        i = np.array(img4.get(g))
        cat_images = im4
        cat_masks = imk4
    return cat_images, cat_masks , i

def coculture_neu(var):
    e = 'EX745_845'
    f = 'Cell_M'
    g = 'index'
    if var == 'ind':
        img1 = h5py.File('Z:/DeyPlay/CCI_Con_N_2ndInj_Full.mat','a')
        imk1 = trans(np.array(img1.get(f)))
        im1 = trans(np.array(img1.get(e)))
        i = np.array(img1.get(g))
        cat_images = im1
        cat_masks = imk1
    elif var == 'id':
        img2 = h5py.File('Z:/DeyPlay/CCI_P110_N_2ndInj_Full.mat','a')
        im2 = trans(np.array(img2.get(e)))    
        imk2 = trans(np.array(img2.get(f)))    
        i = np.array(img2.get(g))
        cat_images = im2
        cat_masks = imk2
    elif var == 'nind':
        img3 = h5py.File('Z:/DeyPlay/Sham_Con_N_2ndInj_Full.mat','a')
        im3 = trans(np.array(img3.get(e)))
        imk3 = trans(np.array(img3.get(f)))
        i = np.array(img3.get(g))
        cat_images = im3
        cat_masks = imk3
    elif var == 'nid':
        img4 = h5py.File('Z:/DeyPlay/Sham_P110_N_2ndInj_Full.mat','a')
        im4 = trans(np.array(img4.get(e)))
        imk4 = trans(np.array(img4.get(f)))
        i = np.array(img4.get(g))
        cat_images = im4
        cat_masks = imk4
    return cat_images, cat_masks , i



def mono_tbi_injury(var):
    img1 = h5py.File('Z:/DeyPlay/Neuron_8h_1stInj_Full.mat','a')
    img2 = h5py.File('Z:/DeyPlay/Neuron_24h_1stInj_Full.mat','a')
    img3 = h5py.File('Z:/DeyPlay/Neuron_48h_1stInj_Full.mat','a')
    img4 = h5py.File('Z:/DeyPlay/Neuron_Control_1stInj_Full.mat','a')
    e = 'EX745_845'
    f = 'Cell_M'
    im1 = trans(np.array(img1.get(e)))
    im2 = trans(np.array(img2.get(e)))     
    im3 = trans(np.array(img3.get(e)))
    im4 = trans(np.array(img4.get(e)))

    imk1 = trans(np.array(img1.get(f)))
    imk2 = trans(np.array(img2.get(f)))     
    imk3 = trans(np.array(img3.get(f)))
    imk4 = trans(np.array(img4.get(f)))

    if var == 'tri_injury_8':
        cat_images = im1
        cat_masks = imk1
    elif var == 'tri_injury_24':
        cat_images = im2
        cat_masks = imk2
    elif var == 'tri_injury_ctrl':
        cat_images = im4
        cat_masks = imk4
    elif var == 'tri_injury_48':
        cat_images = im3
        cat_masks = imk3
    return cat_images, cat_masks 


def tri_tbi_injury(var):
    img1 = h5py.File('Z:/DeyPlay/Microglia_8h_1stInj_Full.mat','a')
    img2 = h5py.File('Z:/DeyPlay/Microglia_24h_1stInj_Full.mat','a')
    img3 = h5py.File('Z:/DeyPlay/Microglia_Control_1stInj_Full.mat','a')
    img4 = h5py.File('Z:/DeyPlay/Microglia_48h_1stInj_Full.mat','a')
    e = 'EX745_845'
    f = 'Cell_M'
    im1 = trans(np.array(img1.get(e)))
    im2 = trans(np.array(img2.get(e)))     
    im3 = trans(np.array(img3.get(e)))
    im4 = trans(np.array(img4.get(e)))
    
    imk1 = trans(np.array(img1.get(f)))
    imk2 = trans(np.array(img2.get(f)))     
    imk3 = trans(np.array(img3.get(f)))
    imk4 = trans(np.array(img4.get(f)))
    
    if var == 'tri_injury_8':
        cat_images = im1
        cat_masks = imk1
    elif var == 'tri_injury_24':
        cat_images = im2
        cat_masks = imk2
    elif var == 'tri_injury_ctrl':
        cat_images = im3 
        cat_masks = imk3
    elif var == 'tri_injury_48':
        cat_images = im4 
        cat_masks = imk4
    return cat_images, cat_masks 

def tri_tbi_tri(var):
    img1 = h5py.File('Z:/DeyPlay/Tricultures_8h_1stInj_Full.mat','a')
    img2 = h5py.File('Z:/DeyPlay/Tricultures_24h_1stInj_Full.mat','a')
    img3 = h5py.File('Z:/DeyPlay/Tricultures_Control_1stInj_Full.mat','a')
    img4 = h5py.File('Z:/DeyPlay/Tricultures_48h_1stInj_Full.mat','a')
    e = 'EX745_845'
    f = 'Cell_M'
    im1 = trans(np.array(img1.get(e)))
    im2 = trans(np.array(img2.get(e)))     
    im3 = trans(np.array(img3.get(e)))
    im4 = trans(np.array(img4.get(e)))
    
    imk1 = trans(np.array(img1.get(f)))
    imk2 = trans(np.array(img2.get(f)))     
    imk3 = trans(np.array(img3.get(f)))
    imk4 = trans(np.array(img4.get(f)))
    
    if var == 'tri_injury_8':
        cat_images = im1
        cat_masks = imk1
    elif var == 'tri_injury_24':
        cat_images = im2
        cat_masks = imk2
    elif var == 'tri_injury_ctrl':
        cat_images = im3 
        cat_masks = imk3
    elif var == 'tri_injury_48':
        cat_images = im4 
        cat_masks = imk4
    return cat_images, cat_masks 


def tri_tbi_ast(var):
    img1 = h5py.File('Z:/DeyPlay/Astrocytes_8h_1stInj_Full.mat','a')
    img2 = h5py.File('Z:/DeyPlay/Astrocytes_24h_1stInj_Full.mat','a')
    img3 = h5py.File('Z:/DeyPlay/Astrocytes_Control_1stInj_Full.mat','a')
    img4 = h5py.File('Z:/DeyPlay/Astrocytes_48h_1stInj_Full.mat','a')
    e = 'EX745_845'
    f = 'Cell_M'
    im1 = trans(np.array(img1.get(e)))
    im2 = trans(np.array(img2.get(e)))     
    im3 = trans(np.array(img3.get(e)))
    im4 = trans(np.array(img4.get(e)))
    
    imk1 = trans(np.array(img1.get(f)))
    imk2 = trans(np.array(img2.get(f)))     
    imk3 = trans(np.array(img3.get(f)))
    imk4 = trans(np.array(img4.get(f)))
    
    if var == 'tri_injury_8':
        cat_images = im1
        cat_masks = imk1
    elif var == 'tri_injury_24':
        cat_images = im2
        cat_masks = imk2
    elif var == 'tri_injury_ctrl':
        cat_images = im3 
        cat_masks = imk3
    elif var == 'tri_injury_48':
        cat_images = im4 
        cat_masks = imk4
    return cat_images, cat_masks 


def mono_generatory_mar_16(var):
    img1 = h5py.File('Z:/DeyPlay/W1_Tensor_Mar16.mat','a')
    img2 = h5py.File('Z:/DeyPlay/W2_Tensor_Mar16.mat','a')
    img3 = h5py.File('Z:/DeyPlay/W3_Tensor_Mar16.mat','a')
    img5 = h5py.File('Z:/DeyPlay/W5_Tensor_Mar16.mat','a')
    img7 = h5py.File('Z:/DeyPlay/W7_Tensor_Mar16.mat','a')
    img8 = h5py.File('Z:/DeyPlay/W8_Tensor_Mar16.mat','a')
    img9 = h5py.File('Z:/DeyPlay/W9_Tensor_Mar16.mat','a')
    img11 = h5py.File('Z:/DeyPlay/W11_Tensor_Mar16.mat','a')
    img14 = h5py.File('Z:/DeyPlay/W14_Tensor_Mar16.mat','a')
    e = 'EX745_845'
    f = 'Shg_Neu_Silk'
    im1 = trans(np.array(img1.get(e)))
    mk1 = trans(np.array(img1.get(f)))
    im2 = trans(np.array(img2.get(e)))
    mk2 = trans(np.array(img2.get(f)))      
    im3 = trans(np.array(img3.get(e)))
    mk3 = trans(np.array(img3.get(f)))
    im5 = trans(np.array(img5.get(e)))
    mk5 = trans(np.array(img5.get(f)))
    im7 = trans(np.array(img7.get(e)))
    mk7 = trans(np.array(img7.get(f)))
    im8 = trans(np.array(img8.get(e)))
    mk8 = trans(np.array(img8.get(f)))
    im9 = trans(np.array(img9.get(e)))
    mk9 = trans(np.array(img9.get(f)))
    im11 = trans(np.array(img11.get(e)))
    mk11 = trans(np.array(img11.get(f)))
    im14 = trans(np.array(img14.get(e)))
    mk14 = trans(np.array(img14.get(f)))
    if var == 'mono_3_5_7_8_9_11_14':
        cat_images = np.concatenate([im3,im5,im7,im8,im9,im11,im14],axis=0)
        cat_nams = np.concatenate([mk3,mk5,mk7,mk8,mk9,mk11,mk14],axis=0)  
        return cat_images, cat_nams    
    elif var =='mono_1_2':
        cat_images = np.concatenate([im1,im2],axis=0)
        cat_nams = np.concatenate([mk1,mk2],axis=0) 
        return cat_images, cat_nams 
    elif var == 'mono_1_2_3_5_7_8_9_11_14':
        cat_images = np.concatenate([im1,im2,im3,im5,im7,im8,im9,im11,im14],axis=0)
        cat_nams = np.concatenate([mk1,mk2,mk3,mk5,mk7,mk8,mk9,mk11,mk14],axis=0)  
        return cat_images, cat_nams  
        
def nams_generator_mar_16(var):
    img1 = h5py.File('Z:/DeyPlay/W1_NAMS_Tensor_Mar16.mat','a')
    img3 = h5py.File('Z:/DeyPlay/W3_NAMS_Tensor_Mar16.mat','a')
    img4 = h5py.File('Z:/DeyPlay/W4_NAMS_Tensor_Mar16.mat','a')
    img5 = h5py.File('Z:/DeyPlay/W5_NAMS_Tensor_Mar16.mat','a')
    img6 = h5py.File('Z:/DeyPlay/W6_NAMS_Tensor_Mar16.mat','a')
    img7 = h5py.File('Z:/DeyPlay/W7_NAMS_Tensor_Mar16.mat','a')
    img8 = h5py.File('Z:/DeyPlay/W8_NAMS_Tensor_Mar16.mat','a')
    img9 = h5py.File('Z:/DeyPlay/W9_NAMS_Tensor_Mar16.mat','a')
    img11 = h5py.File('Z:/DeyPlay/W11_NAMS_Tensor_Mar16.mat','a')
    e = 'EX745_845' 
    f = 'nam_mask' 
    im1 = trans(np.array(img1.get(e)))
    mk1 = trans(np.array(img1.get(f)))
    im3 = trans(np.array(img3.get(e)))
    mk3 = trans(np.array(img3.get(f)))
    im4 = trans(np.array(img4.get(e)))
    mk4 = trans(np.array(img4.get(f)))
    im5 = trans(np.array(img5.get(e)))
    mk5 = trans(np.array(img5.get(f))) 
    im6 = trans(np.array(img6.get(e)))
    mk6 = trans(np.array(img6.get(f)))
    im7 = trans(np.array(img7.get(e)))  
    mk7 = trans(np.array(img7.get(f)))  
    im8 = trans(np.array(img8.get(e)))
    mk8 = trans(np.array(img8.get(f)))
    im9 = trans(np.array(img9.get(e)))
    mk9 = trans(np.array(img9.get(f)))
    im11 = trans(np.array(img11.get(e)))
    mk11 = trans(np.array(img11.get(f)))
    
    if var == 'tri_3_5_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im3,im5,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk3,mk5,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_1_3_5_6_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im1,im3,im5,im6,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk1,mk3,mk5,mk6,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_1_3_5_7_8_9_11':
        cat_images = np.nan_to_num(np.concatenate([im1,im3,im5,im7,im8,im9,im11],axis=0),nan=0.0)
        cat_nams = np.concatenate([mk1,mk3,mk5,mk7,mk8,mk9,mk11],axis=0) 
    elif var == 'tri_6':
        cat_images = np.nan_to_num(im6,nan=0.0)
        cat_nams = mk6
          
    return cat_images, cat_nams

def match_indices(n_slices,steps):
    groups = []
    for i in range(steps):
        group = tuple(j + i for j in range(0, n_slices, steps))
        groups.append(group)
    return groups
        
def uneven_matching_slice(A,n_slices,steps):
    groups = match_indices(n_slices,steps)
    group_slice = A[groups].transpose(0, 2, 3, 1)
    return group_slice


def nams_generator_14():
    img14 = h5py.File('Week14_NAM_Tensor.mat','a')
    e = 'EX745_845' 
    f = 'nam_mask' 
    cat_images = trans(np.array(img14.get(e)))
    cat_nams = trans(np.array(img14.get(f)))
    a = cat_images[0:24,:,:]
    b = cat_nams[0:6,:,:]
    c = cat_images[24:64,:,:]
    d = cat_nams[6:16,:,:]
    g = cat_images[64:100,:,:]
    h = cat_nams[16:25,:,:]
    
    return uneven_matching_slice(a,a.shape[0],b.shape[0]),b,uneven_matching_slice(c,c.shape[0],d.shape[0]),d,uneven_matching_slice(g,g.shape[0],h.shape[0]),h


def cnam_generator(var):
    if var == 'tri_3_5_7_8_9_11':
        img3 = h5py.File('Week3_MANS_Tensor.mat','a')
        img5 = h5py.File('Week5_MANS_Tensor.mat','a')
        img7 = h5py.File('Week7_MANS_Tensor.mat','a')
        img8 = h5py.File('Week8_MANS_Tensor.mat','a')
        img9 = h5py.File('Week9_MANS_Tensor.mat','a')
        img11 = h5py.File('Week11_MANS_Tensor.mat','a')
        e = 'EX745_845' 
        f = 'nam_mask' 
        im3 = trans(np.array(img3.get(e)))
        mk3 = trans(np.array(img3.get(f)))
        im5 = trans(np.array(img5.get(e)))
        mk5 = trans(np.array(img5.get(f))) 
        im7 = trans(np.array(img7.get(e)))  
        mk7 = trans(np.array(img7.get(f)))  
        im8 = trans(np.array(img8.get(e)))
        mk8 = trans(np.array(img8.get(f)))
        im9 = trans(np.array(img9.get(e)))
        mk9 = trans(np.array(img9.get(f)))
        im11 = trans(np.array(img11.get(e)))
        mk11 = trans(np.array(img11.get(f)))
        cat_images = np.concatenate([im3,im5,im7,im8,im9,im11],axis=0)
        cat_nams = np.concatenate([mk3,mk5,mk7,mk8,mk9,mk11],axis=0) 
         
    elif var == 'tri_3_5_7_8_9':
        img3 = h5py.File('Week3_MANS_Tensor.mat','a')
        img5 = h5py.File('Week5_MANS_Tensor.mat','a')
        img7 = h5py.File('Week7_MANS_Tensor.mat','a')
        img8 = h5py.File('Week8_MANS_Tensor.mat','a')
        img9 = h5py.File('Week9_MANS_Tensor.mat','a')
        e = 'EX745_845' 
        f = 'nam_mask' 
        im3 = trans(np.array(img3.get(e)))
        mk3 = trans(np.array(img3.get(f)))
        im5 = trans(np.array(img5.get(e)))
        mk5 = trans(np.array(img5.get(f))) 
        im7 = trans(np.array(img7.get(e)))  
        mk7 = trans(np.array(img7.get(f)))  
        im8 = trans(np.array(img8.get(e)))
        mk8 = trans(np.array(img8.get(f)))
        im9 = trans(np.array(img9.get(e)))
        mk9 = trans(np.array(img9.get(f)))
        cat_images = np.concatenate([im3,im5,im7,im8,im9],axis=0)
        cat_nams = np.concatenate([mk3,mk5,mk7,mk8,mk9],axis=0) 
        
    elif var == 'tri_3_4_5_6_7_8_9':
        img3 = h5py.File('Week3_MANS_Tensor.mat','a')
        img4 = h5py.File('Week4_MANS_Tensor.mat','a')
        img5 = h5py.File('Week5_MANS_Tensor.mat','a')
        img6 = h5py.File('Week6_MANS_Tensor.mat','a')
        img7 = h5py.File('Week7_MANS_Tensor.mat','a')
        img8 = h5py.File('Week8_MANS_Tensor.mat','a')
        img9 = h5py.File('Week9_MANS_Tensor.mat','a')
        e = 'EX745_845' 
        f = 'nam_mask' 
        im3 = trans(np.array(img3.get(e)))
        mk3 = trans(np.array(img3.get(f)))
        im4 = trans(np.array(img4.get(e)))
        mk4 = trans(np.array(img4.get(f))) 
        im5 = trans(np.array(img5.get(e)))
        mk5 = trans(np.array(img5.get(f))) 
        im6 = trans(np.array(img6.get(e)))
        mk6 = trans(np.array(img6.get(f))) 
        im7 = trans(np.array(img7.get(e)))  
        mk7 = trans(np.array(img7.get(f)))  
        im8 = trans(np.array(img8.get(e)))
        mk8 = trans(np.array(img8.get(f)))
        im9 = trans(np.array(img9.get(e)))
        mk9 = trans(np.array(img9.get(f)))
        cat_images = np.concatenate([im3,im4,im5,im6,im7,im8,im9],axis=0)
        cat_nams = np.concatenate([mk3,mk4,mk5,mk6,mk7,mk8,mk9],axis=0) 
           
    elif var == 'tri_11':
        img11 = h5py.File('Week11_MANS_Tensor.mat','a')
        e = 'EX745_845' 
        f = 'nam_mask' 
        cat_images = trans(np.array(img11.get(e)))
        cat_nams = trans(np.array(img11.get(f)))
        
    elif var == 'tri_4_6':
        img4 = h5py.File('Week4_MANS_Tensor.mat','a')
        img6 = h5py.File('Week6_MANS_Tensor.mat','a')
        e = 'EX745_845' 
        f = 'nam_mask' 
        im4 = trans(np.array(img4.get(e)))
        mk4 = trans(np.array(img4.get(f)))
        im6 = trans(np.array(img6.get(e)))
        mk6 = trans(np.array(img6.get(f)))
        cat_images = np.concatenate([im4,im6],axis=0)
        cat_nams = np.concatenate([mk4,mk6],axis=0) 
          
    return cat_images, cat_nams
        

def get_model(n_classes,h,w,c,var):
    if var == 'basic':
        return Multi_Unet(n_classes,h,w,c)
    
    
def twist_tensor(A):
    B = np.concatenate([A[i:i+12, :, :][:, :, :, np.newaxis] for i in range(0, A.shape[0], 12)], axis=-1)
    B = np.transpose(B, (3, 1, 2, 0))
    return B
