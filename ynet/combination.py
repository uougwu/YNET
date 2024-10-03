# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 04:04:03 2023
@author: uugwu01
"""
import numpy as np
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import random
from tensorly.decomposition import tucker
import tensorly as tl
from skimage.filters import threshold_multiotsu
from skimage import morphology, measure
from keras.metrics import MeanIoU
from scipy.ndimage import binary_dilation, binary_closing
from skimage.morphology import remove_small_objects
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.models import load_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from generate_data import mono_tbi_injury,tri_tbi_ast,tri_tbi_injury,nams_generator,tri_tbi_tri,mono_generatory
from tabulate import tabulate
import pandas as pd
#from smooth_tiled_predictions import predict_img_with_smooth_windowing
from scipy import ndimage
from skimage.morphology import dilation, disk
from generate_data import coculture_neu, coculture_na, coculture_nm, coculture_tri
import matplotlib.pyplot as plt
from generate_data import mono_generatory_mar_16
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import napari
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel
import sys
from scipy.io import savemat

def July_Alzer_ctrl(var,outpath):
    
    e = 'EX745_845'
    f = 'index'

    infected = h5py.File(var, 'a')
    data = trans(np.array(infected.get(e)))
    indices = [x * 4 for x in np.array(infected.get(f)).astype(int).tolist()[0]]
    filenames = [f.replace('.mat', '') for f in os.listdir(outpath) if f.startswith('Control')]

    print(filenames)
    print(indices)

    extracted_images = []
    image_array = data.copy()
    start_index = 0

    for idx, index in enumerate(indices):
        z = image_array[start_index:start_index + index, :, :]
        extracted = np.stack([z[i:i + (index // 4), :, :] for i in range(0, z.shape[0], index // 4)], axis=-1)
        image = minmax(extracted)
        model = load_model('Z:/DeyPlay/mono_2024c_enameta_pretrained_neu_alzh_v5.hdf5', compile=False)
        model_nbs = load_model('Z:/DeyPlay/mono_2024c_enameta_pretrained_neu_alzh_v4.hdf5', compile=False)
        ypred = ((model_nbs.predict(image.clip(0.001, 0.15)) + 
                  model.predict(image.clip(0.001, 0.3)))[:, :, :, 1] > 0.5).transpose(1, 2, 0) #0.5 default
        para = filenames[idx]  
        file_path = f'{outpath}/ResultsUpd_Intensity_1n2/Predict_{para}.mat'
        savemat(file_path, {'cell_mask': ypred}, do_compression=True)
        extracted_images.append(image)
        start_index += index

    return np.concatenate(extracted_images, axis=0)

def generate_filenames(base_var, length, suffix):
    base_name = base_var.split(suffix)[0]  
    filenames = [f"{base_name}_s{str(i).zfill(2)}{suffix}" for i in range(length)]
    return filenames

def July_Alzer(var,outpath,Day):
    
    e = 'EX745_845'
    f = 'index'

    infected = h5py.File(var, 'a')
    data = trans(np.array(infected.get(e)))
    indices = [x * 4 for x in np.array(infected.get(f)).astype(int).tolist()[0]]

    base_var = var.replace('.mat', '')
    base_var = os.path.basename(base_var)
    suffix = base_var.split(Day)[1]
    suffix = f"_{suffix}" 

    length = len(indices)
    filenames = generate_filenames(base_var, length, suffix)

    extracted_images = []
    image_array = data.copy()
    start_index = 0

    for idx, index in enumerate(indices):
        z = image_array[start_index:start_index + index, :, :]
        extracted = np.stack([z[i:i + (index // 4), :, :] for i in range(0, z.shape[0], index // 4)], axis=-1)
        image = minmax(extracted)
        
        model = load_model('Z:/DeyPlay/mono_2024c_enameta_pretrained_neu_alzh_v5.hdf5', compile=False)
        model_nbs = load_model('Z:/DeyPlay/mono_2024c_enameta_pretrained_neu_alzh_v4.hdf5', compile=False)
        
        ypred = ((model_nbs.predict(image.clip(0.001, 0.15)) + 
                  model.predict(image.clip(0.001, 0.1)))[:, :, :, 1] > 0.5).transpose(1, 2, 0)
        
        para = filenames[idx]  
        file_path = f'{outpath}/Predict_{para}.mat'
        savemat(file_path, {'cell_mask': ypred}, do_compression=True)
        
        extracted_images.append(image)
        start_index += index

    return np.concatenate(extracted_images, axis=0)


def open_and_correct_images_silk(images, stack_of_masks, save_directory, default_filename):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    corrected_masks = list(stack_of_masks)
    deleted_indices = []  
    current_index = [0]  

    def save_current_and_next():
        corrected_masks[current_index[0]] = viewer.layers['Mask'].data
        if current_index[0] < len(stack_of_masks) - 1:
            current_index[0] += 1
            while current_index[0] in deleted_indices and current_index[0] < len(stack_of_masks) - 1:
                current_index[0] += 1
            viewer.layers['Image'].data = images[current_index[0]]
            viewer.layers['Mask'].data = stack_of_masks[current_index[0]]
        else:
            corrected_masks[current_index[0]] = viewer.layers['Mask'].data
            retained_masks = [corrected_masks[i] for i in range(len(corrected_masks)) if i not in deleted_indices]
            corrected_stack = np.stack(retained_masks, axis=0)
            
            custom_filename = filename_input.text() or default_filename
            
            save_path = os.path.join(save_directory, custom_filename)
            ensure_dir_exists(os.path.dirname(save_path))
            file_path = f'{save_path}.mat'
            if corrected_stack.shape == (1,512,512):
                savemat(file_path, {'cell_mask': to_categorical(corrected_stack[0,:,:],3)[:,:,1],'silk_mask': to_categorical(corrected_stack[0,:,:],3)[:,:,2]}, do_compression=True)
            else:
                savemat(file_path, {'cell_mask': trans(to_categorical(corrected_stack,3)[:,:,:,1].transpose(2,1,0)),'silk_mask': trans(to_categorical(corrected_stack,3)[:,:,:,2].transpose(2,1,0))}, do_compression=True)
            print(f"Corrected masks saved to {save_path}")
            viewer.close()

    def go_back():
        if current_index[0] > 0:
            current_index[0] -= 1
            while current_index[0] in deleted_indices and current_index[0] > 0:
                current_index[0] -= 1
            viewer.layers['Image'].data = images[current_index[0]]
            viewer.layers['Mask'].data = stack_of_masks[current_index[0]]
        else:
            print("You are at the first image. Cannot go back further.")

    def delete_current():
        if current_index[0] not in deleted_indices:
            deleted_indices.append(current_index[0])
            print(f"Deleted image and mask at index {current_index[0]}")
        save_current_and_next()  

    viewer = napari.Viewer()

    # Load the image and mask
    viewer.add_image(images[0], name='Image')
    viewer.add_labels(stack_of_masks[0], name='Mask')
    
    save_button = QPushButton('Save and Next')
    save_button.clicked.connect(save_current_and_next)

    back_button = QPushButton('Back')
    back_button.clicked.connect(go_back)

    delete_button = QPushButton('Delete')
    delete_button.clicked.connect(delete_current)

    filename_input = QLineEdit()
    filename_input.setPlaceholderText("Enter custom filename (e.g., Control123_Day3_noTreat)")
    filename_input.setText(default_filename)

    save_widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QLabel("Custom Filename:"))
    layout.addWidget(filename_input)
    layout.addWidget(save_button)
    layout.addWidget(back_button)
    layout.addWidget(delete_button)
    save_widget.setLayout(layout)
    viewer.window.add_dock_widget(save_widget, area='right')

    napari.run()

def quality_assurance(var):
    e = 'EX745_845'
    g = 'cell_mask'
    infected = h5py.File(var, 'a')
    data = trans(np.array(infected.get(e)))
    image = np.stack([data[i:i + (data.shape[0] // 4), :, :] for i in range(0, data.shape[0], data.shape[0] // 4)], axis=-1)
    mask = np.array(infected.get(g))
    if mask.shape==(512,512):
        masks = mask
    else:
        masks = trans(mask)
    return image,masks

def quality_assurance_duo(var):
    e = 'EX745_845'
    infected = h5py.File(var, 'a')
    data = trans(np.array(infected.get(e)))
    extracted = np.stack([data[i:i + (data.shape[0] // 4), :, :] for i in range(0, data.shape[0], data.shape[0] // 4)], axis=-1)
    image = minmax(extracted)
    model = load_model('Z:/DeyPlay/mono_2024c_enameta_pretrained_neu_alzh_v5.hdf5', compile=False)
    model_nbs = load_model('Z:/DeyPlay/mono_2024c_enameta_pretrained_neu_alzh_v4.hdf5', compile=False)
    mask = ((model_nbs.predict(image.clip(0.0001, 0.15)) + 
                model.predict(image.clip(0.0001, 0.1)))[:, :, :, 1] > 0.5)#transpose(2, 1, 0)
    return image,mask


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def open_and_correct_images(images, stack_of_masks, save_directory, default_filename):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    corrected_masks = list(stack_of_masks)
    deleted_indices = []  
    current_index = [0]  

    def save_current_and_next():
        corrected_masks[current_index[0]] = viewer.layers['Mask'].data
        if current_index[0] < len(stack_of_masks) - 1:
            current_index[0] += 1
            while current_index[0] in deleted_indices and current_index[0] < len(stack_of_masks) - 1:
                current_index[0] += 1
            viewer.layers['Image'].data = images[current_index[0]]
            viewer.layers['Mask'].data = stack_of_masks[current_index[0]]
        else:
            corrected_masks[current_index[0]] = viewer.layers['Mask'].data
            retained_masks = [corrected_masks[i] for i in range(len(corrected_masks)) if i not in deleted_indices]
            corrected_stack = np.stack(retained_masks, axis=0)
            
            custom_filename = filename_input.text() or default_filename
            
            save_path = os.path.join(save_directory, custom_filename)
            ensure_dir_exists(os.path.dirname(save_path))
            file_path = f'{save_path}.mat'
            if corrected_stack.shape == (1,512,512):
                savemat(file_path, {'cell_mask': corrected_stack[0,:,:]}, do_compression=True)
            else:
                savemat(file_path, {'cell_mask': corrected_stack.transpose(2, 1, 0)}, do_compression=True)

            print(corrected_stack.shape)
            print(f"Corrected masks saved to {save_path}")
            viewer.close()

    def go_back():
        if current_index[0] > 0:
            current_index[0] -= 1
            while current_index[0] in deleted_indices and current_index[0] > 0:
                current_index[0] -= 1
            viewer.layers['Image'].data = images[current_index[0]]
            viewer.layers['Mask'].data = stack_of_masks[current_index[0]]
        else:
            print("You are at the first image. Cannot go back further.")

    def delete_current():
        if current_index[0] not in deleted_indices:
            deleted_indices.append(current_index[0])
            print(f"Deleted image and mask at index {current_index[0]}")
        save_current_and_next()  

    viewer = napari.Viewer()

    # Load the image and mask
    viewer.add_image(images[0], name='Image')
    viewer.add_labels(stack_of_masks[0], name='Mask')
    
    save_button = QPushButton('Save and Next')
    save_button.clicked.connect(save_current_and_next)

    back_button = QPushButton('Back')
    back_button.clicked.connect(go_back)

    delete_button = QPushButton('Delete')
    delete_button.clicked.connect(delete_current)

    filename_input = QLineEdit()
    filename_input.setPlaceholderText("Enter custom filename (e.g., Control123_Day3_noTreat)")
    filename_input.setText(default_filename)

    save_widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QLabel("Custom Filename:"))
    layout.addWidget(filename_input)
    layout.addWidget(save_button)
    layout.addWidget(back_button)
    layout.addWidget(delete_button)
    save_widget.setLayout(layout)
    viewer.window.add_dock_widget(save_widget, area='right')

    napari.run()



def trans(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:,:] = x[i,:,:].T
    return y

def shapy(Xtrr):
    return Xtrr.reshape(Xtrr.shape[0]*Xtrr.shape[1],Xtrr.shape[2],Xtrr.shape[3],Xtrr.shape[4])

def patchete(A,patch_size,channel):
    G = []
    for i in range(A.shape[0]):  
        for j in range(A.shape[1]):
            patches = patchify(A[i,j,:,:,:],(patch_size,patch_size,channel),step=patch_size)
            G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5])))
    return np.concatenate(G,axis=0)

# def unpatchete(A):
#     groups = []
#     num_groups = A.shape[0] // 12
#     for i in range(num_groups):  
#         groupy = A[i*12:(i + 1)*12,:,:,:]
#         G = np.reshape(groupy,(3,4,256,256,A.shape[3]))
#         group = []
#         for j in range(3): 
#             patches = unpatchify(np.reshape(G[j,:,:,:,:],(2,2,1,256,256,A.shape[3])),(512, 512, A.shape[3]))
#             group.append(patches)
#         groups.append(np.stack(group,axis=0))
#     return shapy(np.stack(groups,axis=0))



def plot_base_tri(Xtes, y,y_pred, nu, var, custom_name):
    maska = np.sum(Xtes.clip(0.0001, 0.1), axis=3)
    button_next = widgets.Button(description="Next")
    button_back = widgets.Button(description="Back")
    button_delete = widgets.Button(description="Delete")
    button_save = widgets.Button(description="Save")
    output = widgets.Output()
    display(widgets.VBox([widgets.HBox([button_back, button_next, button_delete, button_save])]), output)
    
    deleted_indices = []  

    def update_plot(i):
        with output:
            clear_output(wait=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(34, 8))
            fig.suptitle(var, fontsize=20)

            ax = axes[0]
            im = ax.imshow(maska[i, :, :])
            ax.set_title("Sum of Optical Section", fontsize=15)
            ax.axis("on")
            ax.grid(True)

            ax = axes[1]
            im = ax.imshow(y[i, :, :])
            ax.set_title("Ground Truth", fontsize=15)
            ax.axis("on")
            ax.grid(True)

            ax = axes[2]
            ax.imshow(y_pred[i, :, :] > nu)
            ax.set_title(f"Predicted Predicted Neurons_{i}", fontsize=15)
            ax.axis("on")
            ax.grid(True)

            plt.show()

    def on_button_next_click(b):
        on_button_next_click.i += 1
        if on_button_next_click.i >= Xtes.shape[0]:
            on_button_next_click.i = Xtes.shape[0] - 1
        while on_button_next_click.i in deleted_indices and on_button_next_click.i < Xtes.shape[0] - 1:
            on_button_next_click.i += 1
        update_plot(on_button_next_click.i)

    def on_button_back_click(b):
        on_button_next_click.i -= 1
        if on_button_next_click.i < 0:
            on_button_next_click.i = 0
        while on_button_next_click.i in deleted_indices and on_button_next_click.i > 0:
            on_button_next_click.i -= 1
        update_plot(on_button_next_click.i)

    def on_button_delete_click(b):
        if on_button_next_click.i not in deleted_indices:
            deleted_indices.append(on_button_next_click.i)
            print(f"Deleted image {on_button_next_click.i}")
        on_button_next_click(b)  

    def on_button_save_click(b):
        retained_indices = [i for i in range(Xtes.shape[0]) if i not in deleted_indices]
        Xtes_retained = Xtes[retained_indices]
        print(Xtes_retained.shape)
        with h5py.File(f'Z:/DeyPlay/train_alzh/{custom_name}_images.h5', 'w') as hf:
            hf.create_dataset('images', data=Xtes_retained, compression='gzip')
        
    
    on_button_next_click.i = 0  
    
    button_next.on_click(on_button_next_click)
    button_back.on_click(on_button_back_click)
    button_delete.on_click(on_button_delete_click)
    button_save.on_click(on_button_save_click)
    
    update_plot(0)


def plot_base_duo(Xtes, y, nu, var, custom_name):
    maska = np.sum(Xtes.clip(0.0001, 0.1), axis=3)
    button_next = widgets.Button(description="Next")
    button_back = widgets.Button(description="Back")
    button_delete = widgets.Button(description="Delete")
    button_save = widgets.Button(description="Save")
    output = widgets.Output()
    display(widgets.VBox([widgets.HBox([button_back, button_next, button_delete, button_save])]), output)
    
    deleted_indices = []  

    def update_plot(i):
        with output:
            clear_output(wait=True)
            
            fig, axes = plt.subplots(1, 2, figsize=(34, 8))
            fig.suptitle(var, fontsize=20)

            ax = axes[0]
            im = ax.imshow(maska[i, :, :])
            ax.set_title("Sum of Optical Section", fontsize=15)
            ax.axis("on")
            ax.grid(True)

            ax = axes[1]
            ax.imshow(y[i, :, :] > nu)
            ax.set_title(f"Predicted Predicted Neurons (OS_{i})", fontsize=15)
            ax.axis("on")
            ax.grid(True)

            plt.show()

    def on_button_next_click(b):
        on_button_next_click.i += 1
        if on_button_next_click.i >= Xtes.shape[0]:
            on_button_next_click.i = Xtes.shape[0] - 1
        while on_button_next_click.i in deleted_indices and on_button_next_click.i < Xtes.shape[0] - 1:
            on_button_next_click.i += 1
        update_plot(on_button_next_click.i)

    def on_button_back_click(b):
        on_button_next_click.i -= 1
        if on_button_next_click.i < 0:
            on_button_next_click.i = 0
        while on_button_next_click.i in deleted_indices and on_button_next_click.i > 0:
            on_button_next_click.i -= 1
        update_plot(on_button_next_click.i)

    def on_button_delete_click(b):
        if on_button_next_click.i not in deleted_indices:
            deleted_indices.append(on_button_next_click.i)
            print(f"Deleted image {on_button_next_click.i}")
        on_button_next_click(b)  

    def on_button_save_click(b):
        retained_indices = [i for i in range(Xtes.shape[0]) if i not in deleted_indices]
        Xtes_retained = Xtes[retained_indices]
        print(Xtes_retained.shape)
        with h5py.File(f'Z:/DeyPlay/train_alzh/{custom_name}_images.h5', 'w') as hf:
            hf.create_dataset('images', data=Xtes_retained, compression='gzip')
        
    
    on_button_next_click.i = 0  
    
    button_next.on_click(on_button_next_click)
    button_back.on_click(on_button_back_click)
    button_delete.on_click(on_button_delete_click)
    button_save.on_click(on_button_save_click)
    
    update_plot(0)



def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def open_and_correct_image(stack_of_masks, save_directory, default_filename):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    corrected_masks = list(stack_of_masks)
    deleted_indices = []  
    current_index = [0]  

    def save_current_and_next():
        corrected_masks[current_index[0]] = viewer.layers['Binary Image'].data
        if current_index[0] < len(stack_of_masks) - 1:
            current_index[0] += 1
            while current_index[0] in deleted_indices and current_index[0] < len(stack_of_masks) - 1:
                current_index[0] += 1
            viewer.layers['Binary Image'].data = stack_of_masks[current_index[0]]
        else:
            corrected_masks[current_index[0]] = viewer.layers['Binary Image'].data
            retained_masks = [corrected_masks[i] for i in range(len(corrected_masks)) if i not in deleted_indices]
            corrected_stack = np.stack(retained_masks, axis=0)
            
            custom_filename = filename_input.text() or default_filename
            
            save_path = os.path.join(save_directory, custom_filename)
            ensure_dir_exists(os.path.dirname(save_path))
            #np.save(save_path, corrected_stack)
            file_path = f'{save_path}.mat'
            savemat(file_path, {'cell_mask': corrected_stack.transpose(2,1,0)}, do_compression=True)
        
            print(f"Corrected masks saved to {save_path}")
            viewer.close()

    def go_back():
        if current_index[0] > 0:
            current_index[0] -= 1
            while current_index[0] in deleted_indices and current_index[0] > 0:
                current_index[0] -= 1
            viewer.layers['Binary Image'].data = stack_of_masks[current_index[0]]
        else:
            print("You are at the first image. Cannot go back further.")

    def delete_current():
        if current_index[0] not in deleted_indices:
            deleted_indices.append(current_index[0])
            print(f"Deleted image and mask at index {current_index[0]}")
        save_current_and_next()  

    viewer = napari.Viewer()
    viewer.add_labels(stack_of_masks[0], name='Binary Image')
    
    save_button = QPushButton('Save and Next')
    save_button.clicked.connect(save_current_and_next)

    back_button = QPushButton('Back')
    back_button.clicked.connect(go_back)

    delete_button = QPushButton('Delete')
    delete_button.clicked.connect(delete_current)

    filename_input = QLineEdit()
    filename_input.setPlaceholderText("Enter custom filename (e.g., Control123_Day3_noTreat.npy)")
    filename_input.setText(default_filename)

    save_widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QLabel("Custom Filename:"))
    layout.addWidget(filename_input)
    layout.addWidget(save_button)
    layout.addWidget(back_button)
    layout.addWidget(delete_button)
    save_widget.setLayout(layout)
    viewer.window.add_dock_widget(save_widget, area='right')

    napari.run()


def process_matfiles(var):
    e = 'EX745_845' 
    f  = 'index'
    
    infected = h5py.File(var,'a')   
    data = trans(np.array(infected.get(e)))
    
    indices = [x*4 for x in np.array(infected.get(f)).astype(int).tolist()[0]]
    extracted_images = []
    extracted_imagery = []
    image_array = data.copy()
    start_index = 0
    
    for index in indices:
        z = image_array[start_index:start_index+index, :, :]
        extracted = np.stack([z[i:i+(index//4), :, :] for i in range(0, z.shape[0], index//4)], axis=-1)
        normalized_img = minmax(extracted)
        extracted_imagery.append(normalized_img)
        extracted_images.append(patchete(np.expand_dims(normalized_img,axis=0),256,4))
        start_index += index
    original_shape = np.concatenate(extracted_imagery,axis=0).shape
    return np.concatenate(extracted_images,axis=0),original_shape



# def minmax(A):
#     return (A - A.min()) / (A.max() - A.min())


def unpatchete(patches, original_shape, patch_size):
    n, x, y, z = original_shape
    reconstructed_array = np.zeros(original_shape)
    patches_per_dim_0 = x // patch_size  
    patches_per_dim_1 = y // patch_size  

    patch_idx = 0
    for i in range(n):
        current_patches = patches[patch_idx:patch_idx + patches_per_dim_0 * patches_per_dim_1]
        patch_idx += patches_per_dim_0 * patches_per_dim_1
        reshaped_patches = current_patches.reshape(patches_per_dim_0, patches_per_dim_1, 1, patch_size, patch_size, z)
        reconstructed_array[i] = unpatchify(reshaped_patches, (x, y, z))
    return reconstructed_array



def make_initial_masks(image, image_shape, patch_size,reconstructed_image):
    model_base_tri = load_model('Z:/DeyPlay/triculture_2020b_v0_neu.hdf5',compile=False) 
    model_base_mono = load_model('Z:/DeyPlay/mono_2020b_enameta_mar16_pretrained_v2.hdf5',compile=False)  

    y_pred_train_mono = model_base_mono.predict(image.clip(0.001,0.3)) + model_base_mono.predict(image.clip(0.001,0.1)) 
    masky_silk_mono = y_pred_train_mono[:,:,:,3]>0.01
    y_pred_silk_mono = np.array([remove_small_objects(masky_silk_mono[i], min_size=1000) for i in range(masky_silk_mono.shape[0])])
    y_pred_train_mono[:,:,:,3] = y_pred_silk_mono

    y_pred_train_tri = model_base_tri.predict(reconstructed_image) + model_base_tri.predict(reconstructed_image.clip(0.001,0.1)) 
    masky_silk_tri = y_pred_train_tri[:,:,:,3]>0.01
    y_pred_silk_tri = np.array([remove_small_objects(masky_silk_tri[i], min_size=50) for i in range(masky_silk_tri.shape[0])])
    y_pred_train_tri[:,:,:,3] = y_pred_silk_tri

    ypred_neu = (y_pred_train_tri[:,:,:,2]>0.1) + (unpatchete(y_pred_train_mono,image_shape, patch_size)[:,:,:,2]>0.1)
    ypred_neu[ypred_neu==2]=1
    y_pred_train_tri[:,:,:,2] = ypred_neu

    ypred_silk = (y_pred_train_tri[:,:,:,3]>0.003) + (unpatchete(y_pred_train_mono,image_shape, patch_size)[:,:,:,3]>0.003)
    ypred_silk[ypred_silk==3]=1
    y_pred_train_tri[:,:,:,3] = np.array([remove_small_objects(ypred_silk[i], min_size=450) for i in range(ypred_silk.shape[0])])

    return (1 - (y_pred_train_tri[:, :, :, 3] > 0.01)) * (y_pred_train_tri[:, :, :, 2] > 0.1)


def loading_train_val_test_data(var):
    if var == 'train':
        train_folder = 'Z:/DeyPlay/train_alzh'
        X, y = load_images_and_masks(train_folder)
    elif var == 'val':
        validation_folder = 'Z:/DeyPlay/validation_alzh'
        X, y = load_images_and_masks(validation_folder)

    elif var == 'test':
        test_folder = 'Z:/DeyPlay/test_alzh'
        X, y = load_images_and_masks(test_folder)

    return X,y

def load_images_and_masks(folder_path):
    image_list = []
    mask_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_images.h5'):
            base_name = filename.replace('_images.h5', '')
            mask_filename = base_name + '_masker.npy'
            
            image_path = os.path.join(folder_path, filename)
            with h5py.File(image_path, 'r') as f:
                image_data = np.array(f['images'],dtype=np.float64)
                image_list.append(image_data)
            
            mask_path = os.path.join(folder_path, mask_filename)
            if os.path.exists(mask_path):
                mask_data = np.load(mask_path)
                mask_list.append(mask_data.astype(np.float32))
            else:
                print(f"Mask file not found for {base_name}")

    images_array = np.array(image_list, dtype=object)
    masks_array = np.array(mask_list, dtype=object)
  
    return np.concatenate(images_array,axis=0), to_categorical(np.expand_dims(np.concatenate(masks_array,axis=0),axis=3),2)

# def July_Alzer(n_classes):
#     e = 'EX755_860' 
#     f  = 'index'
    
#     infected = h5py.File('Alhz.mat','a')   
#     data = trans(np.array(infected.get(e)))
    
#     indices = [x*4 for x in np.array(infected.get(f)).astype(int).tolist()[0]]
#     extracted_images = []
#     image_array = data.copy()
#     start_index = 0
#     for index in indices:
#         z = image_array[start_index:start_index+index, :, :]
#         extracted = np.stack([z[i:i+(index//4), :, :] for i in range(0, z.shape[0], index//4)], axis=-1)
#         extracted_images.append(minmax(extracted))
#         start_index += index
         
#     return extracted_images

def clean_silk(y,min_size,classy):
    label_objects, _ = measure.label(y[:,:,:,classy], return_num=True, connectivity=1)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes >= min_size
    mask_sizes[0] = 0  # Background is always labeled as 0, don't remove it
    cleaned_mask = mask_sizes[label_objects]
    y[:,:,:,classy] = cleaned_mask
    return y

def mask_baseline(n_classes,patchsize,eta):
    data, mask = mono_generatory_mar_16('mono_3_5_7_8_9_11_14')
    indices = [x*12 for x in [9,9,7,7,9,8,9,9]]
    extracted_images = []
    extracta_images = []
    image_array = data.copy()
    mask_array = mask.copy()
    start_index = 0
    for index in indices:
        z = image_array[start_index:start_index+index, :, :]
        extracted = matching_slices(z,12,3)
        extracted_images.append(minmax(extracted))
        start_index += index
        
    start_ind = 0
    indies = [x*12 for x in [9,9,7,7,9,8,9,9]]
    for ind in indies:
        zz = mask_array[start_ind:start_ind+ind, :, :]
        extracta = matching_masks(zz,12,3)
        extracta_images.append(extracta)
        start_ind += ind

    X, y = np.concatenate(extracted_images,axis=0),np.concatenate(extracta_images,axis=0)
    X_train,y_train,X_val,y_val,X_test,y_test = Alzer(X,y)    
    Xa,ya,Xv,yv,Xt,yt = X_train,to_categorical(y_train,n_classes),X_val,to_categorical(y_val,n_classes),X_test,to_categorical(y_test,n_classes)
    Xtrain = patchete(shaper(np.delete(Xa,[15,16,17],axis=0),3),patchsize,4)
    ytrain = patchete(shaper(np.delete(ya,[15,16,17],axis=0),3),patchsize,4)
    Xval = patchete(shaper(Xv,eta),patchsize,4)
    yval = patchete(shaper(yv,eta),patchsize,4)
    Xtest = patchete(shaper(Xt,eta),patchsize,4)
    ytest = patchete(shaper(yt,eta),patchsize,4)
    return Xtrain,ytrain,Xval,yval,Xtest,ytest 


def plot_adjusted(H, M,n):
    for m in range(H.shape[0]):
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(np.sum(H, axis=3)[m, :, :])
        plt.title('Original Image')
        plt.grid(True)

        # Adjusted image
        plt.subplot(1, 3, 2)
        plt.imshow(np.sum(H, axis=3)[m, :, :].clip(0.01, 0.25))
        plt.title('Contrast Adjusted Image')
        plt.grid(True)

        # Mask
        plt.subplot(1, 3, 3)
        plt.imshow(M[m, :, :, n])
        plt.title('Mask')
        plt.grid(True)

        plt.tight_layout()  # Adjust layout
        plt.show()

def coculture(XX,yy,indey):
    extracted_images = []
    extracta_masks = []
    image_array = XX.copy()
    mask_array = yy.copy()

    start_index = 0
    indices = [4*x[1][0] for x in indey.astype(int)]
    for index in indices:
        z = image_array[start_index:start_index+index, :, :]
        extracted = np.stack([z[i:i+(index//4), :, :] for i in range(0, z.shape[0], index//4)], axis=-1)
        extracted_images.append(minmax(extracted))
        start_index += index
        
    start_ind = 0
    indies = [x[1][0] for x in indey.astype(int)]
    for ind in indies:
        zz = mask_array[start_ind:start_ind+ind, :, :].astype(int)
        extracta = np.stack([zz[i:i+(ind), :, :] for i in range(0, zz.shape[0], ind)], axis=-1)
        extracta_masks.append(extracta)
        start_ind += ind

    return extracted_images, extracta_masks

def patchetem(A,patch_size):
    G = []
    for i in range(A.shape[0]):  
        patches = patchify(A[i,:,:,:],(patch_size,patch_size,4),step=patch_size)
        G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5])))
    return np.concatenate(G,axis=0)

def cona(n_classes):
    id_im, id_mk, neu_id = coculture_na('id')
    im_id, mk_id = coculture(id_im, id_mk, neu_id)

    nid_im, nid_mk, neu_nid = coculture_na('nid')
    im_nid, mk_nid = coculture(nid_im, nid_mk, neu_nid)

    nind_im, nind_mk, neu_nind = coculture_na('nind')
    im_nind, mk_nind = coculture(nind_im, nind_mk, neu_nind)

    ind_im, ind_mk, neu_ind = coculture_na('ind')
    im_ind, mk_ind = coculture(ind_im, ind_mk, neu_ind)

    print(neu_id,neu_nid,neu_nind,neu_ind)

    extracted_images = [im_id,im_nid,im_nind,im_ind]
    extracta_masks = [mk_id,mk_nid,mk_nind,mk_ind]

    X_train,y_train,X_val,y_val,X_test,y_test = Alzer(extracted_images, extracta_masks)    
    return patchetem(X_train,256),patchetem(to_categorical(y_train,n_classes),256),patchetem(X_val,256),patchetem(to_categorical(y_val,n_classes),256),patchetem(X_test,256),patchetem(to_categorical(y_test,n_classes),256)


def cotri(n_classes):
    id_im, id_mk, neu_id = coculture_tri('id')
    im_id, mk_id = coculture(id_im, id_mk, neu_id)

    nid_im, nid_mk, neu_nid = coculture_tri('nid')
    im_nid, mk_nid = coculture(nid_im, nid_mk, neu_nid)

    nind_im, nind_mk, neu_nind = coculture_tri('nind')
    im_nind, mk_nind = coculture(nind_im, nind_mk, neu_nind)

    ind_im, ind_mk, neu_ind = coculture_tri('ind')
    im_ind, mk_ind = coculture(ind_im, ind_mk, neu_ind)

    extracted_images = [im_id,im_nid,im_nind,im_ind]
    extracta_masks = [mk_id,mk_nid,mk_nind,mk_ind]

    X_train,y_train,X_val,y_val,X_test,y_test = Alzer(extracted_images, extracta_masks)    
    return patchetem(X_train,256),patchetem(to_categorical(y_train,n_classes),256),patchetem(X_val,256),patchetem(to_categorical(y_val,n_classes),256),patchetem(X_test,256),patchetem(to_categorical(y_test,n_classes),256)


def conm(n_classes):
    id_im, id_mk, neu_id = coculture_nm('id')
    im_id, mk_id = coculture(id_im, id_mk, neu_id)

    nid_im, nid_mk, neu_nid = coculture_nm('nid')
    im_nid, mk_nid = coculture(nid_im, nid_mk, neu_nid)

    nind_im, nind_mk, neu_nind = coculture_nm('nind')
    im_nind, mk_nind = coculture(nind_im, nind_mk, neu_nind)

    ind_im, ind_mk, neu_ind = coculture_nm('ind')
    im_ind, mk_ind = coculture(ind_im, ind_mk, neu_ind)

    extracted_images = [im_id,im_nid,im_nind,im_ind]
    extracta_masks = [mk_id,mk_nid,mk_nind,mk_ind]

    X_train,y_train,X_val,y_val,X_test,y_test = Alzer(extracted_images, extracta_masks)    
    return patchetem(X_train,256),patchetem(to_categorical(y_train,n_classes),256),patchetem(X_val,256),patchetem(to_categorical(y_val,n_classes),256),patchetem(X_test,256),patchetem(to_categorical(y_test,n_classes),256)


def coneu(n_classes):
    id_im, id_mk, neu_id = coculture_neu('id')
    im_id, mk_id = coculture(id_im, id_mk, neu_id)

    nid_im, nid_mk, neu_nid = coculture_neu('nid')
    im_nid, mk_nid = coculture(nid_im, nid_mk, neu_nid)

    nind_im, nind_mk, neu_nind = coculture_neu('nind')
    im_nind, mk_nind = coculture(nind_im, nind_mk, neu_nind)

    ind_im, ind_mk, neu_ind = coculture_neu('ind')
    im_ind, mk_ind = coculture(ind_im, ind_mk, neu_ind)

    extracted_images = [im_id,im_nid,im_nind,im_ind]
    extracta_masks = [mk_id,mk_nid,mk_nind,mk_ind]

    X_train,y_train,X_val,y_val,X_test,y_test = Alzer(extracted_images, extracta_masks)    
    return patchetem(X_train,256),patchetem(to_categorical(y_train,n_classes),256),patchetem(X_val,256),patchetem(to_categorical(y_val,n_classes),256),patchetem(X_test,256),patchetem(to_categorical(y_test,n_classes),256)

def patchificasion(Xtrain,ytrain,patch_size,n_classes):
    image = Xtrain
    mask = np.argmax(ytrain, axis=3)
    patches = []
    masks = []
    num_samples = image.shape[0]
    for i in range(num_samples):
        for j in range(patch_size//2, 512 - patch_size//2):  
            for k in range(patch_size//2, 512 - patch_size//2):  
                #if mask[i, j, k] == 3 or mask[i, j, k] == 4:
                patch = image[i, j - patch_size//2: j + patch_size//2 + 1, k - patch_size//2: k + patch_size//2 + 1, :]
                patch_mask = mask[i, j - patch_size//2: j + patch_size//2 + 1, k - patch_size//2: k + patch_size//2 + 1]
                if (np.any(patch_mask == 3) and np.any(patch_mask == 2)) or (np.any(patch_mask == 4) and np.any(patch_mask == 2)):
                    patches.append(patch)
                    masks.append(patch_mask)
    patches = np.array(patches)
    masks = np.array(to_categorical(masks,n_classes))
    return patches, masks

def fatchificasion(Xtrain,ytrain,patch_size,n_classes):
    image = Xtrain
    mask = np.argmax(ytrain, axis=3)
    patches = []
    masks = []
    num_samples = image.shape[0]
    for i in range(num_samples):
        for j in range(patch_size//2, 512 - patch_size//2):  
            for k in range(patch_size//2, 512 - patch_size//2):  
                    patch = image[i, j - patch_size//2: j + patch_size//2 + 1, k - patch_size//2: k + patch_size//2 + 1, :]
                    patch_mask = mask[i, j - patch_size//2: j + patch_size//2 + 1, k - patch_size//2: k + patch_size//2 + 1]
                    patches.append(patch)
                    masks.append(patch_mask)
    patches = np.array(patches)
    masks = np.array(to_categorical(masks,n_classes))
    return patches, masks

def connected_accuracy(y_val,y_pred_val,rho,nu,var,title=None,**kwargs):
    insect = []
    truth = []
    predict = []

    
    for i in range(y_val.shape[0]):
        y_pred = y_pred_val[i, :, :, rho] > nu
        yval = y_val[i, :, :, rho]

        if var == 'undilate':
            true_labels, num_true_objects = ndimage.label(yval)
            predicted_labels, num_predicted_objects = ndimage.label(y_pred)
            overlap_mask = (true_labels > 0) & (predicted_labels > 0)
            overlap_labels, overlap_counts = np.unique(true_labels[overlap_mask], return_counts=True)
            overlap_labels = overlap_labels[0:]
            overlap_counts = overlap_counts[0:]
            num_overlapping_objects = len(overlap_labels)
            insect.append(num_overlapping_objects)
            truth.append(num_true_objects)
            predict.append(num_predicted_objects)
            
        elif var == 'dilate':
            threshold_area = kwargs['area']
            r = kwargs['radius']
            true_labels_pred, num_true_objects_pred = ndimage.label(y_pred)
            tiny_objects_mask = np.zeros_like(y_pred, dtype=bool)
            for label in range(1, num_true_objects_pred + 1):
                if np.sum(true_labels_pred == label) < threshold_area:
                    tiny_objects_mask |= true_labels_pred == label
            dilated_tiny_objects_mask = dilation(tiny_objects_mask, disk(r))  
            y_pred_enlarged = np.logical_or(y_pred, dilated_tiny_objects_mask)
            true_labels, num_true_objects = ndimage.label(yval)
            predicted_labels, num_predicted_objects = ndimage.label(y_pred_enlarged)
            overlap_mask = (true_labels > 0) & (predicted_labels > 0)
            overlap_labels, overlap_counts = np.unique(true_labels[overlap_mask], return_counts=True)
            overlap_labels = overlap_labels[0:]
            overlap_counts = overlap_counts[0:]
            num_overlapping_objects = len(overlap_labels)
            insect.append(num_overlapping_objects)
            truth.append(num_true_objects)
            predict.append(num_predicted_objects)

        # print("Number of objects in true mask:", num_true_objects)
        # print("Number of objects in predicted mask:", num_predicted_objects)
        # print("Number of overlapping objects:", num_overlapping_objects)
    
    recall  = np.sum(insect)/np.sum(truth)
    precision = np.sum(insect)/np.sum(predict)
    Fscore = 2*np.sum(insect)/(np.sum(predict)+np.sum(truth))
    

    data = {'Metric': ['Recall (Sensitivity/TPR)', 'Precision', 'F-Score'],
    'Value': [recall, precision, Fscore]}
    if title:
        spaces = int((37 - len(title)))
        print(' ' * spaces + title)
    df = pd.DataFrame(data)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center', showindex=False))

  

def predicting(Xval,model):
    X = []
    Y = unpatchete(Xval)
    for i in range(Y.shape[0]):
        Z = Y[i,:,:,:]
        predictions_smooth = predict_img_with_smooth_windowing(
            Z,
            window_size=256,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=4,
            pred_func=(
                lambda img_batch_subdiv: model.predict((img_batch_subdiv))))
        X.append(predictions_smooth)
    return np.stack(X,axis=0)

# def mono_neu_base(patch_size):
#     imagey, annon = mono_generatory('mono_3_5_7_8_9_11_14')
#     n_classes = 4
#     imagine = np.delete(matching_slices(imagey),[10,18, 26,37],axis=0)
#     maskine = np.delete(matching_masks(annon),[10,18, 26,37],axis=0)
#     #imagee, annone = mono_generatory('mono_1_2')
#     # image = patchete(np.concatenate([matching_slices(imagee),imagine]),256)
#     # mask = matchete(np.concatenate([matching_masks(annone),maskine]),256)

#     image = patchete(imagine,patch_size)
#     mask = matchete(maskine,patch_size)
#     images = image.reshape(image.shape[0]//12,12,image.shape[1],image.shape[2],4)
#     masks = mask.reshape(image.shape[0]//12,12,image.shape[1],image.shape[2])
#     X_train, y_train, X_val,y_val,X_test,y_test = original(images,masks,n_classes)
#     return X_train, y_train, X_val,y_val,X_test,y_test

def mono_neu_base(patch_size,eta):
    imagey, annon = mono_generatory('mono_3_5_7_8_9_11_14')
    n_classes = 4
    imagine = np.delete(matching_slices(imagey),[10,18, 26,37],axis=0)
    maskine = np.delete(matching_masks(annon),[10,18, 26,37],axis=0)

    image = patchete(imagine,patch_size)
    mask = matchete(maskine,patch_size)

    images = image.reshape(image.shape[0]//eta,eta,image.shape[1],image.shape[2],4)
    masks = mask.reshape(image.shape[0]//eta,eta,image.shape[1],image.shape[2])
    X_train, y_train, X_val,y_val,X_test,y_test = original(images,masks,n_classes)

    # I = np.load('Z:/DeyPlay/downsampled_7wks_mono_img.npy')
    # M = np.argmax(np.load('Z:/DeyPlay/downsampled_7wks_mono_mks.npy'),axis=3)
    # imagines = np.delete(matching_slices(I),[10,18, 26,37],axis=0)
    # maskines = np.delete(matching_masks(M),[10,18, 26,37],axis=0)
    # X_trains, y_trains, X_vals,y_vals,X_tests,y_tests = original(imagines,maskines,n_classes)

    # Xtrain = np.concatenate([X_train,X_trains],axis=0)
    # Xval = np.concatenate([X_val,X_val],axis=0)
    # Xtest = np.concatenate([X_test,X_test],axis=0)

    # ytrain = np.concatenate([y_train,y_trains],axis=0)
    # yval = np.concatenate([y_val,y_val],axis=0)
    # ytest = np.concatenate([y_test,y_test],axis=0)

    return X_train, y_train, X_val,y_val,X_test,y_test

def silk_free(s):
    image, anno = nams_generator('tri_3_4_5_6_7_8_9_11')
    annon = unmasky_makery(anno)
    masks = shapery(matchete(annon,256),12)
    images = shaper(patchete(matching_slices(image),256),12)
    XT, ytrain,XV,yval, XE, ytest = original(images,masks,4)
    modeler = load_model('Z:/DeyPlay/tri_2020b_enameta.hdf5',compile=False)
    Xtrain,Xval,Xtest = silkfeval(XT,XV,XE,modeler,s)
    return Xtrain,ytrain,Xval,yval,Xtest,ytest

# def silk_free(s):
#     image, anno = nams_generator('tri_3_4_5_6_7_8_9_11')
#     annon = unmasky_makery(anno)
#     masks = shapery(matchete(annon,256),12)
#     images = shaper(patchete(matching_slices(image),256),12)
#     XT, ytrain,XV,yval, XE, ytest = original(images,masks,4)
#     modeler = load_model('Z:/DeyPlay/tri_enameta_x.hdf5',compile=False) 
#     Xtrain,Xval,Xtest = silkfeval(XT,XV,XE,modeler,s)

#     dimage,dmask,_ = generation()
#     dmasks  = np.argmax(dmask,axis=4)
#     dmasks[dmasks==2]=0
#     dmasks[dmasks==3]=2
#     dmasks[dmasks==4]=3
#     dimages = standardymtx(matching_slices(dimage),'5d')
#     dXtrainy, dytrainy, dXvaly,dyvaly,dXtesty,_ = driginal(dimages,to_categorical(dmasks,4))
#     dXtrain,dXval,_ = silkfeval(dXtrainy,dXvaly,dXtesty,modeler,s)

#     Xtrains = np.concatenate([Xtrain,dXtrain])
#     ytrains = np.concatenate([ytrain,dytrainy])

#     Xvals = np.concatenate([Xval,dXval])
#     yvals = np.concatenate([yval,dyvaly])

#     return Xtrains,ytrains,Xvals,yvals,Xtrain,ytrain,Xval,yval,Xtest,ytest

def silkal(model,X,s):
    y = model.predict(X)[:,:,:,2]>s
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[3]):
            Z[i,:,:,j] = X[i,:,:,j]*(1-y[i,:,:])
    return Z

def silkfeval(Xtrain,Xval,Xtest,model,s):
    return silkal(model,Xtrain,s), silkal(model,Xval,s),silkal(model,Xtest,s)

def unmasky_makery(annony):
    annony[annony==3]=2
    annony[annony==4]=3
    annony[annony==5]=3
    masks =  to_categorical(shay(matching_masks(annony)),4)
    removed_circles = remove_small_objects(masks[:,:, :, 3]>0, min_size=10,connectivity=0)
    masker = mastify(removed_circles,100)
    annony[annony==3]=0
    M = []
    for i in range(masker.shape[0]):
        s = np.expand_dims(shapy(np.expand_dims(matching_masks(annony),axis=4))[i,:,:,0],axis=0)
        u = 3*np.expand_dims(masker[i,:,:],axis=0)
        M.append(np.max(np.concatenate([s,u]),axis=0))  
    return shapery(np.stack(M,axis=0),3)


# def plot_base_tri(Xtes,ytes,y_pred,nu,s,g,var):
#     maska = np.sum(Xtes,axis=3)
#     for i in range(Xtes.shape[0]):
#         fig, axes = plt.subplots(1, 7, figsize=(30, 4.5))
#         fig.suptitle(var, fontsize=20)
        
#         ax = axes[0]
#         im = ax.imshow(maska[i, :, :])
#         ax.set_title("Aggregated Image",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
        
#         # divider = make_axes_locatable(ax)
#         # cax = divider.append_axes("right", size="5%", pad=0.05)
#         # cbar = plt.colorbar(im, cax=cax)
#         #cbar.set_label('Intensity')
        
#         ax = axes[1]
#         ax.imshow(ytes[i,:,:,2])
#         ax.set_title("True Neuronal Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)

#         ax = axes[2]
#         ax.imshow(ytes[i,:,:,4])
#         ax.set_title("True Silk Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)

#         ax = axes[3]
#         ax.imshow(ytes[i,:,:,3])
#         ax.set_title("True Glial Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
          
#         ax = axes[4]
#         ax.imshow(y_pred[i,:,:,2]>nu)
#         ax.set_title("Predicted Neuronal Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
        
#         ax = axes[5]
#         ax.imshow(y_pred[i,:,:,4]>s)
#         ax.set_title("Predicted Silk Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)

#         ax = axes[6]
#         ax.imshow(y_pred[i,:,:,3]>g)
#         ax.set_title("Predicted Glial Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
        
#         plt.show()

def plot_base_tris(Xtes,ytes,y_pred,nu,g,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("True Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[2]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("True Glial Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
          
        ax = axes[3]
        ax.imshow(y_pred[i,:,:,2]>nu)
        ax.set_title("Predicted Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[4]
        ax.imshow(y_pred[i,:,:,3]>g)
        ax.set_title("Predicted Glial Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        plt.show()

def load_inj_tri(n_classes):
    X_ctrl, y_ctrl = tri_tbi_tri('tri_injury_ctrl')
    X_test_match_ctrl = matching_ctrl(X_ctrl)
   
    X_test_ctrl = shaper(patchete(shaper(X_test_match_ctrl,10),256),40)
    y_test_ctrl = shapery(matchete(shapery(y_ctrl,10),256),40)
    Xtrain_ctrl, ytrain_ctrl, Xval_ctrl,yval_ctrl,Xtest_ctrl,ytest_ctrl = driginal(X_test_ctrl,to_categorical(y_test_ctrl,n_classes))

    X_8h, y_8h = tri_tbi_tri('tri_injury_8')
    X_test_match_8h = matching_inj(X_8h)
    X_test_8h = shaper(patchete(shaper(X_test_match_8h,20),256),80)
    y_test_8h = shapery(matchete(shapery(y_8h,20),256),80)
    Xtrain_8h, ytrain_8h, Xval_8h,yval_8h,Xtest_8h,ytest_8h = driginal(X_test_8h,to_categorical(y_test_8h,n_classes))

    X_24h, y_24h = tri_tbi_tri('tri_injury_24')
    X_test_match_24h = matching_inj(X_24h)
    X_test_24h = shaper(patchete(shaper(X_test_match_24h,20),256),80)
    y_test_24h = shapery(matchete(shapery(y_24h,20),256),80)
    Xtrain_24h, ytrain_24h, Xval_24h,yval_24h,Xtest_24h,ytest_24h = driginal(X_test_24h,to_categorical(y_test_24h,n_classes))

    X_48h, y_48h = tri_tbi_tri('tri_injury_48')
    X_test_match_48h = matching_inj(X_48h)
    X_test_48h = shaper(patchete(shaper(X_test_match_48h,20),256),80)
    y_test_48h = shapery(matchete(shapery(y_48h,20),256),80)
    Xtrain_48h, ytrain_48h, Xval_48h,yval_48h,Xtest_48h,ytest_48h = driginal(X_test_48h,to_categorical(y_test_48h,n_classes))

    Xtrain = np.concatenate([Xtrain_ctrl,Xtrain_8h,Xtrain_24h,Xtrain_48h])
    ytrain = np.concatenate([ytrain_ctrl,ytrain_8h,ytrain_24h,ytrain_48h])

    Xval = np.concatenate([Xval_ctrl,Xval_8h,Xval_24h,Xval_48h])
    yval = np.concatenate([yval_ctrl,yval_8h,yval_24h,yval_48h])

    Xtest = np.concatenate([Xtest_ctrl,Xtest_8h,Xtest_24h,Xtest_48h])
    ytest = np.concatenate([ytest_ctrl,ytest_8h,ytest_24h,ytest_48h])

    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def tri_nam():
    image, annon = nams_generator('tri_3_4_5_6_7_8_9_11')
    masks = shapery(matchete(masky_makery(annon),256),12)
    images = shaper(patchete(matching_slices(image),256),12)
    oXtrain, oytrain, oXval,oyval, _,_ = original(images,masks,5)

    dimage,dmasks,n_classes = generation()
    dimages = standardymtx(matching_slices(dimage),'5d')
    dXtrain, dytrain, dXval,dyval,_,_ = driginal(dimages,dmasks)

    Xtrain = np.concatenate([oXtrain,dXtrain])
    ytrain = np.concatenate([oytrain,dytrain])

    Xval = np.concatenate([oXval,dXval])
    yval = np.concatenate([oyval,dyval])

    return Xtrain, ytrain, Xval,yval,n_classes 


def combine_nams():
    image, annon = nams_generator('tri_3_4_5_6_7_8_9_11')
    masks = shapery(matchete(masky_makery(annon),256),12)
    images = shaper(patchete(matching_slices(image),256),12)
    oXtrain, ytrain, oXval,yval, oXtest, ytest = original(images,masks,5)

    normalize_tr_images = normalise_3mtx(image,12,'norm',**{'axis':0})
    divide_tr_image = divisor(normalize_tr_images,image,'non_thresh',**{'thresh':12000})
    divide_tr_images = shaper(patchete(matching_slices(normalise_3mtx(divide_tr_image,12,'minmax',**{'axis':0})),256),12)
    nXtrain, _, nXval,_, nXtest,_ = original(divide_tr_images,masks,5)

    normalize_tr_images_thresh = normalise_3mtx(image,12,'norm',**{'axis':0})
    divide_tr_images_thresh = divisor(normalize_tr_images_thresh,image,'thresh',**{'thresh':12000})
    divide_tr_images_thresh = shaper(patchete(matching_slices(normalise_3mtx(divide_tr_images_thresh,12,'minmax',**{'axis':0})),256),12)
    tXtrain, _, tXval,_, tXtest,_ = original(divide_tr_images_thresh,masks,5)

    Xtrain = np.concatenate([oXtrain,nXtrain,tXtrain],axis=3)
    Xval = np.concatenate([oXval,nXval,tXval],axis=3)
    Xtest = np.concatenate([oXtest,nXtest,tXtest],axis=3)

    #J = np.expand_dims(standardymtx(shapery(matchete(maxprojection(image),256),12),'4d'),axis=4)


    return Xtrain, ytrain, Xval, yval,Xtest, ytest

def maxprojection(A):
    F = matching_slices(A)
    B = []
    for j in range(F.shape[0]):
        G = np.zeros((F.shape[1],F.shape[2],F.shape[3]))
        for i in range(F.shape[1]):
            G[i,:,:] = np.sum(F[j, i, :, :, :],axis=2)
        H = np.expand_dims(np.max(G, axis=0),axis=0)
        B.append(np.repeat(H,repeats=3,axis=0))
    return np.stack(B,axis=0)

def project_nams():
    image, annon = nams_generator('tri_3_4_5_6_7_8_9_11')
    masks = shapery(matchete(masky_makery(annon),256),12)
    images = np.concatenate([shaper(patchete(matching_slices(image),256),12),
                             np.expand_dims(shapery(matchete(maxprojection(image),256),12),axis=4)],axis=4)
    oXtrain, ytrain, oXval,yval, oXtest, ytest = original(images,masks,5)
    Xtrain = np.concatenate([oXtrain,minmax(np.expand_dims(np.sum(oXtrain,axis=3),axis=3))],axis=3)
    Xval = np.concatenate([oXval,minmax(np.expand_dims(np.sum(oXval,axis=3),axis=3))],axis=3)
    Xtest = np.concatenate([oXtest,minmax(np.expand_dims(np.sum(oXtest,axis=3),axis=3))],axis=3)
    return Xtrain, ytrain, Xval, yval,Xtest, ytest

def project_all_nams():
    image, annon = nams_generator('tri_3_4_5_6_7_8_9_11')
    masks = shapery(matchete(masky_makery(annon),256),12)
    images = np.concatenate([shaper(patchete(matching_slices(image),256),12),
                             np.expand_dims(shapery(matchete(maxprojection(image),256),12),axis=4)],axis=4)
    oXtrain, ytrain, oXval,yval, oXtest, ytest = original(images,masks,5)
    Xtrain = np.concatenate([oXtrain,minmax(np.expand_dims(np.sum(oXtrain,axis=3),axis=3)),
                             minmax(np.expand_dims(np.sum(oXtrain[:,:,:,0:2],axis=3),axis=3)),
                             minmax(np.expand_dims(np.sum(oXtrain[:,:,:,2:4],axis=3),axis=3))],axis=3)
    Xval = np.concatenate([oXval,minmax(np.expand_dims(np.sum(oXval,axis=3),axis=3)),
                           minmax(np.expand_dims(np.sum(oXval[:,:,:,0:2],axis=3),axis=3)),
                           minmax(np.expand_dims(np.sum(oXval[:,:,:,2:4],axis=3),axis=3))],axis=3)
    Xtest = np.concatenate([oXtest,minmax(np.expand_dims(np.sum(oXtest,axis=3),axis=3)),
                            minmax(np.expand_dims(np.sum(oXtest[:,:,:,0:2],axis=3),axis=3)),
                            minmax(np.expand_dims(np.sum(oXtest[:,:,:,2:4],axis=3),axis=3))],axis=3)
    return Xtrain, ytrain, Xval, yval,Xtest, ytest

def project_all_nams_step():
    image, annon = nams_generator('tri_3_4_5_6_7_8_9_11')
    masks = shapery(matchete(masky_makery(annon),256),12)
    images = np.concatenate([shaper(patchete(matching_slices(image),256),12),
                             np.expand_dims(shapery(matchete(maxprojection(image),256),12),axis=4)],axis=4)
    oXtrain, ytrain, oXval,yval, oXtest, ytest = original(images,masks,5)
    Xtrain = np.concatenate([oXtrain,minmax(np.expand_dims(np.sum(oXtrain[:,:,:,0:4],axis=3),axis=3)),
                             minmax(np.expand_dims(np.sum(oXtrain[:,:,:,0:2],axis=3),axis=3)),
                             minmax(np.expand_dims(np.sum(oXtrain[:,:,:,2:4],axis=3),axis=3))],axis=3)
    Xval = np.concatenate([oXval,minmax(np.expand_dims(np.sum(oXval[:,:,:,0:4],axis=3),axis=3)),
                           minmax(np.expand_dims(np.sum(oXval[:,:,:,0:2],axis=3),axis=3)),
                           minmax(np.expand_dims(np.sum(oXval[:,:,:,2:4],axis=3),axis=3))],axis=3)
    Xtest = np.concatenate([oXtest,minmax(np.expand_dims(np.sum(oXtest[:,:,:,0:4],axis=3),axis=3)),
                            minmax(np.expand_dims(np.sum(oXtest[:,:,:,0:2],axis=3),axis=3)),
                            minmax(np.expand_dims(np.sum(oXtest[:,:,:,2:4],axis=3),axis=3))],axis=3)
    return Xtrain, ytrain, Xval, yval,Xtest, ytest


def load_mono_mic(n_classes):
    X_ctrl, y_ctrl = tri_tbi_injury('tri_injury_ctrl')
    X_test_match_ctrl = matching_ctrl(X_ctrl)
   
    X_test_ctrl = shaper(patchete(shaper(X_test_match_ctrl,10),256),40)
    y_test_ctrl = shapery(matchete(shapery(y_ctrl,10),256),40)
    Xtrain_ctrl, ytrain_ctrl, Xval_ctrl,yval_ctrl,Xtest_ctrl,ytest_ctrl = driginal(X_test_ctrl,to_categorical(y_test_ctrl,n_classes))

    X_8h, y_8h = tri_tbi_injury('tri_injury_8')
    X_test_match_8h = matching_inj(X_8h)
    X_test_8h = shaper(patchete(shaper(X_test_match_8h,20),256),80)
    y_test_8h = shapery(matchete(shapery(y_8h,20),256),80)
    Xtrain_8h, ytrain_8h, Xval_8h,yval_8h,Xtest_8h,ytest_8h = driginal(X_test_8h,to_categorical(y_test_8h,n_classes))

    X_24h, y_24h = tri_tbi_injury('tri_injury_24')
    X_test_match_24h = matching_inj(X_24h)
    X_test_24h = shaper(patchete(shaper(X_test_match_24h,20),256),80)
    y_test_24h = shapery(matchete(shapery(y_24h,20),256),80)
    Xtrain_24h, ytrain_24h, Xval_24h,yval_24h,Xtest_24h,ytest_24h = driginal(X_test_24h,to_categorical(y_test_24h,n_classes))

    X_48h, y_48h = tri_tbi_injury('tri_injury_48')
    X_test_match_48h = matching_inj(X_48h)
    X_test_48h = shaper(patchete(shaper(X_test_match_48h,20),256),80)
    y_test_48h = shapery(matchete(shapery(y_48h,20),256),80)
    Xtrain_48h, ytrain_48h, Xval_48h,yval_48h,Xtest_48h,ytest_48h = driginal(X_test_48h,to_categorical(y_test_48h,n_classes))

    Xtrain = np.concatenate([Xtrain_ctrl,Xtrain_8h,Xtrain_24h,Xtrain_48h])
    ytrain = np.concatenate([ytrain_ctrl,ytrain_8h,ytrain_24h,ytrain_48h])

    Xval = np.concatenate([Xval_ctrl,Xval_8h,Xval_24h,Xval_48h])
    yval = np.concatenate([yval_ctrl,yval_8h,yval_24h,yval_48h])

    Xtest = np.concatenate([Xtest_ctrl,Xtest_8h,Xtest_24h,Xtest_48h])
    ytest = np.concatenate([ytest_ctrl,ytest_8h,ytest_24h,ytest_48h])

    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def load_mono_ast(n_classes):
    X_ctrl, y_ctrl = tri_tbi_ast('tri_injury_ctrl')
    X_test_match_ctrl = matching_ctrl(X_ctrl)
    
    X_test_ctrl = shaper(patchete(shaper(X_test_match_ctrl,10),256),40)
    y_test_ctrl = shapery(matchete(shapery(y_ctrl,10),256),40)
    Xtrain_ctrl, ytrain_ctrl, Xval_ctrl,yval_ctrl,Xtest_ctrl,ytest_ctrl = driginal(X_test_ctrl,to_categorical(y_test_ctrl,n_classes))

    X_8h, y_8h = tri_tbi_ast('tri_injury_8')
    X_test_match_8h = matching_inj(X_8h)
    X_test_8h = shaper(patchete(shaper(X_test_match_8h,20),256),80)
    y_test_8h = shapery(matchete(shapery(y_8h,20),256),80)
    Xtrain_8h, ytrain_8h, Xval_8h,yval_8h,Xtest_8h,ytest_8h = driginal(X_test_8h,to_categorical(y_test_8h,n_classes))

    X_24h, y_24h = tri_tbi_ast('tri_injury_24')
    X_test_match_24h = matching_inj(X_24h)
    X_test_24h = shaper(patchete(shaper(X_test_match_24h,20),256),80)
    y_test_24h = shapery(matchete(shapery(y_24h,20),256),80)
    Xtrain_24h, ytrain_24h, Xval_24h,yval_24h,Xtest_24h,ytest_24h = driginal(X_test_24h,to_categorical(y_test_24h,n_classes))

    X_48h, y_48h = tri_tbi_ast('tri_injury_48')
    X_test_match_48h = matching_inj(X_48h)
    X_test_48h = shaper(patchete(shaper(X_test_match_48h,20),256),80)
    y_test_48h = shapery(matchete(shapery(y_48h,20),256),80)
    Xtrain_48h, ytrain_48h, Xval_48h,yval_48h,Xtest_48h,ytest_48h = driginal(X_test_48h,to_categorical(y_test_48h,n_classes))

    Xtrain = np.concatenate([Xtrain_ctrl,Xtrain_8h,Xtrain_24h,Xtrain_48h])
    ytrain = np.concatenate([ytrain_ctrl,ytrain_8h,ytrain_24h,ytrain_48h])

    Xval = np.concatenate([Xval_ctrl,Xval_8h,Xval_24h,Xval_48h])
    yval = np.concatenate([yval_ctrl,yval_8h,yval_24h,yval_48h])

    Xtest = np.concatenate([Xtest_ctrl,Xtest_8h,Xtest_24h,Xtest_48h])
    ytest = np.concatenate([ytest_ctrl,ytest_8h,ytest_24h,ytest_48h])

    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def load_mono_neu(n_classes):
    X_ctrl, y_ctrl = mono_tbi_injury('tri_injury_ctrl')
    X_test_match_ctrl = matching_ctrl(X_ctrl)
    X_test_match_ctrl_del = np.delete(X_test_match_ctrl,todelete([slice(0,10),slice(80,90)]),axis=0)
    y_test_match_ctrl_del = np.delete(y_ctrl,todelete([slice(0,10),slice(80,90)]),axis=0)

    X_test_ctrl = shaper(patchete(shaper(X_test_match_ctrl_del,10),256,4),40)
    y_test_ctrl = shapery(matchete(shapery(y_test_match_ctrl_del,10),256),40)
    Xtrain_ctrl, ytrain_ctrl, Xval_ctrl,yval_ctrl,Xtest_ctrl,ytest_ctrl = driginal(X_test_ctrl,to_categorical(y_test_ctrl,n_classes))

    X_8h, y_8h = mono_tbi_injury('tri_injury_8')
    X_test_match_8h = matching_inj(X_8h)
    X_test_8h = shaper(patchete(shaper(X_test_match_8h,20),256,4),80)
    y_test_8h = shapery(matchete(shapery(y_8h,20),256),80)
    Xtrain_8h, ytrain_8h, Xval_8h,yval_8h,Xtest_8h,ytest_8h = driginal(X_test_8h,to_categorical(y_test_8h,n_classes))

    X_24h, y_24h = mono_tbi_injury('tri_injury_24')
    X_test_match_24h = matching_inj(X_24h)
    X_test_24h = shaper(patchete(shaper(X_test_match_24h,20),256,4),80)
    y_test_24h = shapery(matchete(shapery(y_24h,20),256),80)
    Xtrain_24h, ytrain_24h, Xval_24h,yval_24h,Xtest_24h,ytest_24h = driginal(X_test_24h,to_categorical(y_test_24h,n_classes))

    X_48h, y_48h = mono_tbi_injury('tri_injury_48')
    X_test_match_48h = matching_inj(X_48h)
    X_test_48h = shaper(patchete(shaper(X_test_match_48h,20),256,4),80)
    y_test_48h = shapery(matchete(shapery(y_48h,20),256),80)
    Xtrain_48h, ytrain_48h, Xval_48h,yval_48h,Xtest_48h,ytest_48h = driginal(X_test_48h,to_categorical(y_test_48h,n_classes))


    Xtrain = np.concatenate([Xtrain_ctrl,Xtrain_8h,Xtrain_24h,Xtrain_48h])
    ytrain = np.concatenate([ytrain_ctrl,ytrain_8h,ytrain_24h,ytrain_48h])

    Xval = np.concatenate([Xval_ctrl,Xval_8h,Xval_24h,Xval_48h])
    yval = np.concatenate([yval_ctrl,yval_8h,yval_24h,yval_48h])

    Xtest = np.concatenate([Xtest_ctrl,Xtest_8h,Xtest_24h,Xtest_48h])
    ytest = np.concatenate([ytest_ctrl,ytest_8h,ytest_24h,ytest_48h])

    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def plot_mono_isba(Xtes,ytes,y_pred,y_pred_nisba,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(2, 4, figsize=(27,15))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0,0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[0,1]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("ISBA Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[0,2]
        ax.imshow(y_pred[i,:,:,3]>nu)
        ax.set_title("YNET Neuronal (ISBA) Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[0,3]
        ax.imshow(y_pred_nisba[i,:,:,2]>nu)
        ax.set_title("YNET Neuronal (Baseline) Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[1,0]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[1,1]
        ax.imshow(y_pred[i,:,:,2]>s)
        ax.set_title("YNET Silk (ISBA) Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[1,2]
        ax.imshow(y_pred_nisba[i,:,:,3]>s)
        ax.set_title("YNET Silk (Baseline) Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        axes[1, 3].axis('off')
        plt.tight_layout()
        plt.show()

def plot_injury(Xtes,ytes,y_pred,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("True Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[2]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("True Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
          
        ax = axes[3]
        ax.imshow(y_pred[i,:,:,3]>nu)
        ax.set_title("Predicted Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[4]
        ax.imshow(y_pred[i,:,:,2]>s)
        ax.set_title("Predicted Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        plt.show()

# def plot_base_tri(Xtes,ytes,y_pred,nu,s,var):
#     maska = np.sum(Xtes,axis=3)
#     for i in range(Xtes.shape[0]):
#         fig, axes = plt.subplots(1, 5, figsize=(30, 4.5))
#         fig.suptitle(var, fontsize=20)
        
#         ax = axes[0]
#         im = ax.imshow(maska[i, :, :])
#         ax.set_title("Aggregated Image",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
        
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cbar = plt.colorbar(im, cax=cax)
#         #cbar.set_label('Intensity')
        
#         ax = axes[1]
#         ax.imshow(ytes[i,:,:,2])
#         ax.set_title("True Neuronal Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)

#         ax = axes[2]
#         ax.imshow(ytes[i,:,:,3])
#         ax.set_title("True Silk Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)

#         ax = axes[3]
#         ax.imshow(ytes[i,:,:,4])
#         ax.set_title("True Glial Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
          
#         ax = axes[4]
#         ax.imshow(y_pred[i,:,:,2]>nu)
#         ax.set_title("Predicted Neuronal Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
        
#         ax = axes[5]
#         ax.imshow(y_pred[i,:,:,3]>s)
#         ax.set_title("Predicted Silk Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)

#         ax = axes[6]
#         ax.imshow(y_pred[i,:,:,4]>nu)
#         ax.set_title("Predicted Glial Mask",fontsize=15)
#         ax.axis("on")
#         ax.grid(True)
        
#         plt.show()


def plot_mono_ast(Xtes,ytes,y_pred,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(2, 3, figsize=(27,15))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0,0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[0,1]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("ISBA Astrocytes Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[0,2]
        ax.imshow(y_pred[i,:,:,3]>nu)
        ax.set_title("YNET Astrocytes Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[1,0]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[1,1]
        ax.imshow(y_pred[i,:,:,2]>s)
        ax.set_title("YNET Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()



def plot_mono_mic(Xtes,ytes,y_pred,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("True Microglia Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)

        ax = axes[2]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("True Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
          
        ax = axes[3]
        ax.imshow(y_pred[i,:,:,3]>nu)
        ax.set_title("Predicted Microglia Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[4]
        ax.imshow(y_pred[i,:,:,2]>s)
        ax.set_title("Predicted Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        plt.show()







def plot_mono(Xtes,ytes,y_pred,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("True Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[2]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("True Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[3]
        ax.imshow(y_pred[i,:,:,2]>nu)
        ax.set_title("Predicted Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[4]
        ax.imshow(y_pred[i,:,:,3]>s)
        ax.set_title("Predicted Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        plt.show()

def plot_metric(data):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
 
    axes[0].plot(data['epoch'], data['f1-score'], label='Training F1-Score', color='blue')
    axes[0].plot(data['epoch'], data['val_f1-score'], label='Validation F1-Score', color='orange')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[0].set_title('F1-Score vs Epoch', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True)

    axes[1].plot(data['epoch'], data['iou_score'], label='Training IOU Score', color='green', linestyle='--')
    axes[1].plot(data['epoch'], data['val_iou_score'], label='Validation IOU Score', color='red', linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('IOU Score', fontsize=12, fontweight='bold')
    axes[1].set_title('IOU Score vs Epoch', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True)

    axes[2].plot(data['epoch'], data['loss'], label='Training Loss', color='purple', linestyle='-.')
    axes[2].plot(data['epoch'], data['val_loss'], label='Validation Loss', color='brown', linestyle='-.')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[2].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True)

    plt.tight_layout()

    plt.show()


def plot_Alzh_mono(Xtes,ytes,y_pred,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(ytes[i,:,:,0])
        ax.set_title("True Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[2]
        ax.imshow(ytes[i,:,:,1])
        ax.set_title("True Glial Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[3]
        ax.imshow(y_pred[i,:,:,3]>nu)
        ax.set_title("Predicted Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[4]
        ax.imshow(y_pred[i,:,:,4]>s)
        ax.set_title("Predicted Glial Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        plt.show()



def plot_Alzh(Xtes,ytes,y_pred,nu,s,var):
    maska = np.sum(Xtes,axis=3)
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(var, fontsize=20)
        
        ax = axes[0]
        im = ax.imshow(maska[i, :, :])
        ax.set_title("Aggregated Image",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(ytes[i,:,:,3])
        ax.set_title("True Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[2]
        ax.imshow(y_pred[i,:,:,3]>nu)
        ax.set_title("Predicted Neuronal Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[3]
        ax.imshow(ytes[i,:,:,2])
        ax.set_title("True Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        ax = axes[4]
        ax.imshow(y_pred[i,:,:,2]>s)
        ax.set_title("Predicted Silk Mask",fontsize=15)
        ax.axis("on")
        ax.grid(True)
        
        plt.show()


def Alzion(var):
    if var == 'train':
        X = np.load('Z:/DeyPlay/Xtrain_Alz.npy')
        y = np.load('Z:/DeyPlay/ytrain_Alz.npy')
    elif var == 'val':
        X = np.load('Z:/DeyPlay/Xval_Alz.npy')
        y = np.load('Z:/DeyPlay/yval_Alz.npy')
    elif var == 'test':
        X = np.load('Z:/DeyPlay/Xtest_Alz.npy')
        y = np.load('Z:/DeyPlay/ytest_Alz.npy')
    return X,y

def todelete(slices_to_delete):
    indices_to_delete = []
    for s in slices_to_delete:
        indices_to_delete.extend(np.arange(s.start, s.stop))
    return indices_to_delete

def mask_Alzer(var,n_classes):
    e = 'EX755_860' 
    f  = 'indey'
    g = 'Mask'
    if var == 'infected':
        infected = h5py.File('Infected_BothRounds_Alzheimer.mat','a')   
        data = trans(np.array(infected.get(e)))
        masker = trans(np.array(infected.get(g)))
    elif var == 'control':
        infected = h5py.File('Ctrl_BothRounds_Alzheimer.mat','a') 
        data = trans(np.array(infected.get(e)))
        masker = trans(np.array(infected.get(g)))  
    indices = [x*4 for x in np.array(infected.get(f)).astype(int).tolist()[0]]
    extracted_images = []
    extracta_images = []
    image_array = data.copy()
    mask_array = masker.copy()
    start_index = 0
    for index in indices:
        z = image_array[start_index:start_index+index, :, :]
        extracted = np.stack([z[i:i+(index//4), :, :] for i in range(0, z.shape[0], index//4)], axis=-1)
        extracted_images.append(minmax(extracted))
        start_index += index
        
    start_ind = 0
    indies = [x for x in np.array(infected.get(f)).astype(int).tolist()[0]]
    for ind in indies:
        zz = mask_array[start_ind:start_ind+ind, :, :]
        extracta = np.stack([zz[i:i+(ind), :, :] for i in range(0, zz.shape[0], ind)], axis=-1)
        extracta_images.append(extracta)
        start_ind += ind
      
    X_train,y_train,X_val,y_val,X_test,y_test = Alzer(extracted_images, extracta_images)    
    return X_train,to_categorical(y_train,n_classes),X_val,to_categorical(y_val,n_classes),X_test,to_categorical(y_test,n_classes)

def Alzer(image,masks):
    Xtrr, ytrr, X_test,y_test = train_test_val_Alzer(image,masks,0.10)
    # print(len(Xtrr), len(ytrr), len(X_test),len(y_test))
    X_train, y_train, X_val, y_val = train_test_val_Alzer(Xtrr,ytrr,0.30)
    return np.concatenate(X_train,axis=0),np.concatenate(y_train,axis=0), np.concatenate(X_val,axis=0), np.concatenate(y_val,axis=0),np.concatenate(X_test,axis=0),np.concatenate(y_test,axis=0)

def train_test_val_Alzer(image_dataset,mask_dataset,test_size): 
        
       X_train_indices, X_val_indices, y_train_indices, y_val_indices = train_test_split(
           range(len(image_dataset)), range(len(mask_dataset)),test_size=test_size, random_state=42)
       X_train = [image_dataset[i] for i in X_train_indices]
       X_val = [image_dataset[i] for i in X_val_indices]
       y_train = [mask_dataset[i] for i in y_train_indices]
       y_val = [mask_dataset[i] for i in y_val_indices]
       return X_train, y_train, X_val, y_val






def compress(A):
    h = np.expand_dims(np.max(A[:,:,:,3:5],axis=3),axis=3)
    hh = np.concatenate([A[:,:,:,0:3],h],axis=3)
    return hh

    
def plot_tri(Xtes,y_pred,rho,nu,j,var):
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(30, 8))
        fig.suptitle(var, fontsize=30)
        
        ax = axes[0]
        im = ax.imshow(Xtes[i, :, :])
        ax.set_title("Aggregated Image",fontsize=30)
        ax.grid(True)
        #ax.axis("off")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        # ax = axes[1]
        # ax.imshow(y_pred[i,:,:,j]>s)
        # ax.set_title("Predicted Silk Mask",fontsize=30)
        # #ax.axis("off")
        # ax.grid(True)
        
        ax = axes[1]
        ax.imshow(y_pred[i,:,:,j+1]>rho)
        ax.set_title("Predicted Neuronal Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[2]
        ax.imshow(y_pred[i,:,:,j+2]>nu)
        ax.set_title("Predicted Glial Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True)
        
        plt.show()
        
        
def plot_trial(Xtes,y,y_pred,nu,s,j,var):
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(28, 8.5))
        fig.suptitle(var, fontsize=30)
        
        ax = axes[0]
        im = ax.imshow(Xtes[i, :, :])
        ax.set_title("Aggregated Image",fontsize=30)
        ax.grid(True)
        #ax.axis("off")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(y[i,:,:])
        ax.set_title("Manual Astrocytes Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True) 
          
        ax = axes[2]
        ax.imshow(y_pred[i,:,:,j]>nu)
        ax.set_title("Predicted Astrocytes Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True)
        
        plt.show()
        
        
def plot_mic(Xtes,y,y_pred,nu,s,j,var):
    for i in range(Xtes.shape[0]):
        fig, axes = plt.subplots(1, 4, figsize=(28, 8.5))
        fig.suptitle(var, fontsize=30)
        
        ax = axes[0]
        im = ax.imshow(Xtes[i, :, :])
        ax.set_title("Aggregated Image",fontsize=30)
        ax.grid(True)
        #ax.axis("off")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[1]
        ax.imshow(y[i,:,:])
        ax.set_title("Manual Microglia Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True) 
          
        ax = axes[2]
        ax.imshow(y_pred[i,:,:,j+1]>nu)
        ax.set_title("Predicted Microglia Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[3]
        ax.imshow(y_pred[i,:,:,j])
        ax.set_title("Predicted Silk Mask",fontsize=30)
        #ax.axis("off")
        ax.grid(True) 
        
        plt.show()


def driginalize(image,masks):
    Xtrain, ytrain, Xval,yval = train_test_val_gen(image,masks,0.30)
    X_train = shapy(Xtrain)
    y_train = shapy(ytrain)
    X_val = shapy(Xval)
    y_val = shapy(yval)
    return X_train,y_train,X_val,y_val

def maga(A):
    reshaped_slices = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            g = A[i, j,:,:,:].transpose(2, 0, 1)
            reshaped_slices.append(g)
    return np.array(reshaped_slices)

def Silky(images):
    model = load_model('astrodown_enameta_3_25_v0.hdf5',compile=False) 
    Y =  1-(model.predict(shapy(images))>0.5)
    imagery = shapy(images)
    NoSilk = np.zeros_like(imagery)
    for i in range(Y.shape[0]):
        Silk = np.expand_dims(Y[i,:,:,2],axis=2)
        NoSilk[i,:,:,:] = imagery[i,:,:,:]*Silk
    return NoSilk

def compute_metrics(y_val,y_pred_val,c,nu, title=None):
    intersection = np.logical_and(y_val[:,:,:,c], y_pred_val[:,:,:,c]>nu)
    precision = np.sum(intersection) / np.sum(y_pred_val[:,:,:,c]>nu)
    recall = np.sum(intersection) / np.sum(y_val[:,:,:,c])
    Fscore = 2*(np.sum(intersection))/(np.sum(y_val[:,:,:,c]>nu)+np.sum(y_pred_val[:,:,:,c]>nu))
    
    data = {'Metric': ['Recall (Sensitivity/TPR)', 'Precision', 'F_1 Score'],
        'Value': [recall, precision, Fscore]}
    if title:
        spaces = int((35 - len(title)))
        print(' ' * spaces + title)
    df = pd.DataFrame(data)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center', showindex=False))

def iou_compute(y_val, y_pred_val,c,nu):
    intersection = np.logical_and(y_val[:,:,:,c], y_pred_val[:,:,:,c]>nu)
    #union = np.logical_or(y_val[:,:,:,c], y_pred_val[:,:,:,c]>nu)
    iou_score_p = np.sum(intersection) / np.sum(y_pred_val[:,:,:,c]>nu)
    iou_score_q = np.sum(intersection) / np.sum(y_val[:,:,:,c]>nu)
    return iou_score_p, iou_score_q     

def computing_iou(y_val, y_pred_val,c,nu):
    intersection = np.logical_and(y_val[:,:,:,c], y_pred_val[:,:,:,c]>nu)
    union = np.logical_or(y_val[:,:,:,c], y_pred_val[:,:,:,c]>nu)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score                                     


def cemua(Xtrain,ytrain,factor):
    a = shypy(Xtrain,factor)
    aa = np.reshape(a,(a.shape[0],a.shape[1],a.shape[2],-1))
    b = shypy(ytrain,factor)
    bb = np.reshape(b,(b.shape[0],b.shape[1],b.shape[2],-1))
    x,y =  next(aug_generator(aa, bb,batch_size=Xtrain.shape[0],seed=42))
    xx = np.reshape(x,a.shape)
    yy = np.reshape(y,b.shape)
    return xx,yy

def shypy(img,factor):
    s = img.shape[0]
    h = img.shape[1]
    w = img.shape[2]
    d = img.shape[3]
    img_reshaped = np.reshape(img, (s//factor, factor,h,w,d))
    img_transposed = np.transpose(img_reshaped, (0, 2, 3, 1, 4))
    return  img_transposed

def covan(r,rho):
    r[r == np.inf] = np.nan
    a = np.nan_to_num(r,nan=0)
    b = a*(a>rho)   
    return b 




def matching_waves(A):
    ROI_masks = []
    groups = [(0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11)] 
    for j in range(0,A.shape[0],12):
        group_slices = A[j:j+12,:,:]
        hb = group_slices[0:3,:,:]
        hc = group_slices[3:6,:,:]
        hd = group_slices[6:9,:,:]
        he = group_slices[9:12,:,:]
        
        r1 = covan(hb/hd,2)
        r3 = covan(hc/hd,1.5)
        r2 = covan(hb/he,2)
        r4 = covan(hc/he,1.5)
        r = np.concatenate([r1,r2,r3,r4],axis=0)
               
        s1 = covan(hb/(he+hd),1)
        s2 = covan(hc/(he+hd),0.9)
        s3 = covan((hc*hb)/(hd+he),9*1e-5) #covan(r1+r2+r3+r4,1.5)
        s4 = covan((hc+hb)/(hd+he),9*1e-5)
        s = np.concatenate([s1,s2,s3,s4],axis=0)    
        
        group_slice = minmax(group_slices[groups].transpose(0, 2, 3, 1))
        group_slicer = minmax(r[groups].transpose(0, 2, 3, 1))
        groups_slice = minmax(s[groups].transpose(0, 2, 3, 1))
        
        
        g = np.concatenate([group_slice,group_slicer,groups_slice],axis=3)
        ROI_masks.append(g)
    return np.stack(ROI_masks,axis=0)

def matching_waver(A):
    ROI_masks = []
    groups = [(0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11)] 
    for j in range(0,A.shape[0],12):
        group_slices = A[j:j+12,:,:]
        hb = group_slices[0:3,:,:]
        hc = group_slices[3:6,:,:]
        hd = group_slices[6:9,:,:]
        he = group_slices[9:12,:,:]
        
        r1 = covan(hb/hd,0)
        r3 = covan(hc/hd,0)
        r2 = covan(hb/he,0)
        r4 = covan(hc/he,0)
        r = np.concatenate([r1,r2,r3,r4],axis=0)
               
        s1 = covan(hb/(he+hd),0)
        s2 = covan(hc/(he+hd),0)
        s3 = covan((hc*hb)/(hd+he),0) #covan(r1+r2+r3+r4,1.5)
        s4 = covan((hc+hb)/(hd+he),0)
        s = np.concatenate([s1,s2,s3,s4],axis=0)    
        
        group_slice = minmax(group_slices[groups].transpose(0, 2, 3, 1))
        group_slicer = minmax(r[groups].transpose(0, 2, 3, 1))
        groups_slice = minmax(s[groups].transpose(0, 2, 3, 1))
        
        
        g = np.concatenate([group_slice,group_slicer,groups_slice],axis=3)
        ROI_masks.append(g)
    return np.stack(ROI_masks,axis=0)



def originalize(image):
    Xtrr, Xtest,  ytrr, _ = train_test_split(image,range(len(image)),test_size=0.01,random_state=42)
    Xtrain, Xval,_,_ = train_test_split(Xtrr,ytrr,test_size=0.15,random_state=42)
    return np.concatenate(Xtrain,axis=0),np.concatenate(Xval,axis=0),np.concatenate(Xtest,axis=0)


def tri_tbi_glia(var):
    img1 = h5py.File('Micoglia_8h_1stInj_Full.mat','a')
    img2 = h5py.File('Micoglia_24h_1stInj_Full.mat','a')
    img3 = h5py.File('Micoglia_48h_1stInj_Full.mat','a')
    img4 = h5py.File('Neuron_Control_1stInj_Full.mat','a')
    e = 'EX745_845'
    im1 = trans(np.array(img1.get(e)))
    im2 = trans(np.array(img2.get(e)))     
    im3 = trans(np.array(img3.get(e)))
    im4 = trans(np.array(img4.get(e)))
    if var == 'mono_injury_8':
        cat_images = im1
    elif var == 'mono_injury_24':
        cat_images = im2
    elif var == 'mono_injury_48':
        cat_images = im3
    elif var == 'mono_injury_ctrl':
        cat_images = im4    
    return cat_images 

def masky_makery(annon):
    masks =  to_categorical(shay(masky_maker(annon)),5)
    # removed_circles = remove_small_objects(masks[:,:, :, 3]>0, min_size=10,connectivity=0) # 4
    # masker = mastify(removed_circles,100)
  
    # annon[annon==5]=0
    # annon[annon==4]=0
    
    master = np.load('Z:/DeyPlay/masters.npy')

    M = []
    for i in range(master.shape[0]):
        s = np.expand_dims(shapy(np.expand_dims(matching_masks(annon),axis=4))[i,:,:,0],axis=0)
        v = 4*np.expand_dims(master[i,:,:],axis=0)
        #u = 2*np.expand_dims(masker[i,:,:],axis=0)
        u = 2*np.expand_dims(masks[i,:,:,3],axis=0)
        M.append(np.max(np.concatenate([s,v,u]),axis=0))  
       
    return shapery(np.stack(M,axis=0),3)


def masky_making(annon):
    annon[annon==5]=4
    master = np.load('Z:/DeyPlay/masters.npy')
    master = np.delete(master,[191,192,193],axis=0)
    M = []
    for i in range(master.shape[0]):
        s = np.expand_dims(shapy(np.expand_dims(matching_masks(annon),axis=4))[i,:,:,0],axis=0)
        v = 2*np.expand_dims(master[i,:,:],axis=0)
        M.append(np.max(np.concatenate([s,v]),axis=0))   
        #to_categorical(np.expand_dims(np.stack(M,axis=0),axis=3),5)
    return shapery(np.stack(M,axis=0),3)



def masky_maker(annon):
    annon[annon==5]=4
    master = np.load('Z:/DeyPlay/masters.npy')
    M = []
    for i in range(master.shape[0]):
        s = np.expand_dims(shapy(np.expand_dims(matching_masks(annon),axis=4))[i,:,:,0],axis=0)
        v = 2*np.expand_dims(master[i,:,:],axis=0)
        M.append(np.max(np.concatenate([s,v]),axis=0))   
        #to_categorical(np.expand_dims(np.stack(M,axis=0),axis=3),5)
    return shapery(np.stack(M,axis=0),3)

def shaper(A,eta):
    B = A.reshape(A.shape[0]//eta,eta,A.shape[1],A.shape[2],A.shape[3])
    return B

def shapery(A,eta):
    B = A.reshape(A.shape[0]//eta,eta,A.shape[1],A.shape[2])
    return B



def test_data(image,annon):
    imagine = matching_slices(image)
    imagino = matching_masks(annon)
    X_test = np.concatenate((minmax(imagine[10,:,:,:,:]),minmax(imagine[18,:,:,:,:]),
                             minmax(imagine[26,:,:,:,:]),minmax(imagine[37,:,:,:,:])),axis=0)
    y_test = to_categorical(np.concatenate((imagino[10,:,:,:],imagino[18,:,:,:],
                             imagino[26,:,:,:],imagino[37,:,:,:]),axis=0),4)
    return X_test, y_test

def test_data_tri(): 
    # X_test = np.delete(np.load('test_data_tri.npy'),[6,7,8],axis=0)
    # y_test = np.delete(np.load('test_mask_tri.npy'),[6,7,8],axis=0)
    X_test = np.load('test_data_tri.npy')
    y_test = np.load('test_mask_tri.npy')
    
    y2 = y_test[:,:,:,2].copy()
    y3 = y_test[:,:,:,3].copy()
    y_test[:,:,:,2]=y3
    y_test[:,:,:,3]=y2
    return X_test, y_test


def driginal(image,masks):
    Xtrr, ytrr, Xtest,ytest = train_test_val_gen(image,masks,0.10)
    Xtrain, ytrain, Xval, yval = train_test_val_gen(Xtrr,ytrr,0.30)
    X_train = shapy(Xtrain)
    y_train = shapy(ytrain)
    X_val = shapy(Xval)
    y_val = shapy(yval)
    X_test = shapy(Xtest)
    y_test = shapy(ytest)
    return X_train,y_train,X_val,y_val,X_test,y_test


def original(image,masks,n_classes):
    Xtrr, ytrr, Xtest,ytest = train_test_val_gen(standardymtx(image,'5d'),to_categorical(masks,n_classes),0.10)
    Xtrain, ytrain, Xval, yval = train_test_val_gen(Xtrr,ytrr,0.30)
    X_train = shapy(Xtrain)
    y_train = shapy(ytrain)
    X_val = shapy(Xval)
    y_val = shapy(yval)
    X_test = shapy(Xtest)
    y_test = shapy(ytest)
    return X_train,y_train,X_val,y_val,X_test,y_test

def origin(image,masks,n_classes):
    X_train,y_train,X_test,y_test = train_test_val_gen(image,to_categorical(masks,n_classes),0.25)
    return X_train,y_train,X_test,y_test

def originally(image,masks,n_classes,test_size):
    Xtrain,ytrain, Xval,yval = train_test_val_gen(standardymtx(image,'5d'),to_categorical(masks,n_classes),test_size)
    X_train = shapy(Xtrain)
    y_train = shapy(ytrain)
    X_val = shapy(Xval)
    y_val = shapy(yval)
    return X_train,y_train,X_val,y_val

def originalizer(image,masks,n_classes):
    Xtrain,ytrain, Xval,yval = train_test_val_gen(image,to_categorical(masks,n_classes),0.20)
    X_train = shapy(Xtrain)
    y_train = shapy(ytrain)
    X_val = shapy(Xval)
    y_val = shapy(yval)
    return X_train,y_train,X_val,y_val



def shapy(Xtrr):
    return Xtrr.reshape(Xtrr.shape[0]*Xtrr.shape[1],Xtrr.shape[2],Xtrr.shape[3],Xtrr.shape[4])

def shapier(Xtrr):
    return Xtrr.reshape(Xtrr.shape[0]*Xtrr.shape[1],Xtrr.shape[2],Xtrr.shape[3],Xtrr.shape[4],Xtrr.shape[5])


def shay(Xtrr):
    return Xtrr.reshape(Xtrr.shape[0]*Xtrr.shape[1],Xtrr.shape[2],Xtrr.shape[3])

def aug_generator(X_train, y_train,batch_size, seed):
    
    img_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect') # nearest, constant, reflect, wrap

    mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect',
                          preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(X_train, augment=True, seed=seed)
    image_generator = image_data_generator.flow(X_train,batch_size=batch_size,seed=seed)
    
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(y_train,batch_size=batch_size, seed=seed)
  
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)
        
      
def generation():
    image = np.load('Z:/DeyPlay/downsampled_8wks_image.npy')
    training_masks = np.load('Z:/DeyPlay/downsampled_8wks_mask.npy')
    return image, training_masks, 5

def generate_Tri_inj():
    image = np.load('Z:/DeyPlay/val_Tri_inj.npy')
    training_masks = np.load('Z:/DeyPlay/yval_Tri_inj.npy')
    #return matching_inj(image), matching_minj(training_masks)
    return image, training_masks   

def gemask(var):
    F = np.load('Z:/DeyPlay/training_masks_8wks.npy')
    if var == 'full':
        n_classes = 5
        training_masks = F # full class
        return training_masks, n_classes
    elif var == 'full_shg':
        n_classes = 4
        training_masks = np.argmax(F,axis=3) # to get 4 classes
        training_masks[training_masks==1]=0
        training_masks[training_masks==2]=1
        training_masks[training_masks==3]=2
        training_masks[training_masks==4]=3
        training_masky = to_categorical(training_masks,num_classes=n_classes)
        return training_masky, n_classes
    elif var == 'no_silk':
        n_classes = 4
        training_masks = np.argmax(F,axis=3) # to get 4 classes
        training_masks[training_masks==3]=0
        training_masks[training_masks==4]=3
        training_masky = to_categorical(training_masks,num_classes=n_classes)
        return training_masky, n_classes
    elif var == 'only_silk':
        n_classes = 2
        training_masks = np.argmax(F,axis=3) # to get 4 classes
        training_masks[training_masks==1]=0
        training_masks[training_masks==2]=0
        training_masks[training_masks==3]=1
        training_masks[training_masks==4]=0
        training_masky = to_categorical(training_masks,num_classes=n_classes)
        return training_masky, n_classes

    
def gemasky(var):
    F = np.load('training_masks_6wks.npy')
    if var == 'full':
        n_classes = 5
        training_masks = F # full class
        return training_masks, n_classes
    elif var == 'full_shg':
        n_classes = 4
        training_masks = np.argmax(F,axis=3) # to get 4 classes
        training_masks[training_masks==1]=0
        training_masks[training_masks==2]=1
        training_masks[training_masks==3]=2
        training_masks[training_masks==4]=3
        training_masky = to_categorical(training_masks,num_classes=n_classes)
        return training_masky, n_classes
    elif var == 'no_silk':
        n_classes = 4
        training_masks = np.argmax(F,axis=3) # to get 4 classes
        training_masks[training_masks==3]=0
        training_masks[training_masks==4]=3
        training_masky = to_categorical(training_masks,num_classes=n_classes)
        return training_masky, n_classes

def downsample(array, downsample_factor):
    array_size = np.shape(array)
    im_size_row, im_size_col = array_size[0], array_size[1]
    new_im_size_row = im_size_row // downsample_factor
    new_im_size_col = im_size_col // downsample_factor
    new_array_size = [new_im_size_row, new_im_size_col] + list(array_size[2:])
    
    pixel_idxs_begin_row = np.arange(0, im_size_row, downsample_factor) + 1
    pixel_idxs_end_row = np.arange(0, im_size_row, downsample_factor) + 2
    pixel_idxs_begin_col = np.arange(0, im_size_col, downsample_factor) + 1
    pixel_idxs_end_col = np.arange(0, im_size_col, downsample_factor) + 2
    
    downsampled_array_temp = np.zeros((new_im_size_row, new_im_size_col))
    
    for i in range(new_im_size_row):
        row_begin, row_end = pixel_idxs_begin_row[i], pixel_idxs_end_row[i]
        for j in range(new_im_size_col):
            col_begin, col_end = pixel_idxs_begin_col[j], pixel_idxs_end_col[j]
            window = array[row_begin-1:row_end, col_begin-1:col_end]
            new_pix = np.mean(window, axis=(0, 1))
            downsampled_array_temp[i, j] = new_pix
    
    downsampled_array = np.reshape(downsampled_array_temp, new_array_size)
    return downsampled_array


def maskety(image,annon):
    masky = make_mask(image,annon)
    normalize_tr_images = normalise_3mtx(image,12,'norm',**{'axis':0})
    divide_tr_images = divisor(normalize_tr_images,image,'thresh',**{'thresh':2500})
    divide_tr_images_norm = normalise_3mtx(divide_tr_images,12,'minmax',**{'axis':0})
    mask = make_mask(divide_tr_images_norm,annon)
    images = matching_slices(divide_tr_images_norm)
    training_images_sum = np.sum(images,axis=4).reshape(mask.shape)
    masket = to_categorical(masky,num_classes=5)
    for i in range(training_images_sum.shape[0]):
        mastic = (training_images_sum[i,:,:]*(to_categorical(masky,num_classes=5)[i,:,:,2]))>0
        dilated_mask = binary_dilation(mastic, iterations=3)
        closed_mask = binary_closing(dilated_mask, iterations=3)
        removed_mask  = remove_small_objects(closed_mask,min_size=100)
        masket[i,:,:,2] = removed_mask
    return masket

def patchetey(A,patch_size):
    G = []
    for i in range(A.shape[0]):  
        for j in range(3):
            patches = patchify(A[i,j,:,:],(patch_size,patch_size),step=patch_size)
            G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[2],patches.shape[3])))
    return np.concatenate(G,axis=0)

def downsamplifying(A,factor):
    B = np.zeros((A.shape[0],A.shape[1]//factor,A.shape[2]//factor))
    for i in range(A.shape[0]):
        B[i,:,:] = downsample(A[i,:,:],factor)
    return B

def masksampling(A,factor):
    C = []
    for i in range(A.shape[0]):
        B = np.zeros((A.shape[1]//factor,A.shape[2]//factor,A.shape[3]))
        for j in range(A.shape[3]):
            B[:,:,j] = downsample(A[i,:,:,j],factor)>0
        C.append(B)
    return np.stack(C,axis=0)

def datasampling(A,factor):
    C = []
    for i in range(A.shape[0]):
        B = np.zeros((A.shape[1]//factor,A.shape[2]//factor,A.shape[3]))
        for j in range(A.shape[3]):
            B[:,:,j] = downsample(A[i,:,:,j],factor)
        C.append(B)
    return np.stack(C,axis=0)

def threshing(A,rho,tau,beta):
    group = []
    for j in range(0,A.shape[0],12):
        slices = A[j:j+12,:,:]
        B = np.zeros((12,A.shape[1],A.shape[2]))
        for i in range(slices.shape[0]): 
            if i in [0,1,2,3,4,5]: 
                B[i,:,:] = slices[i,:,:]*(slices[i,:,:]>rho)
            elif i in [6,7,8]:
                B[i,:,:] = slices[i,:,:]*(slices[i,:,:]>tau) #860/460  
            elif i in [9,10,11]:
                B[i,:,:] = slices[i,:,:]*(slices[i,:,:]>beta) #860/460 
        group.append(B)
    return np.stack(group,axis=0)   

def make_mask(image,mask):
    images = matching_slices(image)
    masks = matching_slices(mask)
    training_images = images.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
    training_masks = masks.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
    training_images_sum = np.sum(training_images,axis=3)
    training_masks_max = np.max(training_masks,axis=3)
    
    normalize_tr_images = normalise_3mtx(training_images_sum,3,'norm',**{'axis':0})
    divide_tr_images = divisor(normalize_tr_images,training_images_sum,'thresh',**{'thresh':960})
    divide_tr_images_norm = normalise_3mtx(divide_tr_images,3,'minmax',**{'axis':0})
    
    binary_img = divide_tr_images_norm.copy()>0
    removed_circles = morphology.remove_small_objects(binary_img, min_size=500,connectivity=0)
    masker = mastify(removed_circles,750)
    result_image = training_images_sum.copy() #image.copy()
    result_image[masker] = 0  
    
    # training_masks_maxy = training_masks_max.copy()
    # training_masks_maxy[training_masks_maxy==1]=0
    # training_masks_maxy[training_masks_maxy==4]=1
    # training_masks_maxy[training_masks_maxy==3]=2
    # training_masks_maxy[training_masks_maxy==5]=1
    # training_masks_all = catrify(training_masks_maxy,0*masker) 
    
    # training_masks_maxy = training_masks_max.copy()
    # training_masks_maxy[training_masks_maxy==1]=0
    # training_masks_maxy[training_masks_maxy==4]=1
    # training_masks_maxy[training_masks_maxy==3]=3
    # training_masks_maxy[training_masks_maxy==5]=1
    # training_masks_all = catrify(training_masks_maxy,2*masker) 
    
    training_masks_maxy = training_masks_max.copy()
    training_masks_maxy[training_masks_maxy==4]=2
    training_masks_maxy[training_masks_maxy==3]=4
    training_masks_maxy[training_masks_maxy==5]=2
    training_masks_all = catrify(training_masks_maxy,3*masker) 
    
    return training_masks_all

# def compute_class_iou(values):
#     num_classes = len(values)
#     class_iou = []
#     for i in range(num_classes):
#         true_positive = values[i, i]
#         false_positive = sum(values[:, i]) - true_positive
#         false_negative = sum(values[i, :]) - true_positive
#         denominator = true_positive + false_positive + false_negative

#         if denominator == 0:
#             iou = 0  # Avoid division by zero
#         else:
#             iou = true_positive / denominator

#         class_iou.append(iou)
#     return class_iou

def standardymtx(A,var):
    F = []
    if var== '5d':
        for j in range(A.shape[0]):
            image = A[j,:,:,:,:]
            standardized_image = (image - image.min()) / (image.max() - image.min())
            F.append(standardized_image)
    elif var== '4ds':
        for j in range(A.shape[0]):
            image = A[j,:,:,:,0:4]
            standardized_image = (image - image.min()) / (image.max() - image.min())
            standardized_images = np.concatenate([standardized_image,np.expand_dims(A[j,:,:,:,4],axis=3)],axis=3)
            F.append(standardized_images)
    elif var== 'norms':
        for j in range(A.shape[0]):
            image = A[j,:,:,:,0:4].transpose(0,3,1,2)
            imagey = np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2],image.shape[3]))
            standardized_image = np.reshape(normalize(imagey,axis=0),(image.shape[0],image.shape[1],image.shape[2],image.shape[3])).transpose(0,2,3,1)
            standardized_images = np.concatenate([standardized_image,np.expand_dims(A[j,:,:,:,4],axis=3)],axis=3)
            F.append(standardized_images)
    elif var == '4d':
        for j in range(A.shape[0]):
            image = A[j,:,:,:]
            standardized_image = (image - image.min()) / (image.max() - image.min())
            F.append(standardized_image)
    return np.stack(F,axis=0)

def slice_by_slice_plot(Xtes_plot,ytes,y_pred,rho,j,var):
    for i in range(Xtes_plot.shape[0]):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle(var, fontsize=15)
        
        ax = axes[0,0]
        ax.imshow(Xtes_plot[i,:,:])
        ax.set_title("Monoculture Image")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[0,1]
        ax.imshow(ytes[i,:,:,j]>rho)
        ax.set_title("True Neuronal Mask")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[0,2]
        ax.imshow(((y_pred[i,:,:,j]>rho))*(Xtes_plot[i,:,:]))
        ax.set_title("Segmented Neurons")
        #ax.axis("off")
        ax.grid(True)
        
        
        ax = axes[0,3]
        ax.imshow(y_pred[i,:,:,j]>rho)
        ax.set_title("Predicted Neuronal Mask")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[1,0]
        ax.imshow(Xtes_plot[i,:,:])
        ax.set_title("Monoculture Image")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[1,1]
        ax.imshow(ytes[i,:,:,j+1]>rho)
        ax.set_title("True Silk Mask")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[1,2]
        ax.imshow(((y_pred[i,:,:,j+1]>rho))*(Xtes_plot[i,:,:]))
        ax.set_title("Segmented Silk")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[1,3]
        ax.imshow(y_pred[i,:,:,j+1]>rho)
        ax.set_title("Predicted Silk Mask")
        #ax.axis("off")
        ax.grid(True)
    
      
        plt.show()
        
        
        
        
from mpl_toolkits.axes_grid1 import make_axes_locatable
def slice_by_slice_plotty(Xtes_plot,ytes,y_pred,rho,nu,s,j,var):
    for i in range(Xtes_plot.shape[0]):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle(var, fontsize=15)
        
        ax = axes[0, 0]
        im = ax.imshow(Xtes_plot[i, :, :])
        ax.set_title("Aggregated Image")
        ax.axis("off")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Intensity')
        
        ax = axes[0,1]
        ax.imshow(ytes[i,:,:,j+1]>rho)
        ax.set_title("True Neuronal Mask")
        ax.axis("off")
        #ax.grid(True)
        
        # ax = axes[0,2]
        # ax.imshow(((y_pred[i,:,:,j+1]>rho))*(Xtes_plot[i,:,:]))
        # ax.set_title("Segmented Neurons")
        # #ax.axis("off")
        # ax.grid(True)
        
        ax = axes[0,2]
        ax.imshow((1-(y_pred[i,:,:,j]>rho))*(Xtes_plot[i,:,:]))
        ax.set_title("Segmented Neurons")
        ax.axis("off")
        #ax.grid(True)
        
        ax = axes[0,3]
        ax.imshow(y_pred[i,:,:,j+1]>rho)
        ax.set_title("Predicted Neuronal Mask")
        ax.axis("off")
        #ax.grid(True)
        
        ax = axes[1,0]
        ax.imshow((ytes[i,:,:,j+2]>rho))
        ax.set_title("True Glial Mask")
        ax.axis("off")
        #ax.grid(True)
        
        ax = axes[1,1]
        ax.imshow(y_pred[i,:,:,j+2]>nu)
        ax.set_title("Predicted Glial Mask")
        ax.axis("off")
        #ax.grid(True)
        
        ax = axes[1,2]
        ax.imshow(ytes[i,:,:,j]>rho)
        ax.set_title("True Silk Mask")
        ax.axis("off")
        #ax.grid(True)
    
        ax = axes[1,3]
        ax.imshow(y_pred[i,:,:,j]>s)
        ax.set_title("Predicted Silk Mask")
        ax.axis("off")
        #ax.grid(True)
    
      
        plt.show()
        
        
def slice_by_slice_plot_inj(Xtes_plot,y_pred,rho,s,j,var):
    for i in range(Xtes_plot.shape[0]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(var, fontsize=15)
        
        ax = axes[0,0]
        ax.imshow(Xtes_plot[i,:,:])
        ax.set_title("Monoculture Image")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[0,1]
        ax.imshow(((y_pred[i,:,:,j]>s))*(Xtes_plot[i,:,:]))
        ax.set_title("Segmented Neurons")
        #ax.axis("off")
        ax.grid(True)
        
        
        ax = axes[0,2]
        ax.imshow(y_pred[i,:,:,j]>s)
        ax.set_title("Predicted Neuronal Mask")
        #ax.axis("off")
        ax.grid(True)
        
        ax = axes[1,0]
        ax.imshow(Xtes_plot[i,:,:])
        ax.set_title("Monoculture Image")
        #ax.axis("off")
        ax.grid(True)
              
        ax = axes[1,1]
        ax.imshow(((y_pred[i,:,:,j+1]>rho))*(Xtes_plot[i,:,:]))
        ax.set_title("Segmented Silk")
        #ax.axis("off")
        ax.grid(True)
        
        
        ax = axes[1,2]
        ax.imshow(y_pred[i,:,:,j+1]>rho)
        ax.set_title("Predicted Silk Mask")
        #ax.axis("off")
        ax.grid(True)
   
        plt.show()
        
def slicy_plots(X,Xtes_plot,ytes_pred_plot,y_pred,rho,j,var):
    for i in range(Xtes_plot.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        
        ax = axes.flatten()[0]
        ax.imshow(Xtes_plot[i,:,:])
        #ax.set_title("NAM")
        #ax.axis("off")
        
        ax = axes.flatten()[1]
        ax.imshow(ytes_pred_plot[i,:,:,j]>rho)
        #ax.set_title(" Mask")
        #ax.axis("off")
            
        ax = axes.flatten()[2]
        ax.imshow(((y_pred[i,:,:,j]>rho))*(X[i,:,:,j]))
        #ax.set_title("NAM without Silk")
        #ax.axis("off")
        
        ax = axes.flatten()[3]
        ax.imshow(y_pred[i,:,:,j]>rho)
        #ax.set_title("Silk Mask")
        #ax.axis("off")
        
        ax = axes.flatten()[4]
        ax.imshow(X[i,:,:,j])
        #ax.set_title("Silk Mask")
        #ax.axis("off")
        
        plt.title(var,fontsize=15)
        plt.show()
        
        
def slice_by_slice_plots(raw_X_train,Xtes_plot,ytes_pred_plot,y_pred,rho,j):
    for i in range(Xtes_plot.shape[0]):
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes.grid(True)
        # ax = axes.flatten()[0]
        # ax.imshow(Xtes_plot[i,:,:])
        # #ax.set_title("NAM")
        # #ax.axis("off")
        
        ax = axes.flatten()[0]
        ax.imshow(raw_X_train[i,:,:])
        #ax.set_title("NAM")
        #ax.axis("off")
        
        ax = axes.flatten()[1]
        ax.imshow(ytes_pred_plot[i,:,:,j]>rho)
        #ax.set_title(" Mask")
        #ax.axis("off")
            
        ax = axes.flatten()[2]
        ax.imshow((1-(y_pred[i,:,:,j]>rho))*(raw_X_train[i,:,:]))
        #ax.set_title("NAM without Silk")
        #ax.axis("off")
        
        ax = axes.flatten()[3]
        ax.imshow(y_pred[i,:,:,j]>rho)
        #ax.set_title("Silk Mask")
        #ax.axis("off")

# def scores(y_pred,yval,n_classes):
#     IOU_keras = MeanIoU(num_classes=n_classes)  
#     IOU_keras.update_state(yval, y_pred)
#     mean_IOU = IOU_keras.result().numpy()
#     values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
#     background = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] +values[0,4] + values[1,0]+ values[2,0]+ values[3,0]+ values[4,0])
#     collagen = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
#     Astmic = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[0,2]+ values[1,2]+ values[3,2] + values[4,2])
#     silk = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] +values[3,4] + values[0,3]+ values[1,3]+ values[2,3]+ values[4,3])
#     neuron = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[0,4] + values[1,4]+ values[2,4]+values[3,4])
#     return background, collagen, Astmic,neuron, silk,mean_IOU

def scorey(y_pred,yval,n_classes):
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(yval, y_pred)
    mean_IOU = IOU_keras.result().numpy()
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    # Compute class-specific IoU ratios
    background = values[0, 0] / np.sum(values[0, :])
    collagen = values[1, 1] / np.sum(values[1, :])
    neuron = values[2, 2] / np.sum(values[2, :])
    silk = values[3, 3] / np.sum(values[3, :])
    return background, collagen, neuron,silk,mean_IOU

def scores(y_pred,yval,n_classes):
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(yval, y_pred)
    mean_IOU = IOU_keras.result().numpy()
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    # Compute class-specific IoU ratios
    background = values[0, 0] / np.sum(values[0, :])
    collagen = values[1, 1] / np.sum(values[1, :])
    silk = values[2, 2] / np.sum(values[2, :])
    neuron = values[3, 3] / np.sum(values[3, :])
    glial  = values[4, 4] / np.sum(values[4, :])
    return background, collagen, silk, neuron, glial, mean_IOU

def mastify(A,min_size):
    B = []
    for i in range(A.shape[0]):
        removed_circ = morphology.label(A[i,:,:])
        props = measure.regionprops(removed_circ)
        for prop in props:
            if prop.area < min_size:
               removed_circ[removed_circ == prop.label] = 0
        masker = removed_circ > 0
        B.append(masker)
    return np.stack(B,axis=0)

def stackme(A,B):
    C = []
    for i in range(A.shape[0]):
       C.append(np.concatenate((A[i,:,:,:],B[i,:,:,:]),axis=3))
    return np.stack(C,axis=0)


def standardy(A,var):
    F = []
    if var== '5d':
        for j in range(A.shape[0]):
            image = A[j,:,:,:,:]
            mean = np.mean(image)
            std = np.std(image)
            standardized_image = (image - mean) / std
            F.append(standardized_image)
    elif var == '4d':
        for j in range(A.shape[0]):
            image = A[j,:,:,:]
            mean = np.mean(image)
            std = np.std(image)
            standardized_image = (image - mean) / std
            F.append(standardized_image)
    return np.stack(F,axis=0)



def otsuy(A):
    D = []
    for i in range(A.shape[0]):
        B = A[i,:,:,:,:]   
        otsus2 = threshold_multiotsu(B[B > 0])
        threshold_im = otsus2[1]
        C = B*(1-((B > threshold_im)))
        D.append(C)
    return np.stack(D,axis=0) 
        
def katrina_masks(model, normalize_test_images,cat_masks): 
        X_test = np.expand_dims(normalize_test_images,3)
        pre_diction = model.predict(X_test)
        silk_mask = pre_diction[:,:,:,3] 
        silk_mask[silk_mask<0.5]=0
        silk_mask[silk_mask>=0.5]=1
        final_silk_nam_mask = np.zeros_like(silk_mask)
        for i in range(silk_mask.shape[0]):
            cat_silk_to_nams = np.stack([cat_masks[i,:,:],(2*silk_mask[i,:,:])],axis=0) #.astype(np.int64)
            final_silk_nam_mask[i,:,:] = np.max(cat_silk_to_nams,axis=0)
        return final_silk_nam_mask  
    
def matching_ctrl(A):
    ROI_masks = []
    groups = [(i, i + 10, i + 20, i + 30) for i in range(0, 10)]
    for j in range(0,A.shape[0],40):
        group_slices = A[j:j+40,:,:]
        group_slice = group_slices[groups].transpose(0, 2, 3, 1)
        ROI_masks.append(group_slice)
    return shapy(standardymtx(np.stack(ROI_masks,axis=0),'5d'))

def matching_inj(A):
    ROI_masks = []
    groups = [(i, i + 20, i + 40, i + 60) for i in range(0, 20)]
    for j in range(0,A.shape[0],80):
        group_slices = A[j:j+80,:,:]
        group_slice = group_slices[groups].transpose(0, 2, 3, 1)
        ROI_masks.append(group_slice)
    return shapy(standardymtx(np.stack(ROI_masks,axis=0),'5d'))

def matching_minj(A):
    ROI_masks = []
    groups = [(i, i + 20) for i in range(0, 20)]
    for j in range(0,A.shape[0],40):
        group_slices = A[j:j+40,:,:]
        group_slice = group_slices[groups].transpose(1,2,0)
        ROI_masks.append(group_slice)
    return np.stack(ROI_masks,axis=0)

def matching_slices(A,geta,eta):
    ROI_masks = [] 
    groups = [list(range(i, geta, eta)) for i in range(eta)]
    for j in range(0,A.shape[0],geta):
        group_slices = A[j:j+geta,:,:]
        group_slice = group_slices[groups].transpose(0, 2, 3, 1)
        ROI_masks.append(group_slice)
    return np.stack(ROI_masks,axis=0)

def matching_masks(A,geta,eta):
    ROI_masks = []
    for j in range(0,A.shape[0],geta):
        group_slice = A[j:j+geta,:,:][0:eta,:,:]
        ROI_masks.append(group_slice)
    return np.stack(ROI_masks,axis=0)

def matching_masking(A,eta):
    ROI_masks = []
    for j in range(0,A.shape[0],eta):
        group_slice = A[j:j+eta,:,:]
        ROI_masks.append(group_slice)
    return np.stack(ROI_masks,axis=0)

def masky3D(A):
    group = []
    num_groups = A.shape[0] // 12
    for i in range(num_groups):    
        group.append(np.max(A[i * 12 : (i + 1) * 12, :, :],axis=3))     
    return np.stack(group,axis=0)   

def maskyy(A):
    group = []
    for i in range(A.shape[0]):    
        group.append(np.max(A[i, :, :,:],axis=2))     
    return np.stack(group,axis=0)   

def patchete(A,patch_size,channel):
    G = []
    for i in range(A.shape[0]):  
        for j in range(A.shape[1]):
            patches = patchify(A[i,j,:,:,:],(patch_size,patch_size,channel),step=patch_size)
            G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5])))
    return np.concatenate(G,axis=0)

def patchetem(A,patch_size):
    G = []
    for i in range(A.shape[0]):  
        patches = patchify(A[i,:,:,:],(patch_size,patch_size,4),step=patch_size)
        G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5])))
    return np.concatenate(G,axis=0)

def patchetest(A,patch_size):
    G = []
    for i in range(A.shape[0]): 
        H = []
        for j in range(A.shape[4]):
            patches = patchify(A[i,:,:,:,j],(patch_size,patch_size,4),step=patch_size)
            H.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5])))
        Z = np.stack(H,axis=4)
        G.append(Z)
    return np.concatenate(G,axis=0)

def matchetest(A,patch_size):
    G = []
    for i in range(A.shape[0]):  
        patches = patchify(A[i,:,:],(patch_size,patch_size),step=patch_size)
        G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[2],patches.shape[3])))
    return np.concatenate(G,axis=0)

# def unpatchete(A):
#     groups = []
#     num_groups = A.shape[0] // 12
#     for i in range(num_groups):  
#         groupy = A[i*12:(i + 1)*12,:,:,:]
#         G = np.reshape(groupy,(3,4,256,256,A.shape[3]))
#         group = []
#         for j in range(3): 
#             patches = unpatchify(np.reshape(G[j,:,:,:,:],(2,2,1,256,256,A.shape[3])),(512, 512, A.shape[3]))
#             group.append(patches)
#         groups.append(np.stack(group,axis=0))
#     return shapy(np.stack(groups,axis=0))

def unpatchetey(A,patchsize):
    groups = []
    num_groups = A.shape[0] // 48
    for i in range(num_groups):  
        groupy = A[i*48:(i + 1)*48,:,:,:]
        G = np.reshape(groupy,(3,16,patchsize,patchsize,A.shape[3]))
        group = []
        for j in range(3): 
            patches = unpatchify(np.reshape(G[j,:,:,:,:],(4,4,1,patchsize,patchsize,A.shape[3])),(512, 512, A.shape[3]))
            group.append(patches)
        groups.append(np.stack(group,axis=0))
    return shapy(np.stack(groups,axis=0))


def matchete(A,patch_size):
    G = []
    #A = matching_masks(A)
    for i in range(A.shape[0]):  
        for j in range(A.shape[1]):
            patches = patchify(A[i,j,:,:],(patch_size,patch_size),step=patch_size)
            G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[2],patches.shape[3])))
    return np.concatenate(G,axis=0)

def patchy3D(A,patch_size):
    G = []
    A = reshapy(A)
    for i in range(A.shape[0]):     
        patches = patchify(A[i,:,:],(patch_size,patch_size,12),step=patch_size)
        G.append(np.reshape(patches,(patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],12)))
    return np.concatenate(G,axis=0)

def reshapy(A):  
    num_groups = A.shape[0] // 12
    C = np.zeros((num_groups, A.shape[1], A.shape[2], 12))
    for i in range(num_groups):    
        group = A[i * 12 : (i + 1) * 12, :, :]     
        for k in range(12):          
            C[i, :, :, k] = group[k, :, :]
    return C


    
def patchy(A,patch_size):
    G = []
    for i in range(A.shape[0]):
        patches = patchify(A[i,:,:],(patch_size,patch_size),step=patch_size)
        G.append(np.reshape(patches,(-1,patches.shape[2],patches.shape[3])))
    return np.concatenate(G,axis=0)

def plot_patchy(patchy_tr_images,patchy_tr_masks,n):
    #random_integers = [random.randint(0, patchy_tr_masks.shape[0]) for _ in range(n)]
    for i in range(patchy_tr_masks.shape[0]):
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Train Image')
        plt.imshow(patchy_tr_masks[i,:,:])
        plt.subplot(232)
        plt.title('Train Mask')
        plt.imshow(patchy_tr_images[i,:,:])
        
        
def display_transfered_images(training_images,divide_tr_images,cat_masks_w_silk,n):
    random_integers = [random.randint(0, training_images.shape[0]) for _ in range(n)]
    for i in random_integers:
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Train Image')
        plt.imshow(training_images[i,:,:])
        plt.subplot(232)
        plt.title('Predicted Mask')
        plt.imshow(cat_masks_w_silk[i,:,:])
        plt.subplot(233)
        plt.title('Predicted Mask')
        plt.imshow(divide_tr_images[i,:,:])

        

def extract_class(A):
    #A = np.stack((A[:,:,:,0],A[:,:,:,2],A[:,:,:,3]),axis=-1)
    #A = np.stack((A[:,:,:,2],A[:,:,:,3]),axis=-1)
    A = np.expand_dims(A[:,:,:,2],axis=3)
    return A

def train_test_val_gen(image_dataset,mask_dataset,test_size): 
        
       X_train_indices, X_val_indices, y_train_indices, y_val_indices = train_test_split(
           range(len(image_dataset)), range(len(mask_dataset)),test_size=test_size, random_state=42)
       X_train = np.stack([image_dataset[i] for i in X_train_indices],axis=0)
       X_val = np.stack([image_dataset[i] for i in X_val_indices],axis=0)
       y_train = np.stack([mask_dataset[i] for i in y_train_indices],axis=0)
       y_val = np.stack([mask_dataset[i] for i in y_val_indices],axis=0)
       return X_train, y_train, X_val, y_val

def generate_samples(image,masks,test_size):
    Xtrain,ytrain, Xval,yval = train_test_val_gen(image,masks,test_size)
    X_train = shapy(Xtrain)
    y_train = shapy(ytrain)
    X_val = shapy(Xval)
    y_val = shapy(yval)
    return X_train,y_train,X_val,y_val
    
# def get_model(h,w,c,var,**kwargs):
#     if var == 'basic':
#         return Unet(h,w,c)
#     elif var == 'multi':
#         n_classes = kwargs['n_classes']
#         return Multi_Unet(n_classes,h,w,c)
#     elif var == 'namet':
#         d = kwargs['d']
#         n_classes = kwargs['n_classes']
#         return Unet3D(n_classes,h,w,d,c)
#     elif var == 'enamet':
#         n_classes = kwargs['n_classes']
#         return enamet(n_classes,h,w,c)
#     # elif var == 'encode':
#     #     n_classes = kwargs['n_classes']
#     #     return encodem(n_classes,h,w,c)
#     elif var == 'multiscale':
#         n_classes = kwargs['n_classes']
#         return Scaling_Unet(n_classes,h,w,c)

def standardise(image):
    mean = np.mean(image)
    std = np.std(image)
    standardized_image = (image - mean) / std
    return standardized_image

def minmax(A):
    return (A - A.min()) / (A.max() - A.min())

def normalise_mtx(A,var,**kwargs):
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        if var == 'norm':
            axis = kwargs['axis']
            B[i,:,:] = normalize(A[i,:,:],axis=axis)
        elif var == 'mean':
            B[i,:,:] = standardise(A[i,:,:])
        elif var == 'minmax':
            B[i,:,:] = (A[i,:,:] - A[i,:,:].min()) / (A[i,:,:].max() - A[i,:,:].min())
    return np.stack(B,axis=0)

def normalise_3mtx(A,n_slices,var,**kwargs):
    B = []
    for i in range(0,A.shape[0],n_slices):
        group = A[i:i+n_slices,:,:] 
        if var == 'norm':
            axis = kwargs['axis']
            B.append(normalize(group,axis=axis))
        elif var == 'mean':
            B.append(standardise(group))
        elif var == 'minmax':
            B.append((group - group.min()) / (group.max() - group.min()))
        elif var == 'max':
            B.append(group/group.max())
    G = np.stack(B,axis=0)
    return np.reshape(G,(G.shape[0]*G.shape[1],G.shape[2],G.shape[3]))

def normalise_4mtx(A,var):
    C = np.zeros((A.shape[0],A.shape[1],A.shape[2],A.shape[3],A.shape[4]))
    for j in range(A.shape[0]):
        for i in range(A.shape[1]):
            group = A[j,i,:,:,:] 
            if var == 'norm':
                C[j,i,:,:,:] = normalize(group,axis=2)
            elif var == 'mean':
                C[j,i,:,:,:] = standardise(group)
            elif var == 'minmax':
                C[j,i,:,:,:] = (group - group.min()) / (group.max() - group.min())
    return C

def normalise_3_mtx(A,n_slices,var,**kwargs):
    B = []
    for i in range(0,A.shape[0],n_slices):
        group = A[i:i+n_slices,:,:,:] 
        if var == 'norm':
            axis = kwargs['axis']
            B.append(normalize(group,axis=axis))
        elif var == 'mean':
            B.append(standardise(group))
        elif var == 'minmax':
            B.append((group - group.min()) / (group.max() - group.min()))
        elif var == 'max':
            B.append(group/group.max())
    G = np.stack(B,axis=0)
    return G




def divisor4mtx(norm_A,A,var,**kwargs):
    A = matching_slices(A)
    if var == 'non_thresh':
        for s in range(A.shape[0]):
            h = norm_A/A
        h = np.nan_to_num(h,nan=1)
        return h
    elif var == 'thresh':
        thresh = kwargs['thresh']
        for s in range(A.shape[0]):
            h = norm_A/A
        h = np.nan_to_num(h,nan=1)
        h[h>thresh]=0
        return h*A
    


def divisory(norm_A,H,var,**kwargs):
    h = np.zeros_like(H)
    if var == 'non_thresh':
        for s in range(0,H.shape[0],12):
            A = H[s:s+12,:,:]
            B = np.concatenate([A[6:9,:,:]+A[9:12,:,:],A[6:9,:,:]+A[9:12,:,:],A[0:3,:,:]+A[3:6,:,:],A[0:3,:,:]+A[3:6,:,:]],axis=0)
            h[s:s+12,:,:] = norm_A[s:s+12,:,:]/B
        h[h == np.inf] = np.nan
        h = np.nan_to_num(h,nan=0)
        return h
    elif var == 'thresh':
        thresh = kwargs['thresh']
        for s in range(0,H.shape[0],12):
            A = H[s:s+12,:,:]
            B = np.concatenate([A[6:9,:,:]+A[9:12,:,:],A[6:9,:,:]+A[9:12,:,:],A[0:3,:,:]+A[3:6,:,:],A[0:3,:,:]+A[3:6,:,:]],axis=0)
            h[s:s+12,:,:] = norm_A[s:s+12,:,:]/B
        h[h == np.inf] = np.nan
        h = np.nan_to_num(h,nan=0)
        h[h>thresh]=0
        return h*H
    



def divisor(norm_A,A,var,**kwargs):
    h = np.zeros_like(A)
    if var == 'non_thresh':
        for s in range(A.shape[0]):
            h[s,:,:] = norm_A[s,:,:]/A[s,:,:]
        h[h == np.inf] = np.nan # added 6/5/24
        h = np.nan_to_num(h,nan=0)
        return h
    elif var == 'thresh':
        thresh = kwargs['thresh']
        for s in range(A.shape[0]):
            h[s,:,:] = norm_A[s,:,:]/A[s,:,:]
        h[h == np.inf] = np.nan
        h = np.nan_to_num(h,nan=0)
        h[h>thresh]=0
        return h*A
    # plt.imshow(h, cmap='viridis')
    # plt.show()
    # plt.imshow(tr_masks[s,:,:], cmap='viridis')
    # plt.show()
    # plt.imshow(tr_images[s,:,:], cmap='viridis')
    # plt.show()
    # plt.imshow(h*tr_images[s,:,:], cmap='viridis')
    # plt.show()
    

def combine_slices(A):
    ROI_masks = []
    groups = [(0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11)] 
    for j in range(0,A.shape[0],12):
        argmax_result = np.empty((len(groups), 512, 512).astype(np.int64))
        group_slices = A[j:j+12,:,:]
        for i, group in enumerate(groups):
            group_slice = group_slices[group]
            argmax_result[i] = np.argmax(group_slice,axis=0)
        ROI_masks.append(argmax_result)
    return np.stack(ROI_masks,axis=0)




def combine_add_slices(A):
    ROI_masks = []
    for i in range(0,A.shape[0],12):
        group = A[i:i+12,:,:]
        group_slice = np.zeros((3,A.shape[1],A.shape[1]))
        for j in range(0,12,3):
            group_slice += group[j:j+3,:,:]
        ROI_masks.append(group_slice)
    return np.stack(ROI_masks,axis=0)[0,:,:,:]
            
   
def catty_masks(model, normalize_test_images,cat_masks): 
        X_test = np.expand_dims(normalize_test_images,3)
        pre_diction = model.predict(X_test)
        silk_mask = pre_diction[:,:,:,3] 
        silk_mask[silk_mask<0.5]=0
        silk_mask[silk_mask>=0.5]=1
        final_silk_nam_mask = np.zeros_like(silk_mask)
        for i in range(silk_mask.shape[0]):
            cat_silk_to_nams = np.stack([cat_masks[i,:,:],(2*silk_mask[i,:,:])],axis=0) #.astype(np.int64)
            final_silk_nam_mask[i,:,:] = np.max(cat_silk_to_nams,axis=0)
        return final_silk_nam_mask  
    
def catrify(cat_masks, silk_mask): 
        final_silk_nam_mask = np.zeros_like(cat_masks)
        for i in range(silk_mask.shape[0]):
            cat_silk_to_nams = np.stack([cat_masks[i,:,:],silk_mask[i,:,:]],axis=0) #.astype(np.int64)
            final_silk_nam_mask[i,:,:] = np.max(cat_silk_to_nams,axis=0)
        return final_silk_nam_mask  
    
  
def aggregate_masks(model,cat_images,normalize_test_images,cat_masks):
        X_test = np.expand_dims(normalize_test_images,3)
        pre_diction = model.predict(X_test)
        silk_mask = pre_diction[:,:,:,3]
        silk_mask[silk_mask<=0.5]=0
        silk_mask[silk_mask>0.5]=1
        mask = np.zeros_like(silk_mask)
        for i in range(silk_mask.shape[0]):
            cat_silk_to_nams = np.stack([cat_masks[i,:,:],(2*silk_mask[i,:,:]).astype(np.int64)],axis=0)
            mask[i,:,:] = np.max(cat_silk_to_nams,axis=0)
        silk_mask_only = np.where((mask == 0) | (mask == 1) | (mask == 3) | (mask == 4) | (mask == 5), 0, 1)
        silk_free_image = np.zeros_like(silk_mask)
        for j in range(0,silk_mask.shape[0],12):
            group = silk_mask_only[j:j+12,:,:]
            final_silk_mask = np.max(group,axis=0)
            for k in range(group.shape[0]):
                silk_free_image[j+k,:,:] = cat_images[j+k,:,:]*255*(1-final_silk_mask)
        return silk_free_image
    
def silk_free_image(model,cat_images,cat_masks):
        normalize_test_images = normalize(cat_images,axis=0)
        X_test = np.expand_dims(normalize_test_images,3)
        pre_diction = model.predict(X_test)
        silk_mask = pre_diction[:,:,:,3]
        silk_mask[silk_mask<=0.5]=0
        silk_mask[silk_mask>0.5]=1
        mask = np.zeros_like(silk_mask)
        for i in range(silk_mask.shape[0]):
            cat_silk_to_nams = np.stack([cat_masks[i,:,:],(2*silk_mask[i,:,:]).astype(np.int64)],axis=0)
            mask[i,:,:] = np.max(cat_silk_to_nams,axis=0)
        silk_mask_only = np.where((mask == 0) | (mask == 1) | (mask == 3) | (mask == 4) | (mask == 5), 0, 1)
        return cat_images*255*(1-silk_mask_only)
    
    
def trans(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:,:] = x[i,:,:].T
    return y
        

def remove_slices(cat_images,cat_masks,var):
    removed_masks = np.empty((0, 512, 512), dtype=cat_masks.dtype)
    removed_images = np.empty((0, 512, 512), dtype=cat_images.dtype)
    removed_indices = []
    if var == 'mono':
        for i in range(cat_masks.shape[0] // 12):
            start_index = i * 12
            end_index = (i + 1) * 12
            mask_group = cat_masks[start_index:end_index, :, :]
            if any(len(np.unique(mask_group[k, :, :])) != 4 for k in range(12)):
                #print(f'Removing group {start_index}:{end_index-1}')
                removed_indices.extend(range(start_index, end_index))
                removed_masks = np.concatenate((removed_masks, cat_masks[start_index:end_index, :, :]), axis=0)
                removed_images = np.concatenate((removed_images, cat_images[start_index:end_index, :, :]), axis=0)
        cat_masks = np.delete(cat_masks, removed_indices, axis=0)
        cat_images = np.delete(cat_images, removed_indices, axis=0)
        for i in range(removed_masks.shape[0] // 12):
              start_index = i * 12
              end_index = (i + 1) * 12
              masks_group =  removed_masks[start_index:end_index, :, :]
              if np.any(np.max(masks_group, axis=(1, 2)) != 3):
                  removed_masks = np.delete(removed_masks, range(start_index, end_index), axis=0)
                  removed_images = np.delete(removed_images, range(start_index, end_index), axis=0)
                       
    elif var == 'tri':
        for i in range(cat_masks.shape[0] // 12):
            start_index = i * 12
            end_index = (i + 1) * 12
            mask_group = cat_masks[start_index:end_index, :, :]
            if any(len(np.unique(mask_group[k, :, :])) != 6 for k in range(12)):
                #print(f'Removing group {start_index}:{end_index-1}')
                removed_indices.extend(range(start_index, end_index))
                removed_masks = np.concatenate((removed_masks, cat_masks[start_index:end_index, :, :]), axis=0)
                removed_images = np.concatenate((removed_images, cat_images[start_index:end_index, :, :]), axis=0)
        cat_masks = np.delete(cat_masks, removed_indices, axis=0)
        cat_images = np.delete(cat_images, removed_indices, axis=0)
        
    elif var == 'trima':
        for i in range(cat_masks.shape[0] // 12):
            start_index = i * 12
            end_index = (i + 1) * 12
            mask_group = cat_masks[start_index:end_index, :, :]
            if any(len(np.unique(mask_group[k, :, :])) != 5 for k in range(12)):
                #print(f'Removing group {start_index}:{end_index-1}')
                removed_indices.extend(range(start_index, end_index))
                removed_masks = np.concatenate((removed_masks, cat_masks[start_index:end_index, :, :]), axis=0)
                removed_images = np.concatenate((removed_images, cat_images[start_index:end_index, :, :]), axis=0)
        cat_masks = np.delete(cat_masks, removed_indices, axis=0)
        cat_images = np.delete(cat_images, removed_indices, axis=0)
           
    return cat_images, cat_masks, removed_images, removed_masks

def train_test_val(model,train_images,train_masks,var):
    train_masks_w_silk = catty_masks(model,train_images,train_masks)   
    train_images_removed,train_masks_removed,removed_train_images,removed_train_masks = remove_slices(train_images,train_masks_w_silk,var)
    normalize_cat_images = normalize(train_images_removed,axis=1)
    X_train = np.expand_dims(normalize_cat_images,3)
    y_train = np.expand_dims(train_masks_removed,3)
    return X_train, y_train, removed_train_images,removed_train_masks


# def sequester_plot(Xval,X_vl,y_pred,y_val,rho):
#     for i in range(y_pred.shape[0]):
#         fig, axes = plt.subplots(2, 5, figsize=(15, 10))
#         axes[0,0].imshow(X_vl[i,:,:], cmap='viridis')
#         axes[0,0].set_title('Normalized True Image')
#         axes[0,1].imshow(y_pred[i,:,:,3]>rho, cmap='viridis')
#         axes[0,1].set_title('Predicted Neurons Mask')
#         axes[0,2].imshow(y_pred[i,:,:,4]>rho, cmap='viridis')
#         axes[0,2].set_title('Predicted Microglia Mask')
#         axes[0,3].imshow(y_pred[i,:,:,5]>rho, cmap='viridis')
#         axes[0,3].set_title('Predicted Astrocytes Mask')
#         axes[0,4].imshow(y_pred[i,:,:,2]>rho, cmap='viridis')
#         axes[0,4].set_title('Predicted Silk Mask')
        
#         axes[1,0].imshow(Xval[i,:,:], cmap='viridis')
#         axes[1,0].set_title('True Image')
#         axes[1,1].imshow(y_val[i,:,:,3], cmap='viridis')
#         axes[1,1].set_title('Neurons Mask')
#         axes[1,2].imshow(y_val[i,:,:,4], cmap='viridis')
#         axes[1,2].set_title('Microglia Mask')
#         axes[1,3].imshow(y_val[i,:,:,5], cmap='viridis')
#         axes[1,3].set_title('Astrocytes Mask')
#         axes[1,4].imshow(y_val[i,:,:,2], cmap='viridis')
#         axes[1,4].set_title('Silk Mask')
#         plt.show()
        
# def sequester_plot(Xval,X_vl,y_pred,y_val,rho):
#     for i in range(y_pred.shape[0]):
        
#         fig, axe = plt.subplots(1, 2, figsize=(15, 10))
#         axe[0].imshow(Xval[i,:,:], cmap='viridis')
#         axe[0].set_title('True Image')
#         axe[1].imshow(X_vl[i,:,:], cmap='viridis')
#         axe[1].set_title('Normalized True Image')
        
#         fig, axes = plt.subplots(3, 4, figsize=(15, 10))
#         axes[0,0].imshow(y_val[i,:,:,3], cmap='viridis')
#         axes[0,0].set_title('Neurons Mask')
#         axes[0,1].imshow(y_pred[i,:,:,3]>0.3, cmap='viridis')
#         axes[0,1].set_title('Predicted Neurons Mask')
#         axes[0,2].imshow(y_val[i,:,:,3]*Xval[i,:,:], cmap='viridis')
#         axes[0,2].set_title('Neurons')
#         axes[0,3].imshow((y_pred[i,:,:,3]>0.3)*Xval[i,:,:], cmap='viridis')
#         axes[0,3].set_title('Predicted Neurons')
        
#         axes[1,0].imshow(y_val[i,:,:,4], cmap='viridis')
#         axes[1,0].set_title('Microglia Mask')
#         axes[1,1].imshow(y_pred[i,:,:,4]>rho, cmap='viridis')
#         axes[1,1].set_title('Predicted Microglia Mask')
#         axes[1,2].imshow(y_val[i,:,:,4]*Xval[i,:,:], cmap='viridis')
#         axes[1,2].set_title('Microglia')
#         axes[1,3].imshow((y_pred[i,:,:,4]>rho)*Xval[i,:,:], cmap='viridis')
#         axes[1,3].set_title('Predicted Microglia')
             
#         axes[2,0].imshow(y_val[i,:,:,5], cmap='viridis')
#         axes[2,0].set_title('Astrocytes Mask')
#         axes[2,1].imshow(y_pred[i,:,:,5]>rho, cmap='viridis')
#         axes[2,1].set_title('Predicted Astrocytes Mask')
#         axes[2,2].imshow(y_val[i,:,:,5]*Xval[i,:,:], cmap='viridis')
#         axes[2,2].set_title('Astrocytes')
#         axes[2,3].imshow((y_pred[i,:,:,5]>rho)*Xval[i,:,:], cmap='viridis')
#         axes[2,3].set_title('Predicted Astrocytes')
                      
#         plt.show()
        
def sequester_plot(Xval,y_pred,y_val,rho):
    for i in range(y_pred.shape[0]):
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes[0,0].imshow(y_val[i,:,:,3])
        axes[0,0].set_title('Neurons Mask')
        axes[0,1].imshow(y_pred[i,:,:,3]>0.3)
        axes[0,1].set_title('Predicted Neurons Mask')
        axes[0,2].imshow((y_val[i,:,:,3]>0)*np.sum(Xval[i,:,:,:],axis=2))
        axes[0,2].set_title('Neurons')
        axes[0,3].imshow(np.sum(Xval[i,:,:,:],axis=2))
        axes[0,3].set_title('Predicted Neurons')
        
        axes[1,0].imshow(y_val[i,:,:,4])
        axes[1,0].set_title('Microglia Mask')
        axes[1,1].imshow(y_pred[i,:,:,4]>rho)
        axes[1,1].set_title('Predicted Microglia Mask')
        axes[1,2].imshow((y_val[i,:,:,4]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[1,2].set_title('Microglia')
        axes[1,3].imshow((y_pred[i,:,:,4]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[1,3].set_title('Predicted Microglia')
             
        axes[2,0].imshow(y_val[i,:,:,5], cmap='viridis')
        axes[2,0].set_title('Astrocytes Mask')
        axes[2,1].imshow(y_pred[i,:,:,5]>rho, cmap='viridis')
        axes[2,1].set_title('Predicted Astrocytes Mask')
        axes[2,2].imshow((y_val[i,:,:,5]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[2,2].set_title('Astrocytes')
        axes[2,3].imshow((y_pred[i,:,:,5]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[2,3].set_title('Predicted Astrocytes')
                      
        plt.show()
        
def sequestery_plot(Xval,y_pred,y_val,rho):
    for i in range(y_pred.shape[0]):
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes[0,0].imshow(y_val[i,:,:,1])
        axes[0,0].set_title('Neurons Mask')
        axes[0,1].imshow(y_pred[i,:,:,2]>0.05)
        axes[0,1].set_title('Predicted Neurons Mask')
        axes[0,2].imshow((y_val[i,:,:,3]>0)*np.sum(Xval[i,:,:,:],axis=2))
        axes[0,2].set_title('Neurons')
        axes[0,3].imshow(np.sum(Xval[i,:,:,:],axis=2))
        axes[0,3].set_title('Predicted Neurons')
        
        axes[1,0].imshow(y_val[i,:,:,4])
        axes[1,0].set_title('Microglia Mask')
        axes[1,1].imshow(y_pred[i,:,:,4]>rho)
        axes[1,1].set_title('Predicted Microglia Mask')
        axes[1,2].imshow((y_val[i,:,:,4]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[1,2].set_title('Microglia')
        axes[1,3].imshow((y_pred[i,:,:,4]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[1,3].set_title('Predicted Microglia')
             
        axes[2,0].imshow(y_val[i,:,:,5], cmap='viridis')
        axes[2,0].set_title('Astrocytes Mask')
        axes[2,1].imshow(y_pred[i,:,:,5]>rho, cmap='viridis')
        axes[2,1].set_title('Predicted Astrocytes Mask')
        axes[2,2].imshow((y_val[i,:,:,5]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[2,2].set_title('Astrocytes')
        axes[2,3].imshow((y_pred[i,:,:,5]>rho)*np.sum(Xval[i,:,:,:],axis=2))
        axes[2,3].set_title('Predicted Astrocytes')
                      
        plt.show()
        
        
       
        
def tensorry(A):  
    num_groups = A.shape[0] // 12
    C = []
    for i in range(num_groups):    
        group = A[i * 12 : (i + 1) * 12, :, :]     
        core, factors = tucker(group, rank=(12,12,12),tol=10e-5,random_state=42,init='random') #group.shape[0],512,512
        C.append(tl.tucker_to_tensor((core, factors)))
    return np.stack(C,axis=0).reshape((A.shape[0],A.shape[1],A.shape[2]))
