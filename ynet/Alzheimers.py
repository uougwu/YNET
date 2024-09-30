import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import os
import napari
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel
import sys
import h5py
from patchify import patchify, unpatchify

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

def unpatchete(A):
    groups = []
    num_groups = A.shape[0] // 12
    for i in range(num_groups):  
        groupy = A[i*12:(i + 1)*12,:,:,:]
        G = np.reshape(groupy,(3,4,256,256,A.shape[3]))
        group = []
        for j in range(3): 
            patches = unpatchify(np.reshape(G[j,:,:,:,:],(2,2,1,256,256,A.shape[3])),(512, 512, A.shape[3]))
            group.append(patches)
        groups.append(np.stack(group,axis=0))
    return shapy(np.stack(groups,axis=0))



def plot_base_tri(Xtes, y_pred, nu, var, custom_name):
    maska = np.sum(Xtes.clip(0.0001, 0.09), axis=3)
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
            ax.set_title("Aggregated Image", fontsize=15)
            ax.axis("on")
            ax.grid(True)

            ax = axes[1]
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
            np.save(save_path, corrected_stack)
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


def process_matfiles(var,patches):
    e = 'EX745_845' 
    f  = 'index'
    
    infected = h5py.File(var,'a')   
    data = trans(np.array(infected.get(e)))
    
    indices = [x*4 for x in np.array(infected.get(f)).astype(int).tolist()[0]]
    extracted_images = []
    extracted_imagery = []
    image_array = data.copy()
    start_index = 0
    if patches == 1:
        for index in indices:
            z = image_array[start_index:start_index+index, :, :]
            extracted = np.stack([z[i:i+(index//4), :, :] for i in range(0, z.shape[0], index//4)], axis=-1)
            normalized_img = minmax(np.expand_dims(extracted,axis=0))
            extracted_imagery.append(normalized_img)
            extracted_images.append(patchete(np.expand_dims(normalized_img,axis=0),256,4))
            start_index += index
    original_shape = np.concatenate(extracted_imagery,axis=0).shape
    return np.concatenate(extracted_images,axis=0),original_shape



def minmax(A):
    return (A - A.min()) / (A.max() - A.min())