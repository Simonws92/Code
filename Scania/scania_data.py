# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:19:49 2023

@author: Simon
"""


import numpy as np
import torch
import torch.nn as nn

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import os
import time

import pandas as pd
import csv
import pickle


" Import images and saves them as a dataset "
def resize(img, max_dim=100):
        
        "All images must have three dimensions of shape: 100,100,3"
        if len(np.shape(img)) == 2:
            
            row,col = np.shape(img)
            empty_placeholder = np.zeros((row,col,3))
            empty_placeholder[:,:,0] = img
            img = np.reshape(empty_placeholder, (row,col,3) )
        
        if np.shape(img)[-1]>3:
            img = img[:,:,:3]
            
            
        row,col,c = np.shape(img) # Y,X values
        
        "Rescale images to lower resolutions of maximum 100pixels per max_dimension"
        
        """
        dim_ratio = ( max(row,col) / max_dim )
        if row > col:
            new_col   = int(col/dim_ratio)
            new_row   = max_dim
        else:
            new_row   = int(row/dim_ratio)
            new_col   = max_dim
        
        
        "Divide by 256 for color representation in decimal form"
        img_re = cv2.resize(img, ( new_col , new_row )) / 256 # X,Y values
        """
        
        img_re = cv2.resize(img, ( 100 , 100 )) / 256 # X,Y values
        
        "cv2 inverts color. We must sometimes swap them from BGR to RGB"
        #img_re = np.flip(img_re, axis=-1)
        img_re = np.array(img_re)
        
        return img_re
        
    
        


def preprocess_images(main_PATH=None):
    
    "Training images and target category"
    IMAGES = []
    TARGET = []
    
    FOLDERS = os.listdir(main_PATH)
    
    #FOLDERS = ['cloudy','foggy','rainy','shine','sunrise']
    
    
    "number of categories"
    one_hot = np.zeros(len(FOLDERS)) 
    
    for j,folder in enumerate(FOLDERS):
        
        one_hot = np.zeros(len(FOLDERS))
        one_hot[j] = 1
        
        temp = []
        temp_target = []
        
        folder_PATH = main_PATH + "{}".format(folder) + "/"
        #folder_PATH = "C:/Users/Simon/downloads/scania_dataset/dataset/{}/".format(folder)
    
        for i,file in enumerate(os.listdir(folder_PATH)):
            
            file_PATH = folder_PATH + file
            img = Image.open(file_PATH)
            
            #print(file)
            img    = resize( np.array(img) )
            target = one_hot
            
            
            temp.append( img )
            temp_target.append( target )
            
        IMAGES.append(temp)
        TARGET.append(temp_target)
                
        print(i)
        del i #avoid conflicts with new for loops
        
    return IMAGES, TARGET





def save_images(Images, name):
    
    with open(name, "wb") as file:
        pickle.dump(Images, file)


medium_PATH = "C:/Users/Simon/downloads/scania_dataset/Task_1/dataset/"
#big_PATH    = "C:/Users/Simon/downloads/scania_dataset/Task_1/dataset_big/"

compiled_images = "compiled_images_merged"
target          = "target_merged"



Images, Target = preprocess_images(medium_PATH)

save_images(Images, compiled_images )
save_images(Target, target )
