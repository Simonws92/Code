# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:29:50 2023

@author: Simon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
import random

import pandas as pd
import csv
import pickle

import scania_models as sm


##############################################
# Separate file
# Convert dataset to usable input data for deep learning models...
def load_images(name):
    
    with open(name, "rb") as file:
        images = pickle.load(file)
    
    return images

print("Preparing data...")
DATA = load_images("compiled_images_merged")
target_DATA = load_images("target_merged")


X_images=[]
Y_target=[]
for i in range(len(DATA)):
    X_images += DATA[i]
    Y_target += target_DATA[i]

X_array = torch.tensor(X_images, dtype = torch.float)
Y_array = torch.tensor(Y_target, dtype = torch.float)

"batches, channels, row, col"
X_array = torch.permute(X_array, (0, 3, 1,2))
Y_array = torch.permute(Y_array, ( 0, 1 ))

batch_size = 64
batches = int(len(X_images) / batch_size)
NUM_CAT = np.shape(Y_array)[1] #number of categories: rain, cloud... etc

################################################

def load_model(model,PATH):
    "The model must be initialized beforehand"
    model.load_state_dict(torch.load(PATH))
    return model


def shuffle_data(X,Y, batch_size):
    "shape of X: channels,row,col,batches "
    "shape of Y: one_hot_vector of length 5, batches"
    
    "We shuffle the dataset before training"
    r = [ i for i in range( batch_size ) ]
    
    random.shuffle(r)
    
    data_shuffled = X[r,:,:,:]
    data_target   = Y[r,:]
    
    data_shuffled = torch.tensor(data_shuffled, dtype = torch.float)
    data_target   = torch.tensor(data_target  , dtype = torch.float)
    
    return data_shuffled, data_target


if True:
    
    "Model must be initiated before training/pretraining can start"
    print("Initiating models")
    #CV_model = sm.model(NUM_CAT)
    CV_model = sm.resnet(NUM_CAT)
    
    "If pretrained model exists, we may use them instead"
    RESET = False
    
    "Decide if trained model should replace old model"
    REPLACE_MODEL = True
    
    "Model weights are loaded for pretrained purposes"
    if RESET == False:
        PATH = "C:/Users/Simon/Desktop/CV_resnet_model_1"
        CV_model = load_model(CV_model,PATH)
        
    Loss = []
    avg_Loss = []
    
    "Low momentum due to high oscilation from Adam optimizer"
    b1=0.6
    b2=0.999
    optimizer = torch.optim.Adam(CV_model.parameters(), lr=0.00001 ,betas=(b1,b2))
    
    "https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/"
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=30)
    
    
    
    ### Initiate loss function ###
    mse = torch.nn.MSELoss()
    

    epochs = 100
    Confusion_matrix = np.zeros((NUM_CAT,NUM_CAT))
    
    
    ### Training Start ###
    for e in range(epochs):
        #print("Epoch:", e)
        
        "Data is shuffled at every epoch"
        X,Y = shuffle_data(X_array,Y_array, len(X_images) )
        
        epoch_avg_loss = 0

        for i in range(batches+1):
            
            if i == batches:
                "The final batch must not be forgotten"
                x = X[ i*batch_size: , :,:,:]
                y = Y[ i*batch_size: , :]
            else:
                x = X[ i*batch_size : (i+1)*batch_size , :,:,:  ]
                y = Y[ i*batch_size : (i+1)*batch_size , : ]
            
            
            
            "required input shape: (batch,ch,row,col)"
            output = CV_model.forward(x)
            
            loss = mse(output, y)
            loss_np = loss.detach().numpy() 
            Loss.append(loss_np )
            loss.backward()
            
            epoch_avg_loss += loss_np / batches
            "Update weight values"
            optimizer.step()
            scheduler.step()
            
            "create confusion matrix on the final epoch"
            if e == epochs-1:
                # column: prediction
                # rows:   actual
                
                classification = output.clone()
                cold_vector = classification.max(axis=1)[1]
                hot_vector = torch.nn.functional.one_hot(cold_vector,num_classes=NUM_CAT)
                
                for k in range( len(y) ):
                    index_correct = torch.max(y[k], axis=0)[1]
                    index_predict = torch.max(hot_vector[k],axis=0)[1]

                    Confusion_matrix[index_correct,index_predict]+=1
                    
        print("Avg epoch loss:", epoch_avg_loss, e)
        avg_Loss.append( epoch_avg_loss )



#plt.plot(Loss)
plt.plot(avg_Loss)
#plt.figure()

"test image"
# x_plot = torch.permute(x[1], (1, 2, 0))
# plt.imshow(x_plot.detach().numpy())


"https://stackoverflow.com/questions/40887753/display-matrix-values-and-colormap"
"Classifications: ['cloudy','foggy','rainy','shine','snow'] "
"Shows confusion matrix with values"
fig, ax = plt.subplots()
ax.matshow(Confusion_matrix)
for i in range(NUM_CAT):
    for j in range(NUM_CAT):
        c = Confusion_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')



"Number of correct classifications"
accuracy = np.sum( np.diag(Confusion_matrix) ) / len(X)
print(accuracy)


"Saves the model"
def save_model(model,PATH):
    torch.save( model.state_dict(), PATH )

if REPLACE_MODEL == True:
    PATH = "C:/Users/Simon/Desktop/CV_resnet_model_1"
    save_model(CV_model, PATH)

























