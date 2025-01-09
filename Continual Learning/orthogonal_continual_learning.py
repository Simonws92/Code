#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:28:08 2024

@author: simon
"""

import numpy as np
import struct
from array import array
from os.path  import join

def gs( V ):
    "Gram-Schmidt"
    U = np.random.random( np.shape(V) )
    
    for i in range(np.shape(V)[1]):
        
        u = U[:,i]
        v = V[:,i]
        
        if v.T @ v != 0:
            U[:,i] = u - ( ( v @ u) / (v.T@v)) * v
        else:
            U[:,i] = np.zeros( np.shape(U)[0] )
    
    return U




#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test) 



import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = '/home/simon/Desktop/mnist/archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


#
# Load MINST dataset
#
if "x_train" in dir(locals) and "y_train" in dir(locals):
    pass
else:
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


p = 7 # batch size
X = np.array( x_train[:p] ) 

"Task A"
X = np.reshape(X, (28*28, p)) / 256
Y = X.copy()

"Task B"
X_B = x_train[p:2*p]
X_B = np.reshape( X_B, (28*28, p)) / 256
Y_B = X_B.copy()


n,ne = 28*28, 28*28 # input, output dimensions
H = [n, 200, ne]    # network architecture

W  = np.random.rand( H[1], H[0] ) / np.sqrt(H[1])
WL = np.random.rand( H[2], H[1] ) / np.sqrt(H[2])


beta = 0.
LR = 0.0003
alpha = 1
loss = []
loss_B = []
rec_1 = []

C = 10 # initial


for i in range(500):
    "Forward Task A"
    "Hidden layer"
    zH = np.dot( W,X )
    aH = np.tanh( zH )
    "Output layer"
    zL = np.dot( WL,aH )
    "Task A loss"
    pre_C = zL - Y
    C = pre_C**2
    
    
    "Forward Task B"
    zH_B = np.dot( W,X_B )
    aH_B = np.tanh( zH_B )
    zL_B = np.dot( WL,aH_B )
    "Task B loss"
    pre_C_B = zL_B - Y_B
    C_B = pre_C_B**2
    
    #def compute_gradient(W,WL,pre_C, DC_DZL, )
    
    
    "TASK A: Backprop WL"
    DC_DZL =  2*pre_C
    DC_DWL = DC_DZL @ aH.T
    
    "W"
    DC_DZ = ( WL.T @ DC_DZL ) * (1-np.tanh(zH/5)**2)
    DC_DW = DC_DZ @ X.T # Derivative of C at Z in the direction of X
    
    
    "TASK B: Backprop WL"
    DC_DZL_B =  2*pre_C_B
    DC_DWL_B = DC_DZL_B @ aH_B.T
    
    DC_DZ_B = ( WL.T @ DC_DZL_B ) * (1-np.tanh(zH_B/5)**2)
    DC_DW_B = DC_DZ_B @ X_B.T # Derivative of C at Z in the direction of X
    
    
    
    N  = gs( DC_DW  )  # The orthogonal vectors of X. This derivative is accumulated for as long as we're not converged
    NL = gs( DC_DWL )
    
    N_B  = gs( DC_DW_B  )  # The orthogonal vectors of X. This derivative is accumulated for as long as we're not converged
    NL_B = gs( DC_DWL_B )
    
    "Compare loss with random generated N"
    #N  = np.random.random(( np.shape(DC_DW) ))
    #NL = np.random.random(( np.shape(DC_DWL) ))
    
    """
    The other derivative N is a directional derivative along the orthgonal vector wrt X.
    This causes the update of W to orbit around the local minima of the cost function.
    while orbiting. the orbit N accumulates untill it is launched out of the local minimas 'gravitational pull',
    this acts as a soft reset and the accumulation is reduced rather than 
    """
    
    ORTH = True
    
    "Update"
    #if ORTH == True:
    if np.mean(C) < 0.01:   
        
        LR = 0.00003
        
        I_ = np.ones(np.shape(N_B)) / (28**2)
        #GRAD   = (N_B @ I_.T) @ DC_DW
        
        N_B_sq = N_B @ N_B.T
        GRAD   = N_B_sq @ DC_DW
        
        #DC_DZ_B = ( WL.T @ DC_DZL_B ) * (1-np.tanh(zH_B/5)**2)
        #DC_DW_B = DC_DZ_B @ X_B.T # Derivative of C at Z in the direction of X
        
        
        
        #IL_ = np.ones(np.shape(NL_B)) / (28**2)
        
        GRAD_L = (NL @ NL.T) @  DC_DWL * 0.1
        # GRAD_L =  DC_DWL
        #GRAD_L /= GRAD_L.max()
    else:
        LR = 0.0003
        
        GRAD   = DC_DW
        GRAD_L = DC_DWL
    
    if i == 0:
        W_mom  = GRAD
        WL_mom = GRAD_L
    else:        
        W_mom  = (1-beta) * GRAD   + beta * W_mom
        WL_mom = (1-beta) * GRAD_L + beta * WL_mom
    
    W  = W  - W_mom * LR
    WL = WL - WL_mom * LR
    
    rec_1.append( W[:,50] )
    
    Loss = np.mean(C)
    Loss_B = np.mean(C_B)
    if i % 10 == 0:
        print(Loss, Loss_B, np.mean(GRAD))
    
    loss.append( np.mean(C) )
    loss_B.append( Loss_B )

"Display some weights to see how they are changing"
REC = np.array( rec_1 )
#plt.plot(REC.T)
#plt.plot(loss)

plt.plot(loss)
plt.plot(loss_B)
