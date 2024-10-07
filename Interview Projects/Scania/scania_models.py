# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:32:02 2023

@author: Simon
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
import random

import pandas as pd
import csv
import pickle



"loss 0.035, ~90% accuracy on medium dataset, 300 epochs, lr 0.0001, b1 0.6"
"Simple conv -> FC model"
class model(nn.Module):
    def __init__(self, NUM_CAT):
        super(model, self).__init__()
        
        k = 5
        p = int((k - 1)/2)
        final_output_dim = 3 # final conv have 3 channels to represent RGB
        
        self.conv_1 = torch.nn.Conv2d(3,32, kernel_size = k, padding = p)
        self.conv_2 = torch.nn.Conv2d(32,32, kernel_size = k, padding = p)
        self.conv_3 = torch.nn.Conv2d(32,final_output_dim, kernel_size = k, padding = p)     
        
        self.FC_1 = torch.nn.Linear(final_output_dim * 3**2, 128)
        self.FC_2 = torch.nn.Linear(128, NUM_CAT)
        
        self.pool  = torch.nn.MaxPool2d(kernel_size = 2, stride = 2 )
        self.Dpool = torch.nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.act_fcn = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
    
    def forward(self,x):
        output = self.act_fcn( self.conv_1.forward( x ))
        output = self.pool(output) # -> 50 image size output
        output = self.pool(output) # -> 25
        
        output = self.act_fcn( self.conv_2.forward( output ))
        output = self.pool(output) # -> 12
        output = self.pool(output) # -> 6
        
        output = self.act_fcn( self.conv_3.forward( output ))
        output = self.pool(output) # -> 3
        
        output = torch.flatten( output , start_dim=1)
        
        output_1 = self.act_fcn( self.FC_1.forward( output ) )
        output   = self.softmax( self.FC_2.forward( output_1 ) )
        
        return output
    
    
"""  
class resnet_block(nn.Module):
    def __init__(self):
        super(resnet_block,self).__init__()
        
        def forward():
            
            self.conv_0
            self.conv_1
            x = x + x_og
            return x
"""
    

"loss 0.0284, ~92% accuracy on medium dataset, 200 epochs, lr 0.0001, b1 0.6"
class resnet(nn.Module):
    def __init__(self, NUM_CAT):
        super(resnet,self).__init__()
        k = 5
        p = int((k - 1)/2)
        "resnet: conv conv +skip, pool 2x, repeat"
        self.k = k
        self.p = p

        k = 5
        p = int((k - 1)/2)
        final_output_dim = 3 # final conv have 3 channels to represent RGB
        
        "block 1"
        self.conv_1 = torch.nn.Conv2d(3,32, kernel_size = k, padding = p)
        self.conv_2 = torch.nn.Conv2d(32,32, kernel_size = k, padding = p)
        
        "block 2"
        self.conv_3 = torch.nn.Conv2d(32,32, kernel_size = k, padding = p)     
        self.conv_4 = torch.nn.Conv2d(32,32, kernel_size = k, padding = p)     

        "block 3"
        self.conv_5 = torch.nn.Conv2d(32,32, kernel_size = k, padding = p)     
        self.conv_6 = torch.nn.Conv2d(32,final_output_dim, kernel_size = k, padding = p)     

        
        self.FC_1 = torch.nn.Linear(final_output_dim * 3**2, 128)
        self.FC_2 = torch.nn.Linear(128, NUM_CAT)
        
        self.pool  = torch.nn.MaxPool2d(kernel_size = 2, stride = 2 )
        self.Dpool = torch.nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.act_fcn = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        
        "Create skip connections for resnet blocks"
        "in/out channels are flipped"
        self.w_1 = torch.empty( 32 ,3,5,5)
        self.w_2 = torch.empty( 32,32,5,5)
        self.w_3 = torch.empty( 3, 32,5,5)
        with torch.no_grad():
            self.conv_skip_1 = torch.nn.Conv2d(3,32, kernel_size = k, padding = p)
            self.conv_skip_2 = torch.nn.Conv2d(32,32, kernel_size = k, padding = p)
            self.conv_skip_3 = torch.nn.Conv2d(32,3, kernel_size = k, padding = p)

            self.conv_skip_1.weight = Parameter( nn.init.dirac_(self.w_1) )
            self.conv_skip_2.weight = Parameter( nn.init.dirac_(self.w_2) )
            self.conv_skip_3.weight = Parameter( nn.init.dirac_(self.w_3) )

        
    def forward(self,x):
        
        "block 1"
        output = self.act_fcn( self.conv_1.forward( x.clone() ))
        output = self.act_fcn( self.conv_2.forward( output ))
        output = output + self.conv_skip_1.forward( x )
        
        output = self.pool(output) # -> 50 image size output
        output = self.pool(output) # -> 25
        
        "block 2"
        output_skip = output.clone()
        output = self.act_fcn( self.conv_3.forward( output ))
        output = self.act_fcn( self.conv_4.forward( output ))
        output = output + self.conv_skip_2.forward( output_skip )
        
        output = self.pool(output) # -> 12
        output = self.pool(output) # -> 6
        
        "block 3"
        output_skip = output.clone()
        output = self.act_fcn( self.conv_5.forward( output ))
        output = self.act_fcn( self.conv_6.forward( output ))
        output = output + self.conv_skip_3.forward( output_skip )
        
        output = self.pool(output) # -> 3
        
        
        output = torch.flatten( output , start_dim=1)
        
        output_1 = self.act_fcn( self.FC_1.forward( output ) )
        output   = self.softmax( self.FC_2.forward( output_1 ) )
        
        return output