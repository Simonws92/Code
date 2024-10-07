# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:28:23 2024

@author: Simon
"""


import torch
import numpy as np
import matplotlib.pyplot as plt

def Training(X,Y):
    
    "Define models and parameters"
    nm_layers,hd_size, batch_size = 3,32,64 
    
    model = torch.nn.LSTM(input_size=1, hidden_size=hd_size, num_layers=nm_layers, \
                          bias=True, batch_first=True )
        
    FC_1 = torch.nn.Linear(32, 32)
    FC_2 = torch.nn.Linear(32, 1)
    
    b1=0.6
    b2=0.999
    optimizer      = torch.optim.Adam(model.parameters(), lr=0.0001 ,betas=(b1,b2))
    optimizer_FC_1 = torch.optim.Adam(FC_1.parameters() , lr=0.0001 ,betas=(b1,b2))
    optimizer_FC_2 = torch.optim.Adam(FC_2.parameters() , lr=0.0001 ,betas=(b1,b2))
    
    "Initiate loss function"
    loss_fcn = torch.nn.MSELoss()
    #loss_fcn = torch.nn.L1Loss()
    
    "Convert to torch.tensor"
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    epochs = 1
    training_loss = []
    for e in range(epochs):
        epoch_loss = 0
        
        "We shuffle the training set"
        r = np.arange(0, len(X))
        np.random.shuffle(r)
        r_list = list(r)
        
        number_of_batches = int( len(r_list) / batch_size )
        
        h = torch.randn(nm_layers, batch_size, hd_size)
        c = torch.randn(nm_layers, batch_size, hd_size)
        
        for i in range(number_of_batches):
            
            batch_numbers = r_list[ i*batch_size:(i+1)*batch_size ]
            input_batch  = X[batch_numbers]
            target_batch = Y[batch_numbers]
            
            "DL model"
            "(batch_size, sequence_len, nr_features = 1​) when batch_first=True "
            model_output, (hn, cn) = model(input_batch, (h, c))
            model_output = FC_1(model_output)
            model_output = FC_2(model_output)
            
            
            "Compute backprop and update weights"
            loss = loss_fcn(model_output, target_batch)
            loss.backward()
            optimizer.step()
            optimizer_FC_1.step()
            optimizer_FC_2.step()
            
            epoch_loss += loss.detach().numpy()
        
        training_loss.append(epoch_loss)
        print("epoch loss:", e,epoch_loss)
        
        
        "Plot test sample per epoch"
        
        if e % 10==0:
            plt.clf()
            print(np.shape(target_batch))
            plt.plot( target_batch[0,:,0].detach().numpy()  )
            plt.plot( model_output[0,:,0].detach().numpy()  )
            plt.pause(0.1)
    
        
    return training_loss
