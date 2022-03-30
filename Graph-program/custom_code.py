# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 08:52:55 2022

@author: Simon
"""

from torch import nn,optim

import torchvision.models as models
#def some_custom_function(x):
#    return x+1




def create_fc_wrapper( network_type ):
    H = network_type["H"]
    return create_fc_custom(H)

class create_fc_custom(nn.Module):
    def __init__(self, H):
        
        super(create_fc_custom,self).__init__()
        
        "Initiate fully connected"
        self.H = H
        self.relu = nn.ReLU()
        
        for i in range(len(H)-1):
            exec( 'self.linear{} = nn.Linear(H[{}], H[{}])'.format(i,i,i+1) )
        

    def forward(self,x, AE=0):
        x = x.view((-1 , self.H[0]))
        
        for i in range(len(self.H)-1):
            x = eval( ' self.linear{}(x)'.format(i) ) + 100
            if i==len(self.H)-2:
                "does not perform activation on the final FC layer"
                pass
            else:
                x = self.relu(x)
        
        return x
    
    
    
    

def create_import_wrapper( network_type ):
    print("in here")
    return create_resnet18()


class create_resnet18( nn.Module ):
    def __init__(self):
        
        super(create_resnet18,self).__init__()
        self.resnet18 = models.resnet18()


    def forward(self,x):
        x = self.resnet18(x)
        return x









