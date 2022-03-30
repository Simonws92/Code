# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:16:59 2021

@author: Simon
"""
import numpy as np
def network_templates(type_nr):
    "define nodes"
    "Must define node architecture"
    A_node = np.array([ [0,0,0,0],[1,0,0,0],[1,0,0,0],[0,0,1,0] ])
    #0: no connection, 1: standard-operation connection, 2: node connection
    "args = padding, stride, filter size, pool_mod, nr_kernels = len(k_1)"
    ntw = 0
    if type_nr == 1:
        ntw = {'type': ['conv']*1, 'H': [[3,20,40,60]], 'args': [1,1,3,1], 'A': A_node }
    if type_nr == 2:
        ntw = {'type': ['conv']*1, 'H': [[3,10,20,30]], 'args': [2,1,5,4] } #no pooling
    if type_nr == 3:
        ntw = {'type': ['conv']*1, 'H': [[3,20,40,60]], 'args': [3,1,7,1] }
    if type_nr == 4:
        ntw = {'type': ['conv']*1, 'H': [[30,30,30,30]], 'args': [2,1,5,4] } #no pooling
    if type_nr == 5:
        ntw = {'type': ['conv']*1, 'H': [[30,30,30,60,60,60]], 'args': [2,1,5,1] } #pooling every layer
    if type_nr == 6:
        ntw = {'type': ['conv']*1, 'H': [[60,60,60,60]], 'args': [3,1,7,1] }
    if type_nr == 7:
        ntw = {'type': ['fc'], 'H': [60*(4**2),100,100], 'args': 0, 'A': A_node }
        
    if type_nr == 8:
        ntw = {'type': ['conv'], 'H': [[3,20,40,60,60,60]], 'args': [1,1,3,1] }
    
    p_1 = [1,2,3]
    k_1 = [3,5,7]
    if type_nr == 9:
        ntw = {'type': ['conv'], 'H': [3,20,40,60], 'args': [p_1,1,k_1,1], 'A': A_node }
    if type_nr == 10:
        ntw = {'type': ['conv'], 'H': [60,60,60,60], 'args': [p_1,1,k_1,0], 'A': A_node }
    
    if type_nr == 11:
        ntw = {'type': ['conv'], 'H': [60,60,60,60], 'args': [[1],1,[3],0], 'A': A_node }
    
    if type_nr == 12:
        ntw = {'type': ['fc'], 'H': [100,10], 'args': 0, 'A': A_node }
        
    if type_nr == 13:
        ntw = {'type': ['fc'], 'H': [100,100,100,10], 'args': 0, 'A': A_node }
        
    if type_nr == 14:
        ntw = {'type': ['fc'], 'H': [100,100,100,10], 'args': 0, 'A': A_node }
        
    "USE these for autoencoders"
    
    '''
    #  Example: input = 64x64, output of 20 = 8, input of 21 = 8, output of 21 = 64
    '''
    pool_mod = [0,0,1] #Perform pooling at these layers
    p_2 = [10,11,12]
    k_2 = [21,23,25]
#    if type_nr == 20:
#        ntw = { 'type': ['conv'], 'H':[3,50,100,200], 'args':[p_1,1,k_1,1], 'A':A_node, 'pool': 'max' }
#    if type_nr == 21:
#        ntw = { 'type': ['conv'], 'H':[200,100,50,3], 'args':[p_1,1,k_1,1], 'A':A_node, 'pool': 'upsample' }
#
#    if type_nr == 22:
#        ntw = { 'type': ['conv'], 'H':[200,200], 'args':[p_1,1,k_1,1], 'A':A_node, 'pool': 'none' } 
        
    p_0 = [1]
    k_0 = [3]
    
    p_3 = [2,5]
    k_3 = [5,11]
    
    #20 identity conv
    if type_nr == 200: 
        ntw = { 'type': ['conv'], 'H':[3,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none', 'id': True }

    if type_nr == 201: 
        ntw = { 'type': ['conv'], 'H':[16,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none', 'id': True }

    if type_nr == 202: 
        ntw = { 'type': ['conv'], 'H':[16,3], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none', 'id': True }
    
    #21 resnets
    if type_nr == 210: 
        ntw = { 'type': ['conv'], 'H':[3,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none', 'id': False }

    if type_nr == 211: 
        ntw = { 'type': ['conv'], 'H':[16,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none', 'id': False }

    if type_nr == 212: 
        ntw = { 'type': ['conv'], 'H':[16,3], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none', 'id': False }

    if type_nr == 23:
        ntw = { 'type': ['conv'], 'H':[3,16,16,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none' }
    
    if type_nr == 231:
        ntw = { 'type': ['conv'], 'H':[16,16,16,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'none' }


    #other    
    if type_nr == 24:
        ntw = { 'type': ['conv'], 'H':[3,32,32,16], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'max' }
        
    if type_nr == 25:
        ntw = { 'type': ['conv'], 'H':[16,32,32,3], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'upsample' }        
        
    if type_nr == 26:
        ntw = { 'type': ['conv'], 'H':[3,32,32,32], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'max' }

    if type_nr == 27:
        ntw = { 'type': ['conv'], 'H':[32,32,32,3], 'args':[p_3,1,k_3,1], 'A':A_node, 'pool': 'upsample' }
        

    p_4 = [1,2,3]
    k_4 = [3,5,7]
    if type_nr == 31:
        ntw = { 'type': ['conv'], 'H':[3,16,32,32,32,32], 'args':[p_4,1,k_4,1], 'A':A_node, 'pool': 'max' }

    if type_nr == 32:
        ntw = {'type': ['fc'], 'H': [32,100,100,1], 'args': 0, 'A': A_node, 'pool': 'none' }
        
    if type_nr == 40:
        ntw = {'type': ['fc'], 'H': [10,10,10], 'args': 0, 'A': A_node, 'pool': 'none' }
    
        
    A_node = np.array([ [0,0,0,0],[14,0,0,0],[0,14,0,0],[0,0,14,0] ])
    "Sub_graph type"
    if type_nr == 30:
        ntw = { 'type': ['node'], 'H': [] , 'args':0, 'A': A_node}
        
    if type_nr == 100:
        "path is the source custom code where the graph-program will load the source code from"
        "This particular edge operation is not using pytorch"
        
        "ntw dict may hold any custom arguments as long as 'type' have the argument ['custom']"
        
        #no pytorch
        #ntw = {'type': ['custom'], 'H': [], 'args': 0, 'A': A_node, 'pool': 'none', 'path':'custom_code', 'pytorch': False }
        
        #with pytorch
        #ntw = {'type': ['custom'], 'H': [1,10,1], 'args': 0, 'A': A_node, 'pool': 'none', 'path':'custom_code', 'pytorch': True }
        ntw = {'type': ['custom'], 'H': [1,10,1], 'args': 0, 'A': A_node, 'pool': 'none', 'path':'custom_code', 'function':'create_fc_wrapper' , 'pytorch': True }

#    if type_nr == custom:
#        ntw = { 'type': ['custom'],  }
    
    "Pretrained model loading"
    if type_nr == 110:
        ntw = {'type': ['custom'], 'path':'custom_code', 'function':'create_import_wrapper', 'pytorch': True }

    
    return ntw

































