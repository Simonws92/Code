# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 13:45:02 2021

@author: Simon
"""

import numpy as np
import new_Dense_1_3 as nd
import copy


#x = np.array([ XOR , XOR, XOR ])

"Define graph architecture"
A = np.array([ [0,0,0,0,0],
               [0,0,0,0,0],
               [1,1,0,1,1],
               [0,0,1,0,0],
               [0,0,1,0,0] ])
    
WW = np.array([ [0,0,0,0,0],
                [0,0,0,0,0],
                [1,1,1,1,1],
                [0,0,1,0,0],
                [0,0,1,0,0] ])

Graph_weights = [ [0,0]*5 for i in range(5) ]

"Load weights"
H = [3,5,3,3]
W,b = nd.init_weights(H, A = 0)
b[0] = np.zeros((5,1))
b[1] = np.zeros((3,1))
b[2] = np.zeros((3,1))

for i in range(5):
    for j in range(5):
        if WW[i,j]==1:
            k = np.random.randint(2,10)                                        #number of layers
            H = [3] + [ np.random.randint(5,10) for i in range(k) ] + [3]      #number of neurons
            
            W, b = nd.init_weights(H, A = 0)
            Graph_weights[i][2*j]   = W
            Graph_weights[i][2*j+1] = H
            

def neural_net(W,b,H,x):
    H_act = ['tanh']*(len(H)-2) + ['I']
    z,a = nd.feedforward( x, H, W, b,H_act=H_act)
    return a[-1]

X_c = [ np.random.random((3,1)) ]*2 + [ np.zeros((3,1)) ]*3

"certain x_values may want to be fixed and not reset after every time-step"
x_fixed = [1,1,0,0,0]
x_computed = [0]*5

Z = [0]*5



"pseudo timesteps"
T=10

node_sum = [0]*5

"timestep / stages"
for c in range(2):
    
    "Z are the outputs"
    Z = [0]*5
    
    "X are the inputs"
    X_c = [ np.random.random((3,1)) ]*2 + [ np.zeros((3,1)) ]*3
    X = copy.deepcopy(X_c)
    print("NEXT TIMESTEP")
    "pseudo-timestep"
    for t in range(T):
        #print(X)     
        for i in range(5):
            for j in range(5):
                if A[i,j]!=0 and x_computed[j]==0:
                    #print(i+1,j+1)
                    Z[j] = neural_net( Graph_weights[i][2*j] ,b, Graph_weights[i][2*j+1] , X[j] )
                    X[i] += Z[j]
                    #print(Z[j])
#                    if j == 2:
#                        print("Output from 3", i,j)
#                        print(Z[j])
                    
                    "prevents computations of certain inputs again"
                    if x_fixed[j] == 1:
                        x_computed[j] = 1
                        
                else:
                    pass
    
        
        for I in range(5):
            node_sum[I] += np.sum(X[I])
            
    "resets computed rule"
    x_computed = [0]*5
    
    print("Output at stage:", c, np.round(np.array(Z),3)  )

print("X_input", np.round(np.array(X),3))







