# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:23:17 2019
@author: Simon
"""
#import numpy as np
#import cupy as np
import numpy as np
import numexpr as ne
######################
#ACTIVATION FUNCTIONS
######################
def Act(H_act,z,d): #Work in progress!
    z = np.array(z)
    if H_act=='I':
        if d==0:
            return z
        else:
            return np.ones(np.shape(z))
            
    if H_act=='Lrelu':
        if d==0:
            a = 0.1
            z[z<=0]=a
            return z
        else:
            a = 0.1
            z[z<=0]=a
            z[z>0] =1
            return z        

    if H_act=='relu':
        if d==0:
            z[z<=0]=0
            return z
        else:
            z[z<=0]=0
            z[z>0] =1
            return z
    
    if H_act=='softplus':
        inRanges = (z < 10)
        if d==0:
            return np.log(1 + np.exp(z*inRanges))*inRanges + z*(1-inRanges)
        if d==1:
            z = 1/ (np.exp(-z*inRanges) +1)
            z[z<1e-16]=0
            return z
        if d==2:
            z[abs(z)>4]=4
            z = -np.e**z / (np.e**z + 1)**2
            return z

    if H_act=='tanh':
        #lz = len(z)
        if d==0:
            return np.tanh(z)
        elif d==1:
            return 1- (np.tanh(z))**2
        elif d==2:
            return -2*np.tanh(z)*( 1-np.tanh(z)**2 )
    
    if H_act=='log':
        if d==0:
            z[z<=0]=0
            z = np.log((z)**2+1)
            return z
        elif d==1:
            return 2*z/(z**2+1)
        
    
    if H_act=='softmax':
        #print(z, "prepresoft")
        z = np.array(z)
        z = np.tanh(z/(max(z))) #Otherwise the values will be too large for e**z
        sumz = sum(np.e**z[i] for i in range(len(z)))
        A = np.e**z
        #print(z, "PRESOFT...")
        return A / sumz
    

def adam():
    V_dw = 0 #Initialize V S values.
    V_db = 0
    S_dw = 0
    S_db = 0
    VS = [V_dw, V_db, S_dw, S_db]
    
    b1 = 0.9 #initialize adam values
    b2 = 0.999
    alpha = 0.001
    epsilon = 10e-8
    AD = [b1,b2,alpha,epsilon]
    
    return VS, AD



def nudge_str(delta,N):
    Delta=delta.copy()
    Delta[Delta<=-2]=-2 #prevents overflow
    Delta[Delta>=2]=2
    
    nudge = np.zeros(np.shape(Delta))
    nudge[abs(Delta)<N]=1
    #print(nudge)
    #nudge = ( 2 / ( (1-0.99)**(-Delta/N) + (1-0.99)**(Delta/N)  ))

    return nudge


def backpropagation(H, A, z, y, W, b, C=0, vs=0,lr=0.0001,s=0,H_act=[], optimizer='none'): #i = 0:= epoch iterations, VS=0 if not using ADAM
    N= 0.001 #Nudge priority
    
    VS=vs
    n = len(H)-1
    
    b1 = 0.9 #initialize adam values
    b2 = 0.999
    alpha = 0.001
    epsilon = 10e-8
    delta = 2*(A[-1] - y) * Act(H_act[-1],z[-1],1)      #act_tanh( z[-1],1)
    #print(A[-1])
    delta_H_1 = 2*Act(H_act[-1],z[-1],1)**2                #Delta_H1 and delta_H2 have same dimensions
    delta_H_2 = 2*(A[-1]-y) * Act( H_act[-1], z[-1],2)
    #delta = np.log(A[-1])*y*softmax(z[-1],0) #for softmax loss function
    for p in range(n):#this for loop goes through each W wrt the layers
        if p == 0:
            delta_w = np.dot(delta, A[-2].T )
            if H_act[-1]=='I':
                delta_b = np.average( y-W[-1]@A[-2] ,axis=1 )
            else:
                delta_b = np.average( np.dot( 2*(A[-1]-y), Act(H_act[-1],z[-1],1).T ) , axis=1)
            
            g=0.5/(( A[-2]**2) +1)
            f_H = (nudge_str(delta,N) / (delta_H_1**2 + 1)) @ g.T
        if p >0:
            delta_old = delta
            delta_H_1_old = delta_H_1
            delta_H_2_old = delta_H_2
            w_temp = W[-p]
            der      = Act(H_act[-p-1],z[-p-1],1)
            scnd_der = Act(H_act[-p-1],z[-p-1],2)
            
            delta = np.dot( w_temp.T, delta_old ) * der
            delta_b = np.average(delta, axis=1)
            delta_H_1 = np.dot( w_temp.T , delta_H_1_old) * scnd_der
            delta_H_2 = np.dot( w_temp.T , delta_H_2_old) * (der**2)
            
            #print(delta,"delta_w")
            delta_w = np.dot( delta, A[-2-p].T ) #Note that A[n-p] is always at the end of the product chain when adjusting
            
        
        #print(np.shape(delta_H_1), "H1") 
        #print(np.shape(delta_H_2), "H2")
        
        G = (A[-2-p].T**2)
        delta_H = ( delta_H_1 + delta_H_2 ) @ G
        #f_H = (( nudge_str(delta,N) @G)/( (np.log(delta_H**2+1)) +1 ) ) #@  G #nudge * 2nd der of weight
        f_H = ( nudge_str(delta,N) @G) / (( delta_H**2+1) ) #@  G #nudge * 2nd der of weight
#        print( np.average( delta_H+1), "H")
#        print(np.average(delta), "delta")
#        print(np.average(f_H), "FH")
        
        delta_w = np.tanh( delta_w  ) #Solves exploding derivatives

        r = np.random.random(np.shape(f_H))
        #optimizer = 'SH'
        if optimizer=='SH':
            W[-p-1] -= f_H * lr**2 *np.sign(delta_w) # # / abs(weight_pen+1)  #190, 170 with increased beta.
            #f_H gives strength to the direction of sign(delta_w)
            #r scales the weights randomly
        
        for J in range(3):
            W[-p-1] -= lr*delta_w
            b[-p-1] -= lr*delta_b #default updater
     
        
    #Weight clipping: Any weight values exeeding +-2 will be pushed to +-0.5
    #Regularization technique
        W[-p-1][W[-p-1]>2]=0.5
        W[-p-1][W[-p-1]<-2]=-0.5

    
    return W, b, delta, f_H, delta_H_1, delta_H_2

def dropout(H):
    drop = np.random.random(np.shape(H))*10-1
    drop[drop<=0]=0
    drop[drop>0] =1
    return drop
    

def feedforward(dp,x, H, W, b, drop=0,H_act=[]): #Does a complete feedforward iteration through the network
    
    z = []
    A = [x]
    for i in range(len(H)-1):
        if dp == 1: #MUST FIX BIAS DIMENSIONS
            hidden_layer = np.dot(W[i], A[i] ) + b[i][:,np.newaxis]
        else:
            hidden_layer = np.dot(W[i], A[i]) #+ b[i][:,np.newaxis]
        
        #dropout_matrix = dropout( hidden_layer )
        z.append(  hidden_layer  )
        if i == len(H)-2: #no dropout. Final layer
            A.append( Act(H_act[-1],z[i], 0) )
            
        else:
            #A.append( Act(H_act[i],z[i], 0)* dropout_matrix ) #Dropout
            A.append( Act(H_act[i],z[i], 0) )
    return  z, A

def pre_proccess(X,Y,H,W,b, e): #Need to initialize a batch for the feedforward
    
    for k in range(e): #number of epochs
        
        drop = dropout(H)
        
        for i in range(shape(X)[1]):
            x = X[:,i] 
            y = Y[:,i] 
    
        y, C, z, A = feedforward(x,y, H, W, b, drop)
        W, b = backpropagation(H, A, z, y, C, W, b)
    
    return y, C, z, A


