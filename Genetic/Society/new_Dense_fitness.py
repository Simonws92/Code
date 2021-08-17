# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:23:17 2019
@author: Simon
"""
#import numpy as np
#import cupy as np
import numpy as np

class dense_fc:

    def __init__( self, H):

        self.delta = [0]*len(H) #Saves old delta values to be used for next batching training
    def place_holder(self):
        return self.delta
#solution = dense_fc(H_fc) #Create a class object with an object name
#solution.place_holder() #Calls on class methods aka def

######################
#ACTIVATION FUNCTIONS
######################
def init_weights(H):
    W_fc = []
    for i in range( len(H)-1): #creates dense weight matrices
        W_fc.append( (np.random.random((H[i+1],H[i]))*2 -1  ) * np.sqrt(2/(H[i]) ) ) 
        #creates weight matrices with weights between -1 and 1
    
    b_fc = []
    for i in range( len(H)-1): #creates the bias vectors
        b_fc.append(( np.random.random((H[i+1],1))*2 -1 ) )
        
    return W_fc, b_fc

def init_Cweights(H): #Returns sparse matrices corresponding to convolution
    #By default we use 5x5 filters
    
    
    W_fc = []
    w = np.array()
    for i in range( len(H)-1): #creates dense weight matrices
        W_fc.append( (np.random.random((H[i+1],H[i]))*2 -1  ) * np.sqrt(2/(H[i]) ) ) 
        #creates weight matrices with weights between -1 and 1
    
    b_fc = []
    for i in range( len(H)-1): #creates the bias vectors
        b_fc.append(( np.random.random((H[i+1],1))*2 -1 ) )

    
    
    return W, b
    
    
    
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
        
    if H_act=='sigmoid':
        s = 1/ (1+np.exp(-z))
        if d==0:
            return s
        if d==1:
            return s*(1-s)
    
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


def backpropagation(H, A, z, y, W, b, C=0, vs=0,lr=0.0,s=0,H_act=[], optimizer='none', N=0.001, delta_class='', F=0,COST='mse'): #i = 0:= epoch iterations, VS=0 if not using ADAM
    #N= 0.001 #Nudge priority
    
    n = len(H)-1
    
    b1 = 0.9 #initialize adam values
    b2 = 0.999
    alpha = 0.001
    epsilon = 10e-8
    
    
    eps = 0.1 #if error is less than this, we accept output
    cost_matrix_old = (A[-1] - y)
    #cm_sq = cost_matrix**2
    #binary_error_matrix = np.zeros(np.shape(A[-1]))
    #binary_error_matrix[cm_sq<=eps]=0 #we accept error
    #binary_error_matrix[cm_sq>0] =0.2    
    
    ##Todo: Make output A[-1] to be zero on all places where y is zero. leave nonzero entries of y alone wrt A[-1]
    
    
    #cm = np.exp( (A[-1]-y)**2 ) # cost function
    #cost_matrix = np.tan(1.3 * cost_matrix_old) / 0.6
    #print(cost_matrix)
    if COST == 'mse':
        delta = 2* cost_matrix_old * Act(H_act[-1],z[-1],1)      #act_tanh( z[-1],1)
    else:
        F_old = F[0]
        F_new = F[1]
        delta = ( ( (F_old) / (F_new+0.1)**2 )  * Act(H_act[-1],z[-1],1) )
        #print(sum(sum(abs(delta))), "delta")
        #print( sum(delta[0]), "sum of fitness comparison: Lower is better" )
    #print(A[-1])
    #delta_H_1 = 2*Act(H_act[-1],z[-1],1)**2                #Delta_H1 and delta_H2 have same dimensions
    #delta_H_2 = 2*(A[-1]-y) * Act( H_act[-1], z[-1],2)
    

    
    #delta = np.log(A[-1])*y*softmax(z[-1],0) #for softmax loss function
    for p in range(n):#this for loop goes through each W wrt the layers
        if p == 0:
            delta_w = np.dot(delta, A[-2].T )
            if H_act[-1]=='I':
                delta_b = np.average( y-W[-1]@A[-2] ,axis=1 )
            else:
                delta_b = np.average( np.dot( 2*(A[-1]-y), Act(H_act[-1],z[-1],1).T ) , axis=1)
            
        if p >0:
            delta_old = delta
            #delta_H_1_old = delta_H_1
            #delta_H_2_old = delta_H_2
            w_temp = W[-p]
            der      = Act(H_act[-p-1],z[-p-1],1)
            #scnd_der = Act(H_act[-p-1],z[-p-1],2)
            
            delta = np.dot( w_temp.T, delta_old ) * der
            
            delta_b = np.average(delta, axis=1)
            #delta_H_1 = np.dot( w_temp.T , delta_H_1_old) * scnd_der
            #delta_H_2 = np.dot( w_temp.T , delta_H_2_old) * (der**2)
            
            #print(delta,"delta_w")
            delta_w = np.dot( delta, A[-2-p].T ) #Note that A[n-p] is always at the end of the product chain when adjusting
            

        b1 = 0.999
        b2 = 0.999
        t=1
        for n in range(1):
            #print(np.shape(W[-1-p]), np.shape(delta_w))
            #print(W[-1-p], delta_w)
            delta_w = delta_w.astype('float64')
            W[-p-1] -= lr * delta_w
            delta_b = np.reshape(delta_b, (len(delta_b),1))
            b[-p-1] -= lr*delta_b #default updater
     
        
    #Weight clipping: Any weight values exeeding +-2 will be pushed to +-0.5
    #Regularization technique
#        W[-p-1][W[-p-1]>1]=0.5
#        W[-p-1][W[-p-1]<-1]=-0.5

    
    return W, b, delta

def dropout(H):
    drop = np.random.random(np.shape(H))*10-1
    drop[drop<=0]=0
    drop[drop>0] =1
    return drop
    

def feedforward(dp,x, H, W, b, drop=0,H_act=[]): #Does a complete feedforward iteration through the network
    
    z = []
    A = [x]
    for i in range(len(H)-1):
        if dp == 1:
            hidden_layer = np.dot(W[i], A[i] ) 
            #print(np.shape(hidden_layer), i)
            #print(np.shape(b[i]))
            hidden_layer += b[i]#[:,np.newaxis] 
            

        else:
            hidden_layer = (np.dot(W[i], A[i]) ) #+ b[i][:,np.newaxis] 
            #print("We're in here")
            
        #dropout_matrix = dropout( hidden_layer )
        z.append(  hidden_layer  ) 
        #print(z[i])
        if i == len(H)-2: #no dropout. Final layer
            A.append( Act(H_act[-1],z[i], 0) )
            
        else:
            #print(dropout_matrix)
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


