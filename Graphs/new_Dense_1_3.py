# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:23:17 2019
@author: Simon
VERSION 1.3.2
"""

import numpy as np
import scipy as sp
import matplotlib as plt
import time
from PIL import Image
import matplotlib



class dense_fc:
    '''
    
    The Adam class saves momementum for both weights and biases
    
    '''
    
    def __init__( self, H):
        l=[]
        for i in range(len(H)):
            l.append([0,0,0,0])
        self.Adam = l #for mt and vt values
    def update_Adam(self, layer, w_b, bias_true, value):
        (self.Adam)[layer][w_b + 2*bias_true] = value
    def reset_Adam(self, H): #Occurs after every new epoch
        l=[]
        for i in range(len(H)):
            l.append([0,0,0,0])
        self.Adam = l 
        
################
##### AdaH #####
################
def AdaH(delta_class, p, delta_w1, delta_w2, batch_nr, bias_true=0, vectors = 0, it=0):
    beta1 = 0.99
    beta2 = 0.999
    '''
    if vectors == 1:
        try:
            print(np.shape(  (delta_class.Adam)[-p-1][0 + 2*bias_true]   ))
            old_mt = (delta_class.Adam)[-p-1][it,:][0 + 2*bias_true]
        except:
            old_mt = (delta_class.Adam)[-p-1][it,:][0 + 2*bias_true]
    else:
    '''
    old_mt = (delta_class.Adam)[-p-1][0 + 2*bias_true]
    old_vt = (delta_class.Adam)[-p-1][1 + 2*bias_true]

    ''' Momentum '''
    Mt = beta1 * old_mt + (1-beta1) * (delta_w1)     #gradient part
    Vt = beta2 * old_vt + (1-beta2) * (delta_w2**2)  #hessian part
    
    ''' Adahessian '''
    Mt_bar = Mt / (1 - beta1**(batch_nr+1))
    Vt_bar = Vt / (1 - beta2**(batch_nr+1))
    
    delta_class.update_Adam( -p-1, 0, bias_true, Mt)
    delta_class.update_Adam( -p-1, 1, bias_true, Vt)
    #ad = Mt / ( np.sqrt(Vt) + eps)
    return Vt_bar, Mt_bar

################
##### Adam #####
################
def Adam(delta_class, p, delta_w1, delta_w2, batch_nr, bias_true=0):
    beta1 = 0.9
    beta2 = 0.999
    old_mt = (delta_class.Adam)[-p-1][0 + 2*bias_true]#.copy()
    old_vt = (delta_class.Adam)[-p-1][1 + 2*bias_true]#.copy()

    Mt = beta1 * old_mt + (1-beta1) * (delta_w1)
    Vt = beta2 * old_vt + (1-beta2) * (delta_w2**2)
    
    ''' Adam correction '''
    Mt_bar = Mt / (1 - beta1**(batch_nr+1))
    Vt_bar = Vt / (1 - beta2**(batch_nr+1))
    
    ''' Adam final update '''
    eps = 1e-8
    Mt_bar = Mt_bar / (np.sqrt(Vt_bar) + eps)
    #print(Mt_bar)
    
    delta_class.update_Adam( -p-1, 0, bias_true, Mt)
    delta_class.update_Adam( -p-1, 1, bias_true, Vt)
    return Mt_bar, Vt_bar


def init_weights(H, A = 0):
    W_fc = []
    b_fc = []
    
    "No adjacency defined"
    if A == 0:
        A = [ np.ones(( np.sum(H[I+1]) , np.sum(H[I]) )) for I in range(len(H)-1) ]
    
    for j in range(len(H)-1):
        "Creates a placeholder weight matrix"
        WW = np.zeros(( np.sum(H[j+1]) , np.sum(H[j]) ))
        
        try:
            "graph network"
            "rows"
            k_row = 0
            for i in range( np.shape(A[j])[0]):
                "columns"
                k_col = 0
                for k in range( np.shape(A[j])[1]):
                    w = ( np.random.random((H[j+1][i],H[j][k]))*2 -1  ) * A[j][i,k] #* np.sqrt(1/(H[j+1][I]) )
                    WW[ k_row:k_row+H[j+1][i] , k_col:k_col+H[j][k]  ] = w
                    #figure()
                    #imshow(w)
                    '''
                    for i in range(len(H[j])):
                        w = ( np.random.random((H[j+1][i],H[j][i]))*2 -1  ) * np.sqrt(1/(H[j+1][i]) ) 
                        WW[ k_row:k_row+H[j+1][i] , k_col:k_col+H[j][i]  ] = w
                    '''
                    k_col+=H[j][i]
                k_row+=H[j+1][i]
                    
                
            W_fc.append(WW)
            b_fc.append( ( np.random.random(( np.sum(H[j+1]),1))*2 -1 ) )
        
        
        except:
            "simple network"
            #print("In dense",H[j])
            W_fc.append( (np.random.random(( np.sum(H[j+1]) , np.sum(H[j]) ))*2 -1  ) * np.sqrt(1/( np.sum(H[j]) ) ) )
            b_fc.append( ( np.random.random(( np.sum(H[j+1]),1))*2 -1 ) )
    
    
    '''
    "Deprecated code. Use only for emergencies"
    for j in range(len(H)-1)
        for i in range( len(H[j])-1): #creates dense weight matrices
            W_fc.append( (np.random.random((H[i+1],H[i]))*2 -1  ) * np.sqrt(1/(H[i]) ) )
            #creates weight matrices with weights between -1 and 1
        
        b_fc = []
        for i in range( len(H)-1): #creates the bias vectors
            b_fc.append(( np.random.random((H[i+1],1))*2 -1 ) )
    ''' 
    
    return W_fc , b_fc

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
            z[z<=0]=a*z
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
        
        #z -= z.max()
        '''
        inRanges = (z < 10)
        if d==0:
            return np.log(1 + np.exp(z*inRanges))*inRanges + z*(1-inRanges)
        if d==1:
            z = 1/ (np.exp(-z*inRanges) +1)
            z[z<1e-16]=0
            return z
        '''
        
        #print("max:",z.max(), "softplus")
        #for i in range(np.shape(z)[0]):
        #    z[i] -= z[i].max()
        
        #z -= z.max()
        #inRanges = (z < 10)
        
        if d==0:
            #return np.log(1 + np.exp(z*inRanges))*inRanges + z*(1-inRanges)
            return np.log(1 + np.exp(z))
        if d==1:
            #z = 1/ (np.exp(-z*inRanges) +1)
            z = 1/(np.exp(-z)+1) 
            z[z<1e-16]=0
            return z
        if d==2:
            z[abs(z)>4]=4
            z = -np.e**z / (np.e**z + 1)**2
            return z

    if H_act=='tanh':
        #lz = len(z)
        a = np.tanh(z)
        if d==0:
            return a
        elif d==1:
            return 1- a**2
        elif d==2:
            return -2*a*( 1-a**2 )
        
    
    if H_act=='log':
        if d==0:
            z[z<=0]=0
            z = np.log((z)**2+1)
            return z
        elif d==1:
            return 2*z/(z**2+1)
        
    
    if H_act=='softmax':
        z -= z.max()
        act=np.exp(z)/np.sum( np.exp(z), axis=0 )
        if d==0:
            return act    #Softmax activation
        else:
            return act*(1-act)


def backpropagation(H, A, z, y, W, b, lr=0.001, eta=1,H_act=[], optimizer='none', delta_class='', batch_nr = 0, ch=1, other=0): #i = 0:= epoch iterations, VS=0 if not using ADAM
    
    n = len(H)-1
    
    "mse"
    delta = (A[-1] - y) #* Act(H_act[-1],z[-1],1)
    
    "l1"
    delta = delta / (abs(delta)+1)
    
    delta_2 = delta.copy()
    #figure()
    #plot(delta)
    Weight_updates = [0]*n
    Weight_updates_2 = [0]*n
    Bias_updates = [0]*n
    
    delta_H_1 = 2*Act(H_act[-1],z[-1],1)**2               
    delta_H_2 = 2*(A[-1]-y) * Act( H_act[-1], z[-1],2)
    
    for p in range(n):#this for loop goes through each W wrt the layers
        if p == 0:
            
            if ch==1:
            
                delta_w = np.dot(delta, A[-2].T )
                delta_b = np.sum(delta, axis=1)[:, np.newaxis]
                
                
                ''' The creation of the propagation matrix A_l for the first layer '''
                if optimizer == 'H':
                    A_prev = delta_H_1 + delta_H_2
                    
                    #scnd_der = Act(H_act[-p-1],z[-1],2)
                    #delta_2_old = Act(H_act[-1],z[-1],1)**2 + (A[-1]-y)*scnd_der
                    
            else: #for multiple channels
                A_prev = delta_H_1 + delta_H_2
                
                delta_w = np.zeros(( H[-p-1] , H[-p-2] ))
                for j in range(ch):
                    delta_w += np.dot( delta ,( A[-2-p][:,:,j] ).T )  / ch
                
                delta_b = np.sum(delta, axis=1)[:, np.newaxis]
                
            
        if p >0:
            
            if ch==1:
            
                delta_old = delta
                
                der = Act(H_act[-p-1],z[-p-1],1)
                
                
                pre_delta = np.dot( W[-p].T, delta_old ) #/dp
                delta = pre_delta * der
                
                if  p == 5:
                    delta+=other
                
                #delta_b = np.average(delta, axis=1)[:, np.newaxis]
                delta_b = np.sum(delta, axis=1)[:,np.newaxis]
                
                #delta_H_1 = np.dot( w_temp.T , delta_H_1_old) * scnd_der
                #delta_H_2 = np.dot( w_temp.T , delta_H_2_old) * (der**2)
                
                #print(delta,"delta_w")
                delta_w = np.dot( delta, A[-2-p].T ) #/ dp #Note that A[n-p] is always at the end of the product chain when adjusting
                #delta_w, _ = Adam(delta_class[0], p, delta_w, delta_w, batch_nr, bias_true=0)
                Weight_updates[-1-p] = delta_w
                
            else:
                delta_old = delta
                pre_delta = np.dot( W[-p].T, delta_old ) #/dp
                der = Act(H_act[-p-1],z[-p-1],1)
                delta = np.dot( W[-p].T, delta_old ) #* np.average( der, axis=-1 )
                
                #print(np.shape(delta), "delta")
                #print(np.shape(A[-2-p]), "a")
                delta_w = np.zeros(( H[-p-1] , H[-p-2] ))
                
                for j in range(ch):
                    delta_w +=  np.dot( delta , (A[-2-p][:,:,j]).T ) / ch
                #np.average( delta_w, axis=0 )
                delta_b = np.sum(delta, axis=1)[:,np.newaxis]
                #print(np.shape(delta_w), "shape")
            
            
            ''' The creation of the propagation matrix A_l '''
        if optimizer =='H' and p>0:
            #print(type(scnd_der))
            if ch==1:
                delta_w, _ = Adam(delta_class[0], p, delta_w, delta_w, batch_nr, bias_true=0)
                
                
                
                scnd_der = Act(H_act[-p-1],z[-p-1],2)
                B_l = pre_delta * scnd_der
                A_l =  np.dot( (W[-p]**2).T ,A_prev ) * (1+der**2)  + (B_l)
                #A_l/=A_l.max() #TEST
                
                Hess = np.dot( A_l, (A[-2-p]**2).T )
                
                Hess, _ = Adam(delta_class[1], p, Hess, Hess, batch_nr, bias_true=0)

                #Cleaned Hessian
                #Hess[ abs(Hess)<1e-6 ] = 1
                #Hess[ Hess < -0.1] = 1
                print(np.sum(Hess))
                Hess = abs(Hess)+0.01
                
                Weight_updates[-1-p] = delta_w / Hess#( abs(Hess)+1)
                
                A_prev = A_l.copy()
            
            else:
                delta_w, _ = Adam(delta_class[0], p, delta_w, delta_w, batch_nr, bias_true=0)

                scnd_der = Act(H_act[-p-1],z[-p-1],2)
                B_l = pre_delta * np.average( scnd_der , axis=-1 )
                A_l =  np.dot( (W[-p]**2).T ,A_prev ) * np.average(der**2, axis=-1)  + (B_l)
                #A_l/=A_l.max() #TEST
                
                Hess = np.zeros(( H[-p-1] , H[-p-2] ))
                
                for j in range(ch):
                    Hess += np.dot( A_l , ( A[-2-p][:,:,j]**2 ).T ) / ch
                
                
                #Cleaned Hessian
                Hess[ abs(Hess)<1e-6 ] = 1
                Hess[ Hess < -0.1] = 1
                
                #H_D
                #Hess = abs(Hess)+1
                
                Weight_updates[-1-p] = delta_w / Hess
                
                A_prev = A_l.copy()
                
            

            #Hess=0
            
            
            ##########################
            ###### FULL HESSIAN ######
            ##########################
            
            ####################
            ##### Smoother #####
            ####################
            '''
            old_A = A_prev.copy()
            for I in range(5): #Number of smoothing steps
            
                padding_avg = np.zeros( np.shape(A_l) )
                K = 2
                for i in range(100):
                    for j in range(128):
                        
                        if j==127 or i == 99:
                            if j==127:
                                padding_avg[ i , j ] = np.sum(old_A[ i, -2:-1  ]) / 2
                            if i == 99:
                                padding_avg[ i , j ] = np.sum(old_A[  -2:-1 , j ]) / 2
                        
                        else:
                            padding_avg[ i , j ] = np.sum( old_A[ i: (i+K) , j: (j+K) ] ) / 4
            old_A = padding_avg / padding_avg.max()
            '''
            #####################
            #####################
            
            old_A = A_prev.copy()
            if p==2:
            #I = np.eye( H[-2-p], H[-2-p] )
                Hess_full = [ np.dot(old_A[i] * A[-2-p] , A[-2-p].T)  for i in range(len(A_prev)) ] # N x K x K
            else:
                pass
                #Hess_full = 0
            
            '''
            temp_sol = np.zeros(( np.shape(Hess)[0] , np.shape(Hess)[1] ))
            for j in range( np.shape(Hess)[0] ):
                
                #print(np.shape(Hess[j]), np.shape(delta_w))
                
                pre_sol = np.linalg.solve(Hess[j],-delta_w[j,:].T).T
                
                delta_w_j, _ = Adam(delta_class[2], p, pre_sol, pre_sol, batch_nr, bias_true=0)
                temp_sol[j,:] = delta_w_j
            Weight_updates[-1-p] = temp_sol
            '''
            
            ##### Method 1 #####
            "This method is not training"
            #Vt,Mt = AdaH(delta_class[2], p, delta_w.T, H_i, batch_nr, vectors = 0)
            #Weight_updates[-1-p] = np.linalg.solve(Vt,-Mt).T 

            #########################
            ###### AVG HESSIAN ######
            #########################
            '''
            avg_A = np.average( A_prev, axis = 0 )
            avg_prop_A = np.reshape( avg_A , (1,dp) )
            H_i = np.dot( avg_prop_A * A[-2-p] , A[-2-p].T ) 
            H_i+= np.eye( np.shape(H_i)[0] )
            ##### Method 2 #####
            "This method will work on mnist"
            pre_sol = np.linalg.solve(H_i,-delta_w.T).T
            Vt,Mt = AdaH(delta_class[2], p, pre_sol , pre_sol, batch_nr, vectors = 0)
            delta_w, _ = Adam(delta_class[0], p, pre_sol, pre_sol, batch_nr, bias_true=0)
            Weight_updates[-1-p] = delta_w
            
            
            #Vt is pos def
            #H_i is not pos def
            
            
            
            #x = np.ones((1, np.shape(H_i)[0]))
            #pos_def = (x  @ H_i) @ x.T
            #if pos_def < 0:
            #print("Check pos def", pos_def)
            '''
            
            delta_b, _ = Adam(delta_class[0], p, delta_b, delta_b, batch_nr, bias_true=1)
            Bias_updates[-1-p] = delta_b
            

        else:
            ''' 1st order optimizer '''
            delta_w, _ = Adam(delta_class[0], p, delta_w, delta_w, batch_nr, bias_true=0)
            delta_b, _ = Adam(delta_class[0], p, delta_b, delta_b, batch_nr, bias_true=1)
            Weight_updates[-1-p] = delta_w
            Bias_updates[-1-p] = delta_b
            
            #A_prev = 0
            
#    H_i=0
#    Hess=0
#    Hess_full = 0
    

            

    for I in range(n):
        W[I] -= eta* Weight_updates[I] * lr #+ 0.0001 * W[I] #regularization
        b[I] -= eta* Bias_updates[I]   * lr*0 #+ 0.0001 * b[I] 

    
    delta = np.dot( W[0].T, delta ) #* Act(H_act[0],A[0],1) #/dp#* A[0]
    
    
    return W, b, delta, Hess_full, Hess, Weight_updates 

def dropout(sh):
    drop = np.random.random(sh)*3-1
    #drop = np.array(drop)
    drop[drop<=0]=1
    drop[drop>0] =1
    return drop
    

def feedforward( x, H, W, b, drop=0,H_act=[], ch=1): #Does a complete feedforward iteration through the network
    z = []
    A = [x]
    for i in range(len(H)-1):
        dropout_matrix = dropout(np.shape(W[i]))
        
        
        if ch > 1: #multiple channels
            hidden_layer = np.zeros(( H[i+1] , np.shape(x)[1] , ch ))
            for j in range(ch):
                hidden_layer[:,:,j]  = np.dot( W[i], A[i][:,:,j] )
                #print(np.shape(hidden_layer))
                hidden_layer[:,:,j] += b[i]
                
                
        else:
            hidden_layer = np.dot(W[i]*dropout_matrix, A[i] ) #/ H[i]
    
            hidden_layer += b[i]
                
        z.append(  hidden_layer  )
        
        if i == len(H)-2: #no dropout. Final layer
            
            if ch > 1:
                z[i] = np.average( z[i], axis=-1 )
                A.append( Act(H_act[-1],z[i], 0) )
            else:
                A.append( Act(H_act[-1],z[i], 0) )
            
        else:
            
            A.append( Act(H_act[i],z[i], 0) ) 
    return  z, A



