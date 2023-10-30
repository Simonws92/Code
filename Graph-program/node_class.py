# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 22:43:57 2021

@author: Simon
"""


'''
Distributed parallel package
'''
#import torch.distributed.autograd as dist_autograd
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch import optim
#from torch.distributed.optim import DistributedOptimizer
#from torch.distributed.rpc import RRef



### This import cannot be here... ###
import load_dataset

import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn.parameter import Parameter

import random
import time
import numpy as np
import graph_network_templates as gt
import graph_algorithms as ga
import adj_matrices as am


import importlib

"creates a node of a sub/graph"
"The node is the final step of the sub_graph computation"

####################################
###### Network initialization ######
####################################
"Create custom network"
"Define wrapper for custom-made nodes?"
class create_custom(nn.Module):
    def __init__(self):
        pass
    
"Defining custom function must happen before calling 'create_custom'"
"Define custom operation using graphs"
"custom operation is then cast to pytorch which can use gpu acceleration"
"Edges must be defined manually"
"Custom database must be defined manually"
"Define custom forward operation"

" - adj matrix for forward operation"
" - adj matrix for custom operation"
" - custom inner-product operation"
" - activation functions"

###########################
###### Identity edge ######
###########################
class create_id(nn.Module):
    def __init__(self, network_type):
        super(create_id,self).__init__()
        
        self.H = network_type['H']
        self.id = nn.Identity()

    def forward(self,x):
        x = self.id(x)
        return x




"Creates base FC network"
#############################
###### Fully-connected ######
#############################
class create_fc(nn.Module):
    def __init__(self, H):
        super(create_fc,self).__init__()
        "Initiate fully connected"
        self.H = H
        self.relu      = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.tanh      = nn.Tanh()
        self.sigmoid   = nn.Sigmoid()
        self.softplus  = nn.Softplus()
        
        
        self.dropout = nn.Dropout(p=0.2)
        self.normalize = torch.nn.functional.normalize
        
        
        #self.Batch_norm = nn.BatchNorm2d( )
        
        
        for i in range(len(H)-1):
            exec( 'self.linear{} = nn.Linear(H[{}], H[{}], bias=True)'.format(i,i,i+1) )
    
    
    # Testing
    def srelu(self,x):
        x = self.sigmoid(x) + x/50 + self.softplus(x)
        return x
        

    def forward(self,x, AE=0):
        #x = (x - torch.mean( x,1 )[:,None])
            
        for i in range(len(self.H)-1):
            x = eval( ' self.linear{}(x)'.format(i) )
            #x = self.dropout(x)
            
            #batch_norm = torch.nn.BatchNorm1d( np.shape(x)[1] )
            #x = batch_norm(x)
            
            if i==len(self.H)-2:
                "does not perform activation on the final FC layer"
                pass
                
            else:
                
                #x = self.relu(x)       #348 iterations
                x = self.leakyrelu(x) #318 iterations a=0.2, 375 a=0.1
                #x = self.srelu(x)     #941 iterations
                
            #if len(np.shape(x)) == 4:
                

            
            #eps = 0.001
            #standard_dev = torch.std(x+eps,0, unbiased=False)[:,None]
            
            #x = (x - torch.mean( x,1 )[:,None]) #/ ( standard_dev.detach() )
            #s = torch.sum(x)
            #print(i,s)
            
            #Naive normalization
            #x = x / torch.max(x.detach(),0)[0]
            
        return x


########################
######    CONV    ######
########################
"Creates base conv network"
class create_conv(nn.Module):
    """
    Can we store intermediate convolution values?
    """
    
    "args = padding, stride, filter size, pool_mod" #[2,1,5,4]
    def __init__(self, H, identity, pool_type, dim,*args):
        super(create_conv,self).__init__()
        p=args[0]
        s=args[1]
        k=args[2]
        pool_mod=args[3]
        
        self.H = H
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.pool_mod = pool_mod
        self.pool_type = pool_type
        self.nr_kernels = len(k)
        self.upsample = torch.nn.Upsample( scale_factor=2, mode='bilinear', align_corners = False)
        
        self.dim = dim
        
        for i in range(len(H)-1):
            for j in range(self.nr_kernels):
                
                exec( 'self.conv{}_{} = nn.Conv2d(H[{}],H[{}],k[{}], padding=p[{}], stride=s)'.format(i,j,i,i+1,j,j) ) #For normal CNN only
                if identity == True:
                    
                    "For some reason, Parameter flips the channels"
                    w = eval( 'torch.empty( H[{}],H[{}],k[{}],k[{}])'.format(i+1,i,j,j) )
                    #exec( 'self.conv{}_{}.parameters = nn.init.dirac_(w)'.format(i,j) )
                    with torch.no_grad():
                        exec( 'self.conv{}_{}.weight = Parameter( nn.init.dirac_(w) )'.format(i,j) )
                    #exec( 'self.conv{}_{}.bias = False'.format(i,j))
    
    def forward(self,x, **kwargs):
        pool_type = self.pool_type
        pool_mod = self.pool_mod
        #print( self.parameters )
        for i in range( len(self.H)-1 ):
            
            
            "Loop through all kernel sizes, aka highway-connection"
            if self.nr_kernels > 1:
                x_temp = 0
                for j in range(self.nr_kernels):
                    x_temp += eval( 'F.relu(self.conv{}_{}(x))'.format(i,j) ) #/ (self.nr_kernels)
                    #print(shape(x),shape(x_temp), self.nr_kernels)
                x = x_temp
            if self.nr_kernels == 1:
                x = eval( 'F.relu(self.conv{}_{}(x))'.format(i,0) ) #/ (self.nr_kernels)
            
            
            if pool_mod==0:
                pass
            else:
                if i%pool_mod==0:
                    if pool_type == 'max':
                        x = eval( 'self.pool(x)'.format(i) )
                    if pool_type == 'upsample':
                        x = eval( 'self.upsample(x)'.format(i) )
        
        return x

###########################
######      LSTM     ######
###########################
class create_lstm(nn.Module):
    def __init__(self, A, A_op):
        super(create_lstm,self).__init__()
        #print("in lstm init")
        
        #the interior design of this lstm from network_type
        self.A = A
        self.A_op = A_op
        lstm_block = graphs( self.A, self.A_op, 'lstm_block' )
        graphs.compile_graph( lstm_block )
        self.lstm_block = lstm_block
        
    def forward(self, x_in):
        #print("in lstm fwd")
        "Prefer to wrap this code inside a class"
        "Creates an lstm object using the hybrid graph method"
        ts = np.shape(x_in)[1]
        lstm_out = torch.zeros( np.shape(x_in) )
        timeout = 0
        for t in range(ts):
            
            #print("Timestep in lstm:", t)
            #We assign the first time input to the lstm
            
            
            "We compute the lstm-block for one timestep"
            lstm_in = x_in[:,t,:,:]
            _ = graphs.pseudo_forward_TEST( self.lstm_block, lstm_in, input_nodes = 1 )
            
            "Nodes 0,1,2 are the input nodes X,H,C respectivily "
            internal = self.lstm_block.adj_node_output_9.clone()
            timeout  = self.lstm_block.adj_node_output_11.clone()
            self.lstm_block.node_output_temp_2 = internal
            #We must assign the second time input to the lstm
            
            lstm_out[:,t,:,:] = timeout.clone()
            
            create_lstm.clear_values(self)
            
        return lstm_out
    
    def clear_values(self):
        for i in range( len(self.A) ):
            try:
                exec('self.lstm_block.adj_node_output_{}'.format(i) )
            except:
                pass


############################
######   CUSTOM ATT   ######
############################
class create_attention(nn.Module):
    """
    Graph attention is a node-pairwise model
    
    
    """
    
    def __init__(self, network_type ):
        super(create_attention, self).__init__()
        
        def softmax(X):
            eX = torch.exp(X)
            a  = eX / torch.sum( eX )
            
            return a
        
        
        
        
        

############################
######   CUSTOM RNN   ######
############################
class create_custom_rnn(nn.Module):
    def __init__(self, network_type ):
        super(create_custom_rnn,self).__init__()
        
        #print()
        #print("in custom rnn init")
        
        "We need to create initial Ct and Ht values"
        self.Ct = torch.tensor(0)
        self.Ht = torch.tensor(0)
        self.initial_value = False
        self.snapshot_step = 1
        
        #the interior design of this lstm from network_type
        self.A = network_type['A']
        self.A_op = network_type['A_op']
        rnn_block = graphs( self.A, self.A_op, 'lstm_block' )
        graphs.compile_graph( rnn_block )
        self.rnn_block = rnn_block
        
        self.in_ft  = network_type['H'][0]
        self.out_ft = network_type['H'][-1]
        
        self.network_type = network_type
        
        # We need to call simple_DFS to receive a list of models.
        # LSTM uses GCN depth of 1
        # Then we call the graph traversal algorithm to compute LSTM
        # Currently graph traversal does not respect hierchical traversal order.
        # We need to adjust GCN such that it will respect correct traversal order.
        
        #ga.model_name_converter( self.obj_lstm.List_of_models[0] ) #broken...
        
        
        ### This RNN requires a gcn operator which we must create
        # dist_A depends on the data, not the model used
        
        "We want to implement gcn with layers"
        " Very early alpha... work in progress "
        # self.gcn_op = graphs(network_type['A'], 'obj', gs_dim = self.in_ft , graph_conv=True, graphsage=True, gcn=2)
        # self.gcn_op.dist_A, _ = load_dataset.load_metrla()
        # graphs.compile_graph( self.gcn_op )
    
    def flush(self, x_in):
        "We call this function when the dimensions of the input does not match internal timestep values"
        "Flush old timestep values to avoid conflict with new input"
        
        self.Ct = torch.tensor(0)
        self.Ht = torch.tensor(0)
        self.initial_value = False
        self.snapshot_step = 1
        
        self.in_ft  = self.network_type['H'][0]
        self.out_ft = self.network_type['H'][-1]
        
        # try:
        #     #del self.rnn_block.input_values
        #     delattr( self.rnn_block , "input_values" )
        # except:
        #     pass
        
    def forward(self, x_in):
        
        if np.shape(self.Ht) != np.shape(x_in):
            create_custom_rnn.flush(self, x_in)
        
        "Prefer to wrap this code inside a class"
        "Creates an lstm object using the hybrid graph method"
        
        out_ft = self.out_ft
        if len(np.shape(x_in))>3:
            n  = np.shape(x_in)[-4]
        
        ts = np.shape(x_in)[-3]
        bs = np.shape(x_in)[-2]
        ft = np.shape(x_in)[-1]
        
        if len(np.shape(x_in))==3:
            rnn_in   = torch.zeros( ts, bs, ft )
            rnn_out  = torch.zeros( ts, bs, out_ft )
            timeout  = torch.zeros( bs, out_ft )
            internal = torch.zeros( bs, out_ft )
        
        if len(np.shape(x_in))==4:
            rnn_in   = torch.zeros( n, ts, bs, ft )
            rnn_out  = torch.zeros( n, ts, bs, out_ft )
            timeout  = torch.zeros( n, bs, out_ft )
            internal = torch.zeros( n, bs, out_ft )
            
        if len(np.shape(x_in))==5:
            ch = np.shape(x_in)[0]
            
            rnn_in   = torch.zeros( ch, n, ts, bs, ft )
            rnn_out  = torch.zeros( ch, n, ts, bs, out_ft )
            timeout  = torch.zeros( ch, n, bs, out_ft )
            internal = torch.zeros( ch, n, bs, out_ft )
        
        "Assign initial value for this snapshot"
        if self.initial_value==True:
            pass
            # timeout  = self.Ht
            # internal = self.Ct
        
        # These defines must be defined outside forward...
        self.rnn_block.multiple_input = True
        self.rnn_block.input_node = ['NA',1,2] #list of input nodes
        self.rnn_block.output_node = [9,11]
        self.rnn_block.input_values = [ 0, internal, timeout ]
        
        rnn_in = x_in #currently not used
        
        for t in range(ts):
            #print("Timestep LSTM block:", t)
            
            # """
            # Not needed for now
            # We call a fork on our lstm operator which performs gcn
            # We perform gcn on every lstm timestep
            # Because we need to perform gcn at every timestep,
            # the lstm section is very slow and should be optimized
            # """ 
            
            # """
            # for j in self.gcn_op.sensor_nodes:
            #     if j in self.gcn_op.banned_nodes:
            #         pass
            #     else:
            #         ls = eval('self.gcn_op.ga_object.GCN_list_{}'.format(j))
            #         node.traverse_branch(self.gcn_op, ls, x_in[:,t].detach().clone(), j)
            #         rnn_in[j] = eval('self.gcn_op.adj_node_output_{}'.format(j)).clone()
            # """
            
            "We run the internal state of self.rnn_block"
            # pseudo_forward_TEST is not complete yet.
            #_ = graphs.pseudo_forward_TEST( self.rnn_block, rnn_in, input_nodes = 1 )
            

            if len(np.shape(x_in))==3:
                rnn_input = x_in[t]
                
            if len(np.shape(x_in))==4:
                rnn_input = x_in[:,t]
            
            if len(np.shape(x_in))==5:
                rnn_input = x_in[:,:,t]
                
            _ = node.graph_traversal(self.rnn_block, rnn_input, timestep=t)
            internal = eval('self.rnn_block.adj_node_output_{}'.format(9)).clone()            
            timeout  = eval('self.rnn_block.adj_node_output_{}'.format(11)).clone()
            
            if len(np.shape(x_in))==3:
               rnn_out[t,:,:]      = timeout.clone()
            
            if len(np.shape(x_in))==4:
               rnn_out[:,t,:,:]    = timeout.clone()
               
            if len(np.shape(x_in))==5:
                rnn_out[:,:,t,:,:] = timeout.clone()
            
            #print( internal[0,0] , internal[0,1] )
            
            ### We need to normalize the timeblock dep. output value,
            ### otherwise this time dep value will accumulate and explode
            
            #.detach() for truncated gradient computations
            #self.rnn_block.input_values = [ 0, internal.detach().clone(), timeout.detach().clone() ]
            self.rnn_block.input_values = [ 0, internal, timeout ]
            
            "We save our Ct and Ht values for the next snapshot"
            if t == self.snapshot_step-1:
                self.initial_value = True
                self.Ct = internal.detach().clone()
                self.Ht = timeout.detach().clone()
            
            #print( np.shape(self.Ct) , np.shape(self.Ht) )
            
            create_custom_rnn.clear_values(self)
        
        #rnn_out = rnn_out / torch.max( rnn_out.detach() , 0 )[0]
        
        # if len(np.shape(rnn_out)) == 4:
        #     batch_norm = torch.nn.BatchNorm2d( np.shape(rnn_out)[1] )
        #     rnn_out = batch_norm(rnn_out)
            
        return rnn_out
    
    def clear_values(self):
        
        for i in range( len(self.A) ):
            try:
                exec('del self.rnn_block.adj_node_output_{}'.format(i) )
            except:
                pass


##########################
######      GAT     ######
##########################
class create_GAT(nn.Module):
    
    """
    Creates a GAT module
    """
    
    def __init__(self, network_template):
        
        super(create_gcn, self).__init__()
        
        pass
        

##########################
######      GCN     ######
##########################
class create_gcn(nn.Module):
    
    """
    Important:
        The gcn model is gcn_test.edge_1_0
    """
    
    "args = padding, stride, filter size, pool_mod" #[2,1,5,4]
    def __init__(self, network_template ):
        
        super(create_gcn,self).__init__()
        
        #print("creating gcn...")
        
        "H[0] = gcn channels, H[1] = edge_connection architecture"
        K = network_template["args"][0]
        H = network_template["H"]
        A = network_template["A"] #This A is only for edge_connection aggregator model
        
        self.ls = 0 #placeholder value for graph traversal. Must be given during forward
        self.K = K
        self.H = H[0]
        self.graphsage = True
        self.graph_pool_bool = False
        "The adj matrix for GCN must be defined before calling gcn.forward"
        
        for h in range(len(self.H)-1):
            for i in range(self.H[h+1]):
                for k in range(self.H[h]):
                    "obj_{}_{}_{}, layer, in channel, out channel"
                    exec( 'obj_{}_{}_{} = graphs(A, "obj_{}_{}_{}", graph_conv=True, graphsage=True, gcn=2) '.format(h,i,k,h,i,k))
                    #exec( 'obj_{}_{}_{}.dist_A = dist_A'.format(h,i,k))
                    exec( 'graphs.compile_graph(obj_{}_{}_{})'.format(h,i,k))
                    exec( 'self.obj_{}_{}_{} = obj_{}_{}_{}'.format(h,i,k,h,i,k) )
                    exec( 'self.obj_{}_{}_{}.name = "obj_{}_{}_{}"'.format(h,i,k,h,i,k) )
                    
    
    def save_adj_matrix(self, A, dist_A):
        "This is the adj matrix for the graph traversal"
        self.A = A
        self.dist_A = dist_A
    
    def save_branch_connections(self, bc):
        self.branch_connections = bc
    
    def save_graph_connections(self, A):
        #print("test here")
        "We create an object to store all ls lists for each node"
        self.ga_object = ga.graph_algorithm_class()
        self.connections = []
        
        for j in range(len(A)):
            if j in self.banned_nodes:
                pass
            else:
                list_of_models = self.List_of_models[0]
                
                "this function only returns the last branch_connection"
                ls = self.ga_object.simple_DFS( list_of_models , self.K, j )
                
                "We store all branch connections for each node in metrla dataset"
                self.connections.append(ls)
    
    
    #####################################
    ###### GRAPH POOLING PROTOTYPE ######
    #####################################
    "This code will be executed after GCN"
    def graph_pool(A, X):
        """
        X should have dimension 5.
        X shape: (ch, nodes, bs, ts, ft)
        """
        
        "Converts adj matrix to float matrix"
        A = torch.tensor(A, dtype=torch.float).detach()
        
        # X must be matrixmultiplied with Restriction matrix wrt nodes.
        if len(np.shape(X))==5:
            X=X.permute(0,4,2,3,1) # nodes and ft must switch places
        if len(np.shape(X))==4:
            X=X.permute(1,2,3,0)
        
        # X_node_information must be matrix multiplied with Restriction matrix
        # X should not be repeated due to its large size
        
        N = len(A)
        if N%2==0:
            n = int(N/2)-1
        else:
            n = int((N-1)/2)
        
        # We must produce a restriction matrix
        R = torch.zeros((n,N))
        for i in range(n):
            R[i, 2*i:2*i+3 ] = torch.tensor([1,2,1])
        Re = R.T/2
        
        
        """
        This function samples all incoming connections and relevant data
        for every node. Then collects all connected data into new nodes
        of the new graph of smaller resolution/ fewer nodes.
        """
        
        ch = np.shape(X)[0]
        ft = np.shape(X)[1]
        bs = np.shape(X)[2]
        ts = np.shape(X)[3]
        nodes = N
        new_nodes = n
        
        
        A_Re      = torch.matmul(A, Re)
        A_coarse  = torch.matmul(Re.T , A_Re )
        A_coarse -= torch.diag( torch.diag(A_coarse))
        A_coarse = A_coarse.detach()
        
        
        temp_1 = torch.zeros( ch, ft, bs, ts, nodes, new_nodes )
        
        st = time.time()
        "Computes 1st restriction"
        for l in range(N):
            temp_1[:,:,:,:,l,:] = torch.matmul( X * A[l], Re.detach() )
            
        #print("Time 1:", time.time() - st)
        
        
        "Computes 2nd restriction"
        st = time.time()
        s = 2*n
        temp_2 = torch.zeros( ch, ft, bs, ts, new_nodes, new_nodes)
        Coarse_data = torch.zeros(ch,ft,bs,ts,new_nodes)
        
        st = time.time()
        r_test = torch.matmul( temp_1.permute(0,1,2,3,5,4) , Re.detach())
        
        "We must permute back"
        temp_2 = r_test.permute(0,1,2,3,5,4)
        
        et = time.time()
        #print("time 2:",et - st)
        
        "Distribute values"
        st = time.time()
    
        for l in range(n):
            S3 = torch.sum( temp_2[:,:,:,:, l , l+1: ] * A_coarse[l, l+1:], axis = -1 )
            S2 = torch.sum( temp_2[:,:,:,:, : , l  ] * A_coarse[:, l ], axis = -1 )
            
            Coarse_data[:,:,:,:,l] += S2
            Coarse_data[:,:,:,:,l] += S3
            
            if l != 0:
                S1 = torch.sum( temp_2[:,:,:,:, l , :l ] * A_coarse[l, :l], axis = -1 )
                Coarse_data[:,:,:,:,l] += S1
    
        #print("Time 3:", time.time() - st)
            
        Coarse_data /= Coarse_data.max()
        Coarse_data = Coarse_data.permute(0,4,2,3,1)
        
        ### A_coarse cleaning ###
        A_coarse = np.array(A_coarse)
        A_coarse /= A_coarse.max()        # normalize. prototype
        A_coarse[ A_coarse < 0.1 ] = 0    # Remove low priority connections
        N = len(A_coarse)
        list_of_models = [ (i,j) for i in range(N) for j in range(N) if A_coarse[i,j]>0 ]
        
        ### Check for isolated nodes ###
        isolated_nodes = [ i for i in range(N) if np.sum(A_coarse[i])==0 ]
        isolated_nodes += [ j for j in range(N) if np.sum(A_coarse[:,j])==0 ]
        #print("isolated nodes:",isolated_nodes)
        
        ### Create traverse order ###
        
        "We create an object to store all ls lists for each node"
        "Since we're dealing with GCN, there is no hierarchial traverse order"
        "No need to sort list_of_models"
        ga_object = ga.graph_algorithm_class()
        connections = []
        
        for j in range(len(A_coarse)):
            
            "this function only returns the last branch_connection"
            "We perform single-depth GCN of coarsened graph. Double-depth possible, time consuming"
            ls = ga_object.simple_DFS( list_of_models , 1, j )
            
            "We store all branch connections for each node in metrla dataset"
            connections.append([ls])
        
        return A_coarse, Coarse_data, connections
    
    ##########################
    ###### Prolongation ######
    ##########################
    def graph_prolongation(A_coarse, A_fine, X):
        """
        X should have dimension 5.
        X shape: (ch, nodes, bs, ts, ft)
        """
        
        "Converts adj matrix to float matrix"
        
        # X must be matrixmultiplied with Restriction matrix wrt nodes.
        if len(np.shape(X))==5:
            X=X.permute(0,4,2,3,1) # nodes and ft must switch places
        if len(np.shape(X))==4:
            X=X.permute(1,2,3,0)
        
        # X_node_information must be matrix multiplied with Restriction matrix
        # X should not be repeated due to its large size
        
        N = len(A_fine)
        n = len(A_coarse)
        
        # We must produce a restriction/prolongation matrix
        R = torch.zeros((n,N), dtype=torch.float)
        #print(np.shape(R))
        for i in range(n):
            R[i, 2*i:2*i+3 ] = torch.tensor([1.,2.,1.])
        
        
        P = torch.tensor(R/2, dtype=torch.float).clone().detach().requires_grad_(False)
        A_coarse = torch.tensor(A_coarse, dtype=torch.float).detach().requires_grad_(False)
        A_fine   = torch.tensor(A_fine  , dtype=torch.float).detach().requires_grad_(False)
        #print("ok")
        
        """
        This function samples all incoming connections and relevant data
        for every node. Then collects all connected data into new nodes
        of the new graph of smaller resolution/ fewer nodes.
        """
        
        ch = np.shape(X)[0]
        ft = np.shape(X)[1]
        bs = np.shape(X)[2]
        ts = np.shape(X)[3]
        nodes = n
        new_nodes = N
        
        temp_1 = torch.zeros( ch, ft, bs, ts, nodes, new_nodes )
        
        st = time.time()
        "Computes 1st restriction"
        
        #print("temp_1",np.shape(temp_1))
        #print("X:",np.shape(X))
        #print("A_coarse:", np.shape(A_coarse))
        #print("P:", np.shape(P))
        
        
        for l in range(n):
            temp_1[:,:,:,:,l,:] = torch.matmul( X * A_coarse[l], P )
        #print("Time 1:", time.time() - st)
        
        
        "Computes 2nd restriction"
        st = time.time()
        s = 2*n
        temp_2 = torch.zeros( ch, ft, bs, ts, new_nodes, new_nodes)
        Fine_data = torch.zeros(ch,ft,bs,ts,new_nodes)
        
        st = time.time()
        r_test = torch.matmul( temp_1.permute(0,1,2,3,5,4) , P)
        
        "We must permute back"
        temp_2 = r_test.permute(0,1,2,3,5,4)
        
        et = time.time()
        #print("time 2:",et - st)
        # for l in range(n): #old 2nd restriction
        #     r_test = torch.matmul( temp_1[:,:,:,:,:,l] , Re )
        #     temp_2[:,:,:,:,:,l] = r_test
        
        
        "Distribute values"
        st = time.time()
    
        for l in range(N):
            S3 = torch.sum( temp_2[:,:,:,:, l , l+1: ] * A_fine[l, l+1:], axis = -1 )
            S2 = torch.sum( temp_2[:,:,:,:, : , l  ] * A_fine[:, l ], axis = -1 )
            
            Fine_data[:,:,:,:,l] += S2
            Fine_data[:,:,:,:,l] += S3
            
            if l != 0:
                S1 = torch.sum( temp_2[:,:,:,:, l , :l ] * A_fine[l, :l], axis = -1 )
                Fine_data[:,:,:,:,l] += S1
    
        #print("Time 3:", time.time() - st)
        
        
        ### Not detaching before .max() breaks gradient computations
        Fine_data /= torch.abs(Fine_data).detach().max()
        Fine_data = Fine_data.permute(0,4,2,3,1)
    
        
        #A_coarse must be cleaned
        
    
        return A_fine.detach(), Fine_data

    
    
    def forward(self, x, A=0, compute_nodes = [], ignore_nodes = [], 
                start_layer=0,end_layer=0):
        # GCN traversal must be optimized
        
        banned_nodes = self.banned_nodes
        sensor_nodes = self.sensor_nodes
        connections  = self.traverse_connections
        
        if len(compute_nodes) > 0:
            "if only a specific sensor node is to be computed"
            sensor_nodes = compute_nodes
        if len(ignore_nodes) > 0:
            banned_nodes += ignore_nodes
        
        sensor_nodes = [ sensor_nodes[i] for i in range(len(sensor_nodes)) \
                        if sensor_nodes[i] not in banned_nodes ]
            
        #print("standard sensors:",len(sensor_nodes))

        
        "adj matrix can be inserted with the input, default is none"
        if A==0:
            "Load adj matrix from storage. Must be defined if not with input"
            "This adj matrix is used for traversing the dataset model"
            A      = self.A
            dist_A = self.dist_A
        
        if self.ls == 0:
            # If the current GCN have no defined traversal order
            #print("Could not find ls checkmark... creating new traversal order")
            create_gcn.save_graph_connections(self, A)
        
        
        init_ch, n,ts,bs,ft = np.shape(x)

        "Go through the gcn layers"
        
        st = time.time()
        
        
        compute_nr_layers = len(self.H)-1 #Default
        if end_layer > 0:
            compute_nr_layers = end_layer
        
        #for h in range(len(self.H)-1):
            
        for h in range( start_layer , compute_nr_layers ):
            #print("gcn layer:", h)
            
            in_ch  = self.H[h]
            out_ch = self.H[h+1]
            
            # placeholder output
            x_temp = torch.zeros( out_ch , n , ts , bs , ft )
            gcn_output = torch.zeros( n , ts , bs , ft )
            
            for i in range(self.H[h+1]):
                #print("Computing out channels...")
                
                """
                
                # Now define traverse order for new adj matrix
                A,x = create_gcn.compute_layer(self, x, in_ch, out_ch, h, A)
                
                """
                
                for k in range(self.H[h]):
                    
                    #print("Channel:", i,k)
                    
                    """
                    Performs GCN for a single channel
                    This gcn traversal currently only computes sensor nodes...
                    We may want all nodes to be computed except for bannednodes
                    
                    looping through all sensor nodes is too slow... Needs optimization
                    """
                    
                    "Convert this for loop to multiprocess"
                    st = time.time()
                    for j in sensor_nodes:
                        
                        #print(j)
                        #workaround due to bug wip... due to empty rows in adj matrix
                        # Bannednodes cannot be computed due to no edge connection
                        if j in banned_nodes:
                            pass
                        else:
                            
                            try:
                                # This approach is no longer used. Depricated
                                ls = eval('self.ga_object.GCN_list_{}'.format(j))
                            except:
                                # Must add possibility of adding custom traverse_connections
                                #ls = self.traverse_connections[j]
                                ls = connections[j]
                                
                                #print()
                                #print(j, ls)
                            # We need to give the operation obj_{}_{}_{} the graph traversal adj matrix
                            exec( 'self.obj_{}_{}_{}.A = A'.format(h,i,k) )
                            exec( 'self.obj_{}_{}_{}.dist_A = dist_A'.format(h,i,k) )
                            exec( 'node.traverse_branch(self.obj_{}_{}_{}, ls, x[k].clone(), j)'.format(h,i,k) )
                            gcn_output[j] = eval('self.obj_{}_{}_{}.adj_node_output_{}'.format(h,i,k,j)).clone()
                            
                            
                            "Batch/node normalization"
                            """
                            if gcn_output[j].max() <= 0:
                                div_factor = 0.1
                                
                            else:
                                div_factor = gcn_output[j].max().detach()
                            """
                            
                            ######################################
                            ### Need to normalize wrt Features ###
                            ######################################
                            dims = np.shape(gcn_output[j])
                            ft = dims[-1]
                            
                            
                            for f in range(ft):
                                #print(f,ft)
                                #print(np.shape(gcn_output), np.shape(gcn_output[j]))
                                ft_mean = torch.mean(gcn_output[(j, Ellipsis, f )]).detach()
                                ft_std  = torch.std( gcn_output[(j, Ellipsis, f )]).detach()
                                
                                gcn_output[j][(Ellipsis,f)]-=ft_mean
                                
                                #print("std",ft_std)
                                if ft_std < 0.001:
                                    ft_std=0.001
                                
                                gcn_output[j][(Ellipsis,f)]/=ft_std
                                
                            """
                            gcn_output[j] -= gcn_output[j].detach().mean()
                            #div_factor = gcn_output[j].max() - gcn_output[j].min()
                            div_factor = gcn_output[j].detach().std()
                            if div_factor == 0:  ### in the unlikely event
                                div_factor=0.1
                                #print(div_factor)
                                
                            gcn_output[j]/=div_factor
                            """
                            
                            
                            # Delete temp values to save memory
                            exec('del self.obj_{}_{}_{}.adj_node_output_{}'.format(h,i,k,j))
                    #print("normal time:", time.time()-st)

                    # This sums the gcn layer channels
                    #print("channel gcn output",np.shape(gcn_output))
                    
                    
                    #with torch.no_grad():
                    #    div_factor = ((gcn_output).max() )
                    #    print(div_factor)
                        
                    x_temp[i] += gcn_output #/ div_factor
                    
            x = x_temp #/ torch.abs(x_temp.detach())
            
            #div_factor = torch.abs(x).max(axis=3)[0].detach()
            #print(np.shape(x),np.shape(div_factor))
            #torch.div(x, div_factor)
            #x /= div_factor
        
            " When we pool the graph, we must create a new traverse order "
            "add new sensor and banned nodes"
            if self.graph_pool_bool == True:
                #print("at gcn layer... ", h)
                if h == 0:
                    A,x,connections = create_gcn.graph_pool(A,x)
                    #print("Performing graph pooling...")
                    
                    n = len(A)
                    sensor_nodes = [i for i in range(len(A))]
                    banned_nodes = []
                    #print("pooled x:", np.shape(x))
                    
                if h == 1:
                    #print()
                    #print("Performing graph prolongation")
                    #print("shape of input:", np.shape(x))
                    A,x = create_gcn.graph_prolongation( A , self.A , x )
                    n = len(A)
                    connections  = self.traverse_connections
                    sensor_nodes = self.sensor_nodes
                    banned_nodes = self.banned_nodes
        
        #print("GCN time:", time.time() - st)
                    
        del x_temp
        del gcn_output
            
        return x
    
    
    def compute_layer(self, x, in_ch, out_ch,h,A=0):
        
        "adj matrix can be inserted with the input, default is none"
        if A==0:
            "Load adj matrix from storage. Must be defined if not with input"
            "This adj matrix is used for traversing the dataset model"
            A      = self.A
            dist_A = self.dist_A
        
        if self.ls == 0:
            # If the current GCN have no defined traversal order
            print("Could not find ls checkmark... creating new traversal order")
            create_gcn.save_graph_connections(self, A)
        
        
        init_ch, n,ts,bs,ft = np.shape(x)
        
        "Go through the gcn layers"
        gcn_output = torch.zeros( n , ts , bs , ft )
        
        
        # placeholder output
        x_temp = torch.zeros( out_ch , n , ts , bs , ft )
        
        for i in range(out_ch):
            #print("Computing out channels...")
            
            for k in range(in_ch):
                
                #print("Channel:", i,k)
                
                # Performs GCN for a single channel
                # This gcn traversal currently only computes sensor nodes...
                # We may want all nodes to be computed except for bannednodes
                for j in self.sensor_nodes:
                    
                    if j%100==0:
                        pass
                        #print("Node:", j)
                    #workaround due to bug wip... due to empty rows in adj matrix
                    # Bannednodes cannot be computed due to no edge connection
                    if j in self.banned_nodes:
                        pass
                    else:
                        try:
                            # This approach is no longer used. Depricated
                            ls = eval('self.ga_object.GCN_list_{}'.format(j))
                        except:
                            ls = self.traverse_connections[j]
                        
                        # We need to give the operation obj_{}_{}_{} the graph traversal adj matrix
                        exec( 'self.obj_{}_{}_{}.A = A'.format(h,i,k) )
                        exec( 'self.obj_{}_{}_{}.dist_A = dist_A'.format(h,i,k) )
                        exec( 'node.traverse_branch(self.obj_{}_{}_{}, ls, x[k].clone(), j)'.format(h,i,k) )
                        gcn_output[j] = eval('self.obj_{}_{}_{}.adj_node_output_{}'.format(h,i,k,j)).clone()
                        
                        # Delete temp values to save memory
                        exec('del self.obj_{}_{}_{}.adj_node_output_{}'.format(h,i,k,j))
                        
                # This sums the gcn layer channels
                x_temp[i] += gcn_output / in_ch
        
        x = x_temp
        # When we pool the graph, we must create a new traverse order
        A,x = create_gcn.graph_pool(A,x)
        
        del x_temp
        del gcn_output
        
        return x
            
        

#####################################
###### DEFINE GRAPH STRUCTURES ######
#####################################
class network_variables():
    def __init__(self, LR):
        self.LR = LR
        #self.device = "cpu"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("Device selected by default:", self.device)
    
class graphs(nn.Module):
    
    def __init__(self , A_meta=0, A_op=0, name='', gs_dim = 1,\
                 graph_conv = False, graphsage=False, gcn=1, add_model=True,\
                     list_of_members = [] ):
        
        super (graphs, self).__init__()
        try:
            self.network_settings = ntw.LR
        except:
            self.network_settings = network_variables(0.0)
        
        
        self.device = self.network_settings.device    
        self.LR = self.network_settings.LR
        self.K = gcn #GCN kernel size
        
        self.graph_conv = graph_conv
        self.graphsage = graphsage
        if self.graphsage==True:
            graphs.add_graphsage(self, gs_dim)
        

        self.parent = ''
        self.node_nr = 0

        self.name = name
        self.save = 0
        self.node_count = 0
        self.fc_count   = 0
        self.conv_count = 0
        self.List_of_models = [ [] , [] ]                                      #contains information about the edges
        self.List_of_nodes  = []
        self.A = A_meta #new
        self.dist_A = np.ones(np.shape(self.A))
        
        self.input_nodes = 1
        
        if type(A_meta) == int:
            self.A = np.zeros((1,1))
        
        if A_op == 0:
            "S sum and I id are default"
            self.A_op = [('S','I')]*len(A_meta)
        else:
            self.A_op = A_op

        "Define the adj operation wrt self.A = A"
        self.add_model = add_model
        graphs.adjacency_operation(self)
        self.multiple_input = False
        
        self.add_model = True
        
        
        "The graph must now be sorted"
        #as of sept 2022, sort_graph is broken
        #graphs.sort_graph(self)
        
    def set_kernel_size(self, kernel_size = 1):
        self.K = kernel_size
    def set_learning_rate(self, lr = 0.003):
        self.LR = lr
        
    def add_graphsage(self, gs_dim = 1):
        K = self.K
        
        #define_operations(self, network_type)
        
        opti = 'Adam'
        LR = 0.0000
        
        network_id = 0
        
        "We look for a network_id that satisfies the gs_dim"
        
        for i in range(K):
            
            "We load a network_template by either specifying network_id or dimension dim"
            self.network_type = gt.network_templates( network_id, dim = gs_dim )
            
            exec( 'self.gcn_edge_{}, self.gcn_edge_{}_pytorch = ( self.define_operations( self.network_type ) ) '.format( i, i ) )
            
            exec( 'self.gcn_edge_{}.to(self.device)'.format(i))
            
            exec( 'self.gcn_optimizer_{} = optim.{}( self.gcn_edge_{}.parameters() , lr=LR, weight_decay=0)'.format(i, opti, i) )

        
    
    def adjacency_operation(self, reset = True):
        "Define the adj operation wrt adj matrix A"
        "USAGE: graphs.adjacency_operation( class_obj )"
        major=0
        minor=0
        "This loop must be run for each sub-graph"
        for row in self.A: 
            inc_nodes = []
            out_nodes = []
            "Input for node 'row' "
            for col in row:
                if col!=0:
                    
                    "Define edges in here"
                    if reset == True:
                        
                        #print("DEBUG, TEST HERE")
                        #print("A",self.A[major,minor])
                        graphs.define_edge(self, self.A[major,minor] , major,minor)
                    
                    "this piece creates the list_of_nodes connections"
                    out_nodes.append( minor )
                if self.A.T[major,minor]!=0:
                    inc_nodes.append( minor )
                    
                
                minor+=1
            self.List_of_nodes.append( [inc_nodes, out_nodes] )
            major+=1
            minor=0


    def insert_graph(self, A):
        "USAGE: graphs.insert_graph( class_obj, adj_matrix )"
        "insert a sub-graph inside another graph"
        "This adj matrix is solely for the sub-graph inserted. New adj meta must be defined after"
        exec( 'self.NODE_{} = nc.node(A)'.format(self.node_count) )
        self.node_count += 1
        graphs.adjacency_operation(self)
        
    
    #####
    ##### WORK IN PROGRESS
    #####
    "graph computation in here"
    def forward(self, X):
        
        #print("In main obj forward function")
        #print("Computing:", self.name)
        "Initiates feedforward for graph and nodes"
        
        "We need to call the forward of the self's child function"
        #But we're not defining Sub_graph anywhere
        
        "We need to be able to call the forward function of the member of self"
        
        "the operator is defined by the edge from the adj matrix self.A"
        
        "This list i asumed to be sorted"
        models = self.List_of_models[0]
        
        s = models[0][1]
        exec( "self.output_{} = X".format(s))
        
        for i in range(len(models)):
            t = models[i][0]
            s = models[i][1]
            #print("Computing internal state:", t,s)
            #exec( 'self.edge_{}_{}.forward(X)'.format(t,s))
            
            inp = eval("self.output_{}".format(s))
            try:
                exec( "self.output_{} += self.edge_{}_{}.forward(inp)".format(t,t,s) )
            except:
                exec( "self.output_{}  = self.edge_{}_{}.forward(inp)".format(t,t,s) )
                
            #x = eval( 'self.edge_{}_{}.forward(X)'.format(t,s))
            
        x = eval("self.output_{}".format(t))
        return x
        
    
    
    
    
    def call_opt(self, major, minor, LR = 0.003, opti = 'Adam'):
        "usage ex: graphs.call_opt( obj.Node, 1,0, LR = 0.001 )"
        #exec( 'self.optimizer_{}_{} = optim.Adam( self.edge_{}_{}.parameters() , lr=LR, weight_decay=0)'.format(major, minor, major, minor) )
        
        #### These b's gave best results
        b1 = 0.9
        b2 = 0.99
        #torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False)
        LR = 0
        exec( 'self.optimizer_{}_{} = optim.{}( self.edge_{}_{}.parameters() , lr=LR, betas=(b1,b2), weight_decay=0)'.format(major, minor, opti, major, minor) )
        


    ###############################################
    ###### Define edge connection operations ######
    ###############################################
    def define_operations(self, network_type ):
        "Defines the actual operations of edges between nodes"
        "Returns a class structure with properties used by pytorch"
        #print("in here")
        "runs through network type list"
        _type    = network_type["type"]
        #print(_type)
        
        if _type == ['id']:
            return create_id( network_type ), True
        
        if _type == ['gcn']:
            return create_gcn( network_type ), True
        
        if _type == ['lstm']:
            return create_lstm( network_type["A"], network_type["A_op"] ), True
        
        if _type == ['custom_rnn']:
            return create_custom_rnn( network_type ), True
        
        if _type == ['fc']:
            
            H = network_type["H"]
            A        = network_type["A"]
            args     = network_type["args"]
            pool     = network_type["pool"]
            try:
                identity = network_type["id"]
            except:
                identity = False
                
            #exec( 'self.FC{} =  '.format(self.fc_count) ) #For normal CNN only
            #self.fc_count += 1
            return create_fc(H), True
            
        if _type == ['conv']:
            print("create conv")
            H = network_type["H"]
            A        = network_type["A"]
            args     = network_type["args"]
            pool     = network_type["pool"]
            dim      = network_type["dim"]
            try:
                identity = network_type["id"]
            except:
                identity = False
            
            #exec( 'self.CONV{} = '.format(self.conv_count) ) #For normal CNN only
            #self.conv_count += 1
            return create_conv(H, identity, pool, dim, *args), True

        
        "This section allows for custom codes to be loaded into as nodes/edge connections"
        if _type == ['custom']:
            
            "custom operator must be loaded from its source code to a valid path etc"
            "or operator is loaded from valid source path..."
            
            "load custom source code from a valid path"
            path = network_type["path"]
            function = network_type["function"]
            
            mymodule = importlib.import_module(path)
            
            "function_return is mostly the wrapper function"
            function_return = eval( 'mymodule.{}'.format(function) )
            
            "Must define if pytorch operations are available"
            is_pytorch_available = network_type["pytorch"] #True or False
            
            "calls main custom function to handle custom code executions"
            
            print("DEBUG CUSTOM")
            return function_return( network_type ), is_pytorch_available
            


    def define_edge(self , network_id, major,minor):
        "edge_{'to_node'}_{'from_node'}" #zero-based"
        exec( 'self.network_type = gt.network_templates( network_id )'.format(major,minor) )
        
        "Define optimizer for edges"
        exec( 'self.List_of_models[0].append( ({},{}) ) '.format(major,minor) )

        #st_1 = time.time()
        if self.add_model == True:
            exec( 'self.edge_{}_{}, self.edge_{}_{}_pytorch = ( self.define_operations( self.network_type ) ) '.format( major, minor, major, minor ) )
            
            "graphs.call_opt is only used for valid pytorch class structures"
            if eval( 'self.edge_{}_{}_pytorch'.format( major , minor  ) ) == True:
                
                if self.network_type['type']==['id']:
                    "Identity connections do not carry optimizer"
                    
                    ## We should make an optimization list to use for opt.step()
                    pass
                else:
                    exec( 'graphs.call_opt( self, {}, {}, LR = self.LR)'.format(major, minor) )
            
            "For device placement"
            try:
                #print(self.device)
                exec( '(self.edge_{}_{}.to(self.device) ) '.format( major , minor  ) )
            except:
                pass
            
            #print("DEBUG", self.List_of_models)
            
            exec( 'self.List_of_models[1].append( self.edge_{}_{} ) '.format(major,minor) )
        
        
    def append_model(self, List_of_models, to_node, from_node):
        exec( 'self.edge_inc = "{},{}"'.format( to_node , from_node ) )
        if self.edge_inc in List_of_models: #to prevent duplicates
            pass
        else:
            graphs.define_edge( self, self.A[ to_node , from_node ] , to_node, from_node )
            List_of_models.append( self.edge_inc )
        return List_of_models
    
    
    def add_connection(self, node_1, node_2, conn):
        "Add a connection between two nodes regardless of graph-level"
        
        try:
            self.List_of_nodes[node_1][1].append( node_2 )  #to
            self.List_of_nodes[node_2][0].append( node_1 )  #from
        except:
            print("ERROR: Node values must have a valid id")
            print("input: graph name, to, from, type of connection")
        "Now that we have added a connection, we must add the edge-operation to the graph"
        "Adj matrix must be expanded"
        
        len_A = len(self.A)
        l = self.List_of_nodes
        "If we have more nodes than the adj matrix have registered"
        if len_A < len(l):
            "we must expand the adj matrix"
            A = np.zeros((len_A+1, len_A+1))
            A[ :-1 , :-1 ] = self.A
            self.A = A
        self.A[ node_1 , node_2 ] = conn
        
        "FIX: must fix conv_count etc for totalgraphs"
        graphs.sort_graph(self)
        

    '''
    Currently not in use
    Add nodes manually
    '''
    def insert_node(self, new_node=0):
        "new_node = [ [out], [inc], [type out], [type inc], [recursive = True or False, type = 20]  ]"
        "usage ex: graphs.insert_node( obj.Node , new_node ) or graphs.insert_node( obj , new_node )"
        l = self.List_of_nodes
        m = self.List_of_models
        
        if new_node == 0: #We insert an empty node w/o edges in the parent graph
            self.List_of_nodes.append( [[],[]] )
            
            "Create new adj. matrix..."
            A = np.zeros(( np.shape(self.A)[0] + 1 , np.shape(self.A)[0] + 1 ))
            if len(l) == 1:
                pass
            else:
                A[:-1,:-1] = self.A
                self.A = A
            
        else: #insert node with edges
            out_nodes = new_node[0]
            inc_nodes = new_node[1]
            
            out = np.zeros((len(self.A)+1,1))
            inc = np.zeros((1,len(self.A)+1))
            
            "row based"
            for j,i in enumerate(inc_nodes):
                self.List_of_nodes[i][0].append( len(l) ) #the id of new node = len(l)
                inc[0,i] = new_node[3][j]
                
            "col based"
            for j,i in enumerate(out_nodes):
                self.List_of_nodes[i][-1].append( len(l) )
                out[i,0] = new_node[2][j]
                
            self.List_of_nodes.append( new_node[0:2] )
            
            "Create new adj. matrix..."
            A = np.zeros(( np.shape(self.A)[0] + 1 , np.shape(self.A)[0] + 1 ))
            A[:,-1] = out[:,0]
            A[-1,:] = inc[0,:]
            A[:-1,:-1] = self.A
            self.A = A
            
            if new_node[-1][0] == True:
                self.A[-1,-1] = new_node[-1][1]
            
        graphs.sort_graph( self )
        



    "TBC"
    def subtract_node_number(self, i, k, j):
        
        "First or second entry is subtracted"
        
        temp_int = int( self.List_of_models[0][i][k]  )
        temp_int -= 1
        
        fixed = int( self.List_of_models[0][i][j] )
        
        self.List_of_models[0][i] = '{},{}'.format( fixed , temp_int )
        
        
    
        "Reassigns self.{}_{} to a lower node number"
        exec('self.edge_{}_{} = self.edge_{}_{}'.format( fixed , temp_int , fixed , temp_int+1 ))
        exec('del self.edge_{}_{}'.format( fixed , temp_int+1 ))
        
        exec('self.optimizer_{}_{} = self.optimizer_{}_{}'.format( fixed , temp_int , fixed , temp_int+1 ))
        exec('del self.optimizer_{}_{}'.format( fixed , temp_int+1 ))
        
    "TBC"
    "input is graph and node to be removed from said graph and name of node"
    "ex: graphs.delete_node(obj,3)"
    def delete_node(self, node_del):
        try:
            exec( 'del self.node_{}'.format(node_del))
            
            try:
                exec( 'del self.node_{}_bidirectional'.format(node_del))
            except:
                pass
            
            "Remove col-row at position node_del in adj matrix A"
            A = self.A
            A[node_del,:]   = 0
            A.T[node_del,:] = 0
            A = np.delete(A,node_del,0)
            A = np.delete(A,node_del,1)
            self.A = A
            
            "Resets the List_of_nodes for self wrt new adj-matrix"
            self.List_of_nodes = []
            graphs.adjacency_operation(self, reset = False)
            
                
            
            "Remove all node-related edges"
            "Subtracts model numbers by 1 with values greater than nodel_del"
            "This section should be optimized"
            temp_list = []
            temp_list_2 = []
            for i in range(len(self.List_of_models[0])):
                if str(node_del) in str(self.List_of_models[0][i]):
                    
                    target = self.List_of_models[0][i][0]
                    sender = self.List_of_models[0][i][-1]
                    exec( 'del obj.edge_{}_{}'.format( target , sender ) )
                    exec( 'del obj.optimizer_{}_{}'.format( target , sender ) )
                    
                else:
                    "All node numbers above the node_del must be subtracted by 1"
                    if node_del <  int( self.List_of_models[0][i][0] ):
                        k = 0  #position of major/target
                        j = -1 #position of minor/sender
                        graphs.subtract_node_number(self, i, k, j)
                        
#                        temp_int = int( self.List_of_models[0][i][0] )
#                        temp_int -= 1
#                        
#                        temp_int_2 = int( self.List_of_models[0][i][-1] )
#                        
#                        self.List_of_models[0][i] = '{}_{}'.format(temp_int, temp_int_2)
#                        
#                        "Reassigns self.{}_{} to a lower node number"
#                        exec('self.edge_{}_{} = self.edge_{}_{}'.format(temp_int,temp_int_2 , temp_int+1 , temp_int_2 ))
#                        exec('del self.edge_{}_{}'.format(temp_int+1,temp_int_2))
#                        
#                        exec('self.optimizer_{}_{} = self.optimizer_{}_{}'.format(temp_int,temp_int_2 , temp_int+1 , temp_int_2 ))
#                        exec('del self.optimizer_{}_{}'.format(temp_int+1,temp_int_2))
                        
                        
                    if node_del <  int( self.List_of_models[0][i][-1] ):
                        k = -1
                        j = 0
                        graphs.subtract_node_number(self, i, k, j)
                        
#                        temp_int = int( self.List_of_models[0][i][-1]  )
#                        temp_int -= 1
#                        
#                        temp_int_2 = int( self.List_of_models[0][i][0] )
#                        
#                        self.List_of_models[0][i] = '{}_{}'.format(temp_int_2, temp_int)
#                    
#                        "Reassigns self.{}_{} to a lower node number"
#                        exec('self.edge_{}_{} = self.edge_{}_{}'.format(temp_int_2,temp_int , temp_int_2 , temp_int+1 ))
#                        exec('del self.edge_{}_{}'.format(temp_int_2,temp_int+1))
#                        
#                        exec('self.optimizer_{}_{} = self.optimizer_{}_{}'.format(temp_int_2,temp_int , temp_int_2 , temp_int+1 ))
#                        exec('del self.optimizer_{}_{}'.format(temp_int_2,temp_int+1))
                    

                    
                    temp_list.append(self.List_of_models[0][i])
                    temp_list_2.append( self.List_of_models[1][i] )
                    
                    
                    
            self.List_of_models[0] = temp_list
            self.List_of_models[1] = temp_list_2
            
            
            "All .node_{} must be subtracted by 1"
            i=0
            stop = False
            while stop == False:
                if i > node_del:
                    try:
                        exec( 'temp = self.node_{}'.format(i) )     #Stores node value
                        exec( 'self.node_{} = temp'.format(i-1) )   #Re assigns node value
                        exec( 'del self.node_{}'.format(i) )        #Delete old node value
                        
                    except:
                        stop = True
                i+=1
            
        except:
            print("Node", node_del, "does not exist")
        
        nc.graphs.compile_graph(self)

    
    def delete_edge(self, target, sender):
        try:
            exec( 'del self.edge_{}_{}'.format(target,sender) )
            exec( 'del self.optimizer_{}_{}'.format(target,sender) )
            
            self.A[target,sender]=0
            
        except:
            print("Target/ sender node combination does not exist")
    
    
    
    #####################################
    ###### SORTS GRAPH COMPUTATION ######
    #####################################
    
    ###
    ### This algorithm is not fully tested and sometimes returns incomplete computation lists
    ###
    ### This sorting algorithm needs an overhaul
    
    def sort_graph(self):
        "usage ex: graphs.sort_graph( obj.Node )"
        # print()
        # print("start sort:")
        #print("DEBUG list of nodes", self.List_of_nodes)
        
        l = self.List_of_nodes
        N = len(l)  #nr of nodes
        default = 0 #This is the default node
        min_len = 0
        
        nodes_to_compute = [i for i in range(N)]
        
        for i in range(N): #goes through the list of nodes
            len_list = len( l[i][-1] )
            
            if len_list <= min_len:
                min_len = len_list
                default = i
            # print("default", default)
            
        #print("First node is:", default, "with", min_len, "number of incoming connections and", len(l[default][0]), "number of outgoing connections...")
        "Breadth-first search or depth-first search?"
        List_of_models = []
        List_of_edges  = []
        
        max_it = N+1 #prevents infinite loops
        completed_nodes = [] #prevents nodes from being put in queue a second time
        
        #some outgoing connections may not be computed at all
        #We must make sure all outgoing connections have been computed atleast once
        queue = [default]
        while len(queue)!= 0:
            "Each node will create a queue of which the concurrent nodes will be placed in"
            "When the queue is empty, all nodes are complete"
            "The edge-node connections wrt the nodes in the queue will be computed"


            "next node in queue is the first node in j loop"
            current_node = queue.pop(0)
            
            if current_node == 18:
                print("Queueing 18...")
            
            "compute incoming"
            "Computes all incoming edge-node connections"
            if len( l[current_node][-1] ) != 0:
                for j_0 in range( len( l[current_node][-1] )):
                    
                    rec_node = current_node
                    inc_node = l[current_node][-1][j_0]
                    
                    List_of_models = graphs.append_model(self, List_of_models, rec_node , inc_node )

            "compute outgoing"
            for j_1 in range( len( l[current_node][0] )):
                # print("compute outgoing:",j_1)
                
                rec_node = (l[current_node][0][j_1]) #receiving node
                List_of_models = graphs.append_model(self, List_of_models, rec_node , current_node )
                    
                if rec_node in completed_nodes:
                    pass
                else:
                    queue.append( rec_node )
                    completed_nodes.append( rec_node )
                
                "This will make sure any recursion nodes will be computed asap"
                if self.A[ rec_node , rec_node ] != 0:
                    List_of_models = graphs.append_model(self, List_of_models, rec_node, rec_node)
                
            "We must shuffle the queue in the correct order"
            for j_2 , nodes in enumerate(queue):
                outgoing_for_current = l[nodes][0]
                if len(outgoing_for_current) == 0: #We switch places in the queue
                    queue[-1], queue[j_2] = queue[j_2], queue[-1]

        #print("DEBUG list of models",self.List_of_models[0])
        self.List_of_models[0] = List_of_models
        # print("DEBUG list of models",List_of_models)
    
    
    
    
    
    
    
    
    
    ###########################
    ###### COMPILE_GRAPH ######
    ###########################
    
    #Must be overhauled

    '''
    Compile graph will go through the whole graph and give each member their respective
    parent information etc which will be used when traversing the graph in forward
    '''
    def compile_graph(self):
        
        
        
        "This is a nonbinary-search-tree"
        "self is the obj aka total-graph"
        "compute first sub-graph"
        "TODO: Compile output dimensions for each node"
        
        
        
        temp_graph = self
        
        root_name = self.name
        self.root = self.name
        
        #temp_graph.node_output_0 = node_output_0
        current_node = "temp_graph" #The initial node of the sub-graph in string form
        name = self.name
        current_parent = ""
        
        node_nr = 0
        
        parent_nr = 0
        parent = [""]
        stop = False
        
        self.Computation_List = []
        #old_output_dim = current_node.H
        
        
        while stop == False:
            
            "depth node comes from which model exists in parent"
            "if node_# exists in some sub-graph, we go down one step to this node"
            dir_check = eval('{}'.format(current_node) )
            check = 'node_{}'.format(node_nr) in dir(dir_check)
            if check == True:
                
                parent.append(parent_nr)
                parent_nr = node_nr #We assign a parent_nr to the new node
                current_parent = current_node
                current_node += '.node_{}'.format(node_nr)
                
                
                #print(current_node)
                
                "Checks if this node is bidirectional"
                
                #Debugging
                # print("current parent",current_parent, "node_nr",node_nr)
                # exec( 'print( {}.List_of_nodes)'.format(current_parent   ) )
                # exec( 'print( len( {}.List_of_nodes[:][0]))'.format(current_parent   ) )
                
                # print("checkpoint")
                
                set_1 = eval( 'set({}.List_of_nodes[{}][0])'.format(current_parent,node_nr) )
                set_2 = eval( 'set({}.List_of_nodes[{}][1])'.format(current_parent,node_nr) )
                if len( set_1.intersection(set_2)) != 0:
                    exec( '{}.node_{}_bidirectional = True'.format(current_parent,node_nr))
                    
                
                if eval(current_node).parent == '':
                    eval(current_node).parent = current_parent
                    
                if eval(current_node).node_nr == '':
                    eval(current_node).node_nr = node_nr
                    
                    eval(current_node).name = current_node
                
                    eval(current_node).compiled = True
                    
                    eval(current_node).root = root_name
                
                "If this sub-graph have no nodes, then we must compute its internal state"
                dir_check = eval('{}'.format(current_node) )
                #check = 'node_{}'.format(node_nr) in dir(dir_check)
                check = 'node_0' in dir(dir_check) #A sub-graph must have node_0 to be a sub-graph by def
                if check == False:
                    "We add the current_node to the computation_list"
                    self.Computation_List.append( current_node )
                    
                    "Now we must check the input/output dimensions for this node"
#                    input_dim = current_node.H
#                    if old_output_dim == input_dim:
#                        old_output_dim = input_dim
#                    else:
#                        raise Exception("Dimension error for node", current_node, " Expected dimensions", old_output_dim, "but received", input_dim)
#                    
                    
                    "We go up one level"
                    node_nr, parent_nr, parent, current_node, current_parent \
                        = graphs.traverse_up( node_nr, parent_nr, parent, current_node, current_parent )
                    
                else:
                    parent_nr = node_nr
                    node_nr = 0
                    
            else:
                "We add the current_node to the computation_list"
                self.Computation_List.append( current_node )
                
#                input_dim = current_node.H
#                if old_output_dim == input_dim:
#                    old_output_dim = input_dim
#                else:
#                    raise Exception("Dimension error for node", current_node, " Expected dimensions", old_output_dim, "but received", input_dim)
#                
                
                if parent[-1] != "":
                    "We go up one level"
                    node_nr, parent_nr, parent, current_node, current_parent \
                        = graphs.traverse_up( node_nr, parent_nr, parent, current_node, current_parent )
                else:
                    "We are alredy at the root parent"
                    pass
                    
            
            #print("node and Node nr:", current_node, node_nr)
            
            ###########################
            ###### End condition ######
            ###########################
            "Check condition for ending"
            if len(current_node) < len("temp_graph"):
                current_node = "temp_graph"
                
            #print(parent)
            
            dir_check = eval('{}'.format(current_node) )
            check = 'node_{}'.format(node_nr) in dir(dir_check)
            if parent[-1]=='' and check==False:
                self.Computation_List.append( current_node )
                stop=True
            
        self = temp_graph
        self.compiled = True
        
        #print("Compile done")


    #########################
    ###### END COMPILE ######
    #########################
    
    
    
    #####################################
    ###### Parent graph traversal #######
    #####################################
    "We need to check if this obj have node children"
    def child_check( obj , node_nr):
        
        #dir_check = eval('obj_{}'.format(current_node) )
        check = 'node_{}'.format(node_nr) in dir(obj)
        return check
    
    def traverse_parent_obj(self):
        temp_graph = self
        "We traverse the parent object and the child object using their adj matrices"
        A = self.A
        
        node_nr = 0
        parent_nr = 0
        computation_list = []
        
        current_node   = "temp_graph" #The initial node of the sub-graph in string form
        current_parent = ""
        name = self.name
        parent = [""]
        
        stop = False
        while stop == False:
            print(current_node)
            
            "check if current node have children"
            check = graphs.child_check( eval(current_node) , node_nr)
            if check == True:
                print("We found its child...")
                "Since the obj have at least one child, we continue down"
                
                parent_node  = current_node
                current_node = parent_node + '.node_{}'.format(node_nr)
                
                
            else:
                "This obj have no children, so we can compute its internal_state"
                #compute internal state...
                internal_state = 0
                computation_list.append( current_node )
                "Then we check if it have any siblings"
                
                
                "check for siblings"
                check = graphs.child_check( eval(current_node) , node_nr + 1)
                if check == True:
                    "this node have a sibling,"
                    "so we pass the internal_state to its sibling as input"
                    print("We found its sibling...")
                    
                    len_r = len( '{}'.format(node_nr) )   #We must replace old node nr with new
                    current_node = current_node[:-len_r]
                    
                    node_nr += 1
                    current_node += '{}'.format(node_nr)
                
                else:
                    "This node have no siblings,"
                    "so we must go back up to the parent"                    
                    print("No siblings...")
                    
                    if parent == [""]:
                        print("at the top...")
                    
                    else:
                        node_nr, parent_nr, parent, current_node, current_parent =\
                            graphs.traverse_up( node_nr, parent_nr, parent, current_node, current_parent )
                    
                    
                    
    
    #############################
    ###### Graph traversal ######
    #############################
    def traverse_up( node_nr, parent_nr, parent, current_node, current_parent ):
        "This paragraph of code takes us up one level"
        len_r = len( '.node_{}'.format(node_nr) )
        current_node = current_node[:-len_r]
        
        if current_parent == 'temp_graph':
            current_parent = ''
        else:
            current_parent = current_parent[:-len_r]
    
        print("parent:",parent_nr)
        node_nr = parent_nr + 1    #this is the node we must check if it exists in the new parent
        parent_nr = parent.pop(-1) #when we go up again, we must remove the last parent, as one cannot be its own parent
        
        return node_nr, parent_nr, parent, current_node, current_parent
    

    
        
    def timestep_assign( self, X ):
        "This is the wrapper for a recurrent type neural network model"
        timestep = len(X)
        
        for i in range(timestep):
            "at every new timestep we call the model with new X_t input"
            X_out = graphs.pseudo_forward_TEST( self, X[i].detach() )
            
            "We need the time dep. output and the rec. block output"
        return X_t
    
    
    
    
    def data_primer( self , X , timeseries = True, timestep=1 , start_node = '' , end_node = '', input_nodes = 1, record_timestep=True ):

        "we may want to store timestep outputs"
        X_out_t = torch.zeros( np.shape(X[0]) )
        
        insert_node_value = True        

        self.input_nodes = input_nodes
        if input_nodes > 1:
            for t in range(timestep):
                #print("timestep:", t)
                
                for i in range(input_nodes):
                    """
                    -we may only want to assign node values once
                    -Assigns initial values to input nodes
                    -must assign new values each timestep
                    -node assignment may not be uniform
                    ex: node 1 and 3 may be input nodes, 2 and 4 may not be.
                    workaround: add zero values to dataset at node 2 and 4.
                    """
                    if insert_node_value == True:
                        "we insert new node values for each timestep"
                        exec('self.node_output_{} = X[i,t].detach()'.format(i) )
                    else:
                        "we dont insert new node values for new timesteps"
                        pass
                
                if timeseries == True:
                    "we run the module a number of timesteps"
                    #X_out = graphs.pseudo_forward_TEST( self, X[0,t].detach() , start_node , end_node , input_nodes = input_nodes )
                    X_out = graphs.pseudo_forward_TEST( self, X[0,t] , start_node , end_node , input_nodes = input_nodes )
                    
                    #X[0,t] = self.node_output_1
                    
                    if record_timestep == True:
                        "we store timestep outputs and return it"
                        X_out_t[t] = X_out
                    
                else:
                    X_out = graphs.pseudo_forward_TEST( self, X[0].detach() , start_node , end_node , input_nodes = input_nodes )
                    X[0] = self.node_output_1
                    
                "Each node has two values, one for output and one for input"
                "The input values will be added to the original output when the graph traversal is complete"
                for i in range(input_nodes):
                    "Not every node is bidirectional. Thus those nodes must be skipped wrt the nodes temp values"
                    try:
                        exec( 'self.node_output_{} += self.node_output_{}_temp'.format(i,i) )
                        exec( 'del self.node_output_{}_temp'.format(i) )
                    except:
                        pass
                 
        else:
            "Previous output is the next input"
            "timestep=1 is standard graph traversal"
            for t in range(timestep):
                X_out = graphs.pseudo_forward_TEST( self, X.detach() , start_node , end_node , input_nodes = input_nodes )
                X = self.adj_node_output_1
        
        return X_out_t
    
        
    
    def pseudo_timestep_primer( self , X , timeseries = True, timestep=1 , start_node = '' , end_node = '', input_nodes = 1 ):
        "Required dimensions: node, timestep, batchsize, features"
        
        "This code is a wrapper for the pseudo_timestep for bidir. graphs"
        "it is assumed dataset contains time series information"
        """
        nr of input nodes must be greater or equal to len(X),
            aka value of zero-dim of X = shape(X)[0].
            
        Must allow to insert time dep. input.
        """
        
        self.input_nodes = input_nodes
        if input_nodes > 1:
            
            
            for i in range(input_nodes):
                "Assigns initial values to input nodes"
                "must assign new values each timestep"
                exec('self.node_output_{} = X[i].detach()'.format(i) )

            for t in range(timestep):
                if timeseries == True:
                    "assigns fresh timestep information per input node"
                    X_out = graphs.pseudo_forward_TEST( self, X[:,t].detach() , start_node , end_node , input_nodes = input_nodes )
                    X[:,t] = self.node_output_1
                    
                else:
                    X_out = graphs.pseudo_forward_TEST( self, X[0].detach() , start_node , end_node , input_nodes = input_nodes )
                    X[0] = self.node_output_1
                    
                "Each node has two values, one for output and one for input"
                "The input values will be added to the original output when the graph traversal is complete"
                for i in range(input_nodes):
                    "Not every node is bidirectional. Thus those nodes must be skipped wrt the nodes temp values"
                    try:
                        exec( 'self.node_output_{} += self.node_output_{}_temp'.format(i,i) )
                        exec( 'del self.node_output_{}_temp'.format(i) )
                    except:
                        pass
                
        else:
            "Previous output is the next input"
            "timestep=1 is standard graph traversal"
            for t in range(timestep):
                X_out = graphs.pseudo_forward_TEST( self, X.detach() , start_node , end_node , input_nodes = input_nodes )
                X = self.adj_node_output_1
        
        return X_out







    "Computing all outgoing connections from a specific node in a sub-graph"
    def compute_outgoing(self, sender_node, current_s_internal ):
        
        #print()
        #print("Computing edge connections of:", self.name)
        
        models = self.List_of_models[0]
        target_nodes = []
        j=0
        
        "This for loop may be split into Phases for faster computation"
        for i in range( len( models )):
            #print("Computing model:", models[i])
            if  sender_node == int(models[i][-1]):
                target_node = models[i][0]
                "These node output values must be deleted when the all internal states have been computed"
                #print("Computing edge:", target_node, sender_node)
                
                #print(target_node, sender_node)
                
                
                "the edge_operations are updated as we perform feedforward"
                exec( '{}.optimizer_{}_{}.step()'.format('self', target_node, sender_node) )
                exec( '{}.edge_{}_{}.zero_grad()'.format('self', target_node, sender_node) )


                "Checks if reference node is bidir with its target node"
                "Reference node receive list"
                set_1 = self.List_of_nodes[sender_node][0]
                
                "Target node sender list"
                set_2 = self.List_of_nodes[int(target_node)][0]
                
                if sender_node in set_2 and int(target_node) in set_1:
                    #print(target_node,"Calculating bidirectional node...")
                    try:
                        exec( 'self.node_output_{}_temp += ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                    except:
                        exec( 'self.node_output_{}_temp  = ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                
                
                else:
                    "For directed nodes"
                    #print("Directional")
                    try:
                        exec( 'self.node_output_{} += ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                    except:
                        exec( 'self.node_output_{}  = ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                        
                
                    
                target_nodes.append(target_node)
                
                j+= 1
                
        "if j is zero, then this is the final output for this sub-graph"
        if j == 0:
            exec('self.node_output_{} = current_s_internal'.format(len(self.A)))
        
        #print(np.shape( eval('self.node_output_{}'.format( target_node ))))
        "the target nodes will be returned so we know which nodes to give which input to"
        "target_nodes = receiving nodes"
        n = len(self.A) -1
        current_node_internal = eval('self.node_output_{}'.format(n))
        return current_node_internal


    "This is the current pseudo_forward"
    "Must implement pseudo_timestep for bidirectional graphs - ok"
    "Must allow for multiple input nodes & output nodes"
    "- as workaround, set up bigger adj matrix with a select few nodes with input values" 
    " - obj.node_output_1 etc must be predefined"
    def pseudo_forward_TEST(self, OG_input, start_node = '', end_node='' , input_nodes = 1 ):
        #print("START NEW GRAPH")
        #print("input sum", torch.sum(OG_input))
        
        if start_node == '':
            start_node = self.Computation_List[0]
        if end_node == '':
            end_node = self.Computation_List[-1]
            
        
        root_graph = "temp_graph"
        
        node_nr = 0
        current_parent = ""
        

        "The node_nr for our current node"
        node_nr = self.node_nr
        current_parent = self.parent
        current_node = self.name
        
        temp_graph = self
        
        parent_nr = 0
        init_ = 0
        
        "Checks if the input is graph-based or not"
        gb = False
        
        for j in range(len(self.Computation_List)):
            if start_node == self.Computation_List[j]:
                start = j
            if end_node == self.Computation_List[j]:
                end = j
        
        Computation_List = self.Computation_List[start:end+1]
        #print(Computation_List)
        
        
        for i,(current_node) in enumerate(Computation_List):
            "depth node comes from which model exists in parent"
            
            prev_parent = current_parent
            current_parent = eval(current_node).parent
            node_nr        = eval(current_node).node_nr
            
            #print()
            #print(i, current_node)
            
            "If this sub-graph have no nodes, then we must compute its internal state"
            dir_check = eval(current_node)
            check = 'node_0' in dir(dir_check) #a sub-graph must have node_0 to be a sub-graph by def
            
            #print("PARENT:",current_parent)
            
            if check == False:
                
                "Must give input to the internal state"
                if init_ == 0:
                    "we perform this code once"
                    #print("We compute this:", current_node, i)
                    current_node_internal = node.forward( eval(current_node) , OG_input, graph_based = gb )
                    #print("end...",torch.sum(current_node_internal))
                    #print("end...",np.shape(current_node_internal))

                    init_ = 1
                else:
                    
                    "Must give parent of current_node the correct input value"
                    if current_parent != prev_parent:
                        
                        parent_node_nr = eval(current_parent).node_nr
                        #eval(current_parent).node_output_0 = eval('{}.node_output_{}'.format( prev_parent , parent_node_nr ))
                        exec( 'current_parent.node_output_0 = {}.node_output_{}'.format( prev_parent , parent_node_nr ))
                        
                        #print()
                        #print("Values from:", current_parent, node_nr)
                        "Delete old node_output"
                        #exec( 'del {}.node_output_{}'.format(prev_parent, parent_node_nr) )
                    
                    
                    "We must give the output from previous node to this new node"
                    "The input is at parent as parent.node_output, aka the edge connection"
                    ##print()
                    ##print("compute:", current_node, node_nr)
                    ##current_node_internal = node.forward( eval(current_node) ,  eval( '{}.node_output_{}'.format(current_parent,node_nr) ) , graph_based = gb )
                    
                    #print("Then we compute this:", current_node, i)
                    #print(torch.sum(current_node_internal))
                    current_node_internal = node.forward( eval(current_node) ,  current_node_internal , graph_based = gb )
                    
                    
                    "Delete old node_output"
                    #exec( 'del {}.node_output_{}'.format(current_parent, node_nr) )


            else:
                pass
                #print("IN here")
                "Must give the new child the current_node's output value as input value"
                #eval( Computation_List[i+1] ).parent
                #temp_parent.node_output_0 = current_node_internal
                #exec( 'eval( Computation_List[i+1].parent ).node_outout_0 = current_internal_node' )
            "When we've computed an internal state, we must compute its edge connections"
            "edge connections are stored at parent"
            "input for edge connections are stored at child"
            "node_nr is the current outgoing node aka sender_node"
            
            
            
            return_value = current_node_internal
            #print(current_node)
            if current_parent == '':
                pass
                #print("Compute out for main obj?", i)
                #exec( 'del {}.node_output_{}'.format(current_node, node_nr) )
            
            else:
                if current_parent == self.Computation_List[-1]:
                    "prevents parent object to compute edge conn twice"
                    pass
                
                else:
                    "We dont want this to be computed if it already has been computed"
                    
                    #print("here input...",torch.sum( current_node_internal ))
                    
                    current_node_internal = graphs.compute_outgoing( eval(current_parent) , node_nr, current_node_internal )
                    
                    #print("Here...", current_node, i)
                    #print("here output...",torch.sum( eval(current_parent).node_output_1 ))
                    
                    "Delete old node_output"
                    #exec( 'del {}.node_output_{}'.format(current_parent, node_nr) )
    
                    #exec( 'eval( Computation_List[i+1].parent ).node_outout_0 = current_internal_node' )
                    #exec( 'del {}.node_output_{}'.format(current_parent, node_nr) )
                
        #print("LEAVING GRAPH")
        self = temp_graph
        
        #print(return_value)
        #return return_value
        return current_node_internal
    
    
        
#######################
###### PROTOTYPE ######
#######################
class define_tensor(graphs):
    def __init__(self, x, A=0):
        
        super(graphs, self).__init__()

        "Creates an object with graph structure."
        self.value = x
        self.A = A
        self.tensor_bool = True
        
        #self.graph_object = graphs(self.A, 'graph_object')
        #graphs.compile_graph( self.graph_object ) 
    
    
########################
########################
########################
        

class node(graphs):
    def __init__(self, A , name = '', parent = '', graph_conv = False  ):
        
        super(graphs,self).__init__()
        
        print("in here")
        
        "Fix so we can load LR from global defined fcn"
        try:
            self.network_settings = ntw.LR
        except:
            self.network_settings = network_variables(0.0)
        #self.network_settings = network_variables(0.001)
        
        self.graph_conv = False
        
        self.name = name #Stores the name of this node
        self.parent = parent
        self.node_nr = ''
        #node_nr = int(name)
        #self.parent = name[]
        
        self.save = 0                                                          #May be set to 1 to store certain output values
        self.device = self.network_settings.device    
        self.LR = self.network_settings.LR                                     #loads learning rate for opt
        self.A = A                                                             #the adj. matrix of which this graph is based
        if type(A) == int:
            self.A = np.zeros((1,1))
        self.List_of_models = [ [] , [] ]                                      #contains information about the edges
        self.List_of_nodes  = [  ]                                             #contains inc and out connections wrt notes
        
        "Counts the number of networks in this sub-graph"
        self.fc_count   = 0
        self.conv_count = 0
        
        "Placeholder value"
        self.loss = 0
        
        
        "Define the adj operation wrt self.A = A"
        self.add_model = False
        graphs.adjacency_operation(self)
        
        self.add_model = True
        "The graph must now be sorted"
        graphs.sort_graph(self)

    def node_softmax(self, x):
        exp_a = torch.exp(x)
        sum_exp_a = torch.sum( exp_a )
        a = exp_a / sum_exp_a
        return a

    "This algorithm is executed once per new initiated model"
    "... and only if devices is True"
    "THIS ALG IS NOT COMPLETE"
    def determine_phases(self):
        j=0
        
        "List of phases which will be distributed"
        "phases require an initial model"
        phases = [[]]
        phases[0].append( self.List_of_models[0][0] )
        
        for i in range( len(self.List_of_models[0]) -1 ):
            "Distribute models in their respective phases"
            
            arg_1_1 = self.List_of_models[0][i][0]
            arg_1_2 = self.List_of_models[0][i][-1]
            
            arg_2_1 = self.List_of_models[0][i+1][0]
            arg_2_2 = self.List_of_models[0][i+1][-1]
            
            if arg_1_2 == arg_2_2 or arg_1_1 == arg_2_1:
                print(phases,j)
                print(phases[j])
                phases[j].append( self.List_of_models[0][i+1] )
                
            else:
                print(phases,j)
                print(phases[j])
                phases.append( [ self.List_of_models[0][i+1] ] )
                j+=1
        
        "EX with 3 phases: phases = [ ['0_1', '0_2, '0_3''] , ['2_1' , '3_1'], ['3_2'] ]"
        self.phases = phases
        print(self.phases)
        
        
        
        
    "we'd like this function to be bi-directional"
    "This forward code works well for subgraphs with edges only"
    "This code is used to compute internal states"
    "We'd like to store node_output for every node in the internal state"
    def forward(self, X_input, graph_based = False):
        #print(self.graph_conv)
        #print("test this...")
        if self.graph_conv == True:
            node_output = node.graph_convolution(self, X_input)
        else:
            node_output = node.graph_traversal(self, X_input)
            
        return node_output
    
    




    """
    This function is parallelizable
    """
    def compute_branch(self, ls, X_input, j):
        
        #print(X_input)
        
        "each node will receive a list of models to perform local graph traversal"
        "Each branch aka ls is computed in a linear graph style"
        "ex: (6,0), (6,8), (6,9), and finally (11,6)"
        
        "The last entry of ls has weight W_1, the remaining is W_2"
        #print("ref node:", j, ls)
        old_t = ls[0][0]
        target_neighbors = []
        n = len(ls)+1
        #print(ls)
        for i in range(len( ls )):
            t = ls[i][0] #target node
            s = ls[i][1] #sender node
            
            
            """ graphsage """
            if self.graphsage == True:
                
                #print("inside GS")
                
                if i == len(ls)-1:
                    "The final connection of this branch will have gcn connection 1"
                    
                    self.operation = eval('self.gcn_edge_{}'.format(0))
                else:
                    "The default gcn connection is self.K or in this case 2"
                    self.operation = eval('self.gcn_edge_{}'.format(self.K-1))
                
            else:
                print("not performing gs")
                self.operation = eval( 'self.edge_{}_{}'.format(t,s) )
            
            
                
            "the edge_operations are updated as we perform feedforward"
            "Not used for custom operators not using pytorch"
            if self.graphsage==True or eval( 'self.edge_{}_{}_pytorch'.format( t , s  ) ) == True:
                
                #a node may have multiple incoming nodes aka multiple sender nodes
                #this means that initially, self.node_output_temp_{} is undefined
                
                "All nodes have their own input, some of which may be zero"
                "We multiply with the distance"
                inp = eval( 'X_input[{}].detach()'.format( s )) 
                
                # print()
                # print("input", s)
                # print(inp)
                
                try:
                    pass
                    #Adding this line causes issue with pytorch backprop
                    #inp += eval( 'self.temp_branch_output_{}.detach()'.format( s ))
                    inp += eval( 'self.temp_branch_output_{}'.format( s ))
                    
                    #print("inp shape:",np.shape(inp))
                    #inp += eval('X_input[]'.format(s))
                    #print("from", s, "to", t)
                    
                except:
                    pass
                #print(inp)
                "These are the intermediate branch node values"
                "The accumulation of branch outputs given to the node in question"
                "These are for the branch outputs, not for the reference node output"
                try:
                    #exec( 'self.temp_branch_output_{} += ( self.edge_{}_{} )( inp.clone() )'.format( t, t, s ))
                    #exec( 'self.temp_branch_output_{} += self.dist_A[t,s] * ( self.operation )( inp.clone() )'.format( t ))
                    
                    temp_node_output = (self.operation)(inp.clone())
                    #exec( 'self.temp_branch_output_{} += ( self.operation )( inp.clone() )'.format( t ))
                    exec( 'self.temp_branch_output_{} += temp_node_output / {}'.format(t,n))
                    
                except:
                    #exec( 'self.temp_branch_output_{}  = ( self.edge_{}_{} )( inp.clone() )'.format( t, t, s ))
                    
                    temp_node_output = (self.operation)(inp.clone())
                    #exec( 'self.temp_branch_output_{}  = ( self.operation )( inp.clone() )'.format( t ))
                    exec( 'self.temp_branch_output_{}  = temp_node_output / {}'.format(t,n))
                    
                    
                    #exec( 'self.temp_branch_output_{}  = self.dist_A[t,s] *  ( self.operation )( inp.clone() )'.format( t ))
                
                
                    #Skip connection?
                    #exec('self.temp_branch_output_{} += inp.clone() * self.dist_A[t,s] '.format(t))                
                    
                    
                ####
                #### Store output for GAT. WIP
                ####
                
                GAT = False
                """
                if GAT == True:
                    #print(t,s)
                    "We need to define operator a: ft x ft -> ft"
                    

                    "We need to register all neighboring nodes of target t"
                    target_neighbors.append(s)

                    # Add GAT values for sender and target
                    exec( 'self.GAT_output_{} = temp_node_output'.format(s))
                    exec( 'self.GAT_output_{} = X_input[{}].clone().detach()'.format( t,t )) 
                    
                    switch_target = False
                    if i < len(ls)-1 and old_t != ls[i+1][0]: # out of bounds issue with ls[i+1]
                        switch_target = True
                    old_t = t
                    
                    if switch_target == True:
                        #When the switch occurs, we're done with current target node and must compute its attention
                        "Compute attention in here"
                        for J in target_neighbors: # compute attention of targt t with its neighbors J
                            
                            ft = eval( 'np.shape(self.GAT_output_{})[-1]'.format(t))
                            a = torch.rand(( 2*ft , ft )) #placeholder attention operator
                            # we concat the node values wrt their features
                            concatanated = eval('torch.concat( ( self.GAT_output_{} , self.GAT_output_{} ), -1 )'.format( t , J ))
                            pre_att = torch.matmul( concatanated , a )
                            exec( 'self.E_{}_{} = pre_att'.format(t,J)) # we can perform activation on this line
                        
                            # Define shape of placeholder sum value sum_exp
                            shape_dims = eval( 'np.shape( self.E_{}_{} )'.format(t,J) )
                            
                            
                            
                        sum_exp = torch.zeros( ( shape_dims ) )
                        
                        #print( np.shape(sum_exp) , np.shape(pre_att) )
                        
                        for J1 in target_neighbors:
                            sum_exp += eval( 'torch.exp( self.E_{}_{} )'.format(t,J1))
                        #S_ij = eval('torch.exp( self.E_{}_{} )'.format(t,J1)) / sum_exp
                        
                        pre_act_h = torch.zeros(( np.shape(sum_exp) ))
                        for J2 in target_neighbors:
                            pre_act_h += eval( 'torch.exp( self.E_{}_{} ) * self.GAT_output_{}'.format(t,J2,J2))
                        
                        #hi is the new output for this particular node
                        #hi = sigma ( pre_act_h )
                        hi = pre_act_h / (len(target_neighbors)+1)
                        
                        #hi is the new branch output
                        exec( 'self.temp_branch_output_{} = hi'.format(t))
                        
                        "When we have computed the attention for this target node, we must clear the neighbor list:"
                        target_neighbors = []
                
                ### GAT end
                #########################################
                """

        
        branch_output = eval('self.temp_branch_output_{}.clone()'.format(t)) #* self.dist_A[t,s]
        #print(torch.sum(branch_output))
        
        # print()
        # print("output", t)
        # print(branch_output)
        
        "We must now delete self.temp_branch_output_{} so that new branches with same name" 
        " ...do not carry old info"
        for i in range(len(ls)):
            t = ls[i][0]
            s = ls[i][-1]
            "if t occurs multiple times, then the branch may have already been deleted"
            try:
                exec( 'del self.temp_branch_output_{}'.format(t) )
            except:
                "Has already been deleted"
                pass
            try:
                exec( 'del self.temp_branch_output_{}'.format(s) )
            except:
                "Has already been deleted"
                pass
        
        del inp
        
        
        if torch.sum(X_input)==0:
            pass
            #print("New", i, j)
            #print(X_input)
            #print(branch_output)
        #print("finish")
        return branch_output #* self.dist_A[ t, s ]
    
    
    #####################################################
    def traverse_branch(self, ls, X_input, j):
        "This for loop can be split into multiprocessing"
        "MP branches must be synced and summed at the end"
        n = len(ls)
        #print()
        #print("TRAVERSE BRANCH HERE", len(ls), j)
        for i in range( len(ls) ):
            if i == -1: #if statement made for debugging
                pass
            else:
                "We compute the branches for ref node j"
                "The output of these branches will be accumulated as the node output for node j"
                "the intermediate node values must be aggregated correctly"
                
                "Each of these computations will give a branch output which will be given to ref node j"
                branch_output = node.compute_branch( self, ls[i], X_input, j )
                
                branch_output /= (n+1) #Since we're summing, we need to normalize
                
                "Here we collect all the branches for reference node j and add them together"
                try:
                    exec( 'self.node_output_temp_{} += ( branch_output )'.format( j ))
                except:
                    exec( 'self.node_output_temp_{}  = ( branch_output )'.format( j ))
        
        #print( branch_output , n )
        exec( 'self.adj_node_output_{} = ( self.node_output_temp_{}).clone() '.format( j , j ))
        exec( 'del self.node_output_temp_{}'.format( j ) )
    #####################################################
    
    
    
    ##### This function is bugged, why?. Still bugged?
    "This function allows for bidirectional graph traversal"
    def graph_convolution(self, X_input, graph_based = False):
        
        "We need to call this function directly for each node individually"
        
        list_of_models = ga.model_name_converter( self.List_of_models[0] )
        
        "black_list can be useful for debugging"
        black_list = []
        #print("in graph convolution...")
        for j in range( len(self.A)):
            "We grab the GCN list for the ref node j"
            #print("Ref node:", j)
            
            "Workaround for debugging"
            if j in self.banned_nodes:
                pass
            else:
                "higher loss, fast"
                try:
                    
                    # We will migrate from ga_object.GCN_list_{} to connections[j]
                    #ls = eval('self.ga_object.GCN_list_{}'.format(j))
                    ls = self.traverse_connections[j]
                except:
                    pass
                
                node.traverse_branch(self, ls, X_input, j)
        
        ### This needs to be fixed... Prio... Still bugged?
        "When we're done traversing all nodes and received their respective values"
        "We then add these values to the adj_node_output and delete temp values"
        for j in range(len(self.A)):
            if j in self.banned_nodes:
                pass
            else:
                "We can store nodes from internal state by adding it to our parent class"
                exec(  'self.adj_node_output_{} = ( self.node_output_temp_{}).clone() '.format( j , j ))
                exec( 'del self.node_output_temp_{}'.format( j ) )
            
        nr_nodes = len(self.A)-1
        
        """
        Currently this function only returns the final node in the graph.
        For RNN this function should return the whole graph as one input.
        RNN may consists of internal states or subgraphs.
        """
        return eval('self.adj_node_output_{}'.format(nr_nodes))
        
    
    
    
    
    
    """
    Computes bottom-layer graph (internal state of 1-level graph)
    
    A standard Hierarchial graph traversal function
    
    This function is called when traversing a directional graph is desired
    
    This function is not made for GCN
    
    Timestep independent operator
    """
    def graph_traversal(self, X_input, timestep='na'):
        
        "if our kernel size for gcn is greater than 1, we must add extra edge computations"
        "these extra edge computations will be computed before the usual edge computations wrt k=1 etc"
        "if k > n so that we reach a loop, the edge connections already computed will not be called again (default)"
        "edge connections may only be called once unless specifically stated otherwise."
        
        models_temp = self.List_of_models[0]
        models_len = len( models_temp )
        
        "we need to compute all receiving edge connections to sender_node edge connection"
        "so that sender_node here becomes target_node there"
        "for each sender_node, retreive a list of target_nodes"
        
        inp = X_input
        
        self.node_output_temp_0 = inp
        
        for i in range( models_len ):
            
            t = models_temp[i][0]  #target node
            s = models_temp[i][-1] #sender node
            
            """
            This snippet checks if the target node have been completely computed
            In that case, we can use the activation function on the node if defined.
            """
            if i < models_len-1 and models_temp[i+1][0] != t:
                set_activation = True
            else:
                set_activation = False
            
            "want to insert this code in a function we may call upon"
            if eval( 'self.edge_{}_{}_pytorch'.format( t , s  ) ) == True:
                
                """ If we have multiple input nodes, 
                we need to specify which node has the input """ 
                
                
                """Before computing edge-connection: 
                    We must compute internal state of node"""
                
                if self.multiple_input == True:
                    if s in self.input_node:
                        
                        #print("input at:", t,s)
                        if timestep=='na':
                            inp = self.input_values[s]
                        else:
                            inp = self.input_values[s]
                        
                    else:
                        try:
                            inp = eval( 'self.node_output_temp_{}'.format( s ))
                            
                        except:
                            inp = X_input
                else:
                    inp = eval('self.node_output_temp_{}'.format(s))
                
                
                if self.A_op[t][0] == "M":
                    try:
                        
                        out_0 = eval( '( self.edge_{}_{} )( inp )'.format( t, s ))
                        out_1 = eval( 'self.node_output_temp_{}'.format(t))
                        
                        temp = out_0 * out_1
                        
                        #temp = torch.multiply(out_1, out_0)
                        
                        #temp = (temp - torch.average(temp) ) / torch.max(temp)
                        
                        exec( 'self.node_output_temp_{} = temp'.format(t))
                        #print("after",t,s)
                        
                        #exec( 'self.node_output_temp_{} += ( self.edge_{}_{} )( inp )'.format( t, t, s ) )
                        #exec( 'self.node_output_temp_{} *= ( self.edge_{}_{} )( sigmoid(inp) )'.format( t, t, s ) )
                        
                    except:
                        exec( 'self.node_output_temp_{}  = ( self.edge_{}_{} )( inp )'.format( t, t, s ) )
                        #print("before",t,s)
                        #exec( 'self.node_output_temp_{}  = ( self.edge_{}_{} )( sigmoid(inp) )'.format( t, t, s ) )
                        
                    #print( torch.sum( eval('self.node_output_temp_{}'.format(t)) ))
                    
                else:
                    try:
                        exec( 'self.node_output_temp_{} += ( self.edge_{}_{} )( inp )'.format( t, t, s ) )
                    except:
                        exec( 'self.node_output_temp_{}  = ( self.edge_{}_{} )( inp )'.format( t, t, s ) )
                
                # print()
                # print("target:", t)
                # print(  eval('np.shape(self.node_output_temp_{})'.format(t) ))
                # print( eval('self.node_output_temp_{}'.format(t) ))

                
                """This part will only be used when a node is no longer a target node,
                an activation function will be applied on the node if defined"""
                if set_activation == True:
                    
                    
                    
                    # Pre-activation values should be normalized
                    #print("activation for", t)
                    if self.A_op[t][1] == 'T':
                        # preactivation
                        #print(eval('self.node_output_temp_{}'.format(t)))
                        
                        exec('self.node_output_temp_{} = nn.Tanh()(self.node_output_temp_{}).clone()'.format(t,t))
                        #print("tanh")
                        
                    
                    if self.A_op[t][1] == 'S':
                        # preactivation
                        #print(eval('self.node_output_temp_{}'.format(t)))
                        
                        exec('self.node_output_temp_{} = nn.Sigmoid()( self.node_output_temp_{} ).clone()'.format(t,t))
                        #print("sigmoid")

            
            
        nr_nodes = len(self.A)
        for i in range(nr_nodes):
            try:
                exec( 'self.adj_node_output_{} = self.node_output_temp_{}.clone()'.format(i, i))
                exec( 'del( self.node_output_temp_{})'.format(i))
            except:
                pass

        "By default we return the last node output of this internal graph"
        return eval('self.adj_node_output_{}'.format(nr_nodes-1))

    

def add_list_to_list(L, l, pos):
    for i in range(len(l)):
        L.insert( i+pos, l[i] )
    return L

        
        
        
        
        
        
        
        
        
        
    