# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 00:40:38 2021

@author: Simon
"""


"Pytorch"
import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F


"Pretrained model loading"
import torchvision.models as models
resnet18 = models.resnet18()

resnet18_pt = models.resnet18(pretrained=True)

"Custom dependencies"
import graph_network_templates as gt
import node_class as nc

"Timing"
import time

"Numpy & Scipy"
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
from sklearn.utils import shuffle

"For images & plots"
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageGrab as ig
import cv2

#dtype = torch.float32
#torch.set_default_dtype(torch.float32)
if __name__ == '__main__':
    
    "TODO LIST:"
    "Dedicated transformers (possible with custom operators)"
    "Recursion blocks, lstm etc (use graph nn with timesteps)"
    "Phase and paralelizations 50%"
    "Compute output dimensions of each node/sub-graph etc"
    "-> Will be computed when compiling"
    "Removal of nodes 50%"
    "Decide training = off or on"
    "timestep sensitive nodes - nodes are reset after certain number of timesteps"
    "Mixed timestep sensitive nodes - not every node share the same timestep sensitivity"
    "Allow for node-individual activation functions"
    "Import premade models for nodes"
    "Import pretrained models for nodes"
    "Streamlined services"
    
    
    
    "IMPLEMENTED"
    " - Dedicated resnet blocks"
    " - Make it possible to store and load intermediate values from feedforward"
    " - Start & end feedforward anywhere wrt subnodes"
    " - Addition of nodes and graphs for obj and obj.NODE_0 etc... "
    " - Pseudo_timesteps"
    " - Bidirectional graphs"
    " - Directed/bidirected graph combinations"
    " - Multiple input nodes for graph neural networks"
    " - Add nodes and edges to existing graphs 50%"
    " - Removal of edge connections"
    " - Custom operators"
    " - Must allow for obj only as a neural network"
    " - Import premade models for edge connections"
    " - Import pretrained models for edge connections"

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #dataset = torchvision.datasets.MNIST('mnist_data',transform=T,download=True)
    #dataset = torchvision.datasets.CIFAR10('cifar10_data',transform=T,download=True)
    #dataset = torchvision.datasets.CIFAR100('cifar100_data',transform=T,download=True)
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=2, shuffle=True, pin_memory=True)
    

    ###########################################
    ###### CREATING THE ADJACENCY MATRIX ######
    ###########################################
    
    
    A = np.array([ [0 ,0 ,0 ,0 ,0 ,0 ,0],
                   [22,0 ,0 ,0 ,0 ,0 ,0],
                   [0 ,22,0 ,0 ,0 ,0 ,0],
                   [0 ,22,0 ,0 ,0 ,0 ,0],
                   [0 ,22,0 ,0 ,0 ,0 ,0],
                   [0 ,0 ,22,22,22,0 ,0],
                   [0 ,0 ,0 ,0 ,0 ,22,0]
        ])
    
    A_conv_fc = np.array([ [ 0 , 0 , 0 ],
                           [ 100, 0 , 0 ],
                           [ 0 , 100, 0 ] ])
    
    
    
    #A = np.array([ [0,0,0,0], [20,0,0,0], [0,22,0,0], [0,0,21,0] ])
    A_1 = np.array([ [0] ])
    #A_2 = np.array([ [0,0], [23,0] ]) #normal cnn
    A_2 = np.array([ [0,0], [20,0] ]) #identity
    A_3 = np.array([ [0,0,0], [24,0,0], [24,24,0] ])
    
    
    A_4   = np.array([ [0,0,0,0], [211,0,0,0], [0,211,0,0], [0,0,211,0] ])
    A_4_0 = np.array([ [0,0,0,0], [26,0,0,0], [26,0,0,0], [0,27,27,0] ])
    A_4_1 = np.array([ [0,0,0,0], [24,0,0,0], [24,0,0,0], [0,25,25,0] ])
    
    A_4_4 = np.array([ [0,0,0,0], [23,0,0,0], [0,23,0,0], [0,0,23,0] ])
    
    #A_4_0 = np.array([ [0,0,0], [26,0,0], [0,27,0] ])
    #A_4_1 = np.array([ [0,0,0], [24,0,0], [0,25,0] ])
    
    A_4   = np.array([ [0,0,0,0,0], [211,0,0,0,0], [0,211,0,0,0], [0,0,211,0,0], [0,0,0,211,0] ])
    #A_4 += A_4.T

    A_5   = np.array([ [0,0,0,0,0], [40,0,0,0,0], [0,40,0,0,0], [0,0,40,0,0], [0,0,0,40,0] ])
    A_bidir = A_5.copy()
    A_bidir += A_5.T
    A_bidir[-2,-1] = 0
    A_bidir[-1,0] = 40
    
    
    A_rs18 = np.array([ [0,0], [110,0] ])
    
    "Define adj matrix for common network architectures"
    "Resnet"
    A_resnet_0 = np.array([ [0,0,0], [210,0,0], [200,211,0] ])
    A_resnet   = np.array([ [0,0,0], [211,0,0], [201,211,0] ])
    A_resnet_1 = np.array([ [0,0,0], [211,0,0], [202,212,0] ])
    
    A_resfc  = np.array([ [0,0], [1,0] ])
    A_r18    = np.array([ [0,0,0,0,0,0], [1,0,0,0,0,0], [0,1,0,0,0,0],
                          [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0] ])
    #################################################
    ###### DEFINE GRAPH USING ADJACENCY MATRIX ######
    #################################################
    


    ##########################
    ###### LOAD DATASET ######
    ##########################
    batch_size = 32
    "G: is the SSD"
    dataset = (torchvision.datasets.CIFAR10('G:/Datasets/cifar10_data',transform=T,download=True))
    #dataset = (torchvision.datasets.CIFAR100('G:/Datasets/cifar100_data',transform=T,download=True))
    #testset = torchvision.datasets.CIFAR10('G:/Datasets/cifar10_data', train=False, transform=T,download=True)
    if __name__ == '__main__':
        #DATA = torch.utils.data.DataLoader(dataset,batch_size=BS, shuffle=True, pin_memory=True, num_workers=0) #1: 19.2s
        DATA = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, drop_last = True) #19.5s - 22.5s
    
    

    reset = True
    if reset == True:

        "Loss over time"
        Loss = []
        
        "Assign learning rate"
        ntw = nc.network_variables(0.001)
        
        "Defines a sequential graph"
        #obj = nc.graphs(A_bidir, 'obj')
        obj = nc.graphs(A_rs18)
        
        obj_pt = nc.graphs(A_rs18)
        
        "Some resnet blocks"
#        obj.node_0 = nc.node( A_5 , 'obj.node_0' )
#        obj.node_1 = nc.node( A_5   , 'obj.node_1' )
#        obj.node_2 = nc.node( A_5   , 'obj.node_2' )
#        
##        obj.node_2.node_0 = nc.node( A_resnet , 'obj.node_2.node_0' )
##        obj.node_2.node_1 = nc.node( A_resnet , 'obj.node_2.node_1' )
##        obj.node_2.node_2 = nc.node( A_resnet , 'obj.node_2.node_2' )
#        #obj.node_2.node_3 = nc.node( A_resnet )
#        
#        obj.node_3 = nc.node( A_5   , 'obj.node_3' )
#        obj.node_4 = nc.node( A_5 , 'obj.node_4' )
        
        
        "Define separate graph"
        discr = nc.graphs( A_conv_fc , 'discr')
        
        #discr.node_0 = nc.node( A_conv_fc )
        
        "Compile graphs"
        nc.graphs.compile_graph(obj)
        nc.graphs.compile_graph(obj_pt)
        nc.graphs.compile_graph(discr)
        
    
    cec_loss = nn.L1Loss()
    
    #cec_loss_2 = nn.L1Loss()
    
    target_discr_ones  = torch.ones((13,1000)).to(device)
    target_discr_zeros = torch.zeros((batch_size,1)).to(device)
    
    TEST_input = torch.ones((5, 13, 10)).to(device)
    
    resnet_test = torch.ones((13,3,32,32)).to(device)
    
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    for e in range(1):
        
        print("EPOCH:", e)
        st = time.time()
        
        
        for I,(images,labels) in enumerate(DATA):
            print(I)
            images_gpu = TEST_input
            #images_gpu = images.to(device)
            #labels_gpu = labels.to(device)
            
            "using obj.save = 1, the node_outputs must be deleted manually otherwise the graph will accumulate"

            
            #images = (ds[I*batch_size : (I+1)*batch_size  ]).to(device)
            
            #output = nc.graphs.pseudo_forward(obj, images_gpu )
            "Step 0"
            "initiate number of pseudo_timesteps for obj members, 1 is default"
            obj.pseudo = 2
            
            
            "Step 1"
            "Autoencoder generates the real image"
            "Classified as True"
            #output = nc.graphs.pseudo_forward_TEST( obj   , images_gpu )
            #output = nc.graphs.pseudo_timestep_primer( obj , images_gpu , timestep = 4 , input_nodes = 5 )
            
            
            output = nc.graphs.pseudo_forward_TEST( obj   , resnet_test )
            output_pt = nc.graphs.pseudo_forward_TEST( obj_pt   , resnet_test )
            #output = nc.graphs.pseudo_forward_TEST( discr   , target_discr_ones )

            loss   = cec_loss( output       , target_discr_ones   )
            loss.backward(retain_graph = False)

            break
            
            
            #del obj.node_output_5
            loss   = cec_loss( output       , images_gpu   )
            loss.backward(retain_graph = False)
            
            
            
            output_discr = nc.graphs.pseudo_forward_TEST( discr , output.detach()  )
            loss_2 = cec_loss( output_discr , target_discr_ones )
            loss_2.backward(retain_graph = False)
            
            
            "Step 2"
            "Create a fake image based on the real image"
            "This will be trained to classify as False"
            x = torch.rand((32,16,32,32)).to(device)
            output_fake = nc.graphs.pseudo_forward_TEST( obj , images_gpu  , end_node = 'temp_graph.node_2'     )
            output_fake = nc.graphs.pseudo_forward_TEST( obj , output_fake + x , start_node = 'temp_graph.node_3'     )
            #loss        = cec_loss( output , images_gpu   )
            #loss.backward(retain_graph = True)
            
            output_discr = nc.graphs.pseudo_forward_TEST( discr , output_fake.detach()  )
            loss_3 = cec_loss( output_discr , target_discr_zeros )
            loss_3.backward(retain_graph = False)
            del x
            del obj.node_output_5
#            
#            
#            "Step 3"
#            "Generates a completely random image"
#            "Classified as True"
#            del x
#            x = torch.rand((32,16,32,32)).to(device)
#            output_2 = nc.graphs.pseudo_forward_TEST( obj, x,  start_node ='temp_graph.node_3' )
#            output_discr_2 = nc.graphs.pseudo_forward_TEST( discr , output_2 )
#            
#            loss_4 = cec_loss( output_discr_2 , target_discr_ones )
#            loss_4.backward(retain_graph = False)
#            
#            del obj.node_output_5
#            del x

            
            
            if I%25==0:
                
                clf()
                
                #fig, (ax1, ax2) = plt.subplots(1, 2)
                #fig.suptitle('Horizontally stacked subplots')
                
                "permutes the dims so that we get batch, row, col, channels"
                output_imgs = np.array(output.detach().cpu().permute(0,2,3,1) )
                #output_imgs_2 = np.array(output_2.detach().cpu().permute(0,2,3,1) )
                
                "the node_0 output"
                #node_img = np.array(obj.node_0.node_output_0.detach().cpu().permute(0,2,3,1) )
                #node_img = np.array(obj.node_output_1.detach().cpu().permute(0,2,3,1) )
    
                "The original image"
                #img = np.array(images_gpu.detach().cpu().permute(0,2,3,1) )
                
                "The generated image from x random"
                img = np.array(output_2.detach().cpu().permute(0,2,3,1) )
                
                "We want this to be as close as possible to the og input"
                "Left image"
                fig.add_subplot(1, 2, 1)
                #imshow(output_imgs[0] - output_imgs_2[0])
                imshow(output_imgs[0] / output_imgs[0].max() )
                #imshow( node_img[0] /node_img[0].max() )
                
                "Right image"
                fig.add_subplot(1, 2, 2)
                #imshow(output_imgs_2[0])
                imshow(img[0] / 0.01+img[0].max() )
                
                plt.pause(0.01)
        
            "We need to delete old values to free up memory"
            "... otherwise the backward-computational-graph will accumulate"
            "... and cause a memory leak"
            
            del output
            del images
            
            if I%25==0:
                print()
                print(loss, loss_2, loss_3, loss_4)
                print("Time:", time.time() - st)
                st = time.time()
        print("Time:", time.time() - st)
        
        
        

    
    "insert another sub-graph"
    #graphs.insert_graph(obj,A)
    
    
    
    "Define operations between sub-graphs"
    #graphs.A_meta(obj, A_meta)
    
    #new_node = [ [ 't' , 2 , 7 ] , [ 'r' , 0 , 9 ] , [ 'r' , 1 , 9 ] ]
    #obj.Node.insert_node( new_node )
    