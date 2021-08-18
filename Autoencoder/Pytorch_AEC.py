# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 23:49:30 2021

@author: Simon W S
"""

import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import main_master as m
import new_Dense_1_3 as nd

import time

from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

import matplotlib.pyplot as plt
from sklearn.utils import shuffle


import pytorch_cnn_fork as PTF

####################################
###### Load cifar10/100/mnist ######
####################################

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#dataset = torchvision.datasets.MNIST('mnist_data',transform=T,download=True)
dataset = torchvision.datasets.CIFAR10('cifar10_data',transform=T,download=True)
#dataset = torchvision.datasets.CIFAR100('cifar100_data',transform=T,download=True)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=2, shuffle=True, pin_memory=True)




#######################################
###### Initialize neural network ######
#######################################

class network(nn.Module):
    def __init__(self):
        conv_type = "classic"
        AE = True
        
        
        "Initiate network architecture"
        
        "Conv number of channels per conv layer"
        "conv cifar10 RESNET"
        
        "Encoder"
        self.H_c1 = [3,50,100,120,200] #number of filters
        
        "Decoder"
        self.H_c2 = [200,200,120,120,100,100,50,50,3]
        #self.H_c2 = [200,120,100,50,3]
        
        "FC nr of neurons per layer"
        o=4
        self.output_size = o
        self.H = [self.H_c2[0]*o**2, 100,20,40,20,100,self.H_c2[0]*o**2]
        
        "Define activations"
        #self.H_act = ['relu']*5+['I']
        self.H_act = ['relu']*(len(self.H)-2)+['softmax']
        #Stores values to be used for later
        self.z = []
        self.A = []
        self.n = len(self.H)
        self.delta = 0 #initial placeholder value
        self.labels = 0
        self.weight_updates = [0]*(len(self.H)-1)
        self.fc_mom = 0
        self.batch_nr = 0
        
        super(network,self).__init__()
        
        "Conv: input channels, output channels, filter size "
        "conv_2B{} etc is for convolution of multiple filter sizes"
        if conv_type == "classic":
            "initiate cnn"
            for i in range(len(self.H_c1)-1): #3 for cnn, 4 for resnet
                
                "These padding and kernel size settings works only for imagesize with multiples of 2"
                "These convolution operators retains the spatial dimensions post-conv"
                exec( 'self.conv{} = nn.Conv2d(self.H_c1[{}],self.H_c1[{}],3, padding=1)'.format(i,i,i+1) ) #For normal CNN only
                
                #exec( 'self.conv_b{} = nn.Conv2d(self.H_c1[{}],self.H_c1[{}],5, padding=2)'.format(i,i,i+1) ) #For normal CNN only
                
                #exec( 'self.conv_B{} = nn.Conv2d(self.H_c1[{}],self.H_c1[{}],7, padding=3)'.format(i,i,i+1) ) #For normal CNN only
                
            if AE == True:
                for i in range(len(self.H_c2)-1):
                    "with upsampling"
                    if i%2==0:
                        exec( 'self.conv_2{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],3, padding=1, stride=1)'.format(i,i,i+1) ) #For normal CNN only
                    else:
                        exec( 'self.conv_2A{} = nn.Conv2d(self.H_c2[{}],self.H_c2[{}],3, padding=1)'.format(i,i,i+1) ) #For normal CNN only
                    
                        #exec( 'self.conv_2b{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],5, padding=2, stride=1)'.format(i,i,i+1) ) #For normal CNN only
                        
                        #exec( 'self.conv_2B{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],7, padding=3, stride=1)'.format(i,i,i+1) ) #For normal CNN only
                        
                "without upsampling"
                #exec( 'self.conv_2{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],2, padding=0, stride=2)'.format(i,i,i+1) ) #For normal CNN only
                
                #if i%2==0:
                #    exec( 'self.conv_2{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],3, padding=0, stride=2)'.format(i,i,i+1) ) #For normal CNN only
                #else:
                #    exec( 'self.conv_2{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],3, padding=1, stride=2)'.format(i,i,i+1) ) #For normal CNN only


        
        if conv_type == "resnet":
            for i in range(len(self.H_c1)-1): #3 for cnn, 4 for resnet
                ### The resnet blocks ###
                exec( 'self.conv{}   = nn.Conv2d(self.H_c1[{}],self.H_c1[{}],3, padding=1, padding_mode = "zeros")'.format(i,i,i) )
                exec( 'self.conv_2{} = nn.Conv2d(self.H_c1[{}],self.H_c1[{}],3, padding=1, padding_mode = "zeros")'.format(i,i,i) )
                
                ### Channel reduction ###
                exec( 'self.conv_3{} = nn.Conv2d(self.H_c1[{}],self.H_c1[{}],3, padding=1)'.format(i,i,i+1) )
                
                if AE==True:
                    exec( 'self.conv_T{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],3, padding=1, stride=1)'.format(i,i,i) ) #For normal CNN only
                    exec( 'self.conv_2T{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],3, padding=1, stride=1)'.format(i,i,i) ) #For normal CNN only
                    
                    exec( 'self.conv_3T{} = nn.ConvTranspose2d(self.H_c2[{}],self.H_c2[{}],3, padding=1, stride=1)'.format(i,i,i+1) ) #For normal CNN only
                
            

            
            
        "Define pytorch functions"
        self.pool = nn.MaxPool2d(2,2)
        
        "Initiate fully connected"
        for i in range(self.n-1):
            exec( 'self.linear{} = nn.Linear(self.H[{}], self.H[{}])'.format(i,i,i+1) )
            
            
        self.relu     = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax  = nn.Softmax()
        self.tanh     = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.upsample = torch.nn.Upsample( scale_factor=2, mode='nearest', align_corners=None)
    
    
    ################################
    ###### Feedforward module ######
    ################################
    def forward(self,images, AE=0):
        conv_type = "classic"
        x = images
        if AE != 0:
            FC_start = AE
            #A=[0]*(AE+1)
            self.A[-1] = x
            self.z=[0]*(AE+1)
            #z[-1] = x
            #print("in AE")
            #print(shape(x))
        else:
            
            if conv_type == "resnet":
            ##################
            ##### ResNet #####
            ##################
                for i in range(5):
                    x_temp = x.clone()
                    #print( shape(x), "before padding" )
                    x = eval( '(F.relu(self.conv{}(x)))'.format(i) )
                    x = eval( '(self.conv_2{}(x))'.format(i) )
                    #print( shape(x), "after padding" )
                    
                    x = F.relu(x + x_temp)
                    
                    x = eval( '(self.conv_3{}(x))'.format(i) )
                    
                    #x = nn.BatchNorm2d()
                    if i%2==0:
                        "every other second layer will be pooled"
                        x = self.pool(F.relu(x))
                    #print( shape(x), "after conv" )
                
                    
                #x = self.pool(F.relu(x))
                
            #######################
            ##### Classic CNN #####
            #######################
            if conv_type == "classic":
                for i in range( len(self.H_c1)-1 ):
                    #print(shape(x))
                    #x_b = x.clone()
                    #x_B = x.clone()
                    
                    x = eval( 'F.relu(self.conv{}(x))'.format(i) )
                    #x_b = eval( 'F.relu(self.conv_b{}(x_b))'.format(i) )
                    #x_B = eval( 'F.relu(self.conv_B{}(x_B))'.format(i) )
                    
                    #x = x +x_b #+x_B
                    
                    if i%1==0:
                        x = eval( 'self.pool(x)'.format(i) )
                    else:
                        pass
            batch_size = shape(x)[0]
            x = torch.reshape(x, (batch_size, self.output_size**2*self.H_c2[0]))
            
        "These variables will be stored"
        self.A = [x]
        self.z = [x]
        
        
        #############################
        ###### Fully-connected ######
        #############################
        for i in range(AE,self.n-1):
            #print(AE)
            #print(shape(x), i)
            (self.z).append( eval( 'self.linear{}(self.A[{}])'.format(i,-1) ) )
            if i==self.n-2:
                #(self.A).append( self.softmax( self.z[i+1] ) )
                #temp = self.z[-1].clone()
                #(self.A).append( temp )
                
                "only for AE"
                #print("in here end")
                (self.A).append( self.relu( self.z[i+1] ) )
            else:
                #(self.A).append( self.tanh( self.z[i+1] ))
                #print("in here")
                "Dropout"
                #self.z[i+1] = self.dropout(self.z[i+1])
                (self.A).append( self.relu( self.z[i+1] ) )
                
        x = self.A[-1]
        
        x = x.view(-1,self.H_c2[0],self.output_size,self.output_size)
        #print(shape(x), "start conv T")
        if conv_type == "classic":
            for i in range( len(self.H_c2)-1 ):
                #print(shape(x))
                

                if i%2==0:
                    "perform upsampling"
                    if i > 6:
                        "no pooling"
                        pass
                    else:
                        x = self.upsample(x)
                    x = eval( '(F.relu(self.conv_2{}(x)))'.format(i) )
                else:
                    x = eval( '(F.relu(self.conv_2A{}(x)))'.format(i) )
                
                #x_b = x.clone()
                #x_B = x.clone() 
                #x = eval( '(F.relu(self.conv_2{}(x)))'.format(i) )
                #x_b = eval( 'F.relu(self.conv_2b{}(x_b))'.format(i) )
                #x_B = eval( 'F.relu(self.conv_2B{}(x_B))'.format(i) )
                #x=x+x_b #+x_B
        
        if conv_type == "resnet":
        ##################
        ##### ResNet #####
        ##################
            for i in range( len(self.H_c2)-1 ):
                if i%2==0:
                    "every other second layer will be pooled"
                    
                    x = self.upsample(x)
                
                x_temp = x.clone()
                #print( shape(x), "before padding" )
                x = eval( '(F.relu(self.conv_T{}(x)))'.format(i) )
                x = eval( '(self.conv_2T{}(x))'.format(i) )
                #print( shape(x), "after padding" )
                
                x = F.relu(x + x_temp)
                
                x = eval( '(self.conv_3T{}(x))'.format(i) )
                
                #x = nn.BatchNorm2d()
        #print("done")
        return x


#################################
###### Initiate parameters ######
#################################
#cec_loss = nn.MSELoss
CUSTOM = False
GPU_training = True
reset = False
"Memory leak when reset=True"


TEST = False
if reset == True:
    acc = []
    lot = [] #loss over time
    max_lot = [] #max loss per epoch
    min_acc = []
    lot_test = []
    #cec_loss = nn.CrossEntropyLoss() # cross entropy probability
    #cec_loss = nn.MSELoss()
    cec_loss = nn.L1Loss()
    model = network()
    
    parameters = model.parameters()
    
    
" Compare two different training methods wrt their loss over time "
if False:
    lot_2 = lot.copy()
    max_lot_2 = max_lot.copy()
    min_acc_2 = min_acc.copy()

"initiate optimizer with learning rate"
optimizer = optim.Adam( model.parameters(), lr=0.0001, weight_decay=0)

"GPU training"
if GPU_training == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

if TEST == True:
    epochs = 1
    test_acc = []
    test_loss = []
    max_test_lot = []

    testset = torchvision.datasets.CIFAR10('cifar10_data', train=False, transform=T,download=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=False)
    DATA = testloader
    
else:
    epochs = 10
    DATA = dataloader

"Custom"
identity = PTF.Identity(model)#.apply


###########################
######Import dataset ######
###########################

#data_3 = load('audi_data.npy')
#data_2 = load('bmw_data.npy')
#data = load('ferrari_data.npy')
data = load('faces_data.npy')
data = list(data) #+ list(data_2) + list(data)
data = np.array(data)

#dataset = torch.tensor( np.reshape(data, (len(data),3,96,96)  ))/255

"Dataset dimensions must be: NR_examples, channels, row, col"
dataset = (torch.tensor( data ).permute(0,3,1,2) ) /255
batch_size = 32

##########################
######   Training   ######
##########################
print("Starting training...")
for e in range(200):
    
    
#    "Reduces LR by a factor of 0.1 at every 15 epochs"
#    if e%epochs == 0 and e!=0:
#        lr *= 0.1
#        print("Reducing learning rate to", lr)
    
    
    min_acc_epoch  = 2 #initial minimum accuracy, accuracy can never be above 1.
    max_loss = 0 #must be reset after every new epoch
    start_t = time.time()
    print("Epoch", e)
    
    model.fc_mom[0].reset_Adam(model.H)
    model.fc_mom[1].reset_Adam(model.H)
    model.fc_mom[2].reset_Adam(model.H)
    
    "Shuffle dataset per epoch"
    #testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=True)

    "obtain random entries from the dataset. Dataset is shuffled at every new epoch"
    random_nr = np.arange(0,len(dataset),1)
    np.random.shuffle(random_nr)
    ds = dataset[random_nr]
    
    "if dataset comes from pytorch etc"
    #for i,(images,labels) in enumerate(DATA):
    
    "if custom dataset"
    for i in range( int(len(dataset) / batch_size)-1 ):
        #print("Batch", i)
#        try:
#            img_test, label_test = next(iter(testloader))
#        except:
#            testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=True)
#            img_test, label_test = next(iter(testloader))
        
        images = ds[i*batch_size : (i+1)*batch_size  ]

        start_time = time.time()
        model.batch_nr = i

        "GPU training"
        if GPU_training == True:
            #img_test, label_test = img_test.to(device), label_test.to(device)
           
            "Add noise to input images"
            #images += np.random.normal(0,0.05, (32,32))
            #images=images.float()
            #images, labels = images.to(device), labels.to(device)
            images = images.to(device)

        
        images = Variable(images)
        #labels = Variable(labels)
        
        
        output = model(images) #Forward
        
        #output_2 = model.forward(model.A[3], 3)
        
        #model.labels = torch.nn.functional.one_hot(labels)
        #print(shape(model.labels), shape(model.labels))
        model.zero_grad()
        
        #####################
        ##### Custom BP #####
        #####################
        " Custom for 2nd order "
        if CUSTOM == True:
            for j in range(model.n):
                #print("before", shape(model.A[i]))
                model.A[j] = (model.A[j]).T
                try:
                    model.z[j] = (model.z[j]).T
                except:
                    pass
            
                #print("after", shape(model.A[i]))
                
            "Custom 2nd order"
            output = identity(output)
            
        
        ##########################
        ##### Start Accuracy ##### #Doesnt work for test atm
        ##########################
        " accuracy "
#        accu = 0
#        bs = shape(model.A[-1])[1]
#        for j in range(bs):
#            if torch.sum(abs(model.labels * model.A[-1]-model.labels)[j,:]) < 0.5:
#                accu += 1
#        #print("Accuracy:",accu/batch_size)
#        ac = accu/bs
        
        '''
        if TEST == True:
            test_acc.append(ac)
        else:     
            acc.append(ac)
        
        if ac < min_acc_epoch:
            min_acc_epoch = ac
        ''' 
        
        "Plots example"
        if i%25==0:
            
            clf()
            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #fig.suptitle('Horizontally stacked subplots')
            
            output_imgs = np.array(output.detach().cpu().permute(0,2,3,1) )
            #img = np.array(images.detach().cpu().permute(0,2,3,1) )
            
            #fig.add_subplot(1, 1, 1)
            imshow(output_imgs[0])
            
            #fig.add_subplot(2, 2, 2)
            #imshow(img[0])
            
            plt.pause(0.01)
            
        ##########################
        ##### Calculate loss #####
        ##########################
        #loss = cec_loss(output,labels)
        loss = cec_loss(output,images)
        loss.backward()   
        optimizer.step()
        " The loss is recorded every 25 batches "
        
        #output_test = model(img_test.detach())
        #loss_test = cec_loss(output_test, label_test)
        "For testing only"
        
        if TEST == False:
            lot.append( np.array( loss.detach().cpu()) )
            #lot_test.append( np.array( loss_test.detach().cpu()) )
        
        "free up memory from gpu"
        images = images.cpu()
        

        #print(i)
        if i%25==0:
            print("batch",i)
            print( "LOSS:", np.array([loss.item()]) , "Accuracy:", "na", "Epoch:", e)
            
            #print("Overfit:", (loss_test.item()  - loss.item()) / loss_test.item() )
            
        if loss > max_loss:
            max_loss = loss
        
        #################################
        ##### Automatic derivatives #####
        #################################
        "Dont use both at the same time"
        #loss.backward(retain_graph = True) #Memory heavy
        
        

#                model.weight_updates[0] = 0
#                model.weight_updates[1] = 0
#                model.weight_updates[2] = 0
        #else:

        #else:
        #    test_loss.append( loss.detach().cpu() )
        
        
        
        "Time: 1.2sec CPU, 0.25sec GPU"
    if TEST == False:
        min_acc.append(min_acc_epoch)
        max_lot.append(max_loss)
    
    total_time = time.time() - start_t
    print(total_time)        


"Plots loss over time"
figure()
plt.plot(lot)
    




"plots all training examples of the batch"
if False:
    og_imgs = np.array( images.detach().cpu().permute(0,2,3,1) )
    test_img = og_imgs[2]
    imshow( test_img )    
    
    figure()
    output_imgs = np.array(output.detach().cpu().permute(0,2,3,1) )
    test_img_out =  output_imgs[2]
    imshow( test_img_out)
    
    
if False:
    for i in range(64):
        clf()
        test_img_out =  output_imgs[i]
        imshow( test_img_out)
        plt.pause(0.2)
        plt.show()
        
        
    
if False:
    
    
    for i in range( int(len(dataset) / batch_size)-1 ):
        #print("Batch", i)
    #        try:
    #            img_test, label_test = next(iter(testloader))
    #        except:
    #            testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=True)
    #            img_test, label_test = next(iter(testloader))
        images = ds[i*batch_size : (i+1)*batch_size  ]
    
    
        "GPU training"
        if GPU_training == True:
            #img_test, label_test = img_test.to(device), label_test.to(device)
           
            "Add noise to input images"
            #images += np.random.normal(0,0.05, (32,32))
            #images=images.float()
            #images, labels = images.to(device), labels.to(device)
            images = images.to(device)
    
        
        images = Variable(images)
        #labels = Variable(labels)
        
        
        output = model(images) #Forward
        output_imgs = np.array(output.detach().cpu().permute(0,2,3,1) )
    
        for i in range(64):
            clf()
            test_img_out =  output_imgs[i]
            imshow( test_img_out)
            plt.pause(0.2)
            plt.show()











import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


"Play with the latent-space"
if False:
    A_ls = model.A[3][-1,:]#.copy() #/ abs(A[5][:,10]).max()
    #A_LS = torch.reshape(A_ls, (1,20))
    
    #A_ls = torch.tensor( np.random.random((20,1)) )
    A_ls = A_ls.to(device)
    
    output_2 = model.forward(A_ls, 3)[0]
    
    #img = torch.reshape(output_2, (64,64,3) )
    img = np.array(output_2.detach().cpu().permute(1,2,0) )
    #img /= (img).max()
    #img[img<0.5]=0
    
    fig = plt.figure()
    l = plt.imshow(img / abs(img).max() )
    
    for i in range(20):
        #exec( 'slid{} = Slider(  plt.axes([0.05 , 0.0 + {}/25, 0.1, 0.03])  , "index_{}", -5, 5, valinit=A_ls[{}], valfmt="%d")'.format(i,i,i,i) )
        exec( 'slid{} = Slider(  plt.axes([0.05 , 0.0 + {}/25, 0.1, 0.03])  , "index_{}", -10, 30, valinit=A_ls[{}], valfmt="%d")'.format(i,i,i,i) )
    
    
    def update(val):
        A_ls = model.A[3][-1,:]
        for i in range(20):
            exec( 'id{} = slid{}.val'.format(i,i) )
    
            A_ls[i] = eval( 'id{}'.format(i,i) )
                

        A_ls = A_ls.to(device)

        "For imshow"
        output_2 = model.forward( A_ls, 3)[0]
        img = np.array(output_2.detach().cpu().permute(1,2,0) )
        l.set_data(img /img.max())
        
        model.A[3][-1,:] = A_ls
        
        
    for i in range(20):
        exec( 'slid{}.on_changed(update)'.format(i) )
