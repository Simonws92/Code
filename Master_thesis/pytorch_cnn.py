import torch, torchvision
#from visdom import Visdom
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F

import time

import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

import matplotlib.pyplot as plt


######################################
#    Part I : Write Data Loaders     #
######################################

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#dataset = torchvision.datasets.MNIST('mnist_data',transform=T,download=True)
dataset = torchvision.datasets.CIFAR10('cifar10_data',transform=T,download=True)
#dataset = torchvision.datasets.CIFAR100('cifar100_data',transform=T,download=True)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64, shuffle=True, pin_memory=True)




######################################
#    Part II : Write the Neural Net  #
######################################
torch.cuda.empty_cache()


class network(nn.Module):
    def __init__(self):
        
        "no conv cifar10"
        #self.H = [32**2, 400,200,10]
        "no conv mnist"
        #self.H = [28**2, 400,200,10]
        "conv cifar10 RESNET"
        self.H = [500, 100,100,100,10]
        
        "conv cifar10 CNN"
        #self.H = [2000, 100,100,10]
        
        "conv cifar100"
        #self.H = [500*2**2, 100,100,100]
        
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
        
        
        self.H_c = [3,30,60,120,250, self.H[0] ] #number of filters
        #self.H_c = [3,60,120,250,500, self.H[0] ] #number of filters
        "initiate cnn"
        for i in range(5): #3 for cnn, 4 for resnet
            #exec( 'self.conv{} = nn.Conv2d(self.H_c[{}],self.H_c[{}],3, padding=1)'.format(i,i,i+1) ) #For normal CNN only
            
            ###The resnet blocks###
            exec( 'self.conv{}   = nn.Conv2d(self.H_c[{}],self.H_c[{}],3, padding=1, padding_mode = "zeros")'.format(i,i,i+1) )
            
            #exec( 'self.conv_b{}   = nn.Conv2d(self.H_c[{}],self.H_c[{}],5, padding=2, padding_mode = "zeros")'.format(i,i,i+1) )

            exec( 'self.conv_2{} = nn.Conv2d(self.H_c[{}],self.H_c[{}],3, padding=1, padding_mode = "zeros")'.format(i,i,i) )
            
            exec( 'self.conv_3{} = nn.Conv2d(self.H_c[{}],self.H_c[{}],3, padding=1, padding_mode = "zeros")'.format(i,i,i) )
            
            exec( 'self.Batch_norm_{} = nn.BatchNorm2d( self.H_c[i] )'.format(i) )
            "This convolution must be the identity"
            #exec( 'self.conv_ID{} = nn.Conv2d(self.H_c[{}],self.H_c[{}],1)'.format(i,i,i+1) )
            


        
        "Initiate fully connected"
        for i in range(self.n-1):
            exec( 'self.linear{} = nn.Linear(self.H[{}], self.H[{}])'.format(i,i,i+1) )
        
        "input channels, output channels, filter size "
        self.pool = nn.MaxPool2d(2,2)   
        self.relu     = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax  = nn.Softmax()
        self.tanh     = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        
        
        
        
    def forward(self,images):
        x = images
        
        ##################
        ##### ResNet #####
        ##################
        
        for i in range(5):
            x_temp = x#.clone()
            batch_norm = eval ('(self.Batch_norm_{})'.format(i) )
            x = eval( '(F.relu( batch_norm( self.conv_2{}(x))  ) )'.format(i) )
            x = eval( '( batch_norm( self.conv_3{}(x)) )'.format(i) )
            x = F.relu(x + x_temp)
#            
            x = eval( '(F.relu(self.conv{}(x)))'.format(i) )
            
#            
#            #x = nn.BatchNorm2d()
#            
            x = self.pool(F.relu(x))
        
        #######################
        ##### Classic CNN #####
        #######################
        '''
        for i in range(5):
            #x_b = x.clone()
            x   = eval( 'self.pool(F.relu(self.conv{}(x)))'.format(i) )
            #print( shape(x) )
            
            #x_b = eval( 'self.pool(F.relu(self.conv_b{}(x_b)))'.format(i) )
            #print( shape(x_b) )
            
            #x= x + x_b
            #print( shape(x) )
        '''
            
        
        ##############
        ##### FC #####
        ##############
        x = x.view(-1, self.H[0]) #cifar10
        self.A = [x]
        self.z = [x]
        
        
        for i in range(self.n-1):
            (self.z).append( eval( 'self.linear{}(self.A[{}])'.format(i,-1) ) )
            if i==self.n-2:
                temp = self.z[-1].clone()
                temp = Activations.Act( model, 'softmax',temp.T,0)
                (self.A).append( temp.T )
            
            else:
                (self.A).append( self.relu( self.z[i+1] ) )

                #x = self.tanh(x)

        #print("ff done", shape(self.z[-1]))
        return self.z[-1]
        #return x

################################
##### placeholder for Adam #####
################################
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
        
##############################
##### 2nd order backprop #####
##############################
class Identity(torch.autograd.Function):
    
    def __init__(self ):
        super(Identity, self).__init__()
        
    @staticmethod
    def forward(self, input):
        return input.clone()
    
    @staticmethod
    def backward(self, grad_output):
        
        batch_nr = model.batch_nr
        
        
        grad_input = grad_output.clone()
        
        "Computing gradient for initial layer"
        delta_old = ( model.A[-1] - model.labels.T )
        
        "Loads informations for optimizer"
        delta_class = model.fc_mom
        
        temp = model.A[-2].clone()
        delta_w = torch.matmul( delta_old, temp.T )
        delta_w, _ = Adam_class.Adam_opt( model, delta_class[0], 0, delta_w, delta_w, batch_nr, bias_true=0)
        model.weight_updates[-1] = delta_w
        
        "Computes initial 2nd order delta"
        temp = model.z[-1].clone()
        delta_H_1 = 2*Activations.Act( model, model.H_act[-1],temp,1)**2          #Delta_H1 and delta_H2 have same dimensions
        delta_H_2 = 2*delta_old * Activations.Act( model, model.H_act[-1], temp, 2)
        A_prev = delta_H_1 + delta_H_2

        for p in range(1,model.n-1):
            
            " Load weights "
            W = eval( 'model.linear{}.weight'.format( model.n-p-1 ) )
            
            " Calculate 1st and 2nd derivatives of activations "
            temp = model.z[-p-1].clone()
            der      = Activations.Act( model, model.H_act[-p-1], temp, 1) ### First derivative
            " Relu is not 2times differentiable. Use softplus or something else instead "
            " 2nd der of relu = 1st der of relu as a workaround "
            scnd_der = Activations.Act( model, model.H_act[-p-1], temp, 2) ### Second derivative
            
            " Updates delta function for the next layer "
            pre_delta = torch.matmul(W.T, delta_old )
            delta = pre_delta * der
            delta_old = delta
            
            " The creation of the propagation matrix A_l "
            B_l = pre_delta * scnd_der
            A_l =  torch.matmul( (W.T)**2 ,A_prev ) * (der**2) + B_l
            A_prev = A_l.clone()
            
            " The creation of the Averaged Hessian " #currently not in use
            '''
            Avg_A = torch.mean(A_l, axis = 0)
            Avg_A = torch.reshape(Avg_A, ( len(Avg_A), 1 ))
            
            Avg_A.expand_as(temp)
            H_i = torch.matmul( temp.T , Avg_A * temp ) / batch_size
            '''
            
            temp = model.A[-p-2].clone()
            
            "compute the hessian and the gradient"
            Hess = torch.matmul( A_l,   (temp.T)**2 )
            delta_w = torch.matmul( delta, temp.T )
            
            
            "1st order adam"
#            delta_w, _ = Adam_class.Adam_opt( model, delta_class[0], p, delta_w, delta_w, batch_nr, bias_true=0)
#            Hess, _ = Adam_class.Adam_opt( model, delta_class[1], p, Hess, Hess, batch_nr, bias_true=0)
#            delta_w = delta_w / (abs(Hess)+0.1)
            
            #"2nd order adah"
            delta_w = AdaH_class.AdaH_opt( model, delta_class[1], p, delta_w, Hess, batch_nr, bias_true=0)
            
            "add weight updates"
            model.weight_updates[-1-p] = delta_w
        
        return grad_input
    
################
##### Adam #####
################
class Adam_class:
    def __init__(self):
        pass
    @staticmethod
    def Adam_opt(self, delta_class, p, delta_w1, delta_w2, batch_nr, bias_true=0):
        
        beta1 = 0.9
        beta2 = 0.999
        old_mt = (delta_class.Adam)[-p-1][0 + 2*bias_true]
        old_vt = (delta_class.Adam)[-p-1][1 + 2*bias_true]
    
        Mt = beta1 * old_mt + (1-beta1) * (delta_w1)
        Vt = beta2 * old_vt + (1-beta2) * (delta_w2**2)
        
        ''' Adam correction '''
        Mt_bar = Mt / (1 - beta1**(batch_nr+1))**2
        Vt_bar = Vt / (1 - beta2**(batch_nr+1))**2
        
        ''' Adam final update '''
        eps = 1e-2
        Mt_bar = Mt_bar / (torch.sqrt(Vt_bar) + eps)
        
        delta_class.update_Adam( -p-1, 0, bias_true, Mt)
        delta_class.update_Adam( -p-1, 1, bias_true, Vt)
        return Mt_bar, Vt_bar
    

################
##### AdaH #####
################
class AdaH_class:
    def __init__(self):
        pass
    @staticmethod
    def AdaH_opt(self, delta_class, p, delta_w1, delta_w2, batch_nr, bias_true=0):
        
        beta1 = 0.9
        beta2 = 0.999
        old_mt = (delta_class.Adam)[-p-1][0 + 2*bias_true]
        old_vt = (delta_class.Adam)[-p-1][1 + 2*bias_true]
    
        Mt = beta1 * old_mt + (1-beta1) * (delta_w1)
        Vt = beta2 * old_vt + (1-beta2) * (delta_w2**2)
        
        ''' Adam correction '''
        Mt_bar = Mt / (1 - beta1**(batch_nr+1))**2
        Vt_bar = Vt / (1 - beta2**(batch_nr+1))**2
        
        ''' Adam final update '''
        eps = 1e-2
        
        delta_class.update_Adam( -p-1, 0, bias_true, Mt)
        delta_class.update_Adam( -p-1, 1, bias_true, Vt)
        
        grad = Mt_bar / ( torch.sqrt(Vt_bar) + eps )
        
        return grad


################################
##### Activation functions #####
################################
class Activations:
    def __init__(self):
        pass
    @staticmethod
    def Act(self,H_act,z,d):
        
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
            
        if H_act=='softmax':
            z -= z.max()
            a = torch.exp(z)
            a= a/torch.sum(a, axis=0)
            if d==0:
                return a
            else:
                return a*(1-a)
            
        if H_act == 'tanh':
            temp = torch.tanh(z)
            if d==0:
                return 
            if d==1:
                return 1-temp**2
            if d==2:
                return -2*temp*( 1-temp**2 )


########################################################
###### Initiate parameters and placerholder stats ######
########################################################
CUSTOM = False
GPU_training = True
reset = True
"Memory leak when reset=True" #WIP


TEST = False
if reset == True:
    acc = []
    lot = [] #loss over time
    max_lot = [] #max loss per epoch
    min_acc = []
    lot_test = []
    cec_loss = nn.CrossEntropyLoss()
    model = network()
    parameters = model.parameters()
    max_test_lot = []
    
    
" Use when comparing two different training methods wrt their loss over time "
do_not_touch = True
if do_not_touch == False:
    lot_2 = lot.copy()
    max_lot_2 = max_lot.copy()
    min_acc_2 = min_acc.copy()

#optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(params=model.parameters(),lr=0.003)

"for custom learning rates"
lr     = 0.003 #epoch 1: 1.6 loss #For custom
fc_lr  = 0.003
bi_lr  = 0.003
cnn_lr = 0.003
optimizer = optim.Adam([
            {'params': model.linear0.weight, 'lr': fc_lr},  {'params': model.linear0.bias,   'lr': bi_lr},
            {'params': model.linear1.weight, 'lr': fc_lr},  {'params': model.linear1.bias,   'lr': bi_lr},
            {'params': model.linear2.weight, 'lr': fc_lr},  {'params': model.linear2.bias,   'lr': bi_lr},
            {'params': model.linear3.weight, 'lr': fc_lr},  {'params': model.linear3.bias,   'lr': bi_lr},
#            {'params': model.linear4.weight, 'lr': fc_lr},  {'params': model.linear4.bias,   'lr': bi_lr},
#            {'params': model.linear5.weight, 'lr': fc_lr},  {'params': model.linear5.bias,   'lr': bi_lr},
            
            ###1st RESNET CONV###
            {'params': model.conv0.weight,   'lr': cnn_lr}, {'params': model.conv0.bias,     'lr': cnn_lr},
            {'params': model.conv1.weight,   'lr': cnn_lr}, {'params': model.conv1.bias,     'lr': cnn_lr},
            {'params': model.conv2.weight,   'lr': cnn_lr}, {'params': model.conv2.bias,     'lr': cnn_lr},
            {'params': model.conv3.weight,   'lr': cnn_lr}, {'params': model.conv3.bias,     'lr': cnn_lr},
            {'params': model.conv4.weight,   'lr': cnn_lr}, {'params': model.conv4.bias,     'lr': cnn_lr},
            
            ### 2nd RESNET CONV###
            {'params': model.conv_20.weight,   'lr': cnn_lr}, {'params': model.conv_20.bias,     'lr': cnn_lr},
            {'params': model.conv_21.weight,   'lr': cnn_lr}, {'params': model.conv_21.bias,     'lr': cnn_lr},
            {'params': model.conv_22.weight,   'lr': cnn_lr}, {'params': model.conv_22.bias,     'lr': cnn_lr},
            {'params': model.conv_23.weight,   'lr': cnn_lr}, {'params': model.conv_23.bias,     'lr': cnn_lr},
            {'params': model.conv_24.weight,   'lr': cnn_lr}, {'params': model.conv_24.bias,     'lr': cnn_lr},
            
            ### FINAL CONV OUTSIDE RESNET ###
            {'params': model.conv_30.weight,   'lr': cnn_lr}, {'params': model.conv_30.bias,     'lr': cnn_lr},
            {'params': model.conv_31.weight,   'lr': cnn_lr}, {'params': model.conv_31.bias,     'lr': cnn_lr},
            {'params': model.conv_32.weight,   'lr': cnn_lr}, {'params': model.conv_32.bias,     'lr': cnn_lr},
            {'params': model.conv_33.weight,   'lr': cnn_lr}, {'params': model.conv_33.bias,     'lr': cnn_lr},
            {'params': model.conv_34.weight,   'lr': cnn_lr}, {'params': model.conv_34.bias,     'lr': cnn_lr}
            
        ], lr=0.0, weight_decay=0)

#optimizer = optim.Adam( parameters=parameters() , lr=0.001)
#optimizer = optim.RMSprop(parameters, lr=0.001, alpha=0.99, eps=1e-04, weight_decay=0, momentum=0, centered=False)

"WIP. Please ignore"
'''
#t = "'params': model.linear{}.weight, 'lr': fc_lr"
#optimizer = optim.Adam([ { exec(t) } ], lr=0.003, weight_decay=1e-5)
#eval( '(F.relu(self.conv{}(x)))'.format(0) )
#exec( 'optimizer = optim.Adam([ { "params": model.linear{}.weight, "lr": fc_lr } , lr=0.003)'.format(0) )
#model = network()
#model_inst = set_model(model)
'''


"initiate custom momentum terms"
model.fc_mom = [ dense_fc(model.H), dense_fc(model.H), dense_fc(model.H) ]


"GPU training"
if GPU_training == True:
    network = network()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    model.to(device)

if TEST == True:
    epochs = 1
    test_acc = []
    test_loss = []
    

    testset = torchvision.datasets.CIFAR10('cifar10_data', train=False, transform=T,download=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=False)
    DATA = testloader
    
else:
    epochs = 10
    DATA = dataloader

"Custom"
if CUSTOM == True:
    identity = Identity()

print("Starting training...")
for e in range(18):
    
    min_acc_epoch  = 2 #initial minimum accuracy, accuracy can never be above 1.
    max_loss = 0 #must be reset after every new epoch
    max_test_loss = 0
    
    start_t = time.time()
    print("Epoch", e)
    
    model.fc_mom[0].reset_Adam(model.H)
    model.fc_mom[1].reset_Adam(model.H)
    model.fc_mom[2].reset_Adam(model.H)
    
    "Shuffle dataset per epoch"
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=128, shuffle = True)
    
    testset = torchvision.datasets.CIFAR10('cifar10_data', train=False, transform=T,download=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=True)

    #model.fc_mom[0].reset_Adam(model.H)
    " Full dataset "
    for i,(images,labels) in enumerate(DATA):
    
        #########################
        ###### For testing ######
        #########################
        TEST=True
        if TEST==True:
            try:
                img_test, label_test = next(iter(testloader))
            except:
                testloader = torch.utils.data.DataLoader(testset,batch_size=128, shuffle=True)
                img_test, label_test = next(iter(testloader))
                
            if GPU_training==True:
                img_test, label_test = img_test.to(device), label_test.to(device)
            output_test = model(img_test.detach())
            loss_test = cec_loss(output_test, label_test)
            loss_test = loss_test.detach().cpu()
            #print("TEST:",loss_test)
            lot_test.append( np.array( loss_test ) )




        start_time = time.time()
        model.batch_nr = i

        "GPU training"
        if GPU_training == True:
            "Add noise to input images"
            #images += np.random.normal(0,0.05, (32,32))
            #images=images.float()
            "sends input and corresponding labels to gpu"
            images, labels = images.to(device), labels.to(device)

        
        images = Variable(images)
        labels = Variable(labels)
        
        "feedforward"
        output = model(images)
        
        "one-hot encoding from integers 0-9 to ex [0,0,0,0,1,0,0,0,0,0]"
        model.labels = torch.nn.functional.one_hot(labels, num_classes = 10)
        model.zero_grad()
        
        #####################
        ##### Custom BP #####
        #####################
        " Custom for 2nd order "
        if CUSTOM == True:
            for j in range(model.n):
                model.A[j] = (model.A[j]).T
                try:
                    model.z[j] = (model.z[j]).T
                except:
                    pass
                
            "Custom 2nd order"
            output = identity.apply(output)
            
        
        "Accuracy WIP"
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
            
            
        ##########################
        ##### Calculate loss #####
        ##########################
        loss = cec_loss(output,labels)
        loss.backward()        
        if CUSTOM == True:
            with torch.no_grad():
                "update custom weight updates"
                model.linear0.weight -= model.weight_updates[0]*lr
                model.linear1.weight -= model.weight_updates[1]*lr
#                model.linear2.weight -= model.weight_updates[2]*lr
#                model.linear3.weight -= model.weight_updates[3]*lr
#                model.linear4.weight -= model.weight_updates[4]*lr
#                model.linear5.weight -= model.weight_updates[5]*lr
        

        optimizer.step()

        lot.append( np.array( loss.detach().cpu()) )
        
        if i%25==0:
            print(i)
            print( "LOSS:", np.array([loss.item()]) , "Accuracy:", "na", "Epoch:", e)
            
            #print("Overfit:", (loss_test.item()  - loss.item()) / loss_test.item() )
            
        if loss > max_loss:
            max_loss = loss
        
        if loss_test > max_test_loss:
            max_test_loss = loss_test

    min_acc.append(min_acc_epoch)
    max_lot.append(max_loss)
    
    max_test_lot.append(max_test_loss)
    
    total_time = time.time() - start_t
    print(total_time)        


"Plots loss over time resulsts for training and validation"
plt.plot(max_lot, "-b", label="Training")
plt.plot(max_test_lot, "-r", label="Validation")
plt.title("2nd order, Cifar10, ResNet. Adam on gradient and Hessian")
plt.xlabel("Epoch")
plt.ylabel("Max loss")
plt.legend(loc="upper right")
plt.show()

figure()
plt.plot(lot, "-b", label="Training")
plt.plot(lot_test, "-r", label="Validation")
plt.title("2nd order, Cifar10, ResNet. Adam on gradient and Hessian")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()


"Ignore"
if False:
    #Adahessian results
    lot2 = lot.copy()
    lot_test2 = lot_test.copy()
    max_lot2 = max_lot.copy()
    max_test_lot2 = max_test_lot.copy()


    plt.plot(lot_avg, "-b", label="Training PyTorch")
    plt.plot(lot_avg2, "-r", label="Training AdaHessian")
    plt.title("1st & 2nd order, Cifar10, ResNet")
    #plt.xlabel("Epoch")
    #plt.ylabel("Max loss")
    plt.legend(loc="lower left")
    plt.show()
    
    #figure()
    plt.plot(lot_test_avg, "-p", label="Validation PyTorch")
    plt.plot(lot_test_avg2, "-p", label="Validation AdaHessian")
    #plt.title("2nd order, Cifar10, ResNet. Adam on gradient and Hessian")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.legend(loc="lower left")
    plt.show()



