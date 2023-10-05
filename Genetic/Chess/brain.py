import numpy as np

def sigmoid(z):
    z = np.array(z,dtype='float64')
    A = 1/(1+np.e**(-z))
    return A

def softmax(z):
    #print(z, "prepresoft")
    z = np.array(z)
    z = sigmoid(z/(max(z))) #Otherwise the values will be too large for e**z
    sumz = sum(np.e**z[i] for i in range(len(z)))
    A = np.e**z
    #print(z, "PRESOFT...")
    return A / sumz

def R(z):
    z = np.array(z)
    z[z<=0]=0
    return z

def feedforward(x, H, W, B, dp): #Does a complete feedforward iteration through the network
    z = []
    A = [x]
    for i in range(len(H)-1):
        
        if dp > 1:
            z.append( np.dot( W[i] , A[i]) + B[i][:,np.newaxis]  )
            if i == len(H)-2: #Final layer
                A.append( sigmoid(z[i]) )
                #A.append( softmax(z[i]) )
            else:
                A.append( np.tanh(z[i]) )
                
        else:
            z.append( np.dot( W[i] , A[i]) + B[i]  )
            if i == len(H)-2: #Final layer
                A.append( z[i] )
                #A.append( softmax(z[i]) )
            else:
                A.append( np.tanh(z[i]) )
    return z[-1],A[-1] #Only interested in the output


def breed(male, female, male_bias, female_bias, NR_CH, MUTATION_COEFF, H):
    #male weights, female weights, ...
    male = np.array(male)
    female = np.array(female)
    
    male_bias = np.array(male_bias)
    female_bias = np.array(female_bias)
    #print(shape(male))
    
    #print("Starting breeding...")
    CH_W = [] #Placeholder for storing neural weights for every children
    CH_B = []
    for k in range(NR_CH): #Going trough all characters...
        W = [] #Placeholder for neural weights for one child
        B = []
        for i in range( len(H)-1): #creates 3 weight matrices
            w = np.zeros((H[i+1],H[i]))
            b = np.zeros(H[i+1])
            
            mutation = np.random.randint(-1,2, H[i+1]*H[i] )  #mutation matrix
            mutation = np.reshape( mutation, (H[i+1], H[i])) * MUTATION_COEFF #Slight mutation
            
            mutation_B = np.random.randint(-1,2, H[i+1]) * MUTATION_COEFF #mutation vector for the bias
            
            r = H[i+1]
            c = H[i]
            binary_matrix   = np.random.randint(0,1,(r,c))
            binary_matrix_2 = np.mod( binary_matrix + 1 , 2 )
            
            b = male_bias[i] * binary_matrix[:,0] + female_bias[i] * binary_matrix_2[:,0]
            w = male[i]*binary_matrix + female[i]*binary_matrix_2
                    

            W.append( w+mutation ) #New weight for a child with added mutation
            B.append( b+mutation_B )
            
        CH_W.append( W )
        CH_B.append( B )
    
    
    
    #mutation = random.randint(-1,1) #*MUTATION_COEFF
    #print(mutation)
    #children[:,0:2] = male[0:2]   + mutation
    #children[:,2:4] = female[2:4] + mutation
    return CH_W, CH_B


def get_second_highest(a):
    hi = mid = 0
    a = np.array(a)
    #print(a, "CHECK a")
    for index, x in enumerate(a):
        if x > hi:
            mid = hi
            hi = x
            #print(mid)
        elif x < hi and x > mid:
            lo = mid
            mid = x
            #print(mid, hi, index)
    return mid, index
