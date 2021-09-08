# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:15:50 2019

@author: Simon
"""
start = time.time()
import numpy as np
import numexpr as ne
import time
import pygame

import pyautogui
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import math
import random

import brain as br
import new_Dense as nd
import new_Dense_fitness as nd_F




class Genetic:
    def __init__( self, H, H2, NR_CH, MUTATION_COEFF, TRAINING):
        self.NR_CH = NR_CH 
        self.H = H
        self.H2 = H2
        self.MUTATION_COEFF = MUTATION_COEFF
        self.TRAINING = TRAINING
        
        
        CH_W, CH_B, CH_money, CH_edu, CH_food, CH_fitness, CH_breed, CH_age = self.define_char(H, NR_CH)
        self.CH_W = CH_W
        self.CH_B = CH_B
        self.CH_money = CH_money
        self.CH_edu = CH_edu
        self.CH_food = CH_food
        self.CH_fitness = CH_fitness
        self.CH_breed = CH_breed
        self.CH_age = CH_age
        
        ###################
        ###### Breed ######
        ###################
        "Weights determine whom gets to breed"
        CH_W_2, CH_B_2, _, _, _, _, _, _ = self.define_char(H2, NR_CH, CH_W = [], CH_B = [], CH_money = [],
            CH_edu = [], CH_food = [], CH_fitness = [], CH_breed = [], CH_age=[])
        
        self.CH_W_2 = CH_W_2
        self.CH_B_2 = CH_B_2
        self.Breed_point = [0] * len(CH_W) #Each character earns a breed point if someone wants to breed with this character
        
        #####################
        ###### Fitness ######
        #####################
        self.H_3 = [6,1]
        CH_W_3, CH_B_3, _, _, _, _, _, _ = self.define_char(self.H_3, NR_CH, CH_W = [], CH_B = [], CH_money = [],
            CH_edu = [], CH_food = [], CH_fitness = [], CH_breed = [], CH_age=[])
        self.CH_W_3 = np.array(CH_W_3[0])
        self.CH_B_3 = np.array(CH_B_3[0])
        
        self.Fitness_old = [0]*NR_CH
        self.Fitness_new = 0
        
        self.fot = []
        
        ######################
        ##### Investment #####
        ######################
        self.invest = 0 #Default
        
        #####################
        ##### Statistic #####
        #####################
        self.TOPOP = []
        self.MAXM = []
        self.MAXE = []
        self.MAXFO = []
        self.historical_breed_point = []
        self.dec = []
        self.total_born = []
        self.old_pop = 0
        self.avg_age = []

        self.death_by_hunger = []
        self.death_by_age    = []
    
    def define_char(self,H, NR_CH, CH_W = [], CH_B = [], CH_money = [],
                    CH_edu = [], CH_food = [], CH_fitness = [], CH_breed = [], CH_age=[] ):
        #print(CH_W, "hi")
        start_len = len(CH_W)
        #print("length of W:",start_len)
        for k in range(self.NR_CH):
            
            if start_len==0: #If empty then create new weights and biases
            
                W = [] #Create NR_CH different neural weights 
                B = [] #Create the bias
                for i in range( len(H)-1): #creates 3 weight matrices
                    W.append( np.random.random((H[i+1],H[i]))*2 -1  )
                    #B.append( np.random.random(H[i+1])*2 -1)
                    B.append( np.random.random((H[i+1],1))*2 -1)
                    #creates weight matrices with weights between -1 and 1
                    
                CH_W.append( W )
                CH_B.append( B )
            else:
                pass
            CH_money.append( 0 ) #All characters start with 10 credits, 0 education, 7 days worth of food and 0 fitness
            CH_edu.append( 0 ) 
            CH_food.append( 200 )
            CH_fitness.append( 0 )
            CH_breed.append(0)
            CH_age.append(0)
        return CH_W, CH_B, CH_money, CH_edu, CH_food, CH_fitness, CH_breed, CH_age
        

    
    
    #def update_frame(i, POS,  NR_CH):
    def play_game(self, t): #One character performs one update to its life
        #print("Character:", s)
        H_act = ["tanh"] + ["I"]

        og_len = len(self.CH_food)
        Survived_W = []
        Survived_W_2 =[]
        Survived_B = []
        Survived_B_2 = []
        Survived_edu = []
        Survived_money = []
        Survived_food = []
        Survived_fitness = []
        Survived_age=[]
        Survived_breed=[]
        
        breed = 0
        school = 0
        work = 0
        buy = 0
        invest = 0
        nr_dead = 0
        
        train = 0
        
        print("Number of characters alive:",og_len, "... Time:", t)
        print("lowest food:", min(self.CH_food))
        print("We have invested: ", self.invest)
        for i in range(og_len):
            #print(i, "Food:", self.CH_food[i])
            
            if og_len==0:
                print("We're all dead!")
            
            random_nr = int( np.random.normal(0) ) *10
            if self.CH_food[i] > 0 and self.CH_age[i] < 200 + random_nr:
                #print("Food:", self.CH_food[i], i)
            
                temp = np.reshape( self.CH_W_3 , (6,1) )
                x = np.array([[self.CH_money[i], self.CH_edu[i], self.CH_food[i],self.invest, self.CH_age[i] ]]).T #Must be a N x 1 matrix
                x = np.concatenate((x, np.tanh(temp / np.average(temp)) ))
                
                x = np.reshape( x , (5+6,1) )
                
                z,choices = nd_F.feedforward(1,x, self.H, self.CH_W[i], self.CH_B[i],0, H_act)
                
                max_choice = max(choices[-1])
                
                ''' Characters choice will be taken into consideration
                    with the importance of the s.CH_W_3 weights
                '''
                
                
                #print(max_choice, "Choice")
                if max_choice == choices[-1][0]: #First option: Go to school
                    #print("Go to school")
                    self.CH_edu[i]+=1 #+ int( self.invest / 20 )
                    school += 1
                    
                if max_choice == choices[-1][1]: #Second option: Go to work
                    #print("Go to work")
                    self.CH_money[i]+=2*self.CH_edu[i] +5
                    work += 1
                    
                    if self.CH_edu[i] == 0:
                        train = 1
                        y = np.array([[1,0,0,0,0 ]]).T #We dont want the final entry to happen during this input of A

                    
                elif max_choice == choices[-1][2] and self.CH_money[i] > 0: #Third option: Buy groceries
                    self.CH_food[i] += 10 #+ int( self.invest / 20 ) #Every 10 investment points will increase food by one
                    self.CH_money[i]-=1
                    buy += 1
                    
                elif max_choice == choices[-1][3] and self.CH_age[i]>5: #Fourth option: Breed, but only for mature enough individuals
                    self.CH_fitness[i] += 1
                    self.CH_breed[i] = 1 #Boolean, true or false
                    breed += 1
                
                
                elif max_choice == choices[-1][4]: #Fifth option: Invest
                    if self.CH_money[i] > 0:
                        self.CH_money[i] -= 1
                        self.Breed_point[ i ] += 10 #Those whom invests will gain fitness, increasing their chances of breeding
                        self.invest += 1
                        invest += 1
#                    else:
#                        y = np.array([[self.CH_money[i], self.CH_edu[i], self.CH_food[i],0,t]]).T #We dont want the final entry to happen during this input of A
#                        for j in range(5):
#                            self.CH_W[i], self.CH_B[i], _, _, _, _ = nd.backpropagation(self.H, choices, z, y, self.CH_W[i], self.CH_B[i], C=0, vs=0,lr=0.01,s=0,H_act=H_act, optimizer='none')
#                            z,A = nd.feedforward(1,x, self.H, self.CH_W[i], self.CH_B[i], drop=0,H_act=H_act)
                
                else:
                    "The AI attempted to do one of the resource intensive objects"
                    train = 1
                    y = np.array([[1,1,1,0,0]]).T

                if train == 1:
                    train = 0
                    for j in range(1):
                        
                        self.CH_W[i], self.CH_B[i], _ =\
                            nd_F.backpropagation(self.H, choices, z, y, self.CH_W[i], self.CH_B[i], C=0, vs=0,lr=0.01,s=0,H_act=H_act, optimizer='none')
                        
                        z,A = nd_F.feedforward(1,x, self.H, self.CH_W[i], self.CH_B[i], drop=0,H_act=H_act)
                
                
                
                self.CH_food[i]-=1 #Each day reduces food supplies by one
                if self.CH_food[i] > 0:
                    self.CH_age[i]+=1
                    
                    if self.CH_age[i] >= 200 + random_nr:
                        self.death_by_age[-1] += 1
                    
                    else:
                        Survived_W_2.append( self.CH_W_2[i])
                        Survived_B_2.append( self.CH_B_2[i])
                        Survived_W.append( self.CH_W[i] )
                        Survived_B.append( self.CH_B[i] )
                        Survived_money.append( self.CH_money[i] )
                        Survived_edu.append( self.CH_edu[i] )
                        Survived_food.append( self.CH_food[i] )
                        Survived_fitness.append( self.CH_fitness[i] )
                        Survived_age.append(self.CH_age[i])
                        #Survived_breed_point etc..
                    
                    
                else:
                    pass
                    #print("Character", i, " death by hunger")
                    nr_dead += 1
                    self.death_by_hunger[-1] += 1
            else:
                print("Character", i, " was dead")
                if self.CH_age[i] >= 200 + random_nr:
                    print("You died of old age")
                    
                if self.CH_food[i] <=0:
                    print("You died of hunger...")
        
        print("Number of survivors:",len(Survived_W))
#        print("School: ", school)
#        print("Work: ", work)
#        print("Buy: ", buy)# / (og_len*5))
#        print("Breed: ", breed)# / (og_len*5))
#        print("Invest: ", invest)# / (og_len*5))
#        print("Death by hunger: ", nr_dead)
        self.old_pop = len(self.CH_food)
        
        return Survived_W, Survived_B, Survived_W_2, Survived_B_2, Survived_money, \
             Survived_edu, Survived_food, Survived_fitness, Survived_age, Survived_breed
    
    ##########################
    ##### Breed selector #####
    ##########################
    def breed_selector(self,Survived_W, Survived_B, Survived_W_2, Survived_B_2, Survived_money,Survived_edu, Survived_food, Survived_fitness, Survived_age, Survived_breed):
        
        H_act=["relu"] + ["I"]
        Survived_breed=[]
        #print(len(Survived_W), "number of survivors")
        
        #Compile a list of potential breeders and create new characters
        #-Call function to breed
        NR_CH = len(Survived_W)
        
        NR_children = 0
        breed_list = [ i  for i in range(NR_CH) if self.CH_breed[i]==1 ]
        if sum( self.CH_breed) >= 2:
            #print(breed_list)
            nr_children = int(len(breed_list)/2)
            #print(nr_children, "number of children")
            
            
            #New neural network for each character
            #Each character will decide whom to breed with
            #inputs, food,money,edu etc,
            #output: one neuron which decides whom to breed with
            
            avg_fd = 0
            ch_born = 0
            
            #print(breed_list, "breed_list")
            
            x_1 = np.array([ self.CH_money[i] for i in breed_list ])
            x_2 = np.array([ self.CH_food[i]  for i in breed_list ])
            x_3 = np.array([ self.CH_edu[i]   for i in breed_list ])
            
            #print("RATIO OF BREEDERS:", len(x_1) / len(self.CH_money))
        #    print(x_1)
        #    print(x_2)
        #    print(x_3)
        #    print(nr_children, "nr ch")
            self.dec=[]
            for J in range(len(breed_list)): #Finish later
                
                x = np.array([ x_1, x_2, x_3 ]) #Must be a 3 x N matrix, N = nr of potential breeders
                z,choices = nd_F.feedforward(1,x, self.H2, self.CH_W_2[J], self.CH_B_2[J],0, H_act)
                
                #print(choices[-1][0], "Your choices")
                breeding_choice = numpy.where(choices[-1][0] == np.amax(choices[-1][0]))
                
                #print("Breeding pos of breed_list",breeding_choice[0][0])
                chosen_ch = breed_list[int( breeding_choice[0][0] )]
                #print("Your desired breeding partner is character number: ", breed_list[int( breeding_choice[0][0] )])
                
                self.Breed_point[ chosen_ch ] += 1 #+ self.CH_fitness[ chosen_ch ]
            
                self.dec.append(choices[-1][0] )
            First  = numpy.where( self.Breed_point == np.amax( self.Breed_point ))[0][0]
            Second = self.get_second_highest()[1]
            
            #print(First, Second)
            #print(self.Breed_point)
            par1 = First  #breed_list[First]      #First parent to breed
            par2 = Second #breed_list[Second]     #second parent to breed
            
            "Make it so that the parents can decide how many children will be born"
            NR_children = 4 + int(self.invest / 20) #Each parent breeds a number of children
            NR_children = min( NR_children , 10) #Max 10 children. 
            #print("NR of children born:" , NR_children)
            for J in range(NR_children): #How many children do you want each pair to make
                
                #The first weights
                children_W, children_B = br.breed( self.CH_W[par1], self.CH_W[par2], self.CH_B[par1], self.CH_B[par2], 1, self.MUTATION_COEFF, self.H )
        
                #The second weights
                children_W_2, children_B_2 = br.breed( self.CH_W_2[par1], self.CH_W_2[par2], self.CH_B_2[par1], self.CH_B_2[par2], 1, self.MUTATION_COEFF, self.H2 )
        
                Survived_W.append( children_W[0] )
                Survived_B.append( children_B[0])
                
                Survived_W_2.append( children_W_2[0] )
                Survived_B_2.append( children_B_2[0] )
                
                Survived_money.append( 0 ) #All characters start with 10 credits, 0 education, 7 days worth of food and 0 fitness
                Survived_edu.append( 0 ) 
                
                food_to_give = 20#int( self.CH_food[par1] *0.5 + self.CH_food[par2]*0.5 )
                #self.CH_food[par1] -= int( self.CH_food[par1] *0.5 )
                #self.CH_food[par2] -= int( self.CH_food[par2] *0.5 )
                self.CH_food[par1] -= 10
                self.CH_food[par2] -= 10
                Survived_food.append( food_to_give  )
                #print("Give food for children:", food_to_give)
                Survived_fitness.append( 0 )
                Survived_breed.append(0)
                Survived_age.append(0)
                
                self.total_born[-1] += 1
                
                
                
                avg_fd+=food_to_give #/ nr_children
        else:
            print("No children this round...")
            
        
        
        ch_born = NR_children
        #maxe = max(CH_edu)
        maxm = 0#max(CH_money)
        maxf = 0#avg_fd
        maxfo = ch_born

        
        self.CH_W = Survived_W
        self.CH_B = Survived_B
        self.CH_W_2 = Survived_W_2
        self.CH_B_2 = Survived_B_2
        self.CH_edu = Survived_edu
        self.CH_age = Survived_age
        self.CH_money = Survived_money
        self.CH_food = Survived_food
        self.CH_fitness = Survived_fitness
        #print(NR_CH, "BEFORE")
        self.CH_breed = [0]*len(Survived_W)
        #print(len(self.CH_breed), "AFTER")
        self.hs = self.Breed_point
        #print(self.CH_food)
        #print(self.Breed_point)
        self.Breed_point = [0]*len(Survived_age)
        
        
        return Survived_W, Survived_B, Survived_money, \
             Survived_edu, Survived_food, Survived_fitness, Survived_age, 0, maxm, maxf,maxfo 
    
    
    def get_second_highest(self):
        a = self.Breed_point
        hi = mid = 0
        mid_pos = 0
        for index, x in enumerate(a):
            if x > hi:
                mid = hi
                hi = x
                mid_pos = index-1
                #print(mid)
            elif x < hi and x > mid:
                #lo = mid
                mid = x
                mid_pos = index
                #print(mid, hi, index)
        return mid, mid_pos
    
    #def start_game(NR_CH, H, CH_W, CH_B, MUTATION_COEFF ,TRAINING, CH_money, CH_edu, CH_food, CH_fitness, CH_breed, CH_age ):
    def start_game(self ):
        T = 1000 #timesteps
        MAXE=[]
        MAXM=[]
        MAXF=[]
        MAXFO=[]
        TOPOP=[len(self.CH_W)]
        old_nr_ch = TOPOP[0]
        
        for t in range(T):
            print()
            print("Time", t)
            self.total_born.append(0)
            
            self.death_by_age.append(0)
            self.death_by_hunger.append(0)
            
            if len(self.CH_food) == 0: #Extinction
                print("Extinction")
                return 0, 0, 0, 0,0,1
            
            else:
                
                #######################
                ##### Day choices #####
                #######################
                #CH_W, CH_B, CH_money, CH_edu, CH_food, CH_fitness, CH_age, CH_breed \
                #= play_game( H, CH_W, CH_B, CH_money, CH_edu, CH_food, CH_fitness, CH_breed, CH_age, t )
                
                
                #print("BEFORE GAME:", old_nr_ch)
                Survived_W, Survived_B, Survived_W_2, Survived_B_2, Survived_money,Survived_edu, Survived_food, Survived_fitness, Survived_age, Survived_breed = self.play_game(t)
                nr_ch = len(Survived_W)
                #print("AFTER GAME:", nr_ch)
                #######################################
                ###### Adaptive fitness function ######
                #######################################
                if len(Survived_W) <= 1:
                    print("EXTINCTION")
                    return 0, 0, 0, 0,0,1
                
                big_f = [ ]
                big_F = [ [] , [] ]
                H_act = [ "I" ]
                
                for i in range(nr_ch):
                    
                    "Add constraints here"
                    #self.CH_B_3[0][0][0][0] = 0
                    
                    '''Here we create the fitness output using the fitness function'''
                    X = np.array([  [ Survived_money[i] ], [ Survived_food[i] ], [ Survived_fitness[i] ], [ Survived_edu[i] ] ,[ Survived_age[i] ], [0]  ]) #0 is placeholder
                    

                    
                    f,F = nd.feedforward(1, X , self.H_3, self.CH_W_3, self.CH_B_3,0, H_act)
                    
                    
                    big_f.append(f)
                    big_F[0].append(F[0])
                    big_F[1].append(F[1])
                    
                max_ch = max(nr_ch , old_nr_ch )
                min_ch = min(nr_ch , old_nr_ch )
                if old_nr_ch > nr_ch: #Some people died
                    "Add zeros to current fitness stats"
                    for k in range( old_nr_ch - nr_ch ):
                        #print("here")
                        big_f.append([ np.zeros((6,1)) , np.zeros((1,1))  ])       # Add zeros here for dead or newborn entries ###
                        big_F[0].append( np.zeros((6,1)) )
                        big_F[1].append( np.zeros((1,1)) )
                
                if old_nr_ch <= nr_ch:
                    "Add zeros to previous fitness stats"
                    for k in range( nr_ch - old_nr_ch  ):
                        self.Fitness_old.append(0)
                        #print("in here")
                
                    ''' End fitness forward function '''
                    
                ''' Start fitness backward '''
                                              #Add zero entries for fallen/added people
                
                big_F0 = np.array(big_F[0]).T
                big_F0 = np.reshape(big_F0, (6,max_ch) )                        #The input stats
                big_F1 = np.reshape( np.array(big_F[1]).T , (1,max_ch) )        #The output fitness
                
                new_f = [ np.reshape(big_f, (1,max_ch)) ]                       #Pre act of fitness
                new_F = [ big_F0 , big_F1 ]                                    #post act
                
                
                Fitness_cost = [ self.Fitness_old , np.array(big_F1) ]         #Fitness_old may have more entries than big_F1
                                                                               #    due to deaths
                #print(nr_ch,old_nr_ch, np.shape(big_F1) , np.shape(self.Fitness_old) )
                
                
                self.Fitness_old = [big_F1[0,l] for l in range( len(Survived_W) ) ]
                
                
                
                "Update fitness function"
                self.CH_W_3, self.CH_B_3, _ = \
                nd_F.backpropagation(self.H_3, new_F, new_f, np.ones((1,max_ch)) , self.CH_W_3, self.CH_B_3,lr=0.01,H_act=H_act,delta_class='', F=Fitness_cost, COST='fit')
                
                ''' Weights in self.CH_W_3 represents importance of stats in X
                '''
                
                ##################################
                ###### End fitness function ######
                ##################################
                old_nr_ch = nr_ch
                self.breed_selector(Survived_W, Survived_B, Survived_W_2, Survived_B_2, Survived_money,Survived_edu, Survived_food, Survived_fitness, Survived_age, Survived_breed)
                
                
                self.TOPOP.append(len(self.CH_food))
                self.MAXE.append(max(self.CH_edu))
                self.MAXM.append(max(self.CH_money))
                #MAXF.append(max(self.CH_food))
                self.MAXFO.append(max(self.CH_food))
                self.avg_age.append( average(self.CH_age) )
                
                #CH_breed = [0]*len(CH_edu) #Number of characters

    
        return MAXE,MAXM,MAXF,MAXFO, TOPOP, 0

#if restart == True:
#    CH_W, CH_B, CH_money, CH_edu, CH_food, CH_fitness, CH_breed, CH_age = define_char(H, NR_CH)
#    CH_W_2, CH_B_2, _, _, _, _, _, _ = define_char(H2, NR_CH, CH_W = [], CH_B = [], CH_money = [],
#                CH_edu = [], CH_food = [], CH_fitness = [], CH_breed = [], CH_age=[])

TRAINING = True
NR_CH   = 100
MUTATION_COEFF = 0.001

Input = 5 #edu, money, food, invest, age
Output = 5 #school, work, buy food, breed, invest
nr_stats_prio = 6
H = [Input + nr_stats_prio, 20,Output] #1 input, 2 hidden, 1 output

Input_2  = 3
Output_2 = 1 #Choose whom to breed with, using the same input except age
H2 = [Input_2,40,Output_2]

restart = True

s = Genetic( H, H2, NR_CH, MUTATION_COEFF, TRAINING )
s.start_game()



end = time.time()
print(end - start)








