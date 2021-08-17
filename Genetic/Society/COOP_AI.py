# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:15:50 2019

@author: Simon
"""
start_time = time.time()
import numpy as np
norm = np.linalg.linalg.norm

import time
import pygame

import pyautogui
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import math
import random
#import pygame
#import tkinter as tk
#from tkinter import messagebox

import brain as br #the neural network

start = True
NR_CH   = 10
MUTATION_COEFF = 0.2

Input = 4 #X,Y,food angle & dist, wall x, wall y
Output = 1 #Angle
H = [Input,20, 20,Output] #1 input, 2 hidden, 1 output

restart = True
if restart == True:
    CH_W_coop = []
    CH_B_coop = []
    for k in range(NR_CH):
        W = [] #Create NR_CH different neural weights 
        B = []
        for i in range( len(H)-1): #creates 3 weight matrices
            W.append( np.random.random((H[i+1],H[i]))*2 -1  )
            B.append( np.random.random(H[i+1])*2 -1 )
            #creates weight matrices with weights between -1 and 1
        CH_W_coop.append( W )
        CH_B_coop.append( B )
    



def play_game(H, CH_W,CH_B, s, NR_CH):

    
    
    
    
    return 0



def start_game(NR_CH, H, CH_W, CH_B, MUTATION_COEFF ):
    Fitness_list = []
    fitness = play_game( H, CH_W, CH_B_coop, i ,NR_CH )
    Fitness_list.append(fitness)
        
    Fitness_list = np.array(Fitness_list)
    
    print(Fitness_list[0])
    print(shape(Fitness_list))
    Fitness_list = Fitness_list[0]
    
    second, index = br.get_second_highest(Fitness_list)
    
    parents = ( np.where(Fitness_list == np.amax(Fitness_list)) )[0]
    
    
    if len(parents) == 1:
        children = br.breed( CH_W[parents[0]], CH_W[index], CH_B[parents[0]], CH_B[index], NR_CH, MUTATION_COEFF, H )
    
    else:
        children = br.breed( CH_W[parents[0]], CH_W[parents[1]], CH_B[parents[0]], CH_B[parents[1]], NR_CH, MUTATION_COEFF, H )
    
    return children, Fitness_list

GENERATIONS = 100
fitness_over_time = []
if start == True:
    for i in range(GENERATIONS):
        new_weights_coop, Fitness_list = start_game(NR_CH, H, CH_W_coop, CH_B_coop, MUTATION_COEFF)
        CH_W_coop = new_weights_coop
        print( "Max fitness after generation", i+1, ": " ,  max(Fitness_list))
        
        fitness_over_time.append( max(Fitness_list) )


end_time = time.time()
print(end_time - start_time)



pygame.display.quit()






