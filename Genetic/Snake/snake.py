# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:15:50 2019

@author: Simon
"""
start = time.time()
import numpy as np
import time
import pygame

import pyautogui
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import math
import random

import brain as br
import game_rules as cr
#import pygame
#import tkinter as tk
#from tkinter import messagebox


TRAINING = True
NR_CH   = 100
MUTATION_COEFF = 0.01

Input = 5 #X,Y,food angle & dist, wall x, wall y, time_remain
Output = 1 #Angle
H = [Input,40, 40,Output] #1 input, 2 hidden, 1 output

restart = True
if restart == True:
    CH_W = []
    CH_B = []
    for k in range(NR_CH):
        W = [] #Create NR_CH different neural weights 
        B = [] #Create the bias
        for i in range( len(H)-1): #creates 3 weight matrices
            W.append( np.random.random((H[i+1],H[i]))*2 -1  )
            B.append( np.random.random(H[i+1])*2 -1)
            #creates weight matrices with weights between -1 and 1
        CH_W.append( W )
        CH_B.append( B )
    

#def update_frame(i, POS,  NR_CH):
def play_game(H, W, B, s):
    FPS = 60
    points = 0
    fitness = 0
    i = 0
    ALIVE = True
    food_XY = []
    POS,X_BOT, Y_BOT, = cr.get_init_values()
    #print("Starting snake number: ", s)
    start_snake = time.time()
    
    ###Create surface...
    surface = pygame.display.set_mode((X_BOT,Y_BOT),0)
    color = 200,0,10
    center = 100,200
    radius = 10
    food_color = 120,240,0
    V = 5
    
    F_T_R =[]
    
    while ALIVE==True: #This is the game
        #time.sleep(1./60) #60 frames per second
    
        if (POS[0]<=0) or (POS[1] <= 0) or (POS[0] >= X_BOT) or (POS[1] >= Y_BOT) or i==600+40*fitness:
            #print("You hit the wall...",  "Points: ",fitness, ". Iteration: ",i) #Game over...
            end_snake = time.time()
            #print("Snake alive for: ", end_snake - start_snake, "seconds...")
            return fitness
        
        if len(food_XY)==0 or len(food_XY)==1:
            food_pos_X = np.random.randint(10,X_BOT-10)
            food_pos_Y = np.random.randint(10,Y_BOT-10)
            food_XY.append([food_pos_X, food_pos_Y])    
        
        wall_d = cr.wall_distance(POS, X_BOT, Y_BOT)
        
        if len(food_XY) > 0:
            #print(food_XY)
            food_angle, nr, food_dist = cr.food_distance(food_XY, POS)
        
        #A = np.array([POS[0]/X_BOT, POS[1]/Y_BOT, food_angle/(np.pi), food_dist/X_BOT, wall_d[0], wall_d[1] ])
        time_remain = i /(300+40*points)
        A = np.array([  food_angle/(np.pi), food_dist/X_BOT, wall_d[0], wall_d[1], time_remain ])
        #Input: X_pos, Y_pos, food angle and dist (closest), distance of nearest walls X and Y.
        angle = br.feedforward( A, H, W, B)
        #print(A, "The input...")

        
        DIR = cr.direction(angle, V)
        OLD_POS = POS 
        POS = POS + DIR

        path  = cr.pathfinder(OLD_POS, POS)
        
        pts, food_XY = cr.eat(food_XY, path, nr)
        #points+=pts
        
        if pts >0:
            F_T_R.append(1) #each entry is true value of an apple you're holding
        pts=0 #reset pts counter
        

        if len(F_T_R)>0 and (220<=POS[0]<=280) and (220<=POS[1] <= 280):
            #Return food to receive points
            fitness += 10 #sqrt(points+1)*log(points+1)
            F_T_R.pop()
        

        #pygame.event.wait()
        surface.fill((0,0,0))
        center = POS[0], POS[1]
        pygame.draw.circle(surface, color, center, radius)
        
        #center_old = OLD_POS[0], OLD_POS[1]
        #pygame.draw.circle(surface, color, center_old, radius)

        
        try:
            food_center1 = food_XY[0]
            food_center2 = food_XY[1]
            pygame.draw.circle(surface, food_color, food_center1, radius)
            pygame.draw.circle(surface, food_color, food_center2, radius)
            pygame.display.update()
        except:
            
            try:
                food_center1 = food_XY[0]
                pygame.draw.circle(surface, food_color, food_center1, radius)
                pygame.display.update()
            
            except:
                food_center=0,0      
                pygame.display.update()
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                 sys.exit(0)
    
        i+= 1

    return fitness


def get_second_highest(a):
    hi = mid = 0
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

def start_game(NR_CH, H, CH_W, CH_B, MUTATION_COEFF ,TRAINING ):
    Fitness_list = []
    for i in range(NR_CH):
        
        fitness = play_game( H, CH_W[i], CH_B[i], i )
        Fitness_list.append(fitness)
        
    Fitness_list = np.array(Fitness_list)
    second, index = get_second_highest(Fitness_list)
    
    parents = ( np.where(Fitness_list == np.amax(Fitness_list)) )[0]
    
    '''
    a = -log(0.01) / 1000
    if max(Fitness_list) > 500:
        MUTATION_COEFF = (np.e**(-max(Fitness_list)*a) )*0.1
        print(MUTATION_COEFF)
    '''
    if TRAINING == True:
        if len(parents) == 1:
            children, children_B = br.breed( CH_W[parents[0]], CH_W[index], CH_B[parents[0]], CH_B[index], NR_CH, MUTATION_COEFF, H )
        
        else:
            children, children_B = br.breed( CH_W[parents[0]], CH_W[parents[1]], CH_B[parents[0]], CH_B[parents[1]], NR_CH, MUTATION_COEFF, H )
    else:
        pass
    
    return children, children_B, Fitness_list

GENERATIONS = 1000
fitness_over_time = []
for i in range(GENERATIONS):
    new_weights, new_bias, Fitness_list = start_game(NR_CH, H, CH_W, CH_B, MUTATION_COEFF, TRAINING)
    CH_W = new_weights
    CH_B = new_bias
    print( "Max fitness after generation", i+1, ": " ,  max(Fitness_list))
    
    fitness_over_time.append( max(Fitness_list) )


end = time.time()
print(end - start)


"Run this code when closing the screen otherwise it will crash"s
pygame.display.quit()






