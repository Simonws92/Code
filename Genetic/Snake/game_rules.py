# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:10:45 2019

@author: Simon
"""
import numpy as np
norm = np.linalg.linalg.norm

###Fixed game_rules...

def pathfinder(old_pos, new_pos): #Creates an interpolation polynomial between the two XY positions
                                  #to find a path between new and old XY positions using least squares
    dpos = (abs(old_pos[0] - new_pos[0])+1)*2
    X    = np.linspace(old_pos[0], new_pos[0], dpos) #Mapped to X-axis
    dx   = abs(old_pos[0] - new_pos[0])
    
    x_ran = np.array([ old_pos[0], new_pos[0] ])
    y_ran = np.array([ old_pos[1], new_pos[1] ])    
    
    if dx == 0:
        path = np.array([ x for x in range(min(y_ran), max(y_ran)) ])
        return path
    dy   = abs(old_pos[1] - new_pos[1])
    f = old_pos[1] + (new_pos[1] - old_pos[1]) * (X - old_pos[0]) / dx #The y values in between old and new
    F    = np.zeros([dpos,2])
    F[:,0] = X
    F[:,1] = f
    #F = np.reshape(F,(dpos,2))
    #distance between F and the points...
    #Create a matrix which stores coords of all points between the x_old x_new and y_old y_new


    
    XY_dims_X =len( range( min(x_ran), max(x_ran)+1 ))
    XY_dims_Y =len( range( min(y_ran), max(y_ran)+1 ))
    
    XY_dots = np.array([[ [x,y] for x in range( min(x_ran),max(x_ran)+1 ) ]  for y in range( min(y_ran),max(y_ran)+1 ) ])
    
    
    #print(XY_dots[0]) #for y=0
    #print(XY_dots[1]) #for y=1 etc
    path = []
    #if distance between XY_dots and F is less than or equal (sqrt2)/2 then F passes this pixel...
    XY_vector = np.reshape(XY_dots, ( XY_dims_X*XY_dims_Y , 2 ))
    
    
    ite = 0
    j=0
    prev_passed = 0
    skips = 0
    jump=0
    #print("NEW")
    while j < ((dx+1)*(dy+1)):
    #for j in range((dx+1) *(dy+1)):
        #print("In here now after for loop...")
        #print(j, "THE REAL J")
        j+=1
        #print(j-1, "The chosen j's")
        
        for i in range(len(F)):
            #print(i, "I equals...")
            ite+=1
            
            if norm(XY_vector[j-1]-F[i], 2) <= np.sqrt(2)/2:
                #print("PASSES THIS PIXEL REGION")
                path.append(XY_vector[j-1])
                prev_passed=1
                break #Prevents duplicates in the path list
            
            else: #WIP
                #print("did not pass...")
                if prev_passed==1: #Prevents redundant iterations over cells which are too far away from the interpolation pol.
                    prev_passed=0
                    #print("SKIP THIS COLUMN")
                    skips+=1
                    
        else:
            continue
    path = np.array(path)
    return path

def food_distance(food, POS):
    dist = np.array([ norm(food[i]-POS,2) for i in range(len(food)) ])
    
    food_choice = ( np.where(dist == np.amin(dist)) )[0][0]
    
    dx = POS[0]-food[food_choice][0]
    dy = POS[1]-food[food_choice][1]
    
    food_dist = np.sqrt(dx**2 + dy**2)
    
    #print(food[food_choice], "NEAREST FOOD!")
    if dx != 0:
        return np.arctan(dy/dx) , food_choice, food_dist
    else:
        return (np.pi/2) * np.sign(dy), food_choice, food_dist


def wall_distance(POS, X_BOT, Y_BOT): #Proximity to nearest wall
    if POS[0]< X_BOT/2: #Closer to left wall
        dx = POS[0]/X_BOT
    else:#Closer to right wall
        dx = np.abs(POS[0]-X_BOT) / X_BOT
    
    if POS[1]<Y_BOT/2:
        dy = POS[1]/Y_BOT
    else:
        dy = np.abs(POS[1]-Y_BOT) / Y_BOT
    
    return 2*np.array([dx, dy]) #current distance from walls. 0 or 1 values

def eat(food, path, nr): #Compares the distance between the closest food and your path
    food_t = np.array(food)
    #print(food[nr], "in here now")
    try:
        dist = np.array([ norm(food_t[nr]-path[i],2) for i in range(len(path)) ])
    except:
        dist = np.array([ norm(food_t-path[i],2) for i in range(len(path)) ])
    t = dist<20
    if any(t):
        #print("You eat food...")
        try:
            food.pop(nr)
            return 10, list(food) #You eat some food...
        except: #IF the food list is only size 1
            return 10, [] #You eat some food...
    else:
        return 0, list(food)


def get_init_values():
    POS = np.array([250,250])
    return POS, 500,500

def direction(angle, V):
    return np.array([ int( V*np.cos(angle* 2 * np.pi)+1), int( V*np.sin(angle* 2 * np.pi)+1)])


def new_pos(angle, POS, V):
    
    DIR = direction(angle, V)
    OLD_POS = POS
    POS = POS + DIR
    path  = pathfinder(OLD_POS, POS)
    return POS, path













