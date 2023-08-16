# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:15:50 2019

@author: Simon
"""

import numpy as np
import time
import pygame
import os

import pyautogui
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import math
import random

import brain as br
import chess_rules as chess
#import pygame
#import tkinter as tk
#from tkinter import messagebox

start = time.time()

TRAINING = True
NR_CH   = 5
MUTATION_COEFF = 0.01

Input = 32 #X,Y,food angle & dist, wall x, wall y, time_remain
Output = 32 #Choose a piece and choose where to move
H = [Input, 20,Output] #1 input, 2 hidden, 1 output
#B=1

restart = False

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
    

def PIECE_LIST():
    #name, id, status ( 1 = alive, 0 = dead), current_location
    List_black = [["p",0,1,8],["p",1,1,9],["p",2,1,10],["p",3,1,11],["p",4,1,12],["p",5,1,13],["p",6,1,14],["p",7,1,15],
                  ["r",8,1,0],["r",9,1,7],["kn",10,1,1],["kn",11,1,6],["b",12,1,2],["b",13,1,5],["q",14,1,4],["k",15,1,3]]
    
    
    List_white = [["p",16,1,48],["p",17,1,49],["p",18,1,50],["p",19,1,51],["p",20,1,52],["p",21,1,53],["p",22,1,54],["p",23,1,55],
                  ["r",24,1,56],["r",25,1,63],["kn",26,1,57],["kn",27,1,62],["b",28,1,58],["b",29,1,61],["q",30,1,59],["k",31,1,60]]
    
    return List_black, List_white




def compile_moves(player, AV_pieces, board, groups):
    """
    Compiles a list with all legal moves for every piece.
    
    groups list of boolean 0, 1: [pawn , rook ,knight ,bishop , queen, king]
    """
    legal_moves = np.zeros((32,27))
    for piece_slot in range(16):
        current_loc = AV_pieces[player][piece_slot][-1]
        piece_id = piece_slot + 16*(player) #If white player turns. The piece id will increment by 15.
        
        if (piece_id <= 7 or  (piece_id <= 23 and piece_id >= 16)) and groups[0]==1:
            #print(piece_id, current_loc, "id and loc")
            piece = "pawn"
            board, moves, legal_moves = chess.PAWN(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)
        
        if (piece_id == 8 or piece_id == 9 or piece_id == 24 or piece_id == 25) and groups[1]==1:
            piece = "rook"
            board, moves, legal_moves = chess.ROOK(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)
        
        if (piece_id == 10 or piece_id == 11 or piece_id == 26 or piece_id == 27) and groups[2]==1:
            #print("Knight")
            piece = "knight"
            board, moves, legal_moves = chess.KNIGHT(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)
        
        if (piece_id == 12 or piece_id == 13 or piece_id == 28 or piece_id == 29) and groups[3]==1:
            #print("Bishop")
            piece = "bishop"
            board, moves, legal_moves = chess.BISHOP(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)
        
        if (piece_id == 14 or piece_id == 30) and groups[4]==1:
            piece = "queen"
            board, moves, legal_moves = chess.KING_QUEEN(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)    
            
        if (piece_id == 15 or piece_id == 31) and groups[5]==1:
            piece = "king"
            board, moves, legal_moves = chess.KING_QUEEN(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)  
    
    return board, legal_moves#, piece


def Player_move(player_turn, legal_moves, AV_pieces, board, groups,
                H, W, B, dp, fitness ):
    
    _, decision = br.feedforward(legal_moves, H, W, B, dp)
    "Here we multiply elementwise decision with binary legal_moves to get the full legal decision list"
    legal_moves_binary = legal_moves.copy()
    legal_moves_binary[legal_moves_binary > 0] = 1
    
    decision *= legal_moves_binary
    #print(decision)
    "Now choose piece based on legal decision"
    old_max = 0
    current_max = 0
    for i in range(16): #Goes through all 32 pieces to see which one is the best decision...
        current_max = sum(decision[i+16*player_turn])
        if current_max > old_max:
            piece_choice = i+16*player_turn #which piece we choose
            old_max = current_max
    
    
    "here we choose the most likely move post legal-move check"
    index_nr = list( decision[piece_choice] ).index( max( decision[piece_choice] ) )
    next_loc = int(legal_moves[piece_choice,index_nr])
    
    
    "Play the selected piece to desired location"
    
    "Takes piece placement information"
    "Since the list have 16 slots, we must convert from 32 to 16 etc"
    current_loc = AV_pieces[player_turn][piece_choice - player_turn*16][-1]
    piece = AV_pieces[player_turn][piece_choice - player_turn*16][0]
    
    "Updates piece placement information"
    AV_pieces[player_turn][piece_choice - player_turn*16][-1] = next_loc 
    
    if board[next_loc][-1] == 1:
        #print("Piece captured!")
        fitness+=1
    
    next_row = int( np.floor(next_loc / 8)   )      #Zero based
    next_col = int( next_loc - 8*next_row )      #Zero based
    
    #print(player_turn, piece_choice, piece," to:", next_row, next_col, next_loc)
    #print("from:", current_loc, "to", next_loc)
    
    board[next_loc] = (0,0,0,0,0,0,0,0)
    board[next_loc] = board[current_loc].copy()
    #board[next_loc, 4] = next_loc
    board[current_loc] = (0,0,0,0,0,0,0,0) #leaves a blank tile when leaving
    
    board_gp = board[:,:3]
    board_gp = np.reshape(board_gp, (8,8,3))
    
    plt.clf()
    plt.imshow(board_gp)
    plt.pause(0.01)
    
    
    return board, AV_pieces, fitness

def play_game(H, W, B, W2, B2, s):
    points = 0
    fitness = 0
    
    dp = 8
    #0,1,2: piece
    #3: piece_id
    #4: b/w
    #5: position
    #6: dead/alive
    #7: occupied
    #8: id
    
    # pawn, rook, knight, bishop, queen, king
    #groups = [1,1,0,0,0,0] # Binary list of allowed piece-groups
    groups = [1,1,1,1,1,1]
    
    board = chess.LOAD_CHESS()          #Loads the starting board
    black_p, white_p = PIECE_LIST()     #Loads the starting piece_lists
    AV_pieces = [ black_p, white_p ]
    
    captured = 0
    
    old_max = -1
    "choice is the id of the piece. Every piece have their own id"
    choice = 0 
    choice_move = 0
    
    #print()
    #print("Start")
    for I in range(1,71): #Each character gets 100 turns each
        "Player 1 (Bottom) starts. Player 0 is second"
        #print("TOTAL ITERATION", I)
        
        
        ###
        ### NEW
        ###
        
        "Checks if player have any legal moves"
        player_turn = I%2
        #print("PLAYER:" ,player_turn)
        board, legal_moves = compile_moves(player_turn, AV_pieces, board, groups)
        
        
        if np.average(legal_moves) == 0:
            #print("We cannot move anywhere! We have lost!")
            #print("Rounds:", I, "Fitness:", fitness)
            return fitness
        else:
            board, AV_pieces, fitness = Player_move(player_turn, legal_moves, AV_pieces, board, groups, H, W, B, dp, fitness )
            
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

    for i in range(int(NR_CH / 2)):
        #print(i+1 ," of " , NR_CH)
        fitness = play_game( H, CH_W[2*i], CH_B[2*i], CH_W[1+i*2], CH_B[1+i*2], i )
        Fitness_list.append(fitness)
        
#        print(fitness, "Fitness...")
#        print("Play is: ", play)
#        print("Eat is: ", eat, " ... For pair nr: ", i)
    Fitness_list = np.array(Fitness_list)
    second, index = get_second_highest(Fitness_list)
    
    parents = ( np.where(Fitness_list == np.amax(Fitness_list)) )[0]
    #print(parents[0])
    #print(parents, "the parents")
    #print(second, index, "Check this...")

    #The pairs that received the highest fitness will breed together
    if TRAINING == True:
        if len(parents) == 1:
            children, children_B = br.breed( CH_W[parents[0]], CH_W[parents[0]+1], CH_B[parents[0]], CH_B[parents[0]+1], NR_CH, MUTATION_COEFF, H )
        else:
            children, children_B = br.breed( CH_W[parents[0]], CH_W[parents[1]], CH_B[parents[0]], CH_B[parents[1]], NR_CH, MUTATION_COEFF, H )
    
    '''
    if TRAINING == True:
        if len(parents) == 1:
            children, children_B = br.breed( CH_W[parents[0]], CH_W[index], CH_B[parents[0]], CH_B[index], NR_CH, MUTATION_COEFF, H )
        
        else:
            children, children_B = br.breed( CH_W[parents[0]], CH_W[parents[1]], CH_B[parents[0]], CH_B[parents[1]], NR_CH, MUTATION_COEFF, H )
    else:
        pass
    '''
    
    return children, children_B, Fitness_list #, play, eat



GENERATIONS = 1

if restart == True:
    fit = []
    bt_max = []
    hs_max = []

for i in range(GENERATIONS):
    new_weights, new_bias, Fitness_list = start_game(NR_CH, H, CH_W, CH_B, MUTATION_COEFF, TRAINING)
    CH_W = new_weights
    CH_B = new_bias
    #print( "Max fitness after generation", i+1, ": " ,  max(Fitness_list), " at index: ", list(Fitness_list).index(max(Fitness_list)))

    
    fit.append( max(Fitness_list) )

plt.figure()
plt.plot(fit)


end = time.time()
print(end - start)
#
#
#
#pygame.display.quit()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
W = CH_W[41]
B = CH_B[41]
Z1=[]
Z2=[]
for i in range(1000):
    X = np.array([i]) / 10
    z,A = br.feedforward( X , H, W, B,1)
    Z1.append(z[0])
    Z2.append(z[1])

plot(Z1)
figure()
plot(Z2)
    
Z = np.array(Z)
Z = np.reshape(Z, (100,100))

ax = plt.axes(projection="3d")

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
'''



    
    
    










