# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:26:00 2020

@author: Simon
"""
import numpy as np

def LOAD_CHESS(): #Loads the pieces in correct positions
    #Placing black pieces
    #notations
    #first three determines type of piece
    #4th is black 0 or white 1 player
    #5th is position
    #6th is alive 0 or dead 1
    #7th is occupied 1 or free 0
    board = np.zeros((64,8))
    board[0] = (0,1,0,0,0,0,1,1) #rook
    board[1] = (0,1,1,0,1,0,1,1) #knight
    board[2] = (1,0,0,0,2,0,1,1) #bishop
    board[3] = (1,1,1,0,3,0,1,1) #king
    board[4] = (1,1,0,0,4,0,1,1) #queen
    board[7] = (0,1,0,0,7,0,1,1) #rook
    board[6] = (0,1,1,0,6,0,1,1) #knight
    board[5] = (1,0,0,0,5,0,1,1) #bishop
    for p in range(8,16):
        board[p] = (0,0,1,0,p,0,1,1) #pawns
 
    
    #Placing white pieces
    board[-1] = (0,1,0,1,63,0,1,1) #rook #The second rook
    board[-2] = (0,1,1,1,62,0,1,1) #knight
    board[-3] = (1,0,0,1,61,0,1,1) #bishop
    board[-5] = (1,1,1,1,59,0,1,1) #king
    board[-4] = (1,1,0,1,60,0,1,1) #queen
    board[-8] = (0,1,0,1,56,0,1,1) #rook #The first rook
    board[-7] = (0,1,1,1,57,0,1,1) #knight
    board[-6] = (1,0,0,1,58,0,1,1) #bishop
    for p in range(48,56):
        board[p] = (0,0,1,1,p,0,1,1) #pawns
    
    return board


def check_pos(check_loc, legal_moves, moves, piece_id, board, player, piece=''):
    #print("Checking pos", check_loc, piece_id)
    
    skip = 0
    if check_loc >= 64 or check_loc < 0:
        "illegal move. skipping"
        print(check_loc)
        check_loc = 63
        
    if moves >= 27:
        print(moves)
        moves = 26
    
    else:
        if board[ check_loc , -1] == 0: #Checks if immediate tile is free or occupied
            
            
            
            legal_moves[piece_id, moves ] = check_loc
            moves+=1
            
        elif board[ check_loc , -1] == 1 and board[ check_loc , 3] == np.mod( player+ 1, 2):
            
            if piece == 'pawn': #pawn cannot capture pieces in front of it
                return legal_moves, moves, 0
            
            else:
                #Capture enemy piece
                legal_moves[piece_id, moves] = check_loc
                moves+=1
                #print("Captured enemy piece at ", check_loc)
                return legal_moves, moves, 1
        else:
            return legal_moves, moves, 1
    
    return legal_moves, moves, 0



def PAWN(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves, moves=0):
    #pawns can only go one step forward or defeat another piece to it's forward left/right
    #These two rows of code translate our current position into X and Y axis positions
    current_row = int( np.floor(current_loc / 8)   )   #Zero based
    current_col = int( current_loc - 8*current_row )   #Zero based
    STOP = 0
    for i in range(1,2): #Checking topside
        
        if player == 0: #We go down
            check_loc = current_loc + 8 #(current_row-i)*8+(current_col)

        else: #We go up
            check_loc = current_loc - 8 #(current_row+i)*8+(current_col)
            
        #print("current loc: ", current_loc, "cheking loc:",check_loc, "player: ", player)
        if STOP!=1: # and current_col > col:
            legal_moves, moves, STOP = check_pos(check_loc, legal_moves, moves, piece_id, board, player, piece='pawn')

    #print(legal_moves)
    return board, moves, legal_moves

def ROOK(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves, moves=0):
    #The rook can go only updown and leftright
    move = 0
    captured = 0 #placeholder for capturing pieces,boolean value
    
    #These two rows of code translate our current position into X and Y axis positions
    current_row = int( np.floor(current_loc / 8)   )   #Zero based
    current_col = int( current_loc - 8*current_row )   #Zero based
    
    if piece == "king":
        steps = 2
    else:
        steps = 8
    
    #must check for available moves
    #Goes through all legal moves for rook...
    #First determine rows and col for current pos for rook
    #Check legal moves on the horizontal axis (left and right)
    #legal_moves = np.zeros((32,27))
    
    STOP_right = 0
    STOP_left = 0

    STOP_top = 0
    STOP_bot = 0
    for i in range(1,steps): #Checking topside
        
        check_right = (current_row)*8+(current_col) +i
        
        check_left = (current_row)*8+(current_col) -i
        
        
        if check_right >= 0 and check_right < 64 and STOP_right != 0: # and current_col > col:
            legal_moves, moves, STOP_right = check_pos(check_right, legal_moves, moves, piece_id, board, player)
    
        if check_left < 64 and check_left >= 0 and STOP_left != 1:
            legal_moves, moves, STOP_left = check_pos(check_left, legal_moves, moves, piece_id, board, player)

        
        check_top = (current_row +i)*8+(current_col)
        
        check_bot = (current_row -i)*8+(current_col) 
        
        if check_top >= 0 and check_top < 64 and STOP_top != 1: #failsafe
            legal_moves, moves, STOP_top = check_pos(check_top, legal_moves, moves, piece_id, board, player)
        
        if check_bot < 64 and check_bot >= 0 and STOP_bot != 1:
            legal_moves, moves, STOP_bot = check_pos(check_bot, legal_moves, moves, piece_id, board, player)
    #print(legal_moves)


    return board, moves, legal_moves


def BISHOP(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves, moves = 0):
    #The rook can go only updown and leftright
    move = 0
    captured = 0
    #Must first check if a friendly piece is currently sitting on the "next_loc"
    #Must also check if there are any pieces inbetween current and next location
    
    #These two rows of code translate our current position into X and Y axis positions
    current_row = int( np.floor(current_loc / 8)   )   #Zero based
    current_col = int( current_loc - 8*current_row )   #Zero based

    
    if piece == "king":
        steps = 1
    else:
        steps = 8 #Maximum number of steps
    

    STOP_UR = 0
    STOP_UL = 0
    
    STOP_DR = 0
    STOP_DL = 0
        
    for i in range(1,steps): #Checking topside

        ###UP RIGHT #Correct
        #to do next: add rows and cols to keep boundaries
        check_loc = (current_row-i)*8+(current_col) +i #Goes one step up and then one step right
        row = int( np.floor(check_loc / 8)   )   #Zero based
        col = int( check_loc - 8*row )   #Zero based
        if check_loc >= 0 and STOP_UR != 1 and row >= 0 and col < 8:
            legal_moves, moves, STOP_UR = check_pos(check_loc, legal_moves, moves, piece_id, board, player)
            
        ###UP LEFT #Correct
        check_loc = (current_row-i)*8+(current_col) -i
        row = int( np.floor(check_loc / 8)   )   #Zero based
        col = int( check_loc - 8*row )   #Zero based
        if check_loc >= 0 and STOP_UL != 1 and row >= 0 and col >= 0 and current_col > col:
            legal_moves, moves,  STOP_UL = check_pos(check_loc, legal_moves, moves, piece_id, board, player)
            
        
        ###Down right #Correct
        check_loc = (current_row+i)*8+(current_col) +i
        row = int( np.floor(check_loc / 8)   )   #Zero based
        col = int( check_loc - 8*row )   #Zero based
        if check_loc < 64 and STOP_DR != 1 and row < 8 and col < 8:
            legal_moves, moves, STOP_DR = check_pos(check_loc, legal_moves, moves, piece_id, board, player)
            
            
        ###Down Left #Correct
        check_loc = (current_row+i)*8+(current_col) -i
        row = int( np.floor(check_loc / 8)   )   #Zero based
        col = int( check_loc - 8*row )   #Zero based
        if check_loc >= 0 and check_loc < 64 and STOP_DL != 1 and row < 8 and col >= 0 and current_col > col:
            legal_moves, moves, STOP_DL = check_pos(check_loc, legal_moves, moves, piece_id, board, player)
            
        
    return board, moves, legal_moves



def KING_QUEEN(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves, moves=0):


    board, moves, legal_moves = ROOK(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves)
    board, moves, legal_moves = BISHOP(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves, moves=moves)

    
    return board, moves, legal_moves


def KNIGHT(player, piece, piece_id, current_loc, board, AV_pieces, legal_moves, moves = 0):
    
    
    #The rook can go only updown and leftright
    move = 0
    captured = 0 #placeholder for capturing pieces,boolean value
    
    #These two rows of code translate our current position into X and Y axis positions
    current_row = int( np.floor(current_loc / 8)   )   #Zero based
    current_col = int( current_loc - 8*current_row )   #Zero based
    
    #must check for available moves
    #Goes through all legal moves for rook...
    #First determine rows and col for current pos for rook
    #Check legal moves on the horizontal axis (left and right)
    #legal_moves = np.zeros((32,27))
    
    STOP_right = 0
    STOP_left = 0

    STOP_top = 0
    STOP_bot = 0
    
    mod_row = np.array([-2,-1,1,2])
    mod_col = np.array([1,2,2,1])
    
    switch = 1
    for i in range(8): #We can only pick from 8 mod 2 different places
        ###Down right #Correct
        #for j in range(2):
        
        if i>3:
            switch = -1
        
        check_loc = (current_row+ mod_row[i%4]*switch )*8 + (current_col + mod_col[i%4]*switch )
        
        row = int( np.floor(check_loc / 8)   )   #Zero based
        col = int( check_loc - 8*row )   #Zero based
        
        
        if check_loc >= 0 and check_loc <= 63:
            legal_moves, moves, STOP_DL = check_pos(check_loc, legal_moves, moves, piece_id, board, player)
    
    
    
    return board, moves, legal_moves
    
    











