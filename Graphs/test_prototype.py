# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:43:50 2024

@author: Simon
"""


"Need to create an algorithm which traverses a graph"
import numpy as np
from collections import defaultdict
import time


def compute_traverse_order(A, input_node, output_node):
    graph = defaultdict(list)
    
    # dict of outgoing edge-connections
    edge_connections = defaultdict(list)
    
    # computed connections dict
    computed_connections = {}
    
    # dict of incoming connections
    incoming_connections = {}
    
    
    for i in range(len(A)):
        incoming_connections[i] = []
        computed_connections[i] = []
        
        
        for j in range(len(A)):
            #edge_connections[j] = []
            if A[i,j]!=0:
                "j sends to i"
                graph[j].append(i)
                incoming_connections[i].append(j)
                edge_connections[j].append((i,j))
    
    outgoing_connections = {}
    for i in range(len(A)):
        outgoing_connections[i] = []
        
        for j in range(len(A)):
            if A[j,i] != 0:
                outgoing_connections[i].append(j)
    #print(incoming_connections)
    #print(outgoing_connections)
    
    
    "dict of outgoing edge-connections"
    # edge_connections = {}
    # for sender in graph:
        
    #     edge_connections[sender] = []
        
    #     for target in graph[sender]:
    #         edge_connections[sender].append( (target, sender) )
            
    
    #print(edge_connections)
        


    # Omit connections which have been computed. Empty list = exhausted output node
    # List of outgoing connections
    exhaust_list = edge_connections.copy()
    exhaust_list[output_node] = []
    
    
    #print("Exhaust list to be reduced", exhaust_list)


    current_node = input_node
    initial_node = input_node
    queue = []
    exhausted_nodes = []
    traversal_order = []
    old_traversal_order = []
    
    
    BACKTRACKED = False
    
    
    nr_edges, nr_nodes = np.sum(A), len(A)
    O_nv = 2*( int(nr_edges) + int(nr_nodes) )
    tol = 100
    O_nv += tol
    Iterations = 0
    #print(O_nv - Iterations )
    
    "eigenvalues may be too expensive to compute for larger matrices n>100"
    Eig = np.linalg.linalg.eig(A)
    
    st = time.time()
    
    #while True:
    while Iterations <= O_nv:
        
        if len(traversal_order) == np.sum(A):
            #print("Finished")
            break
        
        #print("\n", Iterations ,"current NODE:", current_node)
        #print("order:",traversal_order)
        #print("queue:",queue)
        #print("exhaust list:", exhaust_list[current_node])
        


        
        
        ###### BACKTRACK ######
        if initial_node != current_node:
            "We ignore this check for the initial node"
            
            
            for sender in incoming_connections[current_node]:
                
                
                "We go through all possible outgoing connections from a non-exhausted node"
                if sender not in exhausted_nodes:
                    current_node = sender
                    #print(sender , " is unfinished! We backtrack")
                    BACKTRACKED = True
                    
                    ### Added in patch 1.5.3 ###
                    # When backtracking from current_node, we must add its outgoing nodes to the queue
                    
                    "We cannot add nodes to queues if they already exist"
                    "if a node already exists in queue. Skip and compute next_node in queue"
                    #print("CHECKING ALL OUTGOING CONNECTIONS FROM", current_node)
                    
                    
                    new_node_queue =  [ outgoing_node for outgoing_node in outgoing_connections[current_node] ]
                    #queue = queue + new_node_queue
                    #break
                    
                    intersection  = [x for x in new_node_queue if x in queue]
                    non_intersect = [x for x in new_node_queue if x not in queue]
                    
                    ### BETA 1.6.3 ###
                    
                    
                    #print(Eig[0])
                    if np.sum(Eig[0]) == 0:
                        #print("Directed")
                        ### BACKUP 1.5.3 ###
                        #queue = queue + new_node_queue ### default backup
                        
                        "We dont add nodes to the queue which already exists in the queue"
                        queue = queue + non_intersect ### added in 1.6.3
                        
                    else:
                        
                        ### BETA 1.6.3 ###
                        #print("Looping")
                        
                        
                        #We need a method of detecting loops
                        ### Start beta
                        intersection = [] #default
                        #if intersection == []: #### beta. Remove if tests are not passed
                        if non_intersect != []:
                            "Continue as usual"
                            #queue = queue + new_node_queue # default 1.5.3
                            queue = queue + non_intersect # beta 1.6.3
                            
                        else:
                            "This else may encounter false-positives of loops"
                            current_node = queue.pop()
                            #print("Detected loop? New node:", current_node)
                            
                            "Now we want to compute all outgoing from new current_node"
                            for sender in outgoing_connections[current_node]:
                                connection = (sender, current_node)
                                if connection not in traversal_order:
                                    traversal_order.append( connection)
                                    exhaust_list[current_node].remove(connection)
                        ### End beta
                    
                    break
                    
                
                else:
                    BACKTRACKED = False
                    "All sender connections have been computed"
                    pass
            
            ### Added in patch 1.5.3 ###
            if incoming_connections[current_node] == [] and current_node != output_node:
                "We add connections from nodes without incoming connections"
                
                "We want to continue backtracking empty-input branches"
                "EX: current_node have incoming nodes from non-input nodes, we want to add them to the queue"
                
                
                for sender in outgoing_connections[current_node]:
                    
                    
                    connection = (sender, current_node)
                    if connection not in traversal_order:
                        traversal_order.append( connection)
                        exhaust_list[current_node].remove(connection)
                
        
        
        
        """
        We may occur a BACKTRACKED == True 
            for nodes with incomplete outgoing connections
        ex: two input nodes
        
        Thus if no changes have been made: We skip to the next node in queue
        
        
        if a node is NOT input_node and has no incoming connections,
            algorithm gets stuck
        """
        
        
        
        """
        We cannot allow backtracked nodes to compute their edges before
            all backtracked nodes are computed
        """
        if current_node != output_node and BACKTRACKED == False:
            for connection in (edge_connections[current_node]):
                
                
                
                """
                We cannot compute outputs from a node if the node still have
                    unfinished incoming connections
                """
                
                target = connection[0]
                target_exh = target in exhausted_nodes
                sender_exh = current_node in exhausted_nodes
                
                if connection in computed_connections[target] or (target_exh and sender_exh) or target == initial_node:
                    "Doesnt seem to do anything"
                    pass
                else:
                    if target not in queue:
                        queue.append(target)
                    computed_connections[target].append(connection)
                    
                    """
                    Before we can compute forward of node X. all incoming nodes of X
                    must have been computed
                    """
                    
                    """
                    We need to make sure that all incoming connections of
                    connection[-1] are finished
                    """
                    if connection in traversal_order:
                        pass
                    else:
                        traversal_order.append( connection )
                        
                exhaust_list[current_node].remove(connection)
        
        
        "We classify exhausted_node when all of its outgoing and incoming connections are computed"
        if exhaust_list[current_node] == []:
            "Move to next node"
            exhausted_nodes.append(current_node)
            
            
            #print("END",queue)
            if queue == []:
                "We go here when the queue is empty. Assumed all other nodes have been finished"
                break
            else:
                "However, other branches directly connected to output_node may be incomplete"
                current_node = queue.pop(0)
        
        ### We clean computed nodes from the queue ###
        temp_keep = []
        for node in queue:
            if exhaust_list[node] != []:
                temp_keep.append(node)
        queue = temp_keep
        
        #print("\n",queue)
        
        Iterations += 1
        
    et = time.time() - st
    
    
    print("Finished in:", Iterations, "iterations...")
    return traversal_order  #, et


"perform BFS with input_node as initial"
def BFS(graph, s, len_A):
    """
    graph: list of connections
    s: initial_node
    """
    
    # Mark all the vertices as not visited
    visited = [False for i in range(len_A)]
 
    # Create a queue for BFS
    queue = []
    traverse_order = [s]
    # Mark the source node as
    # visited and enqueue it
    queue.append(s)
    visited[s] = True
 
    while queue:
        s = queue.pop(0) # removes the first member of the queue which is the initial node        
        for i in graph[s]: # we loop through all connections from this node
            if visited[i] == False: # if we did not already visit this node
                queue.append(i)     # we add this node to the queue
                visited[i] = True   # then we mark this node as visited
                #print(i)
                #traverse_order.append((i,s))
    return visited, traverse_order


def run_test(key, DICT_OF_TESTS):
    print("\n",key)
    A = DICT_OF_TESTS[key]["A"]
    input_node = DICT_OF_TESTS[key]["input_node"]
    output_node = DICT_OF_TESTS[key]["output_node"]
    CORRECT_traversal_order = DICT_OF_TESTS[key]["tr_order"]
    alt_traversal_order = DICT_OF_TESTS[key]["alt"]
    
    
    
    traversal_order = compute_traverse_order(A, input_node, output_node)
         
    #print("Traversal order:",traversal_order)
    
    if traversal_order == CORRECT_traversal_order or traversal_order == alt_traversal_order :
        print("PASSED")
    else:
        print("Fault")
        

if __name__ == '__main__':
    
    ##################
    ### TEST CASES ###
    ##################
    
    alt_traversal_order = []
    
    DICT_OF_TESTS = {}
    

    "ok"
    A = np.array([[0 ,0 ,0 ,0],
                  [1 ,0 ,0 ,0],
                  [0 ,1 ,0 ,1],
                  [1 ,0 ,0 ,0]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0), (3, 0), (2, 1), (2, 3)]
    
    test_1 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_1"] = test_1
    

    
    "ok"
    A = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 1., 1., 0., 0., 0.],
                  [0., 1., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 1., 1., 0.]])
    input_node = 0
    output_node = 7
    CORRECT_traversal_order = [(1, 0), (3, 0), (2, 0), (4, 0), (6, 1), (5, 3), (6, 2), (5, 4), (7, 6), (7, 5)]
    
    test_2 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_2"] = test_2
    
    
    
    "ok"
    A = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 1., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 1., 1., 0.]])
    input_node = 0
    output_node = 7
    CORRECT_traversal_order = [(1, 0), (3, 0), (2, 0), (4, 0), (5, 1), (6, 3), (5, 2), (6, 4), (7, 5), (7, 6)]
    
    test_3 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_3"] = test_3
    
    
    "ok with indent at line 133. multiple outputs"
    A = np.array([[0., 0., 1., 1., 0., 0.],
                  [0., 0., 1., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.]])
    input_node = 5
    output_node = 0
    CORRECT_traversal_order = [(2, 5), (3, 5), (2, 4), (3, 4), (0, 3), (1, 3), (0, 2), (1, 2)]
    
    test_4 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_4"] = test_4
    
    
    "ok. multiple inputs"
    A = np.array([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [1., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 1., 1., 0.]])
    input_node = 0
    output_node = 4
    CORRECT_traversal_order = [(2, 0), (2, 1), (4, 2), (4, 3)]
    
    test_5 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_5"] = test_5
    
    
    
    "ok. multiple inputs"
    A = np.array([[0., 0., 0., 0.],
                  [1., 0., 0., 1.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., 0.]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0), (1, 3), (2, 1)]
    
    test_6 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_6"] = test_6
    
    
    
    "ok"
    "Continued empty-branch. 'HORN'-type graph "
    A = np.array([[0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.],
                  [0., 1., 0., 1., 0.],
                  [0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0.]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0), (2, 1), (3, 4), (2, 3)]
    
    test_7 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_7"] = test_7
    
    
    "ok"
    "semi-horn"
    A = np.array([[0., 0., 0., 0., 0.],
                  [1., 0., 0., 1., 0.],
                  [0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0.]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0), (3, 4), (1, 3), (2, 1)]
    
    test_8 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_8"] = test_8
    
    
    "NOT PASSED WITH 1.6.3"
    "ok. triple input, one output"
    "not O_nv"
    "O_nv + 5 okay"
    A = np.array([[0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 1., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0), (4, 5), (4, 6), (3, 4), (1, 3), (2, 1)]
    
    test_9 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_9"] = test_9
    
    
    
    "ok"
    A = np.array([[0., 0., 0., 0.],
                  [1., 0., 0., 1.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., 0.]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0), (1, 3), (2, 1)]
    
    test_10 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": [] }
    
    DICT_OF_TESTS["test_10"] = test_10
    
    
    
    # Loop. Not working with 1.5.3
    "Passed with 1.6.3"
    A = np.array([[0., 0., 0., 0.,0],
                  [1., 0., 0., 1.,0],
                  [0., 1., 0., 0.,0],
                  [0., 0., 0., 0.,1],
                  [0., 1., 0., 0.,0]])
    input_node = 0
    output_node = 2
    CORRECT_traversal_order = [(1, 0),(3,4), (1, 3),(4,1), (2, 1)]
    alt_traversal_order = [(1, 0),(3,4), (1, 3),(2,1), (4, 1)]
    
    test_11 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": alt_traversal_order }
    
    DICT_OF_TESTS["test_11"] = test_11


    "Passed with 1.6.3, inefficient"
    A = np.array( [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]] )
    
    input_node  = 0
    output_node = 3
    CORRECT_traversal_order = [ (1,0), (2,1),(4,1), (5,4),(14,5),(6,5),(7,6),(12,7),(8,7),(9,8),
                               (10,9), (11,10),(13,12), (14,13),(15,14),(2,15),(3,2) ]
    
    alt = [(1, 0), (2, 1), (4, 1), (5, 4), (6, 5), (14, 5), (7, 6), (8, 7), (12, 7),(9, 8),
           (10, 9), (11, 10), (12, 11), (13, 12), (14, 13), (15, 14), (2, 15), (3, 2)]
    
    test_12 = {"A":A, "input_node":input_node, "output_node": output_node,
              "tr_order": CORRECT_traversal_order, "alt": alt }
    
    DICT_OF_TESTS["test_12"] = test_12
    
    # with open('adj_matrix_test.npy', 'rb') as f:
    #     A = np.load(f)
    

    
    specific = ""
    

    
    if specific == "":
        for key in DICT_OF_TESTS:
            
            run_test(key, DICT_OF_TESTS)
    else:
        run_test(specific, DICT_OF_TESTS)
    
    
    