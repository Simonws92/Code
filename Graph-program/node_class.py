# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 22:43:57 2021

@author: Simon
"""


'''
Distributed parallel package
'''
#import torch.distributed.autograd as dist_autograd
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch import optim
#from torch.distributed.optim import DistributedOptimizer
#from torch.distributed.rpc import RRef





import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn.parameter import Parameter


import time
import numpy as np
import graph_network_templates as gt

import importlib

"creates a node of a sub/graph"
"The node is the final step of the sub_graph computation"

####################################
###### Network initialization ######
####################################
"Create custom network"
"Define wrapper for custom-made nodes?"
class create_custom(nn.Module):
    def __init__(self):
        pass
    
"Defining custom function must happen before calling 'create_custom'"
"Define custom operation using graphs"
"custom operation is then cast to pytorch which can use gpu acceleration"
"Edges must be defined manually"
"Custom database must be defined manually"
"Define custom forward operation"

" - adj matrix for forward operation"
" - adj matrix for custom operation"
" - custom inner-product operation"
" - activation functions"



"Creates base FC network"
class create_fc(nn.Module):
    def __init__(self, H):
        
        super(create_fc,self).__init__()
        
        "Initiate fully connected"
        self.H = H
        self.relu = nn.ReLU()
        #for i,(arg) in enumerate(argv):
        for i in range(len(H)-1):
            exec( 'self.linear{} = nn.Linear(H[{}], H[{}])'.format(i,i,i+1) )
        

    def forward(self,x, AE=0):
        x = x.view((-1 , self.H[0]))
        #############################
        ###### Fully-connected ######
        #############################
        for i in range(len(self.H)-1):
            x = eval( ' self.linear{}(x)'.format(i) )
            if i==len(self.H)-2:
                "does not perform activation on the final FC layer"
                pass
            else:
                x = self.relu(x)
        #x = x.view(-1,self.H_c2[0],self.output_size,self.output_size)
        
        "Following code is only for AE test"
        #x = x.view(-1, 200, 4,4)
        
        return x
        
"Creates base conv network"
class create_conv(nn.Module):
    "args = padding, stride, filter size, pool_mod" #[2,1,5,4]
    def __init__(self, H, identity, pool_type,*args):
        
        p=args[0]
        s=args[1]
        k=args[2]
        pool_mod=args[3]
        
        
        super(create_conv,self).__init__()
        
        self.H = H
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.pool_mod = pool_mod
        self.pool_type = pool_type
        self.nr_kernels = len(k)
        self.upsample = torch.nn.Upsample( scale_factor=2, mode='bilinear', align_corners = False)

        
        for i in range(len(H)-1):
            #print(i)
            #exec( 'self.conv{} = nn.Conv2d(H[{}],H[{}],k[{}], padding=p[{}], stride=s[{}])'.format(i,i,i+1,i,i,i) ) #For normal CNN only
            for j in range(self.nr_kernels):
                exec( 'self.conv{}_{} = nn.Conv2d(H[{}],H[{}],k[{}], padding=p[{}], stride=s)'.format(i,j,i,i+1,j,j) ) #For normal CNN only
                if identity == True:
                    "For some reason, Parameter flips the channels"
                    w = eval( 'torch.empty( H[{}],H[{}],k[{}],k[{}])'.format(i+1,i,j,j) )
                    #exec( 'self.conv{}_{}.parameters = nn.init.dirac_(w)'.format(i,j) )
                    with torch.no_grad():
                        exec( 'self.conv{}_{}.weight = Parameter( nn.init.dirac_(w) )'.format(i,j) )
                    #exec( 'self.conv{}_{}.bias = False'.format(i,j))
                
    
    def forward(self,x, **kwargs):
        pool_type = self.pool_type
        pool_mod = self.pool_mod
        #print( self.parameters )
        for i in range( len(self.H)-1 ):
            
            
            "Loop through all kernel sizes, aka highway-connection"
            if self.nr_kernels > 1:
                x_temp = 0
                for j in range(self.nr_kernels):
                    x_temp += eval( 'F.relu(self.conv{}_{}(x))'.format(i,j) ) #/ (self.nr_kernels)
                    #print(shape(x),shape(x_temp), self.nr_kernels)
                x = x_temp
            if self.nr_kernels == 1:
                x = eval( 'F.relu(self.conv{}_{}(x))'.format(i,0) ) #/ (self.nr_kernels)
            
            
            if pool_mod==0:
                pass
            else:
                if i%pool_mod==0:
                    if pool_type == 'max':
                        x = eval( 'self.pool(x)'.format(i) )
                    if pool_type == 'upsample':
                        x = eval( 'self.upsample(x)'.format(i) )
        
        return x

"Class for node creation"
"Nodes are computational graphs"
"Each node consists of one input and one output sub-node"

class network_variables():
    def __init__(self, LR):
        self.LR = LR
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
class graphs(nn.Module):
    
    def __init__(self , A_meta=0, name=''):
        super (graphs, self).__init__()
        try:
            self.network_settings = ntw.LR
        except:
            self.network_settings = network_variables(0.001)
        
        self.device = self.network_settings.device    
        self.LR = self.network_settings.LR  

        self.parent = ''
        self.node_nr = 0

        self.name = name
        self.save = 0
        self.node_count = 0
        self.fc_count   = 0
        self.conv_count = 0
        self.List_of_models = [ [] , [] ]                                      #contains information about the edges
        self.List_of_nodes  = []
        self.A = A_meta #new
        
        self.input_nodes = 1
        
        if type(A_meta) == int:
            self.A = np.zeros((1,1))

        "Define the adj operation wrt self.A = A"
        self.add_model = False
        graphs.adjacency_operation(self)
        
        self.add_model = True
        "The graph must now be sorted"
        graphs.sort_graph(self)
        
        
    
    def adjacency_operation(self, reset = True):
        "Define the adj operation wrt adj matrix A"
        "USAGE: graphs.adjacency_operation( class_obj )"
        major=0
        minor=0
        "This loop must be run for each sub-graph"
        for row in self.A: 
            inc_nodes = []
            out_nodes = []
            "Input for node 'row' "
            for col in row:
                if col!=0:
                    
                    "Define edges in here"
                    if reset == True:
                        graphs.define_edge(self, self.A[major,minor] , major,minor)
                    
                    "this piece creates the list_of_nodes connections"
                    out_nodes.append( minor )
                if self.A.T[major,minor]!=0:
                    inc_nodes.append( minor )
                    
                
                minor+=1
            self.List_of_nodes.append( [inc_nodes, out_nodes] )
            major+=1
            minor=0


    def insert_graph(self, A):
        "USAGE: graphs.insert_graph( class_obj, adj_matrix )"
        "insert a sub-graph inside another graph"
        "This adj matrix is solely for the sub-graph inserted. New adj meta must be defined after"
        exec( 'self.NODE_{} = nc.node(A)'.format(self.node_count) )
        self.node_count += 1
        graphs.adjacency_operation(self)
        
    
    "graph computation in here"
    def forward(self, X):
        "Initiates feedforward for graph and nodes"
        self.Sub_graph.forward(X)

        
    def call_opt(self, major, minor, LR = 0.003, opti = 'Adam'):
        "usage ex: graphs.call_opt( obj.Node, 1,0, LR = 0.001 )"
        #exec( 'self.optimizer_{}_{} = optim.Adam( self.edge_{}_{}.parameters() , lr=LR, weight_decay=0)'.format(major, minor, major, minor) )
        exec( 'self.optimizer_{}_{} = optim.{}( self.edge_{}_{}.parameters() , lr=LR, weight_decay=0)'.format(major, minor, opti, major, minor) )


    def define_operations(self, network_type ):
        "Defines the actual operations of edges between nodes"
        "Returns a class structure with properties used by pytorch"
        print("in here")
        "runs through network type list"
        _type    = network_type["type"]
        print(_type)
        
        if _type == ['fc']:
            
            H = network_type["H"]
            A        = network_type["A"]
            args     = network_type["args"]
            pool     = network_type["pool"]
            try:
                identity = network_type["id"]
            except:
                identity = False
                
            #exec( 'self.FC{} =  '.format(self.fc_count) ) #For normal CNN only
            self.fc_count += 1
            return create_fc(H), True
            
        if _type == ['conv']:
            
            H = network_type["H"]
            A        = network_type["A"]
            args     = network_type["args"]
            pool     = network_type["pool"]
            try:
                identity = network_type["id"]
            except:
                identity = False
            
            #exec( 'self.CONV{} = '.format(self.conv_count) ) #For normal CNN only
            self.conv_count += 1
            return create_conv(H, identity, pool, *args), True

        if _type == ['custom']:
            
            "custom operator must be loaded from its source code to a valid path etc"
            "or operator is loaded from valid source path..."
            
            "load custom source code from a valid path"
            path = network_type["path"]
            function = network_type["function"]
            
            mymodule = importlib.import_module(path)
            
            function_return = eval( 'mymodule.{}'.format(function) )
            
            "Must define if pytorch operations are available"
            is_pytorch_available = network_type["pytorch"] #True or False
            
            "calls main custom function to handle custom code executions"
            return function_return( network_type ), is_pytorch_available
            


    def define_edge(self , network_id, major,minor):
        "edge_{'to_node'}_{'from_node'}" #zero-based"
        exec( 'self.network_type = gt.network_templates( network_id )'.format(major,minor) )
        
        "Define optimizer for edges"
        
        exec( 'self.List_of_models[0].append( "{}_{}" ) '.format(major,minor) )
        
        #st_1 = time.time()
        if self.add_model == True:
            exec( 'self.edge_{}_{}, self.edge_{}_{}_pytorch = ( self.define_operations( self.network_type ) ) '.format( major, minor, major, minor ) )
            
            "graphs.call_opt is only used for valid pytorch class structures"
            if eval( 'self.edge_{}_{}_pytorch'.format( major , minor  ) ) == True:
                
                exec( 'graphs.call_opt( self, {}, {}, LR = self.LR)'.format(major, minor) )
            
            "For gpu placement"
            try:
                exec( '(self.edge_{}_{}.to(self.device) ) '.format( major , minor  ) )
            except:
                pass
            
            exec( 'self.List_of_models[1].append( self.edge_{}_{} ) '.format(major,minor) )
        
        
    def append_model(self, List_of_models, to_node, from_node):
        exec( 'self.edge_inc = "{}_{}"'.format( to_node , from_node ) )
        if self.edge_inc in List_of_models: #to prevent duplicates
            pass
        else:
            graphs.define_edge( self, self.A[ to_node , from_node ] , to_node, from_node )
            List_of_models.append( self.edge_inc )
        return List_of_models
    
    
    def add_connection(self, node_1, node_2, conn):
        "Add a connection between two nodes regardless of graph-level"
        
        try:
            self.List_of_nodes[node_1][1].append( node_2 )  #to
            self.List_of_nodes[node_2][0].append( node_1 )  #from
        except:
            print("ERROR: Node values must have a valid id")
            print("input: graph name, to, from, type of connection")
        "Now that we have added a connection, we must add the edge-operation to the graph"
        "Adj matrix must be expanded"
        
        len_A = len(self.A)
        l = self.List_of_nodes
        "If we have more nodes than the adj matrix have registered"
        if len_A < len(l):
            "we must expand the adj matrix"
            A = np.zeros((len_A+1, len_A+1))
            A[ :-1 , :-1 ] = self.A
            self.A = A
        self.A[ node_1 , node_2 ] = conn
        
        "FIX: must fix conv_count etc for totalgraphs"
        graphs.sort_graph(self)
        

    '''
    Currently not in use
    Add nodes manually
    '''
    def insert_node(self, new_node=0):
        "new_node = [ [out], [inc], [type out], [type inc], [recursive = True or False, type = 20]  ]"
        "usage ex: graphs.insert_node( obj.Node , new_node ) or graphs.insert_node( obj , new_node )"
        l = self.List_of_nodes
        m = self.List_of_models
        
        if new_node == 0: #We insert an empty node w/o edges in the parent graph
            self.List_of_nodes.append( [[],[]] )
            
            "Create new adj. matrix..."
            A = np.zeros(( np.shape(self.A)[0] + 1 , np.shape(self.A)[0] + 1 ))
            if len(l) == 1:
                pass
            else:
                A[:-1,:-1] = self.A
                self.A = A
            
        else: #insert node with edges
            out_nodes = new_node[0]
            inc_nodes = new_node[1]
            
            out = np.zeros((len(self.A)+1,1))
            inc = np.zeros((1,len(self.A)+1))
            
            "row based"
            for j,i in enumerate(inc_nodes):
                self.List_of_nodes[i][0].append( len(l) ) #the id of new node = len(l)
                inc[0,i] = new_node[3][j]
                
            "col based"
            for j,i in enumerate(out_nodes):
                self.List_of_nodes[i][-1].append( len(l) )
                out[i,0] = new_node[2][j]
                
            self.List_of_nodes.append( new_node[0:2] )
            
            "Create new adj. matrix..."
            A = np.zeros(( np.shape(self.A)[0] + 1 , np.shape(self.A)[0] + 1 ))
            A[:,-1] = out[:,0]
            A[-1,:] = inc[0,:]
            A[:-1,:-1] = self.A
            self.A = A
            
            if new_node[-1][0] == True:
                self.A[-1,-1] = new_node[-1][1]
            
        graphs.sort_graph( self )
        



    "TBC"
    def subtract_node_number(self, i, k, j):
        
        "First or second entry is subtracted"
        
        temp_int = int( self.List_of_models[0][i][k]  )
        temp_int -= 1
        
        fixed = int( self.List_of_models[0][i][j] )
        
        self.List_of_models[0][i] = '{}_{}'.format( fixed , temp_int )
        
        
    
        "Reassigns self.{}_{} to a lower node number"
        exec('self.edge_{}_{} = self.edge_{}_{}'.format( fixed , temp_int , fixed , temp_int+1 ))
        exec('del self.edge_{}_{}'.format( fixed , temp_int+1 ))
        
        exec('self.optimizer_{}_{} = self.optimizer_{}_{}'.format( fixed , temp_int , fixed , temp_int+1 ))
        exec('del self.optimizer_{}_{}'.format( fixed , temp_int+1 ))
        
    "TBC"
    "input is graph and node to be removed from said graph and name of node"
    "ex: graphs.delete_node(obj,3)"
    def delete_node(self, node_del):
        try:
            exec( 'del self.node_{}'.format(node_del))
            
            try:
                exec( 'del self.node_{}_bidirectional'.format(node_del))
            except:
                pass
            
            "Remove col-row at position node_del in adj matrix A"
            A = self.A
            A[node_del,:]   = 0
            A.T[node_del,:] = 0
            A = np.delete(A,node_del,0)
            A = np.delete(A,node_del,1)
            self.A = A
            
            "Resets the List_of_nodes for self wrt new adj-matrix"
            self.List_of_nodes = []
            graphs.adjacency_operation(self, reset = False)
            
                
            
            "Remove all node-related edges"
            "Subtracts model numbers by 1 with values greater than nodel_del"
            "This section should be optimized"
            temp_list = []
            temp_list_2 = []
            for i in range(len(self.List_of_models[0])):
                if str(node_del) in str(self.List_of_models[0][i]):
                    
                    target = self.List_of_models[0][i][0]
                    sender = self.List_of_models[0][i][-1]
                    exec( 'del obj.edge_{}_{}'.format( target , sender ) )
                    exec( 'del obj.optimizer_{}_{}'.format( target , sender ) )
                    
                else:
                    "All node numbers above the node_del must be subtracted by 1"
                    if node_del <  int( self.List_of_models[0][i][0] ):
                        k = 0  #position of major/target
                        j = -1 #position of minor/sender
                        graphs.subtract_node_number(self, i, k, j)
                        
#                        temp_int = int( self.List_of_models[0][i][0] )
#                        temp_int -= 1
#                        
#                        temp_int_2 = int( self.List_of_models[0][i][-1] )
#                        
#                        self.List_of_models[0][i] = '{}_{}'.format(temp_int, temp_int_2)
#                        
#                        "Reassigns self.{}_{} to a lower node number"
#                        exec('self.edge_{}_{} = self.edge_{}_{}'.format(temp_int,temp_int_2 , temp_int+1 , temp_int_2 ))
#                        exec('del self.edge_{}_{}'.format(temp_int+1,temp_int_2))
#                        
#                        exec('self.optimizer_{}_{} = self.optimizer_{}_{}'.format(temp_int,temp_int_2 , temp_int+1 , temp_int_2 ))
#                        exec('del self.optimizer_{}_{}'.format(temp_int+1,temp_int_2))
                        
                        
                    if node_del <  int( self.List_of_models[0][i][-1] ):
                        k = -1
                        j = 0
                        graphs.subtract_node_number(self, i, k, j)
                        
#                        temp_int = int( self.List_of_models[0][i][-1]  )
#                        temp_int -= 1
#                        
#                        temp_int_2 = int( self.List_of_models[0][i][0] )
#                        
#                        self.List_of_models[0][i] = '{}_{}'.format(temp_int_2, temp_int)
#                    
#                        "Reassigns self.{}_{} to a lower node number"
#                        exec('self.edge_{}_{} = self.edge_{}_{}'.format(temp_int_2,temp_int , temp_int_2 , temp_int+1 ))
#                        exec('del self.edge_{}_{}'.format(temp_int_2,temp_int+1))
#                        
#                        exec('self.optimizer_{}_{} = self.optimizer_{}_{}'.format(temp_int_2,temp_int , temp_int_2 , temp_int+1 ))
#                        exec('del self.optimizer_{}_{}'.format(temp_int_2,temp_int+1))
                    

                    
                    temp_list.append(self.List_of_models[0][i])
                    temp_list_2.append( self.List_of_models[1][i] )
                    
                    
                    
            self.List_of_models[0] = temp_list
            self.List_of_models[1] = temp_list_2
            
            
            "All .node_{} must be subtracted by 1"
            i=0
            stop = False
            while stop == False:
                if i > node_del:
                    try:
                        exec( 'temp = self.node_{}'.format(i) )     #Stores node value
                        exec( 'self.node_{} = temp'.format(i-1) )   #Re assigns node value
                        exec( 'del self.node_{}'.format(i) )        #Delete old node value
                        
                    except:
                        stop = True
                i+=1
            
        except:
            print("Node", node_del, "does not exist")
        
        nc.graphs.compile_graph(self)

    
    def delete_edge(self, target, sender):
        try:
            exec( 'del self.edge_{}_{}'.format(target,sender) )
            exec( 'del self.optimizer_{}_{}'.format(target,sender) )
            
            self.A[target,sender]=0
            
        except:
            print("Target/ sender node combination does not exist")
    
    def sort_graph(self):
        "usage ex: graphs.sort_graph( obj.Node )"
        l = self.List_of_nodes
        N = len(l)
        default = 0 #This is the default node
        min_len = 0
        for i in range(N): #goes through the list of nodes
            len_list = len( l[i][-1] )
            
            if len_list <= min_len:
                min_len = len_list
                default = i
        #print("First node is:", default, "with", min_len, "number of incoming connections and", len(l[default][0]), "number of outgoing connections...")
        "Breadth-first search or depth-first search?"
        List_of_models = []
        List_of_edges  = []
        
        max_it = N+1 #prevents infinite loops
        completed_nodes = [] #prevents nodes from being put in queue a second time
        
        queue = [default]
        while len(queue)!= 0:
            
            "Each node will create a queue of which the concurrent nodes will be placed in"
            "When the queue is empty, all nodes are complete"
            "The edge-node connections wrt the nodes in the queue will be computed"


            "next node in queue is the first node in j loop"
            current_node = queue.pop(0)
            "compute incoming"
            "Computes all incoming edge-node connections"
            if len( l[current_node][-1] ) != 0:
                for j_0 in range( len( l[current_node][-1] )):
                    
                    rec_node = current_node
                    inc_node = l[current_node][-1][j_0]
                    
                    List_of_models = graphs.append_model(self, List_of_models, rec_node , inc_node )

            "compute outgoing"
            for j_1 in range( len( l[current_node][0] )):
                
                rec_node = (l[current_node][0][j_1]) #receiving node
                List_of_models = graphs.append_model(self, List_of_models, rec_node , current_node )
                    
                if rec_node in completed_nodes:
                    pass
                else:
                    queue.append( rec_node )
                    completed_nodes.append( rec_node )
                
                "This will make sure any recursion nodes will be computed asap"
                if self.A[ rec_node , rec_node ] != 0:
                    List_of_models = graphs.append_model(self, List_of_models, rec_node, rec_node)
                
            "We must shuffle the queue in the correct order"
            for j_2 , nodes in enumerate(queue):
                outgoing_for_current = l[nodes][0]
                if len(outgoing_for_current) == 0: #We switch places in the queue
                    queue[-1], queue[j_2] = queue[j_2], queue[-1]

        self.List_of_models[0] = List_of_models
        #print(self.List_of_models[0])
        

    '''
    Compile graph will go through the whole graph and give each member their respective
    parent information etc which will be used when traversing the graph in forward
    '''
    def compile_graph(self):
        
        "This is a nonbinary-search-tree"
        "self is the obj aka total-graph"
        "compute first sub-graph"
        "TODO: Compile output dimensions for each node"
        temp_graph = self
        
        root_name = self.name
        self.root = self.name
        
        #temp_graph.node_output_0 = node_output_0
        current_node = "temp_graph" #The initial node of the sub-graph in string form
        name = self.name
        current_parent = ""
        
        node_nr = 0
        
        parent_nr = 0
        parent = [""]
        stop = False
        
        self.Computation_List = []
        #old_output_dim = current_node.H
        

        while stop == False:
            "depth node comes from which model exists in parent"
            "if node_# exists in some sub-graph, we go down one step to this node"
            dir_check = eval('{}'.format(current_node) )
            check = 'node_{}'.format(node_nr) in dir(dir_check)
            if check == True:
                
                parent.append(parent_nr)
                parent_nr = node_nr #We assign a parent_nr to the new node
                current_parent = current_node
                current_node += '.node_{}'.format(node_nr)
                
                
                print(current_node)
                
                "Checks if this node is bidirectional"
                set_1 = eval( 'set({}.List_of_nodes[{}][0])'.format(current_parent,node_nr) )
                set_2 = eval( 'set({}.List_of_nodes[{}][1])'.format(current_parent,node_nr) )
                if len( set_1.intersection(set_2)) != 0:
                    exec( '{}.node_{}_bidirectional = True'.format(current_parent,node_nr))
                    
                
                if eval(current_node).parent == '':
                    eval(current_node).parent = current_parent
                    
                if eval(current_node).node_nr == '':
                    eval(current_node).node_nr = node_nr
                    
                    eval(current_node).name = current_node
                
                    eval(current_node).compiled = True
                    
                    eval(current_node).root = root_name
                
                "If this sub-graph have no nodes, then we must compute its internal state"
                dir_check = eval('{}'.format(current_node) )
                #check = 'node_{}'.format(node_nr) in dir(dir_check)
                check = 'node_0' in dir(dir_check) #A sub-graph must have node_0 to be a sub-graph by def
                if check == False:
                    "We add the current_node to the computation_list"
                    self.Computation_List.append( current_node )
                    
                    "Now we must check the input/output dimensions for this node"
#                    input_dim = current_node.H
#                    if old_output_dim == input_dim:
#                        old_output_dim = input_dim
#                    else:
#                        raise Exception("Dimension error for node", current_node, " Expected dimensions", old_output_dim, "but received", input_dim)
#                    
                    
                    "We go up one level"
                    node_nr, parent_nr, parent, current_node, current_parent \
                        = graphs.traverse_up( node_nr, parent_nr, parent, current_node, current_parent )
                    
                else:
                    parent_nr = node_nr
                    node_nr = 0
                    
            else:
                "We add the current_node to the computation_list"
                self.Computation_List.append( current_node )
                
#                input_dim = current_node.H
#                if old_output_dim == input_dim:
#                    old_output_dim = input_dim
#                else:
#                    raise Exception("Dimension error for node", current_node, " Expected dimensions", old_output_dim, "but received", input_dim)
#                
                
                if parent[-1] != "":
                    "We go up one level"
                    node_nr, parent_nr, parent, current_node, current_parent \
                        = graphs.traverse_up( node_nr, parent_nr, parent, current_node, current_parent )
                else:
                    "We are alredy at the root parent"
                    pass
                    
                
            
            ###########################
            ###### End condition ######
            ###########################
            "Check condition for ending"
            if len(current_node) < len("temp_graph"):
                current_node = "temp_graph"
                
            print(parent)
            
            dir_check = eval('{}'.format(current_node) )
            check = 'node_{}'.format(node_nr) in dir(dir_check)
            if parent[-1]=='' and check==False:
                self.Computation_List.append( current_node )
                stop=True
            
        self = temp_graph
        self.compiled = True


    #########################
    ###### END COMPILE ######
    #########################
    

    def traverse_up( node_nr, parent_nr, parent, current_node, current_parent ):
        "This paragraph of code takes us up one level"
        len_r = len( '.node_{}'.format(node_nr) )
        current_node = current_node[:-len_r]
        
        if current_parent == 'temp_graph':
            current_parent = ''
        else:
            current_parent = current_parent[:-len_r]
    
        node_nr = parent_nr + 1    #this is the node we must check if it exists in the new parent
        parent_nr = parent.pop(-1) #when we go up again, we must remove the last parent, as one cannot be its own parent
        
        return node_nr, parent_nr, parent, current_node, current_parent
    
    "Computing all outgoing connections from a specific node in a sub-graph"
    def compute_outgoing(self, sender_node, current_s_internal ):
        
        #print()
        #print("NAME sub-graph:", self.name)
        
        models = self.List_of_models[0]
        target_nodes = []
        j=0
        
        "This for loop may be split into Phases for faster computation"
        for i in range( len( models )):
            if  sender_node == int(models[i][-1]):
                target_node = models[i][0]
                "These node output values must be deleted when the all internal states have been computed"


                "the edge_operations are updated as we perform feedforward"
                exec( '{}.optimizer_{}_{}.step()'.format('self', target_node, sender_node) )
                exec( '{}.edge_{}_{}.zero_grad()'.format('self', target_node, sender_node) )


                "Checks if reference node is bidir with its target node"
                "Reference node receive list"
                set_1 = self.List_of_nodes[sender_node][0]
                
                "Target node sender list"
                set_2 = self.List_of_nodes[int(target_node)][0]
                
                if sender_node in set_2 and int(target_node) in set_1:
                    #print(target_node,"Calculating bidirectional node...")
                    try:
                        exec( 'self.node_output_{}_temp += ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                    except:
                        exec( 'self.node_output_{}_temp  = ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                
                
                else:
                    "For directed nodes"
                    #print("Directional")
                    try:
                        exec( 'self.node_output_{} += ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                    except:
                        exec( 'self.node_output_{}  = ( self.edge_{}_{} )( current_s_internal )'.format( target_node , target_node, sender_node ))
                        
                
                    
                target_nodes.append(target_node)
                
                j+= 1
                
        "if j is zero, then this is the final output for this sub-graph"
        if j == 0:
            exec('self.node_output_{} = current_s_internal'.format(len(self.A)))
        
        #print(np.shape( eval('self.node_output_{}'.format( target_node ))))
        "the target nodes will be returned so we know which nodes to give which input to"
        "target_nodes = receiving nodes"
        return target_nodes
    
        
        
    
    "This code is a wrapper for the pseudo_timestep for bidir. graphs"
    def pseudo_timestep_primer( self , X , timestep=1 , start_node = '' , end_node = '', input_nodes = 1 ):
        
        self.input_nodes = input_nodes
        if input_nodes > 1:
            
            "Assigns initial values to input nodes"
            for i in range(input_nodes):
                exec('self.node_output_{} = X[i].detach()'.format(i) )        

            for t in range(timestep):
                X_out = graphs.pseudo_forward_TEST( self, X[0].detach() , start_node , end_node , input_nodes = input_nodes )
                X[0] = self.node_output_1
                
                
                "Each node has two values, one for output and one for input"
                "The input values will be added to the original output when the graph traversal is complete"
                for i in range(input_nodes):
                    "Not every node is bidirectional. Thus those nodes must be skipped wrt the nodes temp values"
                    try:
                        exec( 'self.node_output_{} += self.node_output_{}_temp'.format(i,i) )
                        exec( 'del self.node_output_{}_temp'.format(i) )
                    except:
                        pass
                
        else:
            "Previous output is the next input"
            "timestep=1 is standard graph traversal"
            for t in range(timestep):
                X_out = graphs.pseudo_forward_TEST( self, X.detach()    , start_node , end_node , input_nodes = input_nodes )
                X = self.node_output_1
        
        return X_out
    

    "This is the current pseudo_forward"
    "Must implement pseudo_timestep for bidirectional graphs - ok"
    "Must allow for multiple input nodes"
    " - obj.node_output_1 etc must be predefined"
    def pseudo_forward_TEST(self, OG_input, start_node = '', end_node='' , input_nodes = 1 ):
        
        if start_node == '':
            start_node = self.Computation_List[0]
        if end_node == '':
            end_node = self.Computation_List[-1]
            
            
        root_graph = "temp_graph"
        
        node_nr = 0
        current_parent = ""
        

        "The node_nr for our current node"
        node_nr = self.node_nr
        current_parent = self.parent
        current_node = self.name
        
        temp_graph = self
        
        parent_nr = 0
        init_ = 0
        
        temp_graph.input_to_node_0 = OG_input
        
        for j in range(len(self.Computation_List)):
            if start_node == self.Computation_List[j]:
                start = j
            if end_node == self.Computation_List[j]:
                end = j
        
        Computation_List = self.Computation_List[start:end+1]
        #print(Computation_List)
        
        
        for i,(current_node) in enumerate(Computation_List):
            "depth node comes from which model exists in parent"
            
            prev_parent = current_parent
            current_parent = eval(current_node).parent
            node_nr        = eval(current_node).node_nr
            
            #print()
            #print(i, current_node, "node_nr:", node_nr)
            
            "If this sub-graph have no nodes, then we must compute its internal state"
            dir_check = eval(current_node)
            check = 'node_0' in dir(dir_check) #a sub-graph must have node_0 to be a sub-graph by def
            
            #print("PARENT:",current_parent)
            
            if check == False:
                
                "Must give input to the internal state"
                if init_ == 0:
                    current_node_internal = node.forward( eval(current_node) , OG_input )
                    init_ = 1
                else:
                    
                    "Must give parent of current_node the correct input value"
                    if current_parent != prev_parent:
                        
                        parent_node_nr = eval(current_parent).node_nr
                        eval(current_parent).node_output_0 = eval('{}.node_output_{}'.format( prev_parent , parent_node_nr ))
                        
                        "Delete old node_output"
                        #exec( 'del {}.node_output_{}'.format(prev_parent, parent_node_nr) )
                    
                    "The input is at parent as parent.node_output, aka the edge connection"
                    current_node_internal = node.forward( eval(current_node) ,  eval( '{}.node_output_{}'.format(current_parent,node_nr) )  )
                        
                    "Delete old node_output"
                    #exec( 'del {}.node_output_{}'.format(current_parent, node_nr) )


            else:
                pass
                #print("IN here")
                "Must give the new child the current_node's output value as input value"
                #eval( Computation_List[i+1] ).parent
                #temp_parent.node_output_0 = current_node_internal
                #exec( 'eval( Computation_List[i+1].parent ).node_outout_0 = current_internal_node' )
            "When we've computed an internal state, we must compute its edge connections"
            "edge connections are stored at parent"
            "input for edge connections are stored at child"
            "node_nr is the current outgoing node aka sender_node"
            
            
            
            return_value = current_node_internal
            #print(current_node)
            if current_parent == '':
                pass
                #exec( 'del {}.node_output_{}'.format(current_node, node_nr) )
            
            else:
                #print("OUTGOING ",np.shape(current_node_internal), current_parent )
                target_nodes = graphs.compute_outgoing( eval(current_parent) , node_nr, current_node_internal )
                
                "Delete old node_output"
                #exec( 'del {}.node_output_{}'.format(current_parent, node_nr) )

                #exec( 'eval( Computation_List[i+1].parent ).node_outout_0 = current_internal_node' )
                #exec( 'del {}.node_output_{}'.format(current_parent, node_nr) )
                
        self = temp_graph
        return return_value
        
        

class node(graphs):
    def __init__(self, A , name = '', parent = ''  ):
        
        super(graphs,self).__init__()
        
        "Fix so we can load LR from global defined fcn"
        try:
            self.network_settings = ntw.LR
        except:
            self.network_settings = network_variables(0.001)
        #self.network_settings = network_variables(0.001)
        
        self.name = name #Stores the name of this node
        self.parent = parent
        self.node_nr = ''
        #node_nr = int(name)
        #self.parent = name[]
        
        self.save = 0                                                          #May be set to 1 to store certain output values
        self.device = self.network_settings.device    
        self.LR = self.network_settings.LR                                     #loads learning rate for opt
        self.A = A                                                             #the adj. matrix of which this graph is based
        if type(A) == int:
            self.A = np.zeros((1,1))
        self.List_of_models = [ [] , [] ]                                      #contains information about the edges
        self.List_of_nodes  = [  ]                                             #contains inc and out connections wrt notes
        
        "Counts the number of networks in this sub-graph"
        self.fc_count   = 0
        self.conv_count = 0
        
        "Placeholder value"
        self.loss = 0
        
        
        "Define the adj operation wrt self.A = A"
        self.add_model = False
        graphs.adjacency_operation(self)
        
        self.add_model = True
        "The graph must now be sorted"
        graphs.sort_graph(self)


    "This algorithm is executed once per new initiated model"
    "... and only if devices is True"
    "THIS ALG IS NOT COMPLETE"
    def determine_phases(self):
        j=0
        
        "List of phases which will be distributed"
        "phases require an initial model"
        phases = [[]]
        phases[0].append( self.List_of_models[0][0] )
        
        for i in range( len(self.List_of_models[0]) -1 ):
            "Distribute models in their respective phases"
            
            arg_1_1 = self.List_of_models[0][i][0]
            arg_1_2 = self.List_of_models[0][i][-1]
            
            arg_2_1 = self.List_of_models[0][i+1][0]
            arg_2_2 = self.List_of_models[0][i+1][-1]
            
            if arg_1_2 == arg_2_2 or arg_1_1 == arg_2_1:
                print(phases,j)
                print(phases[j])
                phases[j].append( self.List_of_models[0][i+1] )
                
            else:
                print(phases,j)
                print(phases[j])
                phases.append( [ self.List_of_models[0][i+1] ] )
                j+=1
        
        "EX with 3 phases: phases = [ ['0_1', '0_2, '0_3''] , ['2_1' , '3_1'], ['3_2'] ]"
        self.phases = phases
        print(self.phases)
        
        
        
    "This forward code works well for subgraphs with edges only"
    "This code is used to compute internal states"
    def forward(self, X_input):
        
        #print("Name:",self.name)
        
        node_output_0 = X_input
        
        "TODO: create pseudo-timestep for loops in nodes"
        "TODO: Create parallelization algorithm for this for-loop"
        "-     This loop will be split into PHASES"
        "use adj. matrix to determine phases, then split for-loop into phases"
        
        for i in range(len(self.List_of_models[0])):
            target_node  = self.List_of_models[0][i][0]
            sender_node  = self.List_of_models[0][i][-1]
            
            #print("forward:",target_node, sender_node)
            
            "the edge_operations are updated as we perform feedforward"
            "Not used for custom operators not using pytorch"
            if eval( 'self.edge_{}_{}_pytorch'.format( target_node , sender_node  ) ) == True:
                exec( '{}.optimizer_{}_{}.step()'.format('self', target_node, sender_node) )
                exec( '{}.edge_{}_{}.zero_grad()'.format('self', target_node, sender_node) )
            
            try:
                exec( 'node_output_{} += ( self.edge_{}_{} )( node_output_{} )'.format( target_node, target_node, sender_node , sender_node ) )
                
            except:
                exec( 'node_output_{} = ( self.edge_{}_{} )( node_output_{} )'.format( target_node, target_node, sender_node , sender_node ) )

            #print( eval('node_output_{}'.format(target_node)) )    

        nr_nodes = len(self.A)-1
        return eval('node_output_{}'.format(nr_nodes))
    
    
    
    
    
    def genetic_choices():
        
        "add characters"
        "add species (adj matrices)"
        
        
        "New species are based of old species wrt the adj matrix and its edges"
        "must identify which nodes are added/deleted when creating a new species"
        " - node deletion may be depend on the AI"
        " - allow options for both AI based and algorithmic decision for node deletion"
        
        "when new nodes are added, old nodes and edges must be kept"
        "assign id to each edge. This id is unique for all edges"
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    