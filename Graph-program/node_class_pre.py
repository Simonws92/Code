# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:59:22 2023

@author: Simon
"""

import torch

import numpy as np

import node_class as nc

"""
This code compiles traversal order for an object with subnodes
"""



def define_models():
    "We create a precompiled list of traversal"
    "This list may be made manually"
    members = ['obj']
    parents = ['']
    node_nr = [0]
    traversal_order = []
    st = ''
    sub_nodes = 3
    for i in range(sub_nodes):
        
        members += [ 'obj.node_{}'.format(i) ]
        st = 'node_{}'.format(i) + '.'
        members += [ 'obj.'+st+'node_{}'.format(j) for j in range(sub_nodes) ]
        
        "corresponding list of their parents"
        parents += ['obj']
        parents += ['obj.' + 'node_{}'.format(i) for j in range(sub_nodes) ]
        
        "Traversal order must be opposite"
        traversal_order += [ 'obj.'+st+'node_{}'.format(j) for j in range(sub_nodes) ]
        traversal_order += ['obj.node_{}'.format(i)]
        
        "We need to add node number"
        node_nr += [i]
        node_nr += [j for j in range(sub_nodes)]
        
    else:
        traversal_order += ['obj']
        
        
    "CAUTION: traversal_order must be sorted wrt respective nodes and subnodes adj matrix"
    "All outgoing node connections must be computed first"
    
    
    A_1 = np.array([ [0,0,0],[1,0,0],[1,1,0]  ])*40
    
    
    """
    done
    """
    "defines the object subnodes etc. Requires a list of nodes and its subnodes"
    "Then store them in a list"
    obj_list = []
    
    "Dictionary of class-node objects. dict of dict"
    obj_list_dict = {}
    
    for i in range(len(members)):
        
        name_obj = members[i]
        exec( '{} = nc.graphs(A_meta=A_1, name="{}")'.format( name_obj, name_obj ) )
        
        parent_value = parents[i]
        exec( '{}.parent = "{}"'.format( name_obj, parent_value ) )
        
        obj_list.append( [] )
        obj_list[-1] = eval('{}'.format(name_obj))
        
        obj_list[-1].node_id = node_nr[i]
        
        
        "obj_list must be a dict where we can locate entries based on keys"
        
        "dict with information regarding individual class-node objects e.g in/out dims"
        dim_in  = 1 #temp values
        dim_out = 0
        obj_dict = { 'dim_in_out': [dim_in, dim_out], 'model': eval(name_obj), 'id': node_nr[i] }
        
        
        obj_list_dict[ '{}'.format(name_obj) ] = obj_dict
        
    return obj_list, traversal_order, obj_list_dict


"""
done
"""
import matplotlib.pyplot as plt
class draw_class():
    def __init__(self):
        fig,ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.n = len(traversal_order)
        self.circles_created     = []
        self.connections_created = []
        self.level = 0
        
        
    def draw_tree(self, i, traversal_order, obj_list_dict, up_down ):
        "traversal_order: current node-object"
        "obj_list_dict: complete objects list"
        "up_down: -1 for going down or +1 for going up, 0 if same level"
        "creates a tree-like diagram based on class-node objects"
        
        "checks if current obj have parent"
        parent    = obj_list_dict[traversal_order[i]]["model"].parent
        parent_id = obj_list_dict[parent]["id"]
        child_id = obj_list_dict[traversal_order[i]]["id"]
        child = traversal_order[i]
        
        "Create a circle for the main parent"
        circle_info = (self.level, parent)
        if circle_info not in self.circles_created and parent.count(".")==0:
            "render the main_parent-circle"
            self.circles_created.append( circle_info )
            
            x_pos = 0
            parent_circle = plt.Circle(( x_pos  , self.level+1 ), 0.2, color='r' )
            self.ax.add_patch( parent_circle )
        
        "Standard"
        child_circle_info = (self.level, child)
        if child_circle_info not in self.circles_created:
            "render a child-circle"
            self.circles_created.append( child_circle_info )
            
            parent_pos = obj_list_dict[parent]["pos"]
            x_pos = parent_pos + ( child_id * 3**self.level )
            
            child_circle = plt.Circle(( x_pos , self.level ), 0.2, color='b' )
            self.ax.add_patch( child_circle )
            
        connection = (self.level, parent_id, child_id)
        
        if connection not in self.connections_created:

            "Parental graph-tree relationship"
            self.connections_created.append(connection)
            x_start, x_end = (x_pos,parent_pos)
            y_start, y_end = (self.level,self.level+1)
            
            X = x_start, x_end
            Y = y_start, y_end
            
            plt.plot(X,Y , linewidth = 2,color='b')

            
            "Second line plot: Computational relationship"
            neighbor_pos = obj_list_dict[traversal_order[i+1]]["pos"]
            x_start, x_end = (x_pos,neighbor_pos)
            y_start, y_end = (self.level,self.level)
            
            X = x_start, x_end
            Y = y_start, y_end
            
            plt.plot(X,Y , linewidth = 2,color='g')
            
        self.ax.set_ylim(ymax=3)
        self.ax.set_ylim(ymin=-1)
        self.ax.set_xlim(xmax=8.5)
        self.ax.set_xlim(xmin=-0.5)
        
        plt.show()
        plt.pause(1)


def traverse_graph_objects( inp, traversal_order, obj_list_dict ):
    
    n = len(traversal_order)
    current_parent = 0
    draw=True
    up_down = 0
    max_level = 0
    
    if draw == True:
        draw_object = draw_class()
    
    
    for i in range( n-1 ):
        current_count = traversal_order[i].count(".")
        older_count   = traversal_order[i-1].count(".")
        
        if current_count > max_level:
            max_level = current_count

        if current_count > older_count and i > 0:
            "We go down..."
            up_down = -1
        elif current_count == older_count and i > 0:
            up_down = 0
        elif current_count < older_count and i > 0:
            up_down = 1
        
        draw_object.level += up_down
        
        if current_count == 0:
            "top level"
            draw_object.level = max_level
        
        #temp_node will only be computed if its at baselevel
        temp_node   = obj_list_dict[traversal_order[i]]["model"]
        
        # Standard computation
        temp_parent = obj_list_dict[ temp_node.parent ]["model"]
        
        k = temp_node.node_id
        t,s = temp_parent.List_of_models[0][k]
        
        "When moving from one parent to the next, we need to delete old temp_output_{}"
        if current_parent != temp_parent.name:
            "Keeps track how many steps this parent made"
            current_parent = temp_parent.name
            parent_step = 0
        else:
            parent_step += 1
            try:
                exec( "del temp_output_{}".format(t))
            except:
                "nothing to delete"
                pass
        
        edge_op = eval("temp_parent.edge_{}_{}".format(t,s))
        
        #print("Edge_op", t,s)
        for param in edge_op.parameters():
            if len(np.shape(param))==1:
                "bias"
                pass
            else:
                #print(param.size())
                
                if i == 0:
                    out_dim = param.size()[0]
                
                "The output dimension of the previous model"
                out_dim_old = out_dim
                
                "The expected input dimension for this model"
                in_dim = param.size()[1]
                out_dim = param.size()[0]
                
                if out_dim_old != in_dim and i>0:
                    break
                
        
        if 'node_0' in dir(temp_node):
            temp_node.level = 'upper'
        else:
            "If the current node does not have any node members, base level"
            temp_node.level = 'base'

        if i<n:
            
            if temp_node.level == 'base':
                "Computes base-level internal state nodes"
                
                exec("temp_node.temp_output_{} = temp_node.forward( inp )".format(s))

                try:
                    exec("temp_parent.temp_output_{} += edge_op( temp_node.temp_output_{} )".format(t,s))

                    
                except:
                    exec("temp_parent.temp_output_{} = edge_op( temp_node.temp_output_{} )".format(t,s))

                    
            else:
                "Computes general edge-connections"
                try:
                    exec( "temp_parent.temp_output_{} += edge_op.forward( inp )".format(t))
                    
                except:
                    exec( "temp_parent.temp_output_{}  = edge_op.forward( inp )".format(t) )
                    
        else:
            pass
        
        inp = eval( "temp_parent.temp_output_{}".format(t) )
        output = inp
        
        
        if draw == True:
            draw_object.draw_tree( i, traversal_order , obj_list_dict, up_down )
        
    return output




"Early WIP please ignore"
def check_dim_compatibility( traversal_order, obj_list_dict ):
    out_dim = 0
    in_dim  = 0
    stop = False
    for i in range(len(traversal_order)):
        
        "Checks input/output dimensions for traversal order"
        "Checks if dimensions match"
        if stop == True:
            break
        
        model_name = traversal_order[i]
        
        model = obj_list_dict[ model_name ]["model"]
        
        print(model_name)
        
        
        for j,param in enumerate(model.parameters()):
            "checks internal-state dimensions"
            "This list is not sorted..."
            #print(param.shape)
            #print(param.size())
            
            #if len(param.shape)>1:
            #    "ignore bias"
            
            out_dim = in_dim
            in_dim  = param.size()[0]
            
            #print("must be equal:",out_dim, in_dim)
            
            if j == 0 and out_dim !=0 and len(param.shape)>1:
                
                "should ignore bias params"
                if out_dim != param.size():
                    stop = True
                else:
                    pass
            
        else:
            out_dim = param.size()




def define_positions(traversal_order, obj_list_dict):
    max_level = traversal_order[0].count(".")
    for i in range(len(traversal_order)):
        "We work backwards"
        level = max_level - traversal_order[-1-i].count(".")
        
        current_obj = traversal_order[-1-i]
        current_id = obj_list_dict[current_obj]["id"]
        parent = obj_list_dict[current_obj]["model"].parent
        
        
        if parent == '':
            parent_pos = 0
            obj_list_dict[current_obj]["pos"]=parent_pos
        else:
            
            parent_pos = obj_list_dict[parent]["pos"]
            
            x_pos = parent_pos + current_id * 3 ** level
            
            obj_list_dict[current_obj]["pos"] = x_pos
            
            print(current_obj, x_pos )
    
    
    return obj_list_dict



if __name__ == "__main__":


    x = torch.rand((5,2))
    
    "ok"
    obj_list, traversal_order, obj_list_dict = define_models()
    obj_list_dict = define_positions(traversal_order, obj_list_dict)

    "WIP: need to sort param wrt adj"
    #check_dim_compatibility( traversal_order, obj_list_dict )
    
    "ok"
    output = traverse_graph_objects(x, traversal_order, obj_list_dict )
    

    





