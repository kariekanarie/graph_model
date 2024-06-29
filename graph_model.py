# imports
from igraph import Graph
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import statistics
import powerlaw
import random
import scipy as sp
from tqdm import tqdm
from itertools import product
from collections import Counter
import pickle
import csv

# Helper functions

# generate a base_graph and add the roles to the nodes, r values represent the proportion of different roles
def generate_base_graph_fixed_fitness_roles(n0, w0, r0, r1, r2):
    # Create a directed graph with n0 nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n0))

    # Add edges between every pair of nodes with weight w0
    for source in range(n0):
        for target in range(n0):
            if source != target:  # Avoid self-loops
                G.add_edge(source, target, weight=w0)

    # Assign roles to each node based on the provided probabilities
    roles = []
    for _ in range(n0):
        role = random.choices([0, 1, 2], weights=[r0, r1, r2], k=1)[0]
        roles.append(role)
   
    nx.set_node_attributes(G, dict(enumerate(roles)), name="role")

    return G


# I fot this code from nx
def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        if not seq:
            print("empty seq")
            break  # Exit the loop if seq is empty
        x = rng.choice(seq)
        targets.add(x)
    return targets



# functions for tr_mat, to see which roles can or cannot get an incoming or outgoing edge during densification step
def find_vertical_zero_vectors(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0  
    zero_columns = []  
    for j in range(num_cols):
        if all(matrix[i][j] == 0 for i in range(num_rows)):
            zero_columns.append(j)  
    return zero_columns  

def find_horizontal_zero_vectors(matrix):
    num_rows = len(matrix)
    zero_rows = []
    for i in range(num_rows):
        if all(element == 0 for element in matrix[i]):
            zero_rows.append(i)  
    return zero_rows 

# got this code from github -> Bow-Tie
def get_bowtie_components(graph):
    '''Classifying the nodes of a network into a bow-tie structure.
    Here we follow the definition of a bow-tie as in: 
    "Bow-tie Decomposition in Directed Graphs" - Yang et al. IEEE (2011) 
    
    input:  NetworkX directed graph or numpy adjacency matrix
    output: sets of nodes in the specified partitions (following the 
            NetworkX input graph node labelling or labelled according to
            the order of the adjacency matrix [0, n-1])
    '''
    
    # Verify graph input format
    input_formats = [nx.DiGraph, np.ndarray, np.matrix]
    assert type(graph) in input_formats, 'Input should be a NetworkX directed graph or numpy adjacency matrix'
    if type(graph) == nx.classes.digraph.DiGraph:
        G = graph.copy()
    if (type(graph) == np.ndarray) | (type(graph) == np.matrix):
        G = nx.from_numpy_matrix(np.matrix(graph), create_using=nx.DiGraph())
    
    GT = nx.reverse(G, copy=True)
    
    strongly_con_comp = list(nx.strongly_connected_components(G))    
    strongly_con_comp = max(strongly_con_comp, key=len)

    S = strongly_con_comp

    v_any = list(S)[0]
    DFS_G = set(nx.dfs_tree(G,v_any).nodes())
    DFS_GT = set(nx.dfs_tree(GT,v_any).nodes())
    OUT = DFS_G - S
    IN = DFS_GT - S
    V_rest = set(G.nodes()) - S - OUT - IN

    TUBES = set()
    INTENDRILS = set()
    OUTTENDRILS = set()
    OTHER = set()
    for v in V_rest:
        irv = len(IN & set(nx.dfs_tree(GT,v).nodes())) != 0
        vro = len(OUT & set(nx.dfs_tree(G,v).nodes())) != 0
        if irv and vro:
            TUBES.add(v)
        elif irv and not vro:
            INTENDRILS.add(v)
        elif not irv and vro:
            OUTTENDRILS.add(v)
        elif not irv and not vro:
            OTHER.add(v)
            
    return S, IN, OUT, TUBES, INTENDRILS, OUTTENDRILS, OTHER

# the model 
def model_f_roles(G, p, w0, num_iter, dens_param_in, dens_param_out, pr_mat, pr_f, new_node_m, x):
    """
    G = the graph, nx object
    p = proba of generating an edge
    w0 = weight on edges
    num_iter = number of iterations the model will run
    dense_param_in, dense_param_out = the number of incoming and outgoing edges that will be added in the densification step
    tr_mat = transition matrix for probability of edges, based on node-type
    pr_f = probability function of which node-type will be generated {0: ..., 1:..., 2:...}
    new_node_m = the number of incoming and outgoing edges for each role at birth
    x is the number of edges that will get a weight increase

    """
    # keep a list of the edges (in/out) per node 
    rep_in_c0 = [n for n, d in G.in_degree() if G.nodes[n]["role"] == 0 for _ in range(d)]
    rep_out_c0 = [n for n, d in G.out_degree() if G.nodes[n]["role"] == 0 for _ in range(d)]
    rep_in_c1 = [n for n, d in G.in_degree() if G.nodes[n]["role"] == 1 for _ in range(d)]
    rep_out_c1 = [n for n, d in G.out_degree() if G.nodes[n]["role"] == 1 for _ in range(d)]
    rep_in_c2 = [n for n, d in G.in_degree() if G.nodes[n]["role"] == 2 for _ in range(d)]
    rep_out_c2 = [n for n, d in G.out_degree() if G.nodes[n]["role"] == 2 for _ in range(d)]

    # keep a list of the edges in the graph
    edge_array = list(G.edges())

    # Concat them:
    rep_in = [rep_in_c0, rep_in_c1, rep_in_c2]
    rep_out = [rep_out_c0, rep_out_c1, rep_out_c2]

    for _ in tqdm(range(num_iter)):
        # pick random number for proba
        random_number = random.random()
        source = len(G) # source node index
        
        # with probability p, network growth
        if random_number <= p: 
            
            # choose a random role for the new node using the probability function
            role = random.choices(list(pr_f.keys()), weights=list(pr_f.values()))[0] 
            
            for _ in range(new_node_m[role]["m_in"]):
                # Calculate the probability of outgoing edges for the current node's role
                out_role_p = pr_mat[:, role]
                out_role_p_c = out_role_p.copy()

                # Normalize the probabilities
                out_role_p_c /= np.sum(out_role_p_c)
                available_roles = list(range(len(out_role_p_c)))

                # Loop until a valid role for outgoing edges is found
                while True:
                    # Choose a role based on the normalized probabilities
                    out_role = np.random.choice(available_roles, p=out_role_p_c[available_roles])
                    # Check if the chosen role has nodes available for outgoing edges
                    if rep_out[out_role]:
                        break
                    # If not, remove the chosen role from the available roles and check for more options
                    available_roles.remove(out_role)
                    # If no available roles remain or all probabilities are zero, exit the loop
                    if not available_roles or all(el == 0 for el in out_role_p_c[available_roles]):
                        break
                    # Normalize the probabilities again based on available roles
                    out_role_p_c[available_roles] /= np.sum(out_role_p_c[available_roles])

                # If there are nodes available for outgoing edges for the chosen role
                if rep_out[out_role]:
                    # Select a random node from the available nodes for outgoing edges
                    in_target = _random_subset(rep_out[out_role], 1, np.random.default_rng(seed=1))
                    # Add an edge from the selected node to the current node with the specified weight
                    if in_target:
                        G.add_edge(list(in_target)[0], source, weight=w0)
                        # Update the array of repeated outgoing nodes for the chosen role
                        rep_out[out_role].extend(in_target)


            for _ in range(new_node_m[role]["m_out"]):
                # Calculate the probability of incoming edges for the current node's role
                in_role_p = pr_mat[role, :]
                in_role_p_c = in_role_p.copy()

                # Normalize the probabilities
                in_role_p_c /= np.sum(in_role_p_c)
                available_roles = list(range(len(in_role_p_c)))

                # Loop until a valid role for incoming edges is found
                while True:
                    # Choose a role based on the normalized probabilities
                    in_role = np.random.choice(available_roles, p=in_role_p_c[available_roles])
                    # Check if the chosen role has nodes available for incoming edges
                    if rep_in[in_role]:
                        break
                    # If not, remove the chosen role from the available roles and check for more options
                    available_roles.remove(in_role)
                    # If no available roles remain or all probabilities are zero, exit the loop
                    if not available_roles or all(el == 0 for el in in_role_p_c[available_roles]):
                        break
                    # Normalize the probabilities again based on available roles
                    in_role_p_c[available_roles] /= np.sum(in_role_p_c[available_roles])

                # If there are nodes available for incoming edges for the chosen role
                if rep_in[in_role]:
                    # Select a random node from the available nodes for incoming edges
                    out_target = _random_subset(rep_in[in_role], 1, np.random.default_rng(seed=1))
                    # Add an edge from the current node to the selected node with the specified weight
                    if out_target:
                        G.add_edge(source, list(out_target)[0], weight=w0)
                        # Update the array of repeated incoming nodes for the chosen role
                        rep_in[in_role].extend(out_target)


            if G.has_node(source): # if the node is created, update the fitness parameters
                G.nodes[source]['role'] = role
                

                # update the array for the newly created node role   
                if new_node_m[role]["m_in"] != 0: # check whether m_in was eq to 0 for the new node, if not, add to repeated in nodes
                    rep_in[role].extend([source] * new_node_m[role]["m_in"])
                if new_node_m[role]["m_out"] != 0: # check whether m_out was eq to 0 for the new node, if not, add to repeated out nodes
                    rep_out[role].extend([source] * new_node_m[role]["m_out"])
        
 
        # densification
        else: 
            # print('densification')
            # Pick random nodes for increasing an edge weight or adding an edge 
            possible_in_targets = [0, 1, 2] 
            possible_out_sources = [0, 1, 2]

            zero_column_indices = find_vertical_zero_vectors(pr_mat)  # find the columns that contain only zeroes, these should not get an incoming edge
            zero_row_indices = find_horizontal_zero_vectors(pr_mat)  # find the rows that contain only zeroes, these should not get an outgoing edge

            possible_out_sources = [source for source in possible_out_sources if source not in zero_row_indices]  # update, remove the rows that contain only zeroes
            possible_in_targets = [target for target in possible_in_targets if target not in zero_column_indices]  # update, remove the columns that contain only zeroes

            # Filter nodes based on their roles
            in_target_nodes = [node for node, attr in G.nodes(data=True) if attr['role'] in possible_in_targets]
            out_source_nodes = [node for node, attr in G.nodes(data=True) if attr['role'] in possible_out_sources]

            # Adjust dens_param_in and dens_param_out if they are larger than the number of eligible nodes
            if dens_param_in > len(in_target_nodes):
                dens_param_in = len(in_target_nodes)
            
            if dens_param_out > len(out_source_nodes):
                dens_param_out = len(out_source_nodes)

            # Sample nodes for densification
            in_targets = random.sample(in_target_nodes, dens_param_in)  # only allowed to take a role 1 or 2, because 0 cannot get incoming edges
            out_sources = random.sample(out_source_nodes, dens_param_out)  # only take role 0 or 2
            for in_t in in_targets:  
                # the role of the chosen node (random)
                role = G.nodes[in_t]['role'] 
                # find the role of node it should connect to
                out_role_p = pr_mat[:, role] 
                
                out_role_p_c = out_role_p.copy() # copy otherwise things mess up
                out_role_p_c /= np.sum(out_role_p_c) # normalize 
                
                in_s_role = np.random.choice(len(out_role_p), p=out_role_p_c) # choose the role that will connect to in_t

                available_roles = list(range(len(out_role_p_c)))
                    
                # take an out-role set, but not an empty one
                while not rep_out[in_s_role]:
                    # Exclude the chosen role from the options
                    available_roles.remove(in_s_role)
                    if not available_roles or all(el == 0 for el in out_role_p_c[available_roles]) :
                        break  # Exit the loop if no roles are available                        
               
                    out_role_p_c[available_roles] /= np.sum(out_role_p_c[available_roles])
                    in_s_role = np.random.choice(available_roles, p=out_role_p_c[available_roles])
                
                # check again, if it doesn't exist, exit    
                if not rep_out[in_s_role]:
                    break              

                # pick the in source node (a node which will have the outgoing edges)
                in_s = _random_subset(rep_out[in_s_role], 1, np.random.default_rng(seed=1))
                
                # check whether we are creating a self-loop or not, if we do, we pick another random in_source node (with the same role)
                # maybe need to add a check to see whether the sets have at least 2 elems if we pick the same role, otherwise we can still generate a self-loop?
                while next(iter(in_s)) == in_t:
                    nodes_with_role = [node for node, data in G.nodes(data=True) if data['role'] == role]
                    in_t = random.choice(nodes_with_role)

                if in_s:
                    source = list(in_s)[0]
                    if G.has_edge(source, in_t):
                        # If the edge already exists, quit
                        break
                    else:
                        # If the edge doesn't exist, add it with the specified weight
                        G.add_edge(source, in_t, weight=w0) 
                        # print("edge added from: ", G.nodes[source]['role'], ' to ', G.nodes[in_t]['role'] )                   
                        # update the arrays
                        rep_out[G.nodes[source]['role']].extend([source])
                        rep_in[role].extend([in_t])
                            
            for out_s in out_sources:
                # the role of the chosen node (random)
                role = G.nodes[out_s]['role'] 

                # find the role of node it should connect to
                in_role_p = pr_mat[role, :]
                 
                in_role_p_c = in_role_p.copy()
                in_role_p_c /= np.sum(in_role_p_c) # normalize (maybe not necessary)
                out_t_role = np.random.choice(len(in_role_p), p=in_role_p_c) # choose the role that will connect to out_s

                available_roles = list(range(len(out_role_p_c)))
                    
                # take an out-role set, but not an empty one
                while not rep_in[out_t_role]:
                    # print("excl")    
                    # Exclude the chosen role from the options
                    available_roles.remove(out_t_role)
                    if not available_roles or all(el == 0 for el in in_role_p_c[available_roles]) :
                        break  # Exit the loop if no roles are available                        
               
                    in_role_p_c[available_roles] /= np.sum(in_role_p_c[available_roles])
                    out_t_role = np.random.choice(available_roles, p=in_role_p_c[available_roles])
                    
                
                if not rep_in[out_t_role]:
                    break   

                # pick the in source node (a node which will have the outgoing edges)
                out_t = _random_subset(rep_in[out_t_role], 1, np.random.default_rng(seed=1))

                # check whether we are creating a self-loop or not, if we do, we pick another random in_source node (with the same role)
                # maybe need to add a check to see whether the sets have at least 2 elems if we pick the same role, otherwise we can still generate a self-loop!!!
                while out_t == out_s:
                    nodes_with_role = [node for node, data in G.nodes(data=True) if data['role'] == role]
                    out_s = random.choice(nodes_with_role)

                if out_t:
                    target = list(out_t)[0]
                    if G.has_edge(out_s, target):
                        # If the edge already exists, quit
                        break
                    else:
                        # If the edge doesn't exist, add it with the specified weight
                        G.add_edge(out_s, target, weight=w0)
                        # update the arrays
                        rep_out[out_t_role].extend([target])
                        rep_in[role].extend([out_s])


        # Adding weight randomly to get exponential
        rng = np.random.default_rng(seed=42)
        edges = list(G.edges())

        selected_edges = rng.choice(len(edges), size=x, replace=True)
        for idx in selected_edges:
            u, v = edges[idx]
            G[u][v]['weight'] += 1
        # Adding weight preferentially to get power-law
    #     print("kaasje")
    #     selected_edges = _random_subset_tuple(edge_array, x, np.random.default_rng(seed=1))
    #     edge_array.extend(selected_edges)
    # edge_counts = Counter(edge_array)
    # for edge, count in edge_counts.items():
    #     u, v = edge
    #     if G.has_edge(u, v):
    #         if 'weight' in G[u][v]:
    #             G[u][v]['weight'] += count
    #         else:
    #             G[u][v]['weight'] = count
           
        
    return(G)