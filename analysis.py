"""
 A python module that contains functions that computes the various output statistics
"""

import networkx as nx
import community as com
from community.community_louvain import best_partition
from collections import Counter
import numpy as np
from operator import itemgetter
import copy
import matplotlib.pyplot as plt

def compute_preferences(G):
    '''
    Computes mean preference if number of opinions == 2
    '''
    agent_preferences = [G.nodes()[agent]["agent"].preference for agent in G.nodes()]
    return np.mean(agent_preferences)

def compute_opinions(G, opinions):
    '''
    Computes mean opinion if number of opinions == 2
    '''
    if(opinions <=2):
        agent_opinions = [G.nodes()[agent]["agent"].opinion for agent in G.nodes()]
        return np.mean(agent_opinions)
    else:
        return 'opinions>2'

def compute_transitivity(G):
    '''
    Computes transitivity
    '''  
    return nx.transitivity(G)

def compute_majority_opinions(G, num_agents):
    '''
    Computes majority opinion after model run
    '''    
    agent_opinions = [G.nodes()[agent]["agent"].opinion for agent in G.nodes()]
    agent_opinions = Counter(agent_opinions)
    opinion_sizes = [agent_opinions[key] for key in agent_opinions.keys()]

    difference = max(opinion_sizes)/num_agents  

    return difference


def compute_echo_chamber(G, echo_limit):
    '''
    Computes sizes and number of echo chambers
    '''

    global echo_chamber_size, echo_chamber_n, cliques

    # hides edges based on threshold (echo_limit)
    hidden = hide_edges(G, echo_limit)

    # calculate cliques
    cliques = list(nx.enumerate_all_cliques(hidden))

    # select cliques where size >= 3
    large_cliques = [[G.nodes()[node]["agent"].opinion for node in clique] for clique in cliques if len(clique)>2]
    echo_chambers = [echo for echo in large_cliques if len(set(echo)) == 1]

    if len(echo_chambers)>0:
        echo_chamber_size = Counter([len(chamber) for chamber in echo_chambers])
        echo_chamber_n = len(echo_chambers)/len(cliques)
    
    return len(large_cliques)



def hide_edges(input, echo_limit):
    '''
    hides edges based on threshold (echo_limit)
    '''
    G_copy = copy.deepcopy(input)
    edges = []
    for edge in G_copy.edges():
        if G_copy.edges[edge]["trust"] < echo_limit:
            edges.append(edge)
    G_copy.remove_edges_from(edges)
    return G_copy


def compute_radical_opinions(G, num_agents):
    '''
    Computes percentage of agents holding radical opinions (preference > 0.8)
    '''
    radical_counter = 0
    for a in G.nodes():
        if (G.nodes()[a]["agent"].preference > 0.8): radical_counter += 1
    return radical_counter/num_agents

def get_communities(G):
    '''
    Computes communities using community louvain's best partition
    '''
    return best_partition(G, weight='trust')

def community_no(G):
    '''
    Computes largest community
    '''
    community_partitions = get_communities(G)
    return max(community_partitions.values())+1

def compute_silent_spiral(G):
    '''
    Computes 5% least connected (trust) agents. Returns the percentage of least connected agents with majority opinion 
    '''
    opinions_list = []
    node_information = []

    # select all nodes and calculate their connectivity
    for node in G.nodes():
        neighbors =  G.neighbors(node)
        node_opinion = G.nodes[node]['agent'].opinion
        opinions_list.append(node_opinion)
        connectivity = 0        
        for neighbor_node in neighbors:
            connectivity += G.edges[node,neighbor_node]['trust']

        node_information.append([node, node_opinion, connectivity])

    # Sort based on connectivity
    node_information.sort(key=itemgetter(2))

    # calculate majority opinion
    opinion_count = Counter(opinions_list)
    opinion_freq = [opinion_count[i] for i in opinion_count]
    majority_opinion = opinion_freq.index(max(opinion_freq))

    # calculate percentage of silent spiral nodes having majority opinion
    silent_spiral_nodes = [1 if node[1] == majority_opinion else 0 for node in node_information[:int(len(node_information)*.05)]]
    groupsize = len(silent_spiral_nodes)
    perc_of_major = sum(silent_spiral_nodes)/groupsize

    return perc_of_major
    
def average_trust(G):
    '''
    Computes average trust (edge strength)
    '''    
    return np.mean(list(nx.get_edge_attributes(G,'trust').values()))
