"""
 A python module that hosts the code required to visualize the network. 
 The color of the node maps to a specific opinion whereas as
    the strength of the colour indicates the user's belief in that opinion
    The strength of the edge's grey color indicates the trust between two agents 
"""
from pylab import *
import networkx as nx
def visualize_network(G, layout, time, opinions):
    '''
    Returns figure of network.
    ''' 
    opnion_colours =[cm.Blues,cm.Reds,cm.Greens, cm.Oranges, cm.Greys, cm.Purples]
    opnions_dict = dict()
    for i in range(opinions):
        opnions_dict[f"node_{i}_list"] = []
        opnions_dict[f"nodes_preference_{i}_list"] = []
    trusts = list(nx.get_edge_attributes(G,'trust').values())


    for node in G.nodes():
        node_opinion = G.nodes()[node]["agent"].opinion
        node_preference = G.nodes()[node]["agent"].preference

        opnions_dict[f"node_{node_opinion}_list"].append(node)
        opnions_dict[f"nodes_preference_{node_opinion}_list"].append(node_preference)


    nx.draw_networkx_nodes(G, layout, node_size=5, node_color=opnions_dict[f"nodes_preference_{0}_list"], nodelist=opnions_dict[f"node_{0}_list"], cmap=opnion_colours[0])
    nx.draw_networkx_nodes(G, layout, node_size=5, node_color=opnions_dict[f"nodes_preference_{1}_list"], nodelist=opnions_dict[f"node_{1}_list"], cmap=opnion_colours[1])
    if opinions >= 3:
        nx.draw_networkx_nodes(G, layout, node_size=5, node_color=opnions_dict[f"nodes_preference_{2}_list"], nodelist=opnions_dict[f"node_{2}_list"], cmap=opnion_colours[2])
    if opinions >= 4:
        nx.draw_networkx_nodes(G, layout, node_size=5, node_color=opnions_dict[f"nodes_preference_{3}_list"], nodelist=opnions_dict[f"node_{3}_list"], cmap=opnion_colours[3])
    if opinions >= 5:
        nx.draw_networkx_nodes(G, layout, node_size=5, node_color=opnions_dict[f"nodes_preference_{4}_list"], nodelist=opnions_dict[f"node_{4}_list"], cmap=opnion_colours[4])
    if opinions >= 6:
        nx.draw_networkx_nodes(G, layout, node_size=5, node_color=opnions_dict[f"nodes_preference_{5}_list"], nodelist=opnions_dict[f"node_{5}_list"], cmap=opnion_colours[5])

    nx.draw_networkx_edges(G, layout, alpha=0.5, width=[(rep) for rep in trusts])
    axis('off')


    