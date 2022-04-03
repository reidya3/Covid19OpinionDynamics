"""
Create a new Opinion forming model with the given parameters.
Anthony Reidy 2022
"""

import pycxsimulator
from pylab import *
import networkx as nx
import copy as cp
import random 
from heapq import nlargest
import numpy as np
from analysis import *
from visualization import visualize_network
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--num_agents", default=1000, help="Number of agents in network.")
parser.add_option("-l","--n_neighbors",default=4, help="Number of neighbors for each node")
parser.add_option("-t","--network_type",default=2, help="1 for small-world, 2 for scale free")
parser.add_option("-b", "--beta_component", default=0.3, help='if network type is small world (1); this is the beta-component')
parser.add_option("-s", "--similarity_treshold", default= 0.025, help='Range in which similarity holds')
parser.add_option("-i", "--social_influence", default= 0.01 , help="The influence of neighboring agents on the forming of a new preference.")
parser.add_option("-w", "--swingers", default=4, type=int, help='Number of agents which switches opinion, preference, and trust with each timestep.')
parser.add_option("-v", "--malicious", default=0, help=" Number of malicious agents ")
parser.add_option("-e", "--echo_limit", default="0.7", help="Limit for edge strength (weight) for echo chamber calculation.")
parser.add_option("-m", "--all_majority", default=False, help=" If true: all agents except malicious agents have the same opinion")
parser.add_option("-o", "--opinions", default=2, help="Number of opinions")

(opts, args) = parser.parse_args()
def str_to_bool(s: str):
    status = {"True": True,
                "False": False}
    
    return status[s]
"""
The following STATIC hyperparameters are layed out in the foollowing manner:
    <Variable name> = <Variable Value> # <Brief Description>  Range(<RANGE_START>, <RANGE_END>, <RANGE_STEP>)
"""
num_agents = int(opts.num_agents) # Number of agents in network. Range(2, 1000, 1) 
no_of_neighbors = int(opts.n_neighbors) # Number of neighbors for each node. Range(2, 6, 1) 
network_type = int(opts.network_type) # Network type to use.  Range(1, 2, 1) {1:small word, 2: preferential attachment model}
beta_component = float(opts.beta_component) # if network type is small world (1); this is the beta-component. Range(0, 1, 0.3)
similarity_treshold = float(opts.similarity_treshold) # Range in which similarity holds (difference in preference if opinion is shared). Range(0, 0.1, 0.01)
social_influence = float(opts.social_influence) # The influence of neighboring agents on the forming of a new preference. Range(0, 0.1 0.01)
swingers = int(opts.swingers) #  Number of agents which switches opinion, preference, and trust with each timestep. Range(0,100, 2)
malicious_N = int(opts.malicious)  # Number of malicious agents Range(0, 20, 1)
echo_limit = float(opts.echo_limit) #Limit for edge strength (weight) for echo chamber calculation.  Range(0, 0.5, 0.01)
all_majority = str_to_bool(str(opts.all_majority)) # If true: all agents except malicious agents have the same opinion.
opinions = int(opts.opinions)  # Number of opinions, default 2 Range(2,10,1)


class agent():
    '''
    An agent that talks; select his neighbors based on similarity and trust, and updates opinion and preference accordingly.
    '''

    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.opinion = random.randint(0,opinions-1)
        self.preference = set_rand_unifrom_preference()

    def step(self):
        '''
        A model step. Agent talks to other agents
        ''' 
        self.talk()

    def talk(self):
        '''
        Agent sees neighbors. Chooses neighbors to talk with.
        Start talk with seleced neighbors and form opinion.
        '''
        global G
        neighbors =  G.neighbors(self.unique_id)
        selected_neighbors = self.choose_neighbors(neighbors)
        self.form_opinion(selected_neighbors)

    def choose_neighbors(self, neighbors):
        '''
        Choose neighbors to talk with. 
        If opinion and preference is similar: selection based on similarity.
        Else selection on trust.
        If neighbors are similar only talk with these. Else talk based on trust.
        '''
        selected_neighbors = []
        similar_neighbors = []
        for neighbor in neighbors:
            trust = G.edges[self.unique_id,neighbor]['trust']
            neighbor_agent_object = G.nodes()[neighbor]["agent"]
            if((neighbor_agent_object.opinion == self.opinion) and (abs(neighbor_agent_object.preference - self.preference) < get_rand_similarity(similarity_treshold))):
                similar_neighbors.append(neighbor_agent_object)
                self.update_trust(neighbor)
            else:
                if(trust > np.random.uniform(0,1)):
                    selected_neighbors.append(neighbor_agent_object)
                    self.update_trust(neighbor)

        if(len(similar_neighbors) != 0):
            return similar_neighbors
        else:
            return selected_neighbors

    def form_opinion(self, neighbors):

        '''
        An agent forming opinion. Calculate the probability to change    opinion. 
        If probability is high enough select new opinion.
        '''        
        global opinions
        # Get neighbor preferences
        neighbor_preferences = [neighbor.preference for neighbor in neighbors]

        # Initialize array containing preferences for each opinion
        neighbor_opinion_preferences = [[] for i in range(opinions)]

        # Initialize array containing probabilities
        probability_rates = []
        

        # Append all neighbors preferences based on opinion
        for neighbor in neighbors:
            for opinion in range(opinions):
                if neighbor.opinion == opinion:
                    neighbor_opinion_preferences[opinion].append(neighbor.preference)

        # Calculate the probability of being chosen for each opinion
        for opinion in range(opinions):
            if len(neighbor_opinion_preferences[opinion])!=0:
                probability_rates.append(sum(neighbor_opinion_preferences[opinion])/sum(neighbor_preferences))
            else:
                probability_rates.append(0)
        
        # select opinion with highest probability
        max_opinion_idx = probability_rates.index(max(probability_rates))            

        # Only switch opinion if the probability for other opinions is higher than the preference for agent's own opinion.
        if (probability_rates[max_opinion_idx]  < (self.preference*2)/opinions) and self.opinion != max_opinion_idx:
            return self.select_opinion(neighbor_opinion_preferences[self.opinion], switch_opinion = False, new_opinion = self.opinion)

        # If so: Role dice and select new opinion
        else:
            dice = random.uniform(0,1)
            for idx, prob in enumerate(probability_rates):
                dice = dice - prob
                if dice <= 0:
                    return self.select_opinion(neighbor_opinion_preferences[idx], switch_opinion = True, new_opinion = idx)

    def select_opinion(self, preferences, switch_opinion, new_opinion):
        '''
        An agent opinion selection. If agent switches opinion, switch preference.
        '''

        # if agent switches opinion; switch preference
        if switch_opinion:
            self.preference = 1 - self.preference
            
        self.opinion = new_opinion
        self.update_preference(preferences)

    def update_preference(self, neighbors_preference):
        '''
        Update agents preference using the neighbors with shared opinions preferences
        '''
        global social_influence
        self.preference = self.preference + (social_influence * sum([(neighbor - self.preference) for neighbor in neighbors_preference]))

    def update_trust(self, neighbor_position):
        '''
        Update trust between agents by updating edge strenghts
        '''
        global G
        self.update_edge(self.unique_id, neighbor_position)


    def update_edge(self, node1, node2):
        '''
        Update trust between nodes based on previous encounters and times agreed (shared opinions)
        '''
        global G

        # increment total encounters
        G.edges[node1, node2]['total_encounters'] += 1

        # If agents share opinion, increment times agreed
        if(G.nodes()[node1]['agent'].opinion == G.nodes()[node2]['agent'].opinion ):
            G.edges[node1, node2]['times_agreed'] += 1

        # Calculate new trust
        G.edges[node1, node2]['trust'] = G.edges[node1, node2]['times_agreed'] /  G.edges[node1, node2]['total_encounters']  

##############################################
# Helper Functions related to the model rules
##############################################
def set_rand_unifrom_preference():
    '''
    Returns a random preference between [0, 1]
    '''
    return np.random.uniform(0,1)

def get_rand_similarity(similarity_treshold):
    '''
    Returns a sample from gaussian distribution with mean = similarity_treshold
    '''
    lower_treshold = 0.0
    upper_treshold = similarity_treshold*2
    sample = 10
    while sample > upper_treshold or sample < lower_treshold:
        sample = random.gauss(similarity_treshold, np.std([lower_treshold, similarity_treshold, upper_treshold]))
        
    return sample


def initialize_graph(N,no_of_neighbors,network_type, beta_component=None):
        '''
        Returns an initialized network type using: 
        network type (int)
        N (int)
        no_of_neighbors (int)
        beta_comopnent (float)
        '''
        if(network_type == 1):
            return nx.watts_strogatz_graph(N, no_of_neighbors, beta_component)
        elif(network_type == 2):
            return nx.barabasi_albert_graph(N, no_of_neighbors)

def set_malicious():
    '''
    place malicious agents on network
    '''
    global malicious_agents, G

    # Select most connected nodes
    centrality_dict = nx.degree_centrality(G)  
    most_central = nlargest(malicious_N, centrality_dict, key=centrality_dict.get)

    # For the number of malicious agents update the most connected agents switch opinion to 0 and preference to 1
    for a in most_central:
        G.nodes()[a]["agent"].opinion = 0
        G.nodes()[a]["agent"].preference = 1
        malicious_agents.append(a) 

def perturb_network():
    '''
    Perturbs the network: changes opinion, preference and trust for a number of swingers
    '''
    global swingers, num_agents, swingers, G
    agent_nodes = np.random.randint(num_agents, size=(1,swingers))
    for node in agent_nodes:
        agent = G.nodes()[np.random.randint(num_agents)]['agent']
        edges = G.edges(node)
        for edge in edges:
            G.edges[edge]["trust"] = set_rand_unifrom_preference()
        agent.opinion = np.random.randint(2)
        agent.preference = set_rand_unifrom_preference()               

def update_malicious_agents():
    """
    Re-assigns malicious agents their ordinal 
    preference and opinion
    """
    global malicious_N, malicious_agents, G
    if malicious_N > 0: 
        for a in malicious_agents:
            G.nodes()[a]['agent'].opinion = 0
            G.nodes()[a]['agent'].preference = 1
            neighbors =  G.neighbors(a)
            
            for neighbor in neighbors:
                G.edges[a,neighbor]['trust'] = 1  

############################
#PYCX module 
############################

def initialize():
    global time, G, agents, malicious_agents, echo_chamber_n, echo_chamber_size, cliques,    echo_chamber_n_data,percentage_majority_opinion_data,average_trust_data,radical_opinion_data,community_no_data,silent_spiral_data,transitivity_data
    time = 0
    G = initialize_graph(num_agents,no_of_neighbors,network_type, beta_component)
    #random activation of nodes
    for node, unique_num in zip(random.sample(G.nodes(), num_agents), range(num_agents)):
            G.nodes[node]['agent'] = agent(unique_num)
    nx.set_edge_attributes(G, 2, 'total_encounters')
    nx.set_edge_attributes(G, 1, 'times_agreed')
    nx.set_edge_attributes(G, .5, 'trust')
    G.pos = nx.random_layout(G)
    agents = random.sample(G.nodes(),num_agents)
    malicious_agents = []
    echo_chamber_n = 0
    echo_chamber_size = 0
    # find the number of cliques, Based on the algorithm published by Zhang et al. (2005)
        #  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1559964&isnumber=33129&tag=1
    cliques =  len(list(nx.enumerate_all_cliques(G)))

    # Set malicious agents
    set_malicious()

    # Collect data
    echo_chamber_n_data = list()
    percentage_majority_opinion_data = list()
    average_trust_data = list()
    radical_opinion_data = list()
    community_no_data = list()
    silent_spiral_data = list()
    transitivity_data = list()


def observe():
    global G,    echo_chamber_n_data,percentage_majority_opinion_data,average_trust_data,radical_opinion_data,community_no_data,silent_spiral_data,average_trust,transitivity_data

    subplot2grid((3, 3), (0, 0), colspan=2)
    cla()
    visualize_network(G, nx.spring_layout(G, dim=2, seed=42), time, opinions)
    axis('image')
    title('t = ' + str(time))

    subplot2grid((3, 3), (1, 0))
    plot(range(time), community_no_data, label = 'Community N', color="b")
    title("Number of communities")
    ylabel("Count")

    legend()

    subplot2grid((3, 3), (1, 1))
    plot(range(time), silent_spiral_data, label = 'Silent Spiral', color="g")
    plot(range(time), radical_opinion_data, label = 'Percentage of Radical Opinions', color="b")
    title("Silent Spiral and Percentage of agents with the radical Opinions")

    legend()

    subplot2grid((3, 3), (2, 0))
    plot(range(time), average_trust_data, label = 'Average Trust', color="c")
    plot(range(time), percentage_majority_opinion_data, label = 'Percentage with the majority opinion', color="y")
    title("Average Trust and Percentage of users with majority opinion")
    xlabel("Time")
    
    legend()
    subplot2grid((3, 3), (2, 1))
    plot(range(time), echo_chamber_n_data, label = 'Echo Chamber N', color="y")
    plot(range(time), transitivity_data, label = 'Transitivity', color="r")
    title("Echo chambers, and Transitivity")
    xlabel("Time")

    legend()

    show()

def update():
    global time, G,malicious_agents,malicious_N,  echo_chamber_n_data, echo_chamber_size_data,percentage_majority_opinion_data,average_trust_data,radical_opinion_data,community_no_data,silent_spiral_data,average_trust_data,transitivity_data
    time+=1

    #Make all agents step, random activation
    for a in random.sample(G.nodes(), num_agents):
        G.nodes()[a]["agent"].step()
    
    #Pertub Networks
    #changes opinion, preference and trust for a number of swingers
    perturb_network()

     # Update all malicious agents
    update_malicious_agents()

    ## Gather time series statistics
    echo_chamber_n_data.append(compute_echo_chamber(G, echo_limit))
    community_no_data.append(community_no(G))
    radical_opinion_data.append(compute_radical_opinions(G, num_agents))

    silent_spiral_data.append(compute_silent_spiral(G))
    transitivity_data.append(compute_transitivity(G))

    percentage_majority_opinion_data.append(compute_majority_opinions(G, num_agents))
    average_trust_data.append(average_trust(G))


pycxsimulator.GUI().start(func=[initialize, observe, update])