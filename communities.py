import numpy as np
import networkx as nx
import scipy.spatial as spatial
from parameters import beta, w_a, w_i, w_d, b_min_prob, a_min_prob


def create_network(agents, households, build):
    # ALL DISTANCES ARE COMPUTED IN PROBABILITIES - HIGHER VALUES MEAN CLOSER CONTACT
    # TODO - needs to be sped up?, consider individual agent mobility profile (see at the bottom)
    # compute distance between buildings - gravity model probability - does not accout for LU
    # TO CONSIDER - connect only non-residential buildings?
    fs = build[:, 4] # get floor space
    dists = spatial.distance_matrix(build[:, 6:8], build[:, 6:8])**beta # matrix of squared distances between buildings
    dists[dists==0] = 0.00001 # distance between a building an itself = 0.00001
    scores = fs / dists # attraction scores - floorspace / squared distances
    for i in range(len(build)):
        scores[i, i] = 0 # attractivity of a building to itself = 0
    rows_sum = scores.sum(axis=1).reshape((len(build), 1)) # sum of each row for computing probability scores
    bld_prob = scores / rows_sum # compute matrix of probabilities
    del fs, dists, scores, rows_sum, i
    
    # compute distance between agents based on household incomes, agnet ages, and physical distance
    # find household of agent
    agent_hh = (households[:, 0][:, None] == agents[:, 1]).argmax(axis=0)
    hh_income = households[agent_hh, 2] # household incomes
    income_dist = np.abs(np.subtract(hh_income, hh_income.reshape((len(agents), 1)))) # matrix of difference in incomes
    # maximal value is used to noralize all values to be within 0 to 1
    income_dist = 1 - income_dist / np.max(income_dist) # agents from households with similar incomes have higher probabilities
    
    age_dist = np.abs(np.subtract(agents[:, 4], agents[:, 4].reshape((len(agents), 1)))) # matrix of age difference
    age_dist = 1 - age_dist / np.max(age_dist) # agents of similar ages have higher probabilities
    
    hh_home = households[agent_hh, 1] # find homes of households
    agent_homes = (build[:, 0][:, None] == hh_home).argmax(axis=0) # find homes of agents
    # TODO - consider mobility profile
    agent_dists = spatial.distance_matrix(build[agent_homes, 6:8], build[agent_homes, 6:8]) # matrix of distances between agents
    agent_dists = 1 - (agent_dists) / np.max(agent_dists) # closer agents have higher probabilities
    
    agents_prob = w_a * age_dist * w_i * income_dist * w_d * agent_dists # final probability - multiplication of all distances
    agents_prob[agent_hh == agent_hh.reshape((len(agents), 1))] = 1 # identify agents from the same household and set prob to 1
    del agent_hh, hh_income, income_dist, age_dist, hh_home, agent_dists
    
    # create network
    # BUILDNGS are connected based on the probabilities computed above
    # AGENTS are connected to their homes, workplaces, and to other agents
    # WHY USE -np.log(p) as weight in the network? A mathematical trick - because we use probabilities we should use multiplication
    # i.e. p[0->2] = p[0->1] * p[1->2], but network distances are additive - d[0->2] = d[0->1] + d[1->2]
    # Using logorithm rules, we can solve this - log(a*b) = log(a) + log(b), meaning that using log(p) makes addition equivalent to multiplication
    # TO DO - need to differentiate between agent and building nodes for ease of use later on
    G = nx.DiGraph() # directed graph - relations are not symmetric between buildings
    # create edges between buildings weighted by the computed probabilities only if probability>0.5
    edges = [(build[b,0], build[b1,0], -np.log(bld_prob[b, b1])) 
             for b in range(len(build)) for b1 in range(len(build)) if bld_prob[b, b1] > b_min_prob and  b != b1]
    # create edges with probability weight=1 between agents and their homes
    edges.extend([(agents[a, 0], build[agent_homes[a], 0], np.log(1)) for a in range(len(agents))])
    # create edges with probability weight=1 between agents and their workplaces
    edges.extend([(agents[a, 0], agents[a, 12], np.log(1)) for a in range(len(agents)) 
                  if ~np.isnan(agents[a, 12])])
    edges.extend([(agents[a, 0], agents[a, 18], np.log(1)) for a in range(len(agents)) 
                  if ~np.isnan(agents[a, 18])])
    G.add_weighted_edges_from(edges)
    del edges
    # create edges between agents and agents weighted by probabilities if probability>0.5
    # SLOW and potential MEMORY ERROR, need to find a way to handle
    links = np.where(agents_prob>a_min_prob)
    for i in range(len(links[0])):
        if links[0][i] != links[1][i]:
            G.add_weighted_edges_from([(agents[links[0][i], 0],
                                        agents[links[1][i], 0],
                                        -np.log(agents_prob[links[0][i], links[1][i]]))])
    del links, i
    print(len(list(G.edges)))
    
    # OPTION B: CONNECT AGENTS DIRECTLY TO AGENTS HOMES - not so good, what if agent moves?
    # # find maximal probability for each home for each agent based on agent-agent probabilities
    # agent_home_probs = np.zeros((len(agents), len(np.unique(agent_homes))))
    # agent_homes_unique = np.unique(agent_homes)
    # for a in range(len(agents)):
    #     if a%1000==0:
    #         print(a)
    #     order = np.lexsort((agents_prob[a], agent_homes))
    #     groups = agent_homes[order]
    #     data = agents_prob[a][order]
    #     index = np.empty(len(groups), 'bool')
    #     index[-1] = True
    #     index[:-1] = groups[1:] != groups[:-1]
    #     agent_home_probs[a] = data[index]
    #     current_probs = np.array(
    #         [G[agents[a, 0]][bld[b, 0]]['weight'] if bld[b, 0] in G.neighbors(agents[a,0]) 
    #          else 0 for b in agent_homes_unique])
    #     agent_home_probs[a, current_probs > agent_home_probs[a]] = current_probs[
    #         current_probs > agent_home_probs[a]]
    # edges = [(agents[a, 0], bld[agent_homes_unique[b], 0], -np.log(agent_home_probs[a, b])) 
    #          for a in range(len(agents)) for b in range(len(agent_homes_unique))
    #          if agent_home_probs[a, b] > 0.5]
    # G.add_weighted_edges_from(edges)
    # print(len(list(G.edges)))
    
    # find additional fixed activities for agents
    return G

# mobility = 1 + (agents.disability.to_numpy().reshape((len(agents), 1)) +
#                 ((agents.age.to_numpy() == 1)*1).reshape((len(agents), 1)) +
#                 ((agents.age.to_numpy() == 3)*1).reshape((len(agents), 1)) -
#                 agents.car.to_numpy().reshape((len(agents), 1))) / 3