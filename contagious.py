import numpy as np
from create_random_data import create_data
from communities import create_network
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from random import choice
from parameters import k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day
from time import time

agents, households, build, jobs = create_data('data/civ_withCar_bldg_np.csv', 'data/bldg_with_inst_orig.csv')
# agents: 0 id, 1 hh, 2 dis, 3 worker, 4 age, 5 workIn, 6 wp_participant, 7 building, 8  religous(bin), 
# 13 contagious status, 14 contagious risk, 15 exposure risk, 16 contagious day, 17 sick day
G = create_network(agents, households, build) # link agents and buildings
dists = pd.DataFrame({b:nx.shortest_path_length(G, target=b, weight="weight") for b in build[:, 0]}) # dist for each nodes to each building
agents_reg = agents[:, [0,7,12,7]]

infected_L=[]
R_L = []
recoverd_L = []
new_infections_L = []
time_L = []

#contagious routine
day = 0
infected = np.where(agents[:,13]==2)[0]
while len(agents[agents[:, 13] == 2]) > 0:    
# while len(infected) > 0:
    t = time()
    
    agents[agents[:, 13] == 2, 17] = day - agents[agents[:, 13] == 2, 16] # update number of days since infecton for carriers
    
    bld_visits_by_agents = []
    for n in range(len(agents_reg)):
        a = agents_reg[n, 0]
        current_position = agents_reg[n, 1]
        a_nodes = dists.loc[a] # distances from a to buildings
        a_nodes = a_nodes.index[a_nodes<a_dist].to_numpy() # buildings within distance<1
        visits = [current_position]
        if agents_reg[n, 2] > 0:
            for i in agents_reg[n, 2:]:
                i_nodes = dists[i] # distances from NODES to i
                i_nodes = i_nodes.index[(i_nodes<bld_dist) & (i_nodes.index<4000000)].to_numpy() # BUILDINGS with distance <= 6 from i
                for j in range(k):
                    c_nodes = dists.loc[current_position] # distances from current position to buildings
                    c_nodes = c_nodes.index[c_nodes<bld_dist].to_numpy()
                    intersect = np.intersect1d(i_nodes, c_nodes)
                    union = np.union1d(intersect, a_nodes)
                    current_position = choice(union)
                    visits.append(current_position)
            current_position = i
            visits.append(current_position)
        bld_visits_by_agents.append(visits)
    del n, a, current_position, a_nodes, visits, i, i_nodes, j, c_nodes, intersect, union
    bld_visits_by_agents = np.array(bld_visits_by_agents)
    zero = np.zeros([len(bld_visits_by_agents),len(max(bld_visits_by_agents,key = lambda x: len(x)))])
    for i,j in enumerate(bld_visits_by_agents):
        zero[i][0:len(j)] = j
    bld_visits_by_agents = zero
    bld_visits_by_agents[bld_visits_by_agents==0] = np.nan
    del zero
    
    # Alternative approach:
        # find exposed buildings
        # count number of infected visits by building
        # find healthy agents visiting these buildings
        # compute chance of infection - chanceA*chanceC*density of infections in building / norm_factor
        # check if healthy become infected
    # working_agents = agents - NOT NEEDED ANYMORE
    infected = np.where(agents[:,13]==2)[0]
    uninfected = np.where(agents[:,13]<2)[0] 
    
    for a in infected: # working slow, need to verify that it really works, need to continue - ChanceA*ChanceC*interaction_prob/norm_factor etc.       
        if contagious_risk_day[int(agents[a, 17])] < 4:
            exposed = np.where(np.isin(bld_visits_by_agents[uninfected], bld_visits_by_agents[a]))[0]
            a_distances = np.array(list(nx.shortest_path_length(G, target=agents[a, 0], weight='weight').items()))
            a_distances[:, 1] = np.exp(-a_distances[:,1])
            interaction_prob = np.zeros(len(exposed))
            interaction_prob[:] = 0.01 # set interaction prob to be >0 so there's always would be chance for cross-communities infection
            interacted_agents = np.where(np.in1d(agents[uninfected[exposed], 0], a_distances)) # index of exspoed that are connected to a
            interacted_ids = agents[uninfected[exposed], 0] # ids of exposed that are connected to a
            interaction_prob[np.where(interacted_ids[:, None] == a_distances[:, 0][None, :])[0]] = a_distances[
                np.where(interacted_ids[:, None] == a_distances[:, 0][None, :])[1], 1]
            # interaction_prob = []
            # for e in exposed:
            #     try:
            #         interaction_prob.append([np.exp(-nx.shortest_path_length(G, working_agents[e, 0], working_agents[a, 0], weight='weight'))])
            #     except nx.NetworkXNoPath:
            #         interaction_prob.append([0])
            # interaction_prob = np.array(interaction_prob)
            interaction_prob = np.column_stack((exposed,interaction_prob))
            interaction_prob = np.column_stack((interaction_prob, agents[exposed, 14]))
           
            infect_prob = interaction_prob[:, 1] * interaction_prob[:, 2] *  contagious_risk_day[int(agents[a, 17])] * norm_factor
            rand_cont = np.random.random(len(exposed))
            new_infected = np.unique(exposed[infect_prob > rand_cont]) # new infected agents - unique since some agents appear more than once
            # infected.extend(new_infected) this will mean that these agents can infect in this day already, let's assume that they can't infect
            agents[np.isin(agents[:, 0], agents[uninfected[new_infected]]), 13] = 2
            agents[np.isin(agents[:, 0], agents[uninfected[new_infected]]), 16] = day # set status and day of infection
            uninfected = np.where(agents[:,13]<2)[0]
            # infected += len(new_infected) this may be the source of the error - it adds the value of len(new_infected) to each value in infected
                          
    print('day: ', day+1)
    print('\tinfected: ', len(agents[agents[:, 13] == 2]))
    new_infections = len(agents[agents[:, 13] == 2]) - len(infected)
    if len(infected) > 0:
        R = new_infections / len(infected)
    else:
        R = 0
    print('\tnew infections: ', new_infections)
    print('\tR: ', R)
    print('\t',time()-t)
    infected_L.append(len(agents[agents[:, 13] == 2]))
    new_infections_L.append(new_infections)
    R_L.append(R)
    time_L.append(time()-t)    
    agents[(agents[:, 13] == 2) & (agents[:, 17] == recover), 13] = 5
    print('\trecovered: ', len(agents[agents[:, 13] == 5])) 
    recoverd_L.append(len(agents[agents[:, 13] == 5]))
    day += 1
  
    
results = pd.DataFrame({'Infected Agents':infected_L,'R':R_L,'New Agents Infected':new_infections_L,'Recoverd Agents':recoverd_L, 'Run Time':time_L})
results.to_csv(r'C:\Users\Amir Cahal\Documents\research\code\code and data\2.csv')


