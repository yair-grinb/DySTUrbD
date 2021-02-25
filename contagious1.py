import numpy as np
from create_random_data import create_data
from communities import create_network
import networkx as nx
from random import choice
from parameters import k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day
from time import time
from scipy.sparse.csgraph import shortest_path
import json


for sim in range(8, 20):
    outputs = {'Sim':sim, 'Results':{}}
    T = time()
    t = time()
    agents, households, build, jobs = create_data('data/civ_withCar_bldg_np.csv', 'data/bldg_with_inst_orig.csv')
    # agents: 0 id, 1 hh, 2 dis, 3 worker, 4 age, 5 workIn, 6 wp_participant, 7 building, 8 random socio-economic status, 
    # 13 contagious status, 14 contagious risk, 15 exposure risk, 16 contagious day, 17 sick day
    G = create_network(agents, households, build) # link agents and buildings
    print(time() - t)
    outputs['network_time'] = time()-t
    
    t = time()
    nodes = np.array(G.nodes())
    g = nx.to_scipy_sparse_matrix(G) # convert to scipy sparse matrix - faster route calculations
    dists = shortest_path(g, directed=True, return_predecessors=False) # get shortest paths between all nodes
    np.fill_diagonal(dists, np.inf) # distance from node to self is infinity
    agent_nodes = nodes[np.isin(nodes, agents[:, 0])]
    interaction_prob = np.exp(-dists[np.where(np.isin(nodes, agents[:, 0]))[0]][:,
                                     np.where(np.isin(nodes, agents[:, 0]))[0]])
    agents_reg = agents[:, [0,7,12,7]]
    
    #contagious routine
    day = 0
    
    print(time() - t)
    outputs['interaction_time'] = time()-t
    
    bld_visits_by_agents = []
    for n in range(len(agents_reg)):
        a = agents_reg[n, 0]
        current_position = agents_reg[n, 1]
        a_nodes = nodes[(dists[np.where(nodes==a)][0]<a_dist) & (np.isin(nodes, build[:, 0]))] # buildings within distance<1 from a
        visits = [current_position]
        if agents_reg[n, 2] > 0:
            for i in agents_reg[n, 2:]:
                i_nodes = nodes[(dists[:, np.where(nodes==i)[0][0]]<bld_dist) & (nodes<4000000)] # buildings within bld_dist from i
                for j in range(k):
                    c_nodes = nodes[(dists[np.where(nodes==current_position)][0]<bld_dist) &
                                    (nodes<4000000)] # buildings within bld_dist from current position
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
    
    while len(agents[agents[:, 13] == 2]) > 0:    
    #while day<14:
        t = time()
        
        agents[agents[:, 13] == 2, 17] = day - agents[agents[:, 13] == 2, 16] # update number of days since infecton for carriers
        
        
        infected = np.where(agents[:,13]==2)[0]
        uninfected = np.where(agents[:,13]<2)[0] 
        infected_blds = bld_visits_by_agents[infected] # building visits by infected
        uninfected_blds = bld_visits_by_agents[uninfected] # building visits by uninfected
        # compare the two arrays by flattening them and reshaping infected_blds
        exposure = 1*np.array([np.isin(infected_blds, uninfected_blds[i]).any(axis=1) 
                               for i in range(len(uninfected_blds))])
        # comp = np.transpose(
        #     (uninfected_blds.flatten() == infected_blds.flatten().reshape((len(infected_blds.flatten()), 1))))*1
        # exposed_idx = list(np.where(comp)) # indices where true - exposure of uninfected to infected
        # # by diving by the number of visits per agent we get indices at the agent level
        # exposed_idx[0] = (exposed_idx[0]/uninfected_blds.shape[1]).astype(int)
        # exposed_idx[1] = (exposed_idx[1]/uninfected_blds.shape[1]).astype(int)
        # exposure = np.zeros((len(uninfected), len(infected))) # contain exposure data
        # exposure[tuple(exposed_idx)] = 1 # uninfected that were in the same building with infected are exposed
        
        infection_prob = interaction_prob[uninfected][:, infected] * agents[
            uninfected, 14].reshape((len(uninfected), 1)) 
        infection_prob *= contagious_risk_day[
                agents[infected, 17].astype(int)] 
        infection_prob *= exposure 
        infection_prob *= norm_factor
        rand_cont = np.random.random(infection_prob.shape)
        new_infected = np.where((infection_prob > rand_cont).any(axis=1))
        agents[uninfected[new_infected], 13] = 2
        agents[uninfected[new_infected], 16] = day
                          
        print('day: ', day+1)
        print('\tinfected: ', len(agents[agents[:, 13] == 2]))
        new_infections = len(agents[agents[:, 13] == 2]) - len(infected)
        R = new_infections / len(infected)
        print('\tnew infections: ', new_infections)
        print('\tR: ', R)
        agents[(agents[:, 13] == 2) & (agents[:, 17] == recover), 13] = 5
        print('\trecovered: ', len(agents[agents[:, 13] == 5])) 
        outputs['Results'][day] = {'Infected': len(agents[agents[:, 13] == 2]),
                        'New_infections': new_infections,
                        'Recovered': len(agents[agents[:, 13] == 5]),
                        'R': R}
        if day==0:
            outputs['Results'][day]['Total_infected'] = outputs['Results'][day]['Infected']
        else:
            outputs['Results'][day]['Total_infected'] = outputs['Results'][day-1]['Total_infected'] + new_infections
        outputs['Results'][day]['Susceptible'] = len(agents) - outputs['Results'][day]['Total_infected']
        del infected, uninfected, infected_blds, uninfected_blds, exposure, infection_prob
        del rand_cont, new_infected, new_infections, R
        print('\t'+str(round(time()-t)))
        outputs['Results'][day]['Time'] = time()-t
        day += 1
    print((time()- T)/3600)
    outputs['Total_time'] = time()-T
    with open('outputs/sim'+str(sim)+'.json', 'w') as outfile:
        json.dump(outputs, outfile)
