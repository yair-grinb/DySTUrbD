import numpy as np
from create_random_data import create_data
from communities import create_network
import networkx as nx
from random import choice
from parameters import k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day, quarantine, diagnosis
from time import time
from scipy.sparse.csgraph import shortest_path
import json
from os import mkdir

mkdir('outputs')
scenario_name = ''


def compute_R(agents, infected):
    new_infections = len(agents[agents[:, 13] == 2]) +len(agents[agents[:, 13]==4])+len(agents[agents[:,13]==3.5]) - infected
    sum_I = 0.
    for i in range(1, recover+1):
        sum_I += agents[agents[:, 17] == i].shape[0] * contagious_risk_day.pdf(i)
    if sum_I > 0:
        R = new_infections / sum_I
    else:
        R = 0
    return R


def compute_vis_R(agents):
    new_known_infections = len(agents[(agents[:, 17] == diagnosis)])
    sum_I = 0.
    for i in range(diagnosis+1, recover+1):
        sum_I += agents[(agents[:, 13] == 4) & (agents[:, 17] == i)].shape[0] * contagious_risk_day.pdf(i - diagnosis)
    if sum_I > 0:
        R = new_known_infections / sum_I
    else:
        R = 0
    return R


for sim in range(1,31):
    outputs = {'Sim':sim, 'Results':{'Stats':{}, 'Buildings':{}, 'SAs':{}, 'IO_mat':{}}}
    T = time()
    t = time()
    agents, households, build, jobs = create_data('data/civ_withCar_bldg_np.csv', 'data/bldg_with_inst_orig.csv')
    # agents: 0 id, 1 hh, 2 dis, 3 worker, 4 age, 5 workIn, 6 wp_participant, 7 building, 8 random socio-economic status, 
    # 13 contagious status, 14 contagious risk, 15 exposure risk, 16 contagious day, 17 sick day, 18 more regular activities, 19 quarantine start day, 20 num of days in quarantine
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
    agents_reg = agents[:, [0,7,12,18,7]]
    
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
        for i in agents_reg[n, 2:]:
            if ~np.isnan(i):
                i_nodes = nodes[(dists[:, np.where(nodes==i)[0][0]]<bld_dist) & (nodes<4000000)] # buildings within bld_dist from i
                for j in range(k):
                    if np.random.randint(2) > 0:
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
    
    while len(agents[agents[:, 13] == 2]) + len(agents[agents[:, 13] == 3.5]) + len(agents[agents[:, 13] == 4])> 0:    
    #while day<14:
        t = time()
        # get buildings status - open/close == 1/0 and multiply bld_visits_by_agents by it
        bld_visits = bld_visits_by_agents * build[
            np.argmax(build[:, 0][None, :] == bld_visits_by_agents[:, :, None], axis=2), 10]
        # check if agents are in isolation and if yes - all activities but first (home) become zero
        bld_visits[:, 1:] = bld_visits[:, 1:] * ((agents[:, 13] != 3) & (agents[:, 13] != 4)
                                                 & (agents[:, 13] != 3.5)).reshape((len(agents), 1))
        bld_visits[bld_visits==0] = np.nan
        agents[(agents[:, 13] == 2) | (agents[:, 13] == 4) | (agents[:, 13] == 3.5), 17] = day - agents[(agents[:, 13] == 2) | (agents[:, 13] == 4) | (agents[:, 13] == 3.5), 16] # update number of days since infecton for carriers
        agents[(agents[:, 13] == 3) | (agents[:, 13] == 3.5), 20] = day - agents[(agents[:, 13] == 3)  | (agents[:, 13] == 3.5), 19] # update number of days since infecton for carriers

        
        infected = np.where((agents[:,13]==2) | (agents[:,13]==4) | (agents[:, 13]==3.5))[0]
        uninfected = np.where((agents[:,13]<2) | (agents[:,13]==3))[0]
        infected_quar = len(np.where(agents[:,13]==4)[0]) # total diagnosed agents in quarantine            
        infected_blds = bld_visits[infected] # building visits by infected
        uninfected_blds = bld_visits[uninfected] # building visits by uninfected
        # compare the two arrays by flattening them and reshaping infected_blds
        exposure = 1*np.array([np.isin(infected_blds, uninfected_blds[i]).any(axis=1) 
                               for i in range(len(uninfected_blds))])
        
        infection_prob = interaction_prob[uninfected][:, infected] * agents[
            uninfected, 14].reshape((len(uninfected), 1)) 
        infection_prob *= contagious_risk_day.pdf(agents[infected, 17])
        # infection_prob *= contagious_risk_day[
        #         agents[infected, 17].astype(int)] 
        infection_prob *= exposure 
        infection_prob *= norm_factor
        rand_cont = np.random.random(infection_prob.shape)
        infections = infection_prob > rand_cont
        new_infected = np.where(infections.any(axis=1) & (agents[uninfected, 13]<2))
        new_quarantined_infected = np.where((infection_prob > rand_cont).any(axis=1) & 
                                            (agents[uninfected, 13]==3))
        agents[uninfected[new_infected], 13] = 2
        agents[uninfected[new_infected], 16] = day
        agents[uninfected[new_quarantined_infected], 13] = 3.5
        agents[uninfected[new_quarantined_infected], 16] = day
        # this allows tracing infection chains - who infected whom and when
        agents[uninfected[np.where(infections)[0]], 21] = agents[infected[np.where(infections)[1]], 0]
        if len(agents[agents[:, 13] == 4]) > 0: # if there are sick agents in quarantine
            new_quar = np.where((np.isin(agents[:, 1],agents[agents[:,13]==4, 1])) & 
                                (agents[:, 13]==1))[0] # find their hh members
            agents[new_quar, 13] = 3 # set them to be healthy and in quarantine
            agents[new_quar, 19] = day # record atart fay of quarantine per agent
            new_quar_infected = np.where((np.isin(agents[:, 1],agents[agents[:,13]==4, 1])) & 
                                (agents[:, 13]==2))[0] # find their hh members
            agents[new_quar_infected, 13] = 3.5 # set them to be healthy and in quarantine
            agents[new_quar_infected, 19] = day # record atart fay of quarantine per agent

                          
        print('day: ', day+1)
        print('\tinfected: ', len(agents[agents[:, 13] == 2])+len(agents[agents[:, 13]==4])+len(agents[agents[:,13]==3.5]))
        R = compute_R(agents, len(infected))
        new_infections = len(agents[agents[:, 13] == 2]) +len(agents[agents[:, 13]==4])+len(agents[agents[:,13]==3.5]) - len(infected)
        agent_sas = build[np.where(agents[:, 7][:, None] == build[:, 0][None, :])[1], 3]
        sas_R = {}
        outputs['Results']['SAs'][day] = {}
        for sa in np.unique(agent_sas):
            sa_agents = agents[agent_sas == sa]
            sa_infected = len(np.where(((sa_agents[:,13]==2) | (sa_agents[:,13]==4) | 
                                        (sa_agents[:, 13]==3.5)) & (sa_agents[:, 16] != day))[0])
            sas_R[sa] = compute_R(sa_agents, sa_infected)
        outputs['Results']['SAs'][day]['R'] = sas_R
        del sa_agents, sa_infected
        print('\tnew infections: ', new_infections)
        print('\tquarantined: ', len(agents[agents[:, 13]==3])+len(agents[agents[:,13]==4])+len(agents[agents[:,13]==3.5]))
        print('\tR: ', R)
        vis_R = compute_vis_R(agents)
        new_known_infections = len(agents[(agents[:, 17] == diagnosis)])
        sas_vis_R = {}
        for sa in np.unique(agent_sas):
            sa_agents = agents[agent_sas == sa]
            sas_vis_R[sa] = compute_vis_R(sa_agents)
        outputs['Results']['SAs'][day]['vis_R'] = sas_vis_R
        del sa_agents
        agents[(agents[:, 13] == 3) & (agents[:, 19] == quarantine), 13] = 1 # end of quarantine for helathy agents, can steel be infected
        agents[(agents[:, 13] == 3.5) & (agents[:, 19] == quarantine), 13] = 2 # end of quarantine for infected undiscovered agents
        agents[((agents[:, 13] == 2) | (agents[:, 13] == 3.5)) & (agents[:, 17] == diagnosis), 13] = 4 # sick agents begin quarantine after for days
        agents[(agents[:, 13] == 4) & (agents[:, 17] == recover), 13] = 5 # sick agents in quarantine recover 
        agents[(agents[:, 13] == 1) | (agents[:, 13] == 2) | (agents[:, 13] == 5), 20] = 0
        print('\tVis_R:' ,vis_R)
        # if 1 < vis_R < 2: # if visible R is between 1 & 2 close half of the commercial and public buildings 
        #     indices = np.random.choice(np.arange(build[build[:,1]>=3,10].size), replace=False,
        #                    size=int(build[build[:,1]>=3,10].size * 0.5))
        #     indices = np.append(indices,indices2)
        #     build[indices,10] = 0 
        # elif vis_R > 2: # if visible R is greater than 2 close all commercial and public buildings
        if vis_R > 1:
            build[build[:, 1] >= 3, 10] = 0 
        else: # if visible R smaller than 1 set all buildings as open
            build[:,10] = 1
        print('\trecovered: ', len(agents[agents[:, 13] == 5])) 
        outputs['Results']['Stats'][day] = {'Infected': len(infected)+new_infections,
                        'New_infections': new_infections,
                        'Recovered': len(agents[agents[:, 13] == 5]),
                        'Quarantined': len(agents[agents[:,13]==3])+len(agents[agents[:,13]==4])+len(agents[agents[:,13]==3.5]),
                        'R': R,
                        'Known_R': vis_R}
        outputs['Results']['Buildings'][day] = []
        for b in build:
            b_pop = agents[agents[:, 7] == b[0]]
            if b_pop.shape[0] > 0:
                b_infected = b_pop[(b_pop[:, 13] == 2) | (b_pop[:, 13] == 3.5) | 
                                   (b_pop[:, 13] == 4)].shape[0]
                outputs['Results']['Buildings'][day].append([b[0], b_infected, 
                                                             b_infected / b_pop.shape[0]])
        if day==0:
            outputs['Results']['Stats'][day]['Total_infected'] = outputs['Results']['Stats'][day]['Infected']
        else:
            outputs['Results']['Stats'][day]['Total_infected'] = outputs['Results']['Stats'][day-1]['Total_infected'] + new_infections
        outputs['Results']['Stats'][day]['Susceptible'] = len(agents) - outputs['Results']['Stats'][day]['Total_infected']
        del infected, uninfected, infected_blds, uninfected_blds, exposure, infection_prob, infected_quar
        del rand_cont, new_infected, new_infections, R, vis_R, #indices, indices2
        print('\t'+str(round(time()-t)))
        outputs['Results']['Stats'][day]['Time'] = time()-t
        sas = np.unique(build[:, 3])
        infect_mat = {sa:{sa2: 0 for sa2 in sas} for sa in sas}
        infected_sas = agent_sas[agents[:, 16]==day]
        infecting_sas = agent_sas[np.where(agents[agents[:, 16]==day, 21][:, None] ==
                                           agents[:, 0][None, :])[1]]
        if infecting_sas.size > 0:
            for sa in sas:
                counts = np.unique(infecting_sas[infected_sas == sa], return_counts=True)
                for i in range(len(counts[0])):
                    infect_mat[sa][counts[0][i]] = int(counts[1][i])
        outputs['Results']['IO_mat'][day] = infect_mat
        day += 1
    print((time()- T)/3600)
    outputs['Results']['Infections chain'] = agents[:, [0, 16, 21]].tolist()
    outputs['Total_time'] = time()-T
    with open('outputs/sim'+str(sim)+scenario_name, 'w') as outfile:
        json.dump(outputs, outfile)
