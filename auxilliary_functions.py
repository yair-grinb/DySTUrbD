import networkx as nx
import global_variables as gv
import numpy as np
from random import random, choice
import scipy.spatial as spatial
from parameters import beta, w_a, w_i, w_d, b_min_prob, a_min_prob, k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day, quarantine, diagnosis, scenario_codes, risk_by_age, hospital_recover
from scipy.sparse.csgraph import shortest_path

def compute_network():
    r_dists = dict(nx.all_pairs_dijkstra_path_length(gv.graph, weight='weight'))
    gv.dists = []
    for i in range(len(gv.junctions)):
        gv.dists.append([])
        for j in range(len(gv.junctions)):
            if gv.junctions[j] in r_dists[gv.junctions[i]]:
                gv.dists[-1].append(r_dists[gv.junctions[i]][gv.junctions[j]])
            else:
                gv.dists[-1].append(np.inf)
    gv.dists = np.array(gv.dists)
    gv.routes = dict(nx.all_pairs_dijkstra_path(gv.graph, weight='weight'))


# TODO: Paste create_network from communities(done), verify it works

def create_social_network(agents, households, build):
    # ALL DISTANCES ARE COMPUTED IN PROBABILITIES - HIGHER VALUES MEAN CLOSER CONTACT
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
    
    return G

def get_path(o, d):
    r = gv.routes[gv.junctions[o]][gv.junctions[d]]
    route = [np.where(gv.roads[:, 0] == gv.graph[r[i]][r[i + 1]]['fid'])[0] for i in range(len(r) - 1)]
    return route


# TODO: replace code with routine creation from contagious - keep the computation of paths on roads
# wasn't entirely sure where is the path computation so i've comeented everything
def create_routines(agents_reg,G):
    nodes = np.array(G.nodes())
    g = nx.to_scipy_sparse_matrix(G) # convert to scipy sparse matrix - faster route calculations
    dists = shortest_path(g, directed=True, return_predecessors=False) # get shortest paths between all nodes
    np.fill_diagonal(dists, np.inf) # distance from node to self is infinity
    bld_visits_by_agents = []
    for n in range(len(agents_reg)):
        a = agents_reg[n, 0]
        current_position = agents_reg[n, 1]
        a_nodes = nodes[(dists[np.where(nodes==a)][0]<a_dist) & (np.isin(nodes, gv.bldgs[:, 0]))] # buildings within distance<1 from a
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
    
    # path = []
    # hh = gv.households[gv.households[:, 0] == gv.indivs[i, 1].astype(int)][0]
    # activities = int(round((3. - (gv.indivs[i, 3] == 1) * 0.5) * random() * (1 + (
    #         hh[3] - (gv.indivs[i, 4] != 2) - gv.indivs[i, 2]) / 3.) * 2. + gv.indivs[i, 5], 0))
    
    # if activities < 0:
    #     activities = 0
    
    # loc = np.where(gv.bldgs[:, 0] == hh[1])[0][0]
    # for j in range(activities):
    #     dest = np.nan
    #     if j == 0 and gv.indivs[i, 5] == 1:
    #         job = gv.jobs[gv.jobs[:, -1] == gv.indivs[i, 7]][0]
    #         if ~np.isinf(gv.dists[gv.bldgs[loc, 13]][gv.bldgs[int(job[1]), 13]]):
    #             dest = int(job[1])
    #     if np.isnan(dest):
    #         pref = random()
    #         pot_bldgs = gv.bldgs[(gv.bldgs[:, 0] != gv.bldgs[loc, 0]) & (gv.bldgs[:, 1] != 0) & (
    #             ~np.isinf(gv.dists[gv.bldgs[loc, 13]][gv.bldgs[:, 13].tolist()]))]
    #         if len(pot_bldgs) > 0:
    #             max_dist = np.max(np.ma.masked_invalid(gv.dists[gv.bldgs[loc, 13]]))
    #             loc_dists = gv.dists[gv.bldgs[loc, 13]][pot_bldgs[:, 13].tolist()]
    #             loc_dists = loc_dists / max_dist
    #             loc_dists = 1. - loc_dists * (1 + (hh[3] - (gv.indivs[i, 4] != 2) - gv.indivs[i, 2]) / 3.)
    #             fs_scores = pot_bldgs[:, 4] / np.max(gv.bldgs[gv.bldgs[:, 1] > 1, 4])
    #             fs_scores = 1. - fs_scores
    #             fs_scores = (pot_bldgs[:, 1] > 1) * fs_scores
    #             scores = (pot_bldgs[:, 11] + loc_dists + fs_scores) / (2. + (pot_bldgs[:, 1] > 1))
    #             if np.any(pref > scores):
    #                 dest_idx = np.random.choice(len(pot_bldgs[pref > scores]))
    #                 b_dest = pot_bldgs[dest_idx]
    #                 dest = int(np.where(gv.bldgs[:, 0] == b_dest[0])[0][0])
    #     if ~np.isnan(dest):
    #         path.extend(get_path(gv.bldgs[loc, -2], gv.bldgs[dest, -2]))
    #         loc = dest
    # dest = np.where(gv.bldgs[:, 0] == hh[1])[0][0]
    # if ~np.isinf(gv.dists[gv.bldgs[loc, 13]][gv.bldgs[dest, 13]]):
    #     path.extend(get_path(gv.bldgs[loc, -2], gv.bldgs[dest, -2]))
    # gv.routines[gv.indivs[i, 0]] = path


def compute_building_value(b, boo):
    neigh = gv.bldgs[((gv.bldgs[b][np.newaxis, 6] - gv.bldgs[:, 6])**2. + (gv.bldgs[b][np.newaxis, 7] -
                                                                           gv.bldgs[:, 7])**2.) <= 100.**2.]
    gv.bldgs[b, [10, 11, 14]] = [np.nan, len(neigh[neigh[:, 1] == 0]) * 1. / len(neigh), np.nan]
    
    if gv.bldgs[b, 1] <= 1:
        zone = gv.zones[gv.zones[:, 0] == gv.bldgs[b, 3]][0]
        z_nres = zone[6]
        z_res = zone[5]
        if z_res == 0:
            z_res = 1
        if z_nres == 0:
            z_nres = 1
        nres_100 = len(neigh[neigh[:, 1] > 1]) * 1.
        res_100 = len(neigh[neigh[:, 1] == 1]) * 1.
        if res_100 == 0:
            res_100 = 1
        if nres_100 == 0:
            nres_100 = 1
        gv.bldgs[b, 10] = zone[3] * ((nres_100 / res_100) / (z_nres / z_res)) * gv.bldgs[b, 4]
        if boo:
            med_wtp = np.median(gv.households[:, 2]) / 3.
            ap_val = gv.bldgs[b, 10] / gv.bldgs[b, 12]
            mean_ap_val = np.sum(gv.bldgs[gv.bldgs[:, 1] <= 1, 10]) / np.sum(gv.bldgs[gv.bldgs[:, 1] <= 1, 12])
            gv.bldgs[b, -1] = med_wtp * (1 + (ap_val - mean_ap_val) / (gv.stdResVal * 12.))
