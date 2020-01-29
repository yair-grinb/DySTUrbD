import csv
import global_variables as gv
import numpy as np
import networkx as nx
import random
import model_parameters as mp
from auxilliary_functions import compute_network, create_routines, compute_building_value
from output_functions import save_init_snapshot
from time import time


def create_agent_class(f, idxs, app_idx=None):
    entities = []
    reader = csv.reader(open(f))
    for row in reader:
        e = [float(row[i]) for i in idxs]
        if app_idx is not None:
            e += app_idx
        entities.append(e)
    del reader, e, row, i
    return np.array(entities)


def create_world():
    t = time()
    # create zones - data structure: id, inMig, inMigCh, HPrice, prevPop, prevRes, prevNRes
    print 'getting zones', time() - t
    gv.zones = create_agent_class(gv.zonesFile, range(4), [0, 0, 0])
    
    # create roads - data structure: id, juncA_x, juncA_y, juncB_x, juncB_y, length, free, civsHere, civsAvg
    print 'getting roads', time() - t
    gv.roads = create_agent_class(gv.roadsFile, range(6), [1, 0, 0])
    
    # create road network graph and junctions
    gv.graph = nx.Graph()
    for i in range(gv.roads.shape[0]):
        gv.graph.add_edge((gv.roads[i, 1], gv.roads[i, 2]), (gv.roads[i, 3], gv.roads[i, 4]), weight=gv.roads[i, 5], fid=gv.roads[i, 0])
    gv.junctions = list(gv.graph.nodes)
    gv.junctions_array = np.array(gv.junctions)
    gv.roads_juncs = np.array([[[float(row[i]), float(row[i+1])] for i in [1, 3]] for row in csv.reader(open(gv.roadsFile))])
    gv.traff_hist = np.array([[] for i in range(gv.roads.shape[0])])
    compute_network()
        
    # create individuals - data structure: id, hh, dis, employed, age, employed locally, workforce participation, job, expected income,
    # workplace preference, search length, income
    print 'getting individuals', time() - t
    gv.indivs = create_agent_class(gv.indivsFile, [0, 1, 2, 3, 4, 5, 3], [np.nan, 0, random.random(), 0, 0])
    gv.indivs[(gv.indivs[:, 3] == 0) & (gv.indivs[:, 5] == 1), 3] = 1
    gv.indivs[(gv.indivs[:, 3] == 1) & (gv.indivs[:, 6] == 0), 6] = 1
    
    # compute parameter values
    mp.avgAge = np.mean(gv.indivs[:, 4])
    mp.stdAge = np.std(gv.indivs[:, 4], ddof=1)
    mp.age1chance = len(gv.indivs[gv.indivs[:, 4] == 1]) * 1. / len(gv.indivs)
    mp.age2chance = len(gv.indivs[gv.indivs[:, 4] == 2]) * 1. / len(gv.indivs)
    mp.disChance = len(gv.indivs[gv.indivs[:, 2] == 1]) * 1. / len(gv.indivs)
    mp.employChance = len(gv.indivs[gv.indivs[:, 3] == 1]) * 1. / len(gv.indivs)
    mp.workInChance = len(gv.indivs[gv.indivs[:, 5] == 1]) * 1. / len(gv.indivs[gv.indivs[:, 3] == 1])
    inds_in_hhs = np.unique(gv.indivs[:, 1], return_counts=True)[1]
    mp.hhSize = np.mean(inds_in_hhs)
    mp.hhSTD = np.std(inds_in_hhs, ddof=1)
    
    # create households - data structure: id, home, income, car
    print 'getting households', time() - t
    gv.households = create_agent_class(gv.householdsFile, range(4))
    
    mp.carChance = len(gv.households[gv.households[:, 3] == 1]) * 1. / len(gv.households)
    
    # create buildings - data structure: id, lu, floors, stat, fs, init_lu, x, y, counter, dem, value, neigh, apartments, 
    # nearest_junc, m_price
    gv.bldgs = create_agent_class(gv.bldgsFile, [0, 1, 2, 3, 4, 1, 5, 6], [0, False, 0, 0, 0, None, 0])
    
    # compute number of apartments
    for i in range(gv.bldgs.shape[0]):
        residents = np.size(gv.households[gv.households[:, 1] == gv.bldgs[i, 0]][:, 1])
        gv.bldgs[i, 12] = residents
    del i, residents
    
    # correct floors
    gv.bldgs[np.where(gv.bldgs[:, 2] == 0), 2] = 1
    
    # find nearest junction
    gv.bldgs[:, -2] = np.argmin((gv.bldgs[:, np.newaxis, 6] - gv.junctions_array[:, 0]) ** 2. + (
            gv.bldgs[:, np.newaxis, 7] - gv.junctions_array[:, 1]) ** 2., axis=1)
    
    # residential with no households to empty
    gv.bldgs[(gv.bldgs[:, 12] == 0) & (gv.bldgs[:, 1] == 1), 1] = 0
    
    # compute potential apartments for non-residential buildings
    gv.bldgs[gv.bldgs[:, 1] != 1, 12] = (gv.bldgs[gv.bldgs[:, 1] != 1, 4] / 90.).astype(int)
    gv.bldgs[gv.bldgs[:, 12] < 1, 12] = 1
    
    # init traffic history array
    gv.bldgs_traff_dist = np.zeros(gv.bldgs.shape[0])
    
    # update zone data
    for z in range(len(gv.zones)):
        gv.zones[z, 5] = len(gv.bldgs[(gv.bldgs[:, 3] == gv.zones[z, 0]) & (gv.bldgs[:, 1] == 1)]) * 1.
        gv.zones[z, 6] = len(gv.bldgs[(gv.bldgs[:, 3] == gv.zones[z, 0]) & (gv.bldgs[:, 1] > 1)]) * 1.
        gv.zones[:, 4] = np.sum(gv.bldgs[(gv.bldgs[:, 3] == gv.zones[z, 0]) & (gv.bldgs[:, 1] == 1), 12])
    
    # create jobs - data structure: wage, location, employee, id
    jobs_num = np.round_((np.choose(gv.bldgs[:, 1].astype(int), (gv.bldgs[:, np.newaxis, 4] * np.array(mp.jobs_per_m)
                                                                 ).transpose())).tolist(), 0).astype(int)
    gv.jobs = np.array([[np.random.normal(mp.avgIncome, mp.stdIncome), i, np.nan] for i in range(
        len(jobs_num)) for j in range(jobs_num[i])])
    while len(gv.jobs[gv.jobs[:, 0] <= 0]) > 0: 
        gv.jobs[gv.jobs[:, 0] <= 0, 0] = np.random.normal(mp.avgIncome, mp.stdIncome, len(gv.jobs[gv.jobs[:, 0] <= 0]))
    gv.jobs = np.concatenate((gv.jobs, np.array(range(len(gv.jobs)))[np.newaxis, :].T), axis=1)
    
    # compute expected income
    for h in gv.households: 
        members = gv.indivs[:, 1] == h[0]
        workers = gv.indivs[:, 3] == 1
        if len(gv.indivs[members & workers]) > 0:
            gv.indivs[members & workers, 8] = h[2] / len(gv.indivs[members & workers])
        else:
            gv.indivs[members & workers, 8] = 0
    del members, workers
    
    # match individuals to workplaces
    print 'finding workplaces', time() - t
    local_workers = np.where(gv.indivs[:, 5] == 1)[0]
    av_jobs = np.where(np.isnan(gv.jobs[:, 2]))[0]
    while len(local_workers) > 0 and len(av_jobs) > 0:
        c = random.choice(local_workers)
        j = av_jobs[np.argmin(np.abs(gv.indivs[c, 8] - gv.jobs[av_jobs, 0]))]
        home_junc = gv.bldgs[gv.bldgs[:, 0] == gv.households[gv.households[:, 0] == gv.indivs[c, 1], 1], -2].tolist()
        com_dist = gv.dists[home_junc][0][gv.bldgs[int(gv.jobs[j, 1]), -2]]
        max_dist = np.max(np.ma.masked_invalid(gv.dists[home_junc][0])) 
        rel_wage = (gv.jobs[j, 0] - np.min(gv.jobs[av_jobs, 0])) / (np.max(gv.jobs[av_jobs, 0]) - np.min(gv.jobs[av_jobs, 0]))
        pref = (com_dist / max_dist + rel_wage) / 2.
        gv.indivs[c, [7, 8, 9, 11]] = [gv.jobs[j, -1], gv.jobs[j, 0], pref, gv.jobs[j, 0]]
        gv.jobs[j, 2] = gv.indivs[c, 0]
        local_workers = np.where((gv.indivs[:, 5] == 1) & (np.isnan(gv.indivs[:, 7])))[0]
        av_jobs = np.where(np.isnan(gv.jobs[:, 2]))[0]
    del c, j, home_junc, com_dist, max_dist, rel_wage, pref, local_workers, av_jobs
    
    gv.indivs[(gv.indivs[:, 5] == 1) & (np.isnan(gv.indivs[:, 7])), 5:7] = [0, 0]
    gv.indivs[(gv.indivs[:, 3] == 1) & (gv.indivs[:, 5] == 0), 11] = gv.indivs[(gv.indivs[:, 3] == 1) & (gv.indivs[:, 5] == 0), 8]
    
    # update household income
    for h in range(len(gv.households)): 
        gv.households[h, 2] = np.sum(gv.indivs[gv.indivs[:, 1] == gv.households[h, 0], -1])
    del h
    
    # unoccupied jobs are occupied by in-commuter
    gv.jobs[np.isnan(gv.jobs[:, 2]), 2] = 0
    
    # create routines
    print 'creating routines', time() - t
    for i in range(len(gv.indivs)):
        create_routines(i)
    
    mp.worker_residing_outside = len(gv.jobs[gv.jobs[:, 2] == 0, 3]) * 1. / len(gv.jobs)
    
    print 'computing building values', time() - t
    for b in range(len(gv.bldgs)):
        compute_building_value(b, False)
    ap_vals = gv.bldgs[gv.bldgs[:, 1] <= 1, 10] / gv.bldgs[gv.bldgs[:, 1] <= 1, 12]
    ap_vals_r = np.repeat(ap_vals, gv.bldgs[gv.bldgs[:, 1] <= 1, 12].astype(int))
    gv.stdResVal = np.std(ap_vals_r, ddof=1)
    med_wtp = np.median(gv.households[:, 2]) / 3.
    gv.bldgs[gv.bldgs[:, 1] <= 1, -1] = med_wtp * (1 + (ap_vals - np.mean(ap_vals_r)) / (gv.stdResVal * 12.))
    
    save_init_snapshot()
    
    for b in gv.bldgs: 
        gv.bldgs_values[b[0]] = [b[10]]
    for z in gv.zones: 
        gv.zones_hps[z[0]] = [z[3]]
    for r in gv.roads: 
        gv.rds_civs[r[0]] = []
    
    print 'world created', time() - t
