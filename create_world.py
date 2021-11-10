import csv
import global_variables as gv
import numpy as np
from create_random_data import create_data
import model_parameters as mp
from auxilliary_functions import create_routines, compute_building_value, create_social_network # , compute_network
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
    del reader, e, row, 
    return np.array(entities)


def create_world():
    t = time()
    # create zones - data structure: id, inMig, inMigCh, HPrice, prevPop, prevRes, prevNRes
    print ('getting zones', time() - t)
    # statistical areas
    gv.zones = create_agent_class(gv.zonesFile, range(4), [0, 0, 0])
    print ('getting individuals, households, buildings and jobs', time() - t)
    gv.indivs, gv.households, gv.bldgs, gv.jobs = create_data('data/civ_withCar_bldg_np.csv', 'data/bldg_with_inst_orig.csv')
    
    # TODO: Remove once new land use change mechanism is inserted    

    gv.visits_hist = np.array([[] for i in range(gv.bldgs.shape[0])])
    #TODO - compare with create random data    
    # create individuals - data structure: id, hh, dis, employed, age, employed locally, workforce participation, job, expected income,
    # workplace preference, search length, income
    
    # gv.indivs = create_agent_class(gv.indivsFile, [0, 1, 2, 3, 4, 5, 3], [np.nan, 0, random.random(), 0, 0])
    # gv.indivs[(gv.indivs[:, 3] == 0) & (gv.indivs[:, 5] == 1), 3] = 1
    # gv.indivs[(gv.indivs[:, 3] == 1) & (gv.indivs[:, 6] == 0), 6] = 1

    
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
    mp.carChance = len(gv.households[gv.households[:, 3] == 1]) * 1. / len(gv.households)
    
    # # compute number of apartments
    for i in range(gv.bldgs.shape[0]):
        residents = np.size(gv.households[gv.households[:, 1] == gv.bldgs[i, 0]][:, 1])
        gv.bldgs[i, 13] = residents
    del i, residents
    
    # # correct floors
    gv.bldgs[np.where(gv.bldgs[:, 2] == 0), 2] = 1
    
    # residential with no households to empty
    gv.bldgs[(gv.bldgs[:, 13] == 0) & (gv.bldgs[:, 1] == 1), 1] = 0
    
    # compute potential apartments for non-residential buildings
    gv.bldgs[gv.bldgs[:, 13] != 1, 13] = (gv.bldgs[gv.bldgs[:, 1] != 1, 4] / 90.).astype(int)
    gv.bldgs[gv.bldgs[:, 13] < 1, 13] = 1
    
    # init visits history array
    gv.bldgs_visits_dist = np.zeros(gv.bldgs.shape[0])
    
    # update zone data
    for z in range(len(gv.zones)):
        gv.zones[z, 5] = len(gv.bldgs[(gv.bldgs[:, 3] == gv.zones[z, 0]) & (gv.bldgs[:, 1] == 1)]) * 1.
        gv.zones[z, 6] = len(gv.bldgs[(gv.bldgs[:, 3] == gv.zones[z, 0]) & (gv.bldgs[:, 1] > 1)]) * 1.
        gv.zones[:, 4] = np.sum(gv.bldgs[(gv.bldgs[:, 3] == gv.zones[z, 0]) & (gv.bldgs[:, 1] == 1), 13])
    
    #TODO - compare with create random data
    # create jobs - data structure: wage, location, employee, id
    # jobs_num = np.round_((np.choose(gv.bldgs[:, 1].astype(int), (gv.bldgs[:, np.newaxis, 4] * np.array(mp.jobs_per_m)
    #                                                              ).transpose())).tolist(), 0).astype(int)
    # gv.jobs = np.array([[np.random.normal(mp.avgIncome, mp.stdIncome), i, np.nan] for i in range(
    #     len(jobs_num)) for j in range(jobs_num[i])])
    # while len(gv.jobs[gv.jobs[:, 0] <= 0]) > 0: 
    #     gv.jobs[gv.jobs[:, 0] <= 0, 0] = np.random.normal(mp.avgIncome, mp.stdIncome, len(gv.jobs[gv.jobs[:, 0] <= 0]))
    # gv.jobs = np.concatenate((gv.jobs, np.array(range(len(gv.jobs)))[np.newaxis, :].T), axis=1)
    
    # # compute expected income
    # for h in gv.households: 
    #     members = gv.indivs[:, 1] == h[0]
    #     workers = gv.indivs[:, 3] == 1
    #     if len(gv.indivs[members & workers]) > 0:
    #         gv.indivs[members & workers, 8] = h[2] / len(gv.indivs[members & workers])
    #     else:
    #         gv.indivs[members & workers, 8] = 0
    # del members, workers
    
    # match individuals to workplaces
    # print ('finding workplaces', time() - t)
    # local_workers = np.where(gv.indivs[:, 5] == 1)[0]
    # av_jobs = np.where(np.isnan(gv.jobs[:, 2]))[0]
    # while len(local_workers) > 0 and len(av_jobs) > 0:
    #     c = random.choice(local_workers)
    #     j = av_jobs[np.argmin(np.abs(gv.indivs[c, 8] - gv.jobs[av_jobs, 0]))]
    #     home_junc = gv.bldgs[gv.bldgs[:, 0] == gv.households[gv.households[:, 0] == gv.indivs[c, 1], 1], -2].tolist()
    #     com_dist = gv.dists[home_junc][0][gv.bldgs[int(gv.jobs[j, 1]), -2]]
    #     max_dist = np.max(np.ma.masked_invalid(gv.dists[home_junc][0])) 
    #     rel_wage = (gv.jobs[j, 0] - np.min(gv.jobs[av_jobs, 0])) / (np.max(gv.jobs[av_jobs, 0]) - np.min(gv.jobs[av_jobs, 0]))
    #     pref = (com_dist / max_dist + rel_wage) / 2.
    #     gv.indivs[c, [7, 8, 9, 11]] = [gv.jobs[j, -1], gv.jobs[j, 0], pref, gv.jobs[j, 0]]
    #     gv.jobs[j, 2] = gv.indivs[c, 0]
    #     local_workers = np.where((gv.indivs[:, 5] == 1) & (np.isnan(gv.indivs[:, 7])))[0]
    #     av_jobs = np.where(np.isnan(gv.jobs[:, 2]))[0]
    # del c, j, home_junc, com_dist, max_dist, rel_wage, pref, local_workers, av_jobs
    
    # gv.indivs[(gv.indivs[:, 5] == 1) & (np.isnan(gv.indivs[:, 7])), 5:7] = [0, 0]
    # gv.indivs[(gv.indivs[:, 3] == 1) & (gv.indivs[:, 5] == 0), 11] = gv.indivs[(gv.indivs[:, 3] == 1) & (gv.indivs[:, 5] == 0), 8]
    
    # # update household income
    # for h in range(len(gv.households)): 
    #     gv.households[h, 2] = np.sum(gv.indivs[gv.indivs[:, 1] == gv.households[h, 0], -1])
    # del h
    
    # # unoccupied jobs are occupied by in-commuter
    # gv.jobs[np.isnan(gv.jobs[:, 2]), 2] = 0
    
    print ('creating social network', time() - t)
    gv.graph = create_social_network(gv.indivs, gv.households, gv.bldgs)
    agents_reg = gv.indivs[:, [0,7,12,18,7]]        
   
    # create routines
    print ('creating routines', time() - t)
    gv.routines = dict(zip(gv.indivs[:, 0], create_routines(agents_reg)))
    mp.worker_residing_outside = len(gv.jobs[gv.jobs[:, 2] == 0, 3]) * 1. / len(gv.jobs)
    
    print ('computing building values', time() - t)
    for b in range(len(gv.bldgs)):
        compute_building_value(b, False)
    ap_vals = gv.bldgs[gv.bldgs[:, 1] <= 1, 11] / gv.bldgs[gv.bldgs[:, 1] <= 1, 13]
    ap_vals_r = np.repeat(ap_vals, gv.bldgs[gv.bldgs[:, 1] <= 1, 13].astype(int))
    gv.stdResVal = np.std(ap_vals_r, ddof=1)
    med_wtp = np.median(gv.households[:, 2]) / 3.
    gv.bldgs[gv.bldgs[:, 1] <= 1, 14] = med_wtp * (1 + (ap_vals - np.mean(ap_vals_r)) / (gv.stdResVal * 12.))
    
    save_init_snapshot()
    
    for b in gv.bldgs: 
        gv.bldgs_values[b[0]] = [b[11]]
    for z in gv.zones: 
        gv.zones_hps[z[0]] = [z[3]]
    
    gv.bld_dists = ((gv.bldgs[:, np.newaxis, 6] - gv.bldgs[:, 6]) ** 2. + (gv.bldgs[:, np.newaxis, 7] - gv.bldgs[:, 7]) ** 2.) ** 0.5
    
    print ('world created', time() - t)
    
