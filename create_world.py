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
    
    print ('creating social network', time() - t)
    gv.graph = create_social_network(gv.indivs, gv.households, gv.bldgs)
    gv.agents_reg = gv.indivs[:, [0,7,12,18,7]]        
   
    # create routines
    print ('creating routines', time() - t)
    gv.routines = dict(zip(gv.indivs[:, 0], create_routines(gv.agents_reg)))
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
    
