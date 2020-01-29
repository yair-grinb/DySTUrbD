import networkx as nx
import global_variables as gv
import numpy as np
from random import random


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


def get_path(o, d):
    r = gv.routes[gv.junctions[o]][gv.junctions[d]]
    route = [np.where(gv.roads[:, 0] == gv.graph[r[i]][r[i + 1]]['fid'])[0] for i in range(len(r) - 1)]
    return route


def create_routines(i):
    path = []
    hh = gv.households[gv.households[:, 0] == gv.indivs[i, 1].astype(int)][0]
    activities = int(round((3. - (gv.indivs[i, 3] == 1) * 0.5) * random() * (1 + (
            hh[3] - (gv.indivs[i, 4] != 2) - gv.indivs[i, 2]) / 3.) * 2. + gv.indivs[i, 5], 0))
    
    if activities < 0:
        activities = 0
    
    loc = np.where(gv.bldgs[:, 0] == hh[1])[0][0]
    for j in range(activities):
        dest = np.nan
        if j == 0 and gv.indivs[i, 5] == 1:
            job = gv.jobs[gv.jobs[:, -1] == gv.indivs[i, 7]][0]
            if ~np.isinf(gv.dists[gv.bldgs[loc, 13]][gv.bldgs[int(job[1]), 13]]):
                dest = int(job[1])
        if np.isnan(dest):
            pref = random()
            pot_bldgs = gv.bldgs[(gv.bldgs[:, 0] != gv.bldgs[loc, 0]) & (gv.bldgs[:, 1] != 0) & (
                ~np.isinf(gv.dists[gv.bldgs[loc, 13]][gv.bldgs[:, 13].tolist()]))]
            if len(pot_bldgs) > 0:
                max_dist = np.max(np.ma.masked_invalid(gv.dists[gv.bldgs[loc, 13]]))
                loc_dists = gv.dists[gv.bldgs[loc, 13]][pot_bldgs[:, 13].tolist()]
                loc_dists = loc_dists / max_dist
                loc_dists = 1. - loc_dists * (1 + (hh[3] - (gv.indivs[i, 4] != 2) - gv.indivs[i, 2]) / 3.)
                fs_scores = pot_bldgs[:, 4] / np.max(gv.bldgs[gv.bldgs[:, 1] > 1, 4])
                fs_scores = 1. - fs_scores
                fs_scores = (pot_bldgs[:, 1] > 1) * fs_scores
                scores = (pot_bldgs[:, 11] + loc_dists + fs_scores) / (2. + (pot_bldgs[:, 1] > 1))
                if np.any(pref > scores):
                    dest_idx = np.random.choice(len(pot_bldgs[pref > scores]))
                    b_dest = pot_bldgs[dest_idx]
                    dest = int(np.where(gv.bldgs[:, 0] == b_dest[0])[0][0])
        if ~np.isnan(dest):
            path.extend(get_path(gv.bldgs[loc, -2], gv.bldgs[dest, -2]))
            loc = dest
    dest = np.where(gv.bldgs[:, 0] == hh[1])[0][0]
    if ~np.isinf(gv.dists[gv.bldgs[loc, 13]][gv.bldgs[dest, 13]]):
        path.extend(get_path(gv.bldgs[loc, -2], gv.bldgs[dest, -2]))
    gv.routines[gv.indivs[i, 0]] = path


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
