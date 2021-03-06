import numpy as np
import global_variables as gv
import model_parameters as mp
import random
from auxilliary_functions import compute_network, create_routines, compute_building_value
import math
from scipy.stats import norm, percentileofscore
from time import time
import gc


def leave_town(h_idx, b_idx):
    hh_jobs = np.where(gv.jobs[:, -1] == gv.indivs[(gv.indivs[:, 1] == gv.households[h_idx, 0]) & (gv.indivs[:, 5] == 1), 7])[0]
    gv.jobs[hh_jobs, 2] = np.nan
    gv.jobs[hh_jobs][np.random.rand(len(hh_jobs)) <= mp.worker_residing_outside, 2] = 0
    
    for j in hh_jobs:
        # output - 'j', tick, id, bldg id old, bldg id new, worker id old, worker id new, wage
        gv.data.append(['j', gv.tick, gv.jobs[j, -1], gv.bldgs[b_idx, 0], gv.bldgs[b_idx, 0],
                        gv.indivs[gv.indivs[:, 7] == gv.jobs[j, -1], 0], gv.jobs[j, 2], gv.jobs[j, 0]])
    
    # delete agents
    members = np.where(gv.indivs[:, 1] == gv.households[h_idx, 0])[0]
    for c in members:
        # output - 'i', tick, id, hh id, workforce participation old, workforce participation new, employed old,
        # employed new, employed locally old, employed locally new, income old, income new, job old, job new
        gv.data.append(['i', gv.tick, gv.indivs[c, 0], h_idx, gv.indivs[c, 6], 0,
                        gv.indivs[c, 3], 0, gv.indivs[c, 5], 0, gv.indivs[c, 11], 0, gv.indivs[c, 7], np.nan])
    gv.indivs = np.delete(gv.indivs, members, 0)
    
    # output - 'h', tick, id, building id old, building id new
    gv.data.append(['h', gv.tick, gv.households[h_idx, 0], gv.bldgs[b_idx, 0], np.nan])
    # delete household
    gv.households = np.delete(gv.households, h_idx, 0)


def compute_diff(hh, b):
    soc_diff = None
    dist_score = None
    
    members = gv.indivs[gv.indivs[:, 1] == hh[0]]
    neigh = gv.bldgs[np.where((gv.bldgs[:, 6] - b[6])**2. + (gv.bldgs[:, 7] - b[7])**2. <= 100.**2.)[0], 0]
    hhs_in_neigh = gv.households[np.in1d(gv.households[:, 1], neigh)]
    if len(hhs_in_neigh) > 0:
        mean_inc = np.mean(hhs_in_neigh[:, 2])
        std_inc = np.std(hhs_in_neigh[:, 2], ddof=1)
        is_in_neigh = gv.indivs[np.in1d(gv.indivs[:, 1], hhs_in_neigh)]
        mean_age = np.mean(is_in_neigh[:, 4])
        std_age = np.std(is_in_neigh[:, 4], ddof=1)
        diff_age = norm.pdf((np.mean(members[:, 4]) - mean_age) / std_age, 0, 1)
        diff_inc = norm.pdf((hh[2] - mean_inc) / std_inc, 0, 1)
        soc_diff = (diff_age / norm.pdf(0, 0, 1) + diff_inc / norm.pdf(0, 0, 1)) / 2.    
    
    members_jobs = members[members[:, 5] == 1, 7]
    if len(members_jobs) > 0:
        jobs_juncs = gv.bldgs[gv.jobs[np.in1d(gv.jobs[:, -1], members_jobs), 1].astype(int), 13]
        mean_commute = np.mean(np.ma.masked_invalid(gv.dists[b[13]][jobs_juncs.astype(int)]))
        dist_score = mean_commute / np.max(np.ma.masked_invalid(gv.dists[b[13]]))
    
    return soc_diff, dist_score


def find_new_home(h_idx, in_mig):
    found = False
    
    hh_counts = np.array([len(gv.households[gv.households[:, 1] == b[0]]) for b in gv.bldgs])
    choice_set = np.where(((gv.bldgs[:, 1] == 0) & (gv.bldgs[:, 9] == 0) & (gv.bldgs[:, 5] != 4)) |
                          ((gv.bldgs[:, 1] == 1) & (gv.bldgs[:, 0] != gv.households[h_idx, 1]) &
                           (hh_counts < gv.bldgs[:, 12])) & (gv.bldgs[:, 14] <= gv.households[h_idx, 2] * mp.wtp))[0]
    if len(choice_set) > 0:
        i = 0
        pref = None
        soc_diff = None
        dist_score = None
        
        if not in_mig:
            # compute preferences
            soc_diff, dist_score = compute_diff(gv.households[h_idx], gv.bldgs[gv.bldgs[:, 0] == gv.households[h_idx, 1]][0])
            if soc_diff is None:
                soc_diff = random.random()
            if dist_score is None:
                dist_score = random.random()
            pref = (soc_diff + dist_score) / 2
        else:
            pref = random.random()
        
        while i < 100 and len(choice_set) > 0:
            bi = random.choice(range(len(choice_set)))
            b = choice_set[bi]
            soc_diff2, dist_score2 = compute_diff(gv.households[h_idx], gv.bldgs[b])
            if soc_diff2 is None:
                soc_diff2 = 0.5
            if dist_score2 is None:
                dist_score2 = 0.5
            score = (soc_diff2 + dist_score2) / 2.
            
            if score >= pref:
                if not in_mig:
                    # output - 'h', tick, id, building id old, building id new
                    gv.data.append(['h', gv.tick, gv.households[h_idx, 0], gv.households[h_idx, 1], gv.bldgs[b, 0]])
                gv.households[h_idx, 1] = gv.bldgs[b, 0]
                return True
            i += 1
            choice_set = np.delete(choice_set, bi, 0)
        
        if not found and not in_mig:
            leave_town(h_idx, np.where(gv.bldgs[:, 0] == gv.households[h_idx, 1])[0])
    
    return found


def change_lu(b, old, new):
    gv.bldgs[b, 1] = new
    
    if old == 1:
        # determine if residents leave or relocate
        residents = np.where(gv.households[gv.households[:, 1] == gv.bldgs[b, 0]])[0]
        rands = np.random.rand(len(residents))
        
        for r in range(len(residents)):
            if rands[r] < 0.5:
                leave_town(residents[r], b)
            else:
                found = find_new_home(residents[r], False)
                # if found a home
                if found:
                    # if new home is unoccupied - change land use
                    home_idx = np.where(gv.bldgs[:, 0] == gv.households[residents[r], 1])[0][0]
                    if gv.bldgs[home_idx, 1] == 0:
                        change_lu(home_idx, 0, 1)
                        # output - 'b', tick, id, old lu, new lu, old apartments, new apartments
                        gv.data.append(['b', gv.tick, gv.bldgs[home_idx, 0], 0, 1, 0, gv.bldgs[home_idx, 12]])

    if old != 0:
        # make agents employed in building unemployed
        jobs_in_b = np.where(gv.jobs[:, 1] == b)[0]
        if len(jobs_in_b) > 0:
            emp_in_b = np.where(np.in1d(gv.indivs[:, 7], gv.jobs[jobs_in_b, -1]))[0]
            if len(emp_in_b) > 0:
                for e in emp_in_b:
                    # output - 'i', tick, id, hh id, workforce participation old, workforce participation new, employed,
                    # old, employed new, employed locally old, employed locally new, income old, income new, job old,
                    # job new
                    gv.data.append(['i', gv.tick, gv.indivs[e, 0], gv.households[gv.households[:, 0] == gv.indivs[e, 1], 0], 
                                    1, 1, 1, 0, 1, 0, gv.indivs[e, 11], 0., gv.indivs[e, 7], np.nan])
                    create_routines(e)
                gv.indivs[emp_in_b[:, None], [3, 5, 7, 11]] = np.repeat([[0, 0, np.nan, 0]], len(emp_in_b), axis=0)
            for j in jobs_in_b:
                # output - 'j', tick, id, bldg id old, bldg id new, worker id old, worker id new, wage
                gv.data.append(['j', gv.tick, gv.jobs[j, -1], gv.bldgs[b, 0], np.nan, gv.jobs[j, 2], np.nan, gv.jobs[j, 0]])
            gv.jobs = np.delete(gv.jobs, jobs_in_b, 0)
    
    if new != 0:
        # create new jobs
        jobs_num = int(math.ceil(mp.jobs_per_m[int(new)] * gv.bldgs[b, 4]))
        new_jobs = np.array([[np.random.normal(mp.avgIncome, mp.stdIncome), b, np.nan] for i in range(jobs_num)])
        while np.any(new_jobs[:, 0] <= 0):
            new_jobs[new_jobs[:, 0] <= 0, 0] = np.random.normal(mp.avgIncome, mp.stdIncome, len(new_jobs[new_jobs[:, 0] <= 0]))
        # allocate in commuters to new jobs
        new_jobs[np.random.rand(len(new_jobs)) <= mp.worker_residing_outside, 2] = 0
        # define ids for new jobs
        job_id = int(np.max(gv.jobs[:, -1])) + 1
        new_jobs = np.concatenate((new_jobs, np.array(range(job_id, job_id + len(new_jobs)))[np.newaxis, :].T), axis=1)
        gv.jobs = np.concatenate([gv.jobs, new_jobs])
        for j in new_jobs:
            # output - 'j', tick, id, bldg id old, bldg id new, worker id old, worker id new, wage
            gv.data.append(['j', gv.tick, j[-1], np.nan, gv.bldgs[b, 0], np.nan, j[2], j[0]])
    
    if new <= 1:
        if gv.bldgs[b, 12] == 0:
            gv.bldgs[b, 12] = math.floor(gv.bldgs[b, 4] / 90.)
        if gv.bldgs[b, 12] == 0:
            gv.bldgs[b, 12] = 1
    
    if (new in [0, 1]) and (old not in [0, 1]):
        compute_building_value(b, True)
    
    gv.changedLU = True


def create_event():
    # determine epicenter
    center = random.choice(gv.junctions)
    
    # identify effected buildings
    e_dists = ((gv.bldgs[:, 6] - center[0])**2. + (gv.bldgs[:, 7] - center[1])**2.)**0.5
    rand = np.random.rand(len(gv.bldgs))
    effect_scores = e_dists * 10.**mp.intensity / (mp.event_m * gv.bldgs[:, 2] * np.abs(np.log10(e_dists.astype(float))))
    gv.bldgs[rand <= effect_scores, 9] = 1.
    
    # compute secondary effects - roads, residents, jobs
    for b in np.where(gv.bldgs[:, 9] == 1)[0]:
        # output - 'b', tick, id, old lu, new lu, old apartments, new apartments
        gv.data.append(['b', gv.tick, gv.bldgs[b, 0], gv.bldgs[b, 1], 0, gv.bldgs[:, 12], 0])
        
        # find effected roads
        near_j = gv.junctions_array[gv.bldgs[b, 13]]
        nearby_roads = (((gv.roads_juncs[:, 0] == near_j)[:, 0]) | (gv.roads_juncs[:, 1] == near_j)[:, 1])
        for r in gv.roads[nearby_roads]:
            if gv.graph.has_edge((r[1], r[2]), (r[3], r[4])):
                gv.graph.remove_edge((r[1], r[2]), (r[3], r[4]))
            # output - 'r', tick, id, previous_state, new_state
            gv.data.append(['r', gv.tick, r[0], 1, 0])
        
        # make roads unavailable
        gv.roads[nearby_roads, 6] = 0
    
    compute_network()

    for b in np.where(gv.bldgs[:, 9] == 1)[0]:
        # change building land-use
        change_lu(b, gv.bldgs[b, 1], 0)

    # compute new routines
    for i in range(gv.indivs.shape[0]):
        create_routines(i)
    
    gv.changedLU = False


def hh_step(h_idx):
    gv.households[h_idx, 2] = np.nansum(gv.indivs[gv.indivs[:, 1] == gv.households[h_idx, 0], 11])
    if random.random() < mp.outMigChance:
        leave_town(h_idx, np.where(gv.bldgs[gv.bldgs[:, 0] == gv.households[h_idx, 1]])[0])
        return True
    elif random.random() < gv.zones[gv.zones[:, 0] == gv.bldgs[gv.bldgs[:, 0] == gv.households[h_idx, 1], 3], 2]:
        found = find_new_home(h_idx, False)
        if found:
            if gv.bldgs[gv.bldgs[:, 0] == gv.households[h_idx, 1], 1] == 0:
                change_lu(np.where(gv.bldgs[:, 0] == gv.households[h_idx, 1])[0][0], 0, 1)
                gv.data.append(['b', gv.tick, gv.households[h_idx, 1], 0, 1, 0,
                                gv.bldgs[gv.bldgs[:, 0] == gv.households[h_idx, 1], 12]])
            for i in np.where(gv.indivs[:, 1] == gv.households[h_idx, 0])[0]:
                create_routines(i)
            return False
        else:
            return True
    return False


def find_wp(i):
    if "labor_market" not in mp.lags or gv.tick < mp.events_set[0] or ~np.any(
            [mp.events_set[k] <= gv.tick < mp.events_set[k] + mp.lags['labor_market'] for k in range(len(mp.events_set))]):
        av_jobs = np.where(np.isnan(gv.jobs[:, 2]).astype(float))[0]
        home = gv.bldgs[gv.bldgs[:, 0] == gv.households[gv.households[:, 0] == gv.indivs[i, 1], 1]][0]
        max_d = np.max(np.ma.masked_invalid(gv.dists[home[13]]))
        min_w = np.min(gv.jobs[av_jobs, 0])
        max_w = np.max(gv.jobs[av_jobs, 0])
        for k in range(min(7, len(av_jobs))):
            j_ix = random.choice(range(len(av_jobs)))
            j = av_jobs[j_ix]
            d = gv.dists[home[13]][gv.bldgs[gv.jobs[j, 1].astype(int), 13]]
            score = 1. - d / max_d
            score += (gv.jobs[j, 0] - min_w) / (max_w - min_w + 0.000001)
            score = score / 2.
            if score >= gv.indivs[i, 9]:
                # output - 'i', tick, id, hh id, workforce participation old, workforce participation new, employed old,
                # employed new, employed locally old, employed locally new, income old, income new, job old, job new
                gv.data.append(['i', gv.tick, gv.indivs[i, 0], gv.indivs[i, 1], 1, 1, 0, 1, 0, 1, 0, gv.jobs[j, 0],
                                np.nan, gv.jobs[j, -1]])
                gv.indivs[i, [3, 5, 7, 8, 9, 10, 11]] = [1, 1, gv.jobs[j, -1], gv.jobs[j, 0], score, 0., gv.jobs[j, 0]]
                # output - 'j', tick, id, bldg id old, bldg id new, worker id old, worker id new, wage
                gv.data.append(['j', gv.tick, gv.jobs[j, -1], gv.bldgs[gv.jobs[j, 1].astype(int), 0],
                                gv.bldgs[gv.jobs[j, 1].astype(int), 0], np.nan, gv.indivs[i, 0], gv.jobs[j, 0]])
                gv.jobs[j, 2] = gv.indivs[i, 0]
                create_routines(i)
                break
            av_jobs = np.delete(av_jobs, j_ix, 0)

        if np.isnan(gv.indivs[i, 7]):
            gv.indivs[i, 10] += 1
            if 1 - math.exp(-gv.indivs[i, 10] / mp.search_length) > random.random():
                # output - 'i', tick, id, hh id, workforce participation old, workforce participation new, employed old,
                # employed new, employed locally old, employed locally new, income old, income new, job old, job new
                gv.data.append(['i', gv.tick, gv.indivs[i, 0], gv.indivs[i, 1], 1, 1, 0, 1, 0, 0, 0, gv.indivs[i, 8], np.nan, 0])
                gv.indivs[i, [3, 5, 7, 10, 11]] = [1, 0, 0, 0, gv.indivs[i, 8]]
                create_routines(i)
            elif 1 - math.exp(-gv.indivs[i, 10] / mp.search_length) > random.random():
                gv.data.append(['i', gv.tick, gv.indivs[i, 0], gv.indivs[i, 1], 1, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan])
                gv.indivs[i, [6, 10]] = [0, 0]


def i_step(i):
    if gv.indivs[i, 6] == 1 and gv.indivs[i, 3] == 0:
        if len(gv.jobs[np.isnan(gv.jobs[:, 2].astype(float))]) > 0:
            find_wp(i)
        else:
            gv.indivs[i, 10] += 1


def road_step(r):
    if gv.roads[r, 6] == 0:
        nearby_bldgs = gv.bldgs[np.all(
            (gv.junctions_array[gv.bldgs[:, 13].astype(int)] == gv.roads_juncs[r, 0]), axis=1) |
                                np.all((gv.junctions_array[gv.bldgs[:, 13].astype(int)] == gv.roads_juncs[r, 1]), axis=1), 9]
        if not np.any(nearby_bldgs):
            gv.roads[r, 6] = 1
            # output - 'r', tick, id, previous_state, new_state
            gv.data.append(['r', gv.tick, gv.roads[r, 0], 0, 1])
            gv.graph.add_edge((gv.roads[r, 1], gv.roads[r, 2]), (gv.roads[r, 3], gv.roads[r, 4]), weight=gv.roads[r, 5],
                              fid=gv.roads[r, 0])
            return True
    return False


def bldg_step(b):
    if gv.bldgs[b, 9] == 1 and gv.bldgs[b, 4] / (gv.bldgs[b, 2] * 5.) <= gv.bldgs[b, 8]:
        gv.bldgs[b, 9] = 0
        if gv.bldgs[b, 5] == 4:
            gv.data.append(['b', gv.tick, gv.bldgs[b, 0], 0, 4, 0, 0])
            change_lu(b, 0, 4)
    
    if gv.tick > 30:
        z_traff = percentileofscore(gv.bldgs_traff_dist, gv.bldgs_traff_dist[b])
        z_fs = percentileofscore(gv.bldgs[gv.bldgs[:, 1] == 3, 4], gv.bldgs[b, 4])
        score = z_traff - z_fs
        
        if gv.bldgs[b, 1] < 2 and gv.bldgs[b, 9] == 0 and 20 < score < 40:
            # output - 'b', tick, id, old lu, new lu, old apartments, new apartments
            gv.data.append(['b', gv.tick, gv.bldgs[b, 0], gv.bldgs[b, 1], 3, (gv.bldgs[:, 1] == 1) * (gv.bldgs[:, 12]), 0])
            change_lu(b, gv.bldgs[b, 1], 3)
        elif gv.bldgs[b, 1] == 3 and -40 < score < -20:
            gv.data.append(['b', gv.tick, gv.bldgs[b, 0], 3, 0, 0, 0])
            change_lu(b, 3, 0)
    
    if gv.bldgs[b, 1] == 1 and len(gv.households[gv.households[:, 1] == gv.bldgs[b, 0]]) == 0:
        gv.data.append(['b', gv.tick, gv.bldgs[b, 0], 1, 0, gv.bldgs[:, 12], 0])
        change_lu(b, 1, 0)
    
    compute_building_value(b, True)


def in_migration():
    immig = np.sum(gv.bldgs[gv.bldgs[:, 1] == 1, 12]) - len(gv.households)
    immig = int(round(immig * np.random.normal(mp.inMigChance, 1. - mp.inMigChance), 0))
    
    h_id = np.max(gv.households[:, 0]) + 1
    for i in range(immig):
        hh = [h_id, np.nan, np.random.normal(np.mean(gv.households[:, 2]), np.std(gv.households[:, 2], ddof=1)),
              1. * (random.random() < mp.carChance)]
        while hh[1] <= 0:
            hh[1] = np.random.normal(np.mean(gv.households[:, 2]), np.std(gv.households[:, 2], ddof=1))
        gv.households = np.append(gv.households, [hh], axis=0)
        civ_id = np.max(gv.indivs[:, 0]) + 1
        c_num = int(round(np.random.normal(mp.hhSize, mp.hhSTD), 0))
        if c_num <= 0:
            c_num = 1
        for j in range(c_num):
            k = random.random()
            ind = [civ_id, h_id, 1. * (random.random() < mp.disChance), 0,
                   3. - 1 * (k < mp.age2chance) - 1 * (k < mp.age1chance), 0, 0, np.nan, 0, random.random(), 0, 0]
            if j == 0:
                ind[3:8] = [1, 3. - 1. * (k < mp.age2chance), ind[5], 1, 0]
            elif ind[4] > 1:
                ind[3] = 1. * (random.random() > mp.employChance)
                if ind[3] == 1:
                    ind[5:7] = [1. * (random.random() <= mp.workInChance), 1]
            gv.indivs = np.append(gv.indivs, [ind], axis=0)
            civ_id += 1
        
        gv.indivs[(gv.indivs[:, 1] == h_id) & (gv.indivs[:, 3] == 1), 8] = hh[2] / float(
            len(gv.indivs[(gv.indivs[:, 1] == h_id) & (gv.indivs[:, 3] == 1), 8]))
        found = find_new_home(-1, True)
        
        if not found:
            gv.indivs = np.delete(gv.indivs, np.where(gv.indivs[:, 1] == h_id), axis=0)
            gv.households = np.delete(gv.households, len(gv.households) - 1, axis=0)
        else:
            gv.data.append(['h_im', gv.tick] + list(gv.households[-1]))
            if gv.bldgs[gv.bldgs[:, 0] == gv.households[-1, 1], 1] == 0:
                change_lu(np.where(gv.bldgs[:, 0] == gv.households[-1, 1])[0][0], 0, 1)
                gv.data.append(['b', gv.tick, gv.households[-1, 1], 0, 1, 0, gv.bldgs[gv.bldgs[:, 0] == gv.households[-1, 1], 12]])
                
            for k in np.where(gv.indivs[:, 1] == gv.households[-1, 0])[0]:
                if gv.indivs[i, 3] == 1 and gv.indivs[k, 5] == 0:
                    gv.indivs[k, 11] = gv.indivs[k, 8]
                else:
                    if len(gv.jobs[np.isnan(gv.jobs[:, 2].astype(float))]) > 0:
                        find_wp(k)
                    if np.isnan(gv.indivs[k, 7]):
                        gv.indivs[k, [3, 5]] = [0, 0]
                    elif gv.indivs[k, 7] == 0:
                        gv.indivs[k, 5] = 0
                create_routines(k)
                gv.data.append(['i_im', gv.tick] + list(gv.indivs[k]))
            gv.households[-1, 2] = np.sum(gv.indivs[(gv.indivs[:, 1] == h_id) & (gv.indivs[:, 3] == 1), 11])
            h_id += 1
            

def zone_step(z):
    prev_pop = gv.zones[z, 4] + 1. * (gv.zones[z, 4] == 0)
    pres = gv.zones[z, 5] + 1. * (gv.zones[z, 5] == 0)
    pnres = gv.zones[z, 6] + 1. * (gv.zones[z, 6] == 0)
    z_bldgs = gv.bldgs[gv.bldgs[:, 3] == gv.zones[z, 0]]
    residents = len(gv.households[np.in1d(gv.households[:, 1], z_bldgs[:, 0])])
    pop = residents + 1. * (residents == 0)
    resb = len(z_bldgs[z_bldgs[:, 1] == 1])
    res = resb + 1. * (resb == 0)
    nresb = len(z_bldgs[z_bldgs[:, 1] > 1])
    nres = nresb + 1. * (nresb == 0)
    hp = gv.zones[z, 3] * (1 + math.log((pop / prev_pop + pres / res + nres / pnres) / 3., 10))
    gv.zones[z, 3:7] = [hp, pop, res, nresb]


def labor_market_model():
    fss = np.sum(gv.bldgs[gv.bldgs[:, 1] > 0, 4])
    pos = 1. - float(len(gv.jobs[np.isnan(gv.jobs[:, 2].astype(float))])) / float(len(gv.jobs))
    i_inc = mp.avgIncome
    if pos != gv.i_vacancies or fss != gv.i_value:
        if gv.i_vacancies == 0:
            gv.i_vacancies = 0.00001
        if gv.i_value == 0:
            gv.i_value = 1.
        if pos == 0:
            pos = 0.00001
        mp.avgIncome = mp.avgIncome * fss ** mp.alpha / ((gv.i_value ** mp.alpha) * (pos / gv.i_vacancies) ** mp.beta)
    gv.jobs[np.isnan(gv.jobs[:, 2].astype(float)), 0] += mp.avgIncome - i_inc
    gv.jobs[np.isnan(gv.jobs[:, 2].astype(float)) & (gv.jobs[:, 0] <= 0)] = np.min(
        gv.jobs[(np.isnan(gv.jobs[:, 2].astype(float))) & (gv.jobs[:, 0] > 0)])
    
    free_spots = 1. * len(gv.jobs[np.isnan(gv.jobs[:, 2].astype(float))])
    chance = mp.avgIncome * 0.5 / i_inc
    pots = np.where((gv.indivs[:, 4] == 2) & (gv.indivs[:, 6] == 0))[0]
    while free_spots > 0 and len(pots) > 0:
        if random.random() < chance:
            c = random.choice(pots)
            gv.indivs[c, [6, 8]] = [1, np.random.normal(mp.avgIncome, mp.stdIncome)]
            gv.data.append(['i', gv.tick, gv.indivs[c, 0], gv.indivs[c, 1], 0, 1, 0, 0, 0, 0, 0, 0, np.nan, np.nan])
            free_spots -= 1
            pots = np.delete(pots, np.where(pots == c)[0][0])
    
    if free_spots > 0:
        pots = np.where((gv.indivs[:, 4] == 3) & (gv.indivs[:, 6] == 0))[0]
        while free_spots > 0 and len(pots) > 0:
            if random.random() < chance:
                c = random.choice(pots)
                gv.indivs[c, [6, 8]] = [1, np.random.normal(mp.avgIncome, mp.stdIncome)]
                gv.data.append(['i', gv.tick, gv.indivs[c, 0], gv.indivs[c, 1], 0, 1, 0, 0, 0, 0, 0, 0, np.nan, np.nan])
                free_spots -= 1
                pots = np.delete(pots, np.where(pots == c)[0][0])


def general_step():    
    gv.changedLU = False
    
    gv.i_vacancies = 1. - float(len(gv.jobs[np.isnan(gv.jobs[:, 2].astype(float))])) / float(len(gv.jobs))
    gv.i_value = np.sum(gv.bldgs[gv.bldgs[:, 1] != 0, 4])
    
    if gv.tick in mp.events_set:
        create_event()
    
    t = time()
    
    h = 0
    while h < len(gv.households):
        left = hh_step(h)
        if not left:
            h += 1
    
    for i in range(len(gv.indivs)):
        i_step(i)
    
    freed = False
    for r in range(len(gv.roads)):
        freed2 = road_step(r)
        if freed2:
            freed = True
    
    # compute traffic on roads
    gv.traff_hist = np.append(gv.traff_hist, np.array([0 for i in range(len(gv.roads))])[np.newaxis].T, axis=1)
    visits = [j for i in gv.indivs[:, 0] for j in gv.routines[i]]
    visits_freq = np.unique(visits, return_counts=True)
    if len(visits_freq[1]) > 0:
        gv.traff_hist[visits_freq[0], -1] += visits_freq[1]

    if gv.tick > 30:
        gv.traff_hist = gv.traff_hist[:, 1:]
    gv.roads[:, 8] = np.mean(gv.traff_hist, axis=1)
    
    if freed:
        compute_network()
    
    for b in range(len(gv.bldgs)):    
        nearby_traffic = gv.roads[(np.all(gv.roads_juncs[:, 0] == gv.junctions_array[gv.bldgs[b, 13]], axis=1)) |
                                  (np.all(gv.roads_juncs[:, 1] == gv.junctions_array[gv.bldgs[b, 13]], axis=1)), 8]
        gv.bldgs_traff_dist[b] = np.mean(nearby_traffic)
    
    if gv.tick > mp.events_set[0]:
        gv.bldgs[:, 8] += 1
    for b in range(len(gv.bldgs)):
        bldg_step(b)
    if "in_migration" not in mp.lags or gv.tick < mp.events_set[0] or ~np.any(
            [mp.events_set[k] <= gv.tick < mp.events_set[k] + mp.lags['in_migration'] for k in range(len(mp.events_set))]):
        in_migration()
    
    for z in range(len(gv.zones)):
        zone_step(z)

    if "labor_market" not in mp.lags or gv.tick < mp.events_set[0] or ~np.any(
            [mp.events_set[k] <= gv.tick < mp.events_set[k] + mp.lags['labor_market'] for k in range(len(mp.events_set))]):
        if len(gv.jobs[np.isnan(gv.jobs[:, 2].astype(float))]) > 0:
            labor_market_model()
    
    if freed or gv.changedLU:
        for i in range(len(gv.indivs)):
            create_routines(i)
    
    ap_vals = gv.bldgs[gv.bldgs[:, 1] <= 1, 10] / gv.bldgs[gv.bldgs[:, 1] <= 1, 12]
    ap_vals_r = np.repeat(ap_vals, gv.bldgs[gv.bldgs[:, 1] <= 1, 12].astype(int))
    gv.stdResVal = np.std(ap_vals_r, ddof=1)
    
    for b in gv.bldgs:
        gv.bldgs_values[b[0]].append(b[10])
    for z in gv.zones:
        gv.zones_hps[z[0]].append(z[3])
    for r in gv.roads:
        gv.rds_civs[r[0]].append(r[8])
    
    gv.avg_incms.append(mp.avgIncome)
    
    print gv.tick, round(time() - t, 2), len(gv.bldgs[gv.bldgs[:, 1] == 0]),
    print len(gv.bldgs[gv.bldgs[:, 1] == 1]), len(gv.bldgs[gv.bldgs[:, 1] == 3]), len(gv.bldgs[gv.bldgs[:, 1] == 4]),
    print len(gv.households), len(gv.indivs), np.mean(gv.indivs[:, 8]), np.mean(gv.traff_hist[:, -1]),
    print np.std(gv.traff_hist[:, -1], ddof=1)
    
    gc.collect()


def run_model():
    gv.tick = 1
    while gv.tick <= mp.sim_stop:
        general_step()
        gv.tick += 1
