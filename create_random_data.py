import numpy as np
import pandas as pd
from parameters import jobs_per_m, avgIncome, stdIncome, infection_prob, admission_prob, mortality_prob


def create_data(ind_file, blds_file): 
    #import data
    agents = np.array(pd.read_csv(ind_file, header=None).values)
    agents = agents[:, [0, 11, 2, 3, 6, 12, 3, 1]] # id, hh, dis, worker, age, workIn, wf_participant, building
    agents = np.append(agents, np.zeros((len(agents), 5)), axis= 1) # id, hh, dis, worker, age, workIn, wp_participant, building, religous(bin), job ID, expected income, actual income, job building
    agents[:, [8, 9, 12]] = np.nan
    agents[agents[:, 4]==1, 3] = 0
    agents[agents[:, 4]==1, 5] = 0
    agents[agents[:, 4]==1, 6] = 0
    agents[(agents[:, 3] == 0) & (agents[:, 5] == 1), 3] = 1 
    agents[(agents[:, 3] == 1) & (agents[:, 6] == 0), 6] = 1
    
    households = pd.read_csv(ind_file, header=None)
    households = households[[11, 1, 10, 13]].groupby(11).first().reset_index().to_numpy() # id, home, income, car
    
    #religous agents calculations
    households = np.append(households, np.random.randint(0,2,(len(households), 1)), axis=1)
    for h in households:
        members = agents[:, 1] == h[0]
        agents[members, 8] = h[4]
        
    build=np.array(pd.read_csv(blds_file).values)
    build = build[:, [6, 2, 3, 12, 4, 2, 14, 15,17, 18]] # id, lu, floors, stat, fs, init_lu, x, y, USG group,USG code
    build = np.append(build, np.ones((len(build), 1)), axis= 1) # building status (closed/open)
    build = np.append(build, np.zeros((len(build), 4)), axis= 1) # value, neigh. asstes, rent
    
    agents[agents[:, 4]==3, 4] = np.random.randint(66, 90, len(agents[agents[:, 4]==3]))
    agents[agents[:, 4]==2, 4] = np.random.randint(19, 65, len(agents[agents[:, 4]==2]))
    agents[agents[:, 4]==1, 4] = np.random.randint(1, 18, len(agents[agents[:, 4]==1]))
    

    # create jobs in buildings
    jobs_num = np.round_((np.choose(build[:, 1].astype(int), 
        (build[:, np.newaxis, 4] * np.array(jobs_per_m)).transpose())).tolist(), 0).astype(int)
    jobs = np.array([[np.random.normal(avgIncome, stdIncome), i, np.nan] for i in range(len(jobs_num)) 
                        for j in range(jobs_num[i])])
    while len(jobs[jobs[:, 0] <= 0]) > 0: 
        jobs[jobs[:, 0] <= 0, 0] = np.random.normal(avgIncome, stdIncome, len(jobs[jobs[:, 0] <= 0]))
    jobs = np.concatenate((jobs, np.array(range(len(jobs)))[np.newaxis, :].T), axis=1)
    
    # compute expected income
    for h in households: 
        members = agents[:, 1] == h[0]
        workers = agents[:, 3] == 1
        if len(agents[members & workers]) > 0:
            agents[members & workers, 10] = h[2] / len(agents[members & workers])
        else:
            agents[members & workers, 10] = 0
    
    # match individuals to workplaces
    local_workers = np.where(agents[:, 5] == 1)[0]
    av_jobs = np.where(np.isnan(jobs[:, 2]))[0]
    while len(local_workers) > 0 and len(av_jobs) > 0:
        c = np.random.choice(local_workers)
        j = av_jobs[np.argmin(np.abs(agents[c, 10] - jobs[av_jobs, 0]))]
        agents[c, 9:13] = [jobs[j, 3], jobs[j, 0], jobs[j, 0], build[int(jobs[j, 1]), 0]]
        jobs[j, 2] = agents[c, 0]
        local_workers = np.where((agents[:, 5] == 1) & (np.isnan(agents[:, 9])))[0]
        av_jobs = np.where(np.isnan(jobs[:, 2]))[0]
    
    # unoccupied jobs are occupied by in-commuter
    jobs[np.isnan(jobs[:, 2]), 2] = 0
    
    # unnemployed agents are removed from workforce
    agents[(agents[:, 5] == 1) & (np.isnan(agents[:, 9])), 5:7] = [0, 0]
    agents[(agents[:, 3] == 1) & (agents[:, 5] == 0), 11] = agents[
        (agents[:, 3] == 1) & (agents[:, 5] == 0), 10]
    
    # update household income
    for h in range(len(households)): 
        households[h, 2] = np.sum(agents[agents[:, 1] == households[h, 0], 11])
    del h
    
    #creation of regular activities besides work
    elementry = build[build[:,9]==5310,0]
    elementry_rel = build[build[:,9]==5312,0]
    high_schools = build[build[:,9]==5338,0]
    high_schools_rel = build[build[:,9]==5523,0]
    high_schools_rel = build[build[:,9]==5525,0]
    kinder = build[build[:,9]==5305,0]
    kinder_rel = build[build[:,9]==5300,0]
    religious = build[np.isin(build[:,9], [5501, 5521]), 0]
    yeshiva = build[build[:,9]==5340,0]
    etc = build[np.isin(build[:,9], 
                        [6512, 6520, 6530, 6600, 5740, 5760, 5600, 5700, 5202, 5253]),0]
    rel_etc = np.append(etc,religious)
    
    #inserting all non-working agents their activities
    agents[(agents[:, 8] == 0) & (agents[:, 4] <19), 12] = np.random.choice(high_schools, len(agents[(agents[:, 8] == 0) & (agents[:, 4] <19)]))
    agents[(agents[:, 8] == 0) & (agents[:, 4] <15), 12] = np.random.choice(elementry, len(agents[(agents[:, 8] == 0) & (agents[:, 4] <15)]))
    agents[(agents[:, 8] == 0) & (agents[:, 4] <7), 12] = np.random.choice(kinder, len(agents[(agents[:, 8] == 0) & (agents[:, 4] <7)]))
    agents[(agents[:, 8] == 1) & (agents[:, 4] < 25) & (np.isnan(agents[:, 12])), 12] = np.random.choice(yeshiva, len(agents[(agents[:, 8] == 1) & (agents[:, 4] < 25) & (np.isnan(agents[:, 12]))]))
    agents[(agents[:, 8] == 1) & (agents[:, 4] < 19), 12] = np.random.choice(high_schools_rel, len(agents[(agents[:, 8] == 1) & (agents[:, 4] < 19)]))
    agents[(agents[:, 8] == 1) & (agents[:, 4] < 15), 12] = np.random.choice(elementry_rel, len(agents[(agents[:, 8] == 1) & (agents[:, 4] < 15)]))
    agents[(agents[:, 8] == 1) & (agents[:, 4] < 7), 12] = np.random.choice(kinder_rel, len(agents[(agents[:, 8] == 1) & (agents[:, 4] < 7)]))
    agents[(np.isnan(agents[:, 12])) , 12] = np.random.choice(etc, len(agents[(np.isnan(agents[:, 12]))])) * np.random.randint(2, size=len(agents[np.isnan(agents[:, 12])]))
    agents[agents[:, 12]==0, 12] = np.nan

    
    epidemic = np.zeros((len(agents), 5))
    epidemic[:,0] = 1
    # epidemic[:, 1] = risk_func(agents[:, 4].astype(int))
    epidemic[:, 2] = np.random.random(len(epidemic))
    epidemic[np.random.choice(range(len(epidemic)), 20, replace=False), 0] = 2 # choose only out-commuters
    # also consider the effect of jobs occupied by in-commuters
    
    agents = np.append(agents, epidemic, axis=1)
    
    #create more regular activities per agent
    agents = np.append(agents, np.zeros((len(agents), 11)), axis= 1)
    agents[:,18] = np.nan
    agents[agents[:, 8] == 1, 18] = np.random.choice(rel_etc, len(agents[agents[:, 8] == 1])) * np.random.randint(2, size=len(agents[agents[:, 8] == 1]))
    agents[agents[:, 8] == 0, 18] = np.random.choice(etc, len(agents[agents[:, 8] == 0])) * np.random.randint(2, size=len(agents[agents[:, 8] == 0]))
    agents[agents[:, 18] == 0, 18] = np.nan
    
    #add residence stat zone per agent
    for b in build:
        blds = agents[:, 7] == b[0]
        agents[blds, 22] = b[3]
        
    #add infection prob by age per agent
    for inf in infection_prob:
        agents[(agents[:, 4] >= inf[0]) & (agents[:, 4] < inf[1]), 14] = np.random.normal(
            inf[2], inf[3], len(agents[(agents[:, 4] >= inf[0]) & (agents[:, 4] < inf[1])]))
    agents[agents[:, 14] < 0, 14] = 0
    
    #add admission prob by age per agent
    for inf in admission_prob:
        agents[(agents[:, 4] >= inf[0]) & (agents[:, 4] < inf[1]), 23] = np.random.normal(
            inf[2], inf[3], len(agents[(agents[:, 4] >= inf[0]) & (agents[:, 4] < inf[1])]))
    agents[agents[:, 23] < 0, 23] = 0
       
    #add mortality prob by age per agent
    for inf in mortality_prob:
        agents[(agents[:, 4] >= inf[0]) & (agents[:, 4] < inf[1]), 26] = np.random.normal(
            inf[2], inf[3], len(agents[(agents[:, 4] >= inf[0]) & (agents[:, 4] < inf[1])])) 
    agents[agents[:, 26] < 0, 26] = 0
    
    return agents, households, build, jobs
