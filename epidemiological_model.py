import numpy as np
from parameters import k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day, quarantine, diagnosis, scenario_code, hospital_recover, name
import global_variables as gv

# a script containing all the necessary functions for the epidemiological model

# function 1: update the buildings' status in the network - close/open

def update_buildings_status():
        # get buildings status - open/close == 1/0 and multiply bld_visits_by_agents by it
        gv.bld_visits = gv.routines * gv.bldgs[
            np.argmax(gv.bldgs[:, 0][None, :] == gv.gv.routines[:, :, None], axis=2), 10]
        # check if agents are in isolation and if yes - all activities but first (home) become zero
        gv.bld_visits[:, 1:] = gv.bld_visits[:, 1:] * ((gv.indivs[:, 13] != 3) & (gv.indivs[:, 13] != 4)
                                                  & (gv.indivs[:, 13] != 3.5)).reshape((len(gv.indivs), 1))
        # check if agents are admitted or dead and if yes - all activities become zero
        gv.bld_visits[:, 0:] = gv.bld_visits[:, 0:] * ((gv.indivs[:, 13] != 5) & (gv.indivs[:, 13] != 7)).reshape((len(gv.indivs), 1))
        # set None values for slots with no activity
        gv.bld_visits[gv.bld_visits==0] = np.nan
    
# function 2: update counts of days since infection/quarantine for each agent in the network

def update_counts():
    # update number of days since infecton for carriers
    gv.indivs[(gv.indivs[:, 13] == 2) | 
           (gv.indivs[:, 13] == 4) | 
           (gv.indivs[:, 13] == 3.5)| 
           (gv.indivs[:, 13] == 5), 17] = gv.tick - gv.indivs[(gv.indivs[:, 13] == 2) | 
                  (gv.indivs[:, 13] == 4) | 
                  (gv.indivs[:, 13] == 3.5) | 
                  (gv.indivs[:, 13] == 5), 16]
    # update number of days since quarantine for carriers
    gv.indivs[(gv.indivs[:, 13] == 3) |
           (gv.indivs[:, 13] == 3.5) |
           (gv.indivs[:, 13] == 4), 20] = gv.tick - gv.indivs[(gv.indivs[:, 13] == 3)  |
                    (gv.indivs[:, 13] == 3.5) |
                    (gv.indivs[:, 13] == 4), 19] 
    # update number of days since admission                                                 
    gv.indivs[(gv.indivs[:, 13] == 5), 25] = gv.tick - gv.indivs[(gv.indivs[:, 13] == 5), 24]                                                  
    # document all active infected agents
    gv.infected = np.where((gv.indivs[:,13]==2) | (gv.indivs[:,13]==4) | (gv.indivs[:, 13]==3.5))[0]
    # document all uninfected agents
    gv.uninfected = np.where((gv.indivs[:,13]<2) | (gv.indivs[:,13]==3))[0]
    # document all admitted agents
    gv.admitted = np.where(gv.indivs[:,13]==5)[0]
    # document all unadmitted agents
    gv.unadmitted = np.where(gv.indivs[:,13]!=5)[0]
    # document all  infected and in quarantine agents
    gv.infected_quar = len(np.where(gv.indivs[:,13]==4)[0]) # total diagnosed agents in quarantine    
    # building visits by infected agents
    gv.infected_blds = gv.bld_visits[gv.infected] 
    # building visits by uninfected agents
    gv.uninfected_blds = gv.bld_visits[gv.uninfected]                                                 

# function 3: hospitaliztion of agents calculations

def calculate_hospitalizations():
        # agents infected less than 4 days or more than two weeks
        no_admission_slot = np.where((gv.indivs[:,16] < 4) |
                                     (gv.indivs[:,16] > 14) |
                                     (gv.indivs[:, 13] >= 5) )[0] 
        # array of all agents' admission probability
        admission_prob = gv.indivs[:,23] 
        # in order to prevent changes on original array 'agents', a copy of admissions prob array is required
        admission_prob = admission_prob.copy() 
        # set probability 0 for all susceptible agents
        admission_prob[gv.uninfected] = 0 
        # set probability 0 for all agents infected less than 4 days or more than 2 weeks
        admission_prob[no_admission_slot] = 0 
        # caclulate random factor for admission
        rand_admission = np.random.random(admission_prob.shape)
        # calculate hospitalized & unhospitalized agents
        admissions = admission_prob > rand_admission 
        # calculate new admissions
        gv.new_admissions = np.where(admissions == True)
        # set agent's status as hospitalized
        gv.indivs[admissions, 13] = 5 
        # record admission day 
        gv.indivs[admissions, 24] = gv.tick 
    
# function 4: mortality of agents calculations
    
def calculate_mortality():
        # agents admitted less than 3 days
        no_death_slot = np.where(gv.indivs[:,24] < 3)[0] 
        # array of all agents' death probability
        death_prob = gv.indivs[:,26] 
        # in order to prevent changes on original array 'agents', a copy of death probs array is required
        death_prob = death_prob.copy() 
        # set death prob for all unadmitted agents as zero
        death_prob[gv.unadmitted] = 0 
        # set death prob for all agents admitted 2 days or less as zero
        death_prob[no_death_slot] = 0 
        # caclulate random factor for death
        rand_death = np.random.random(death_prob.shape) 
        # calculate deaths amongst hospitalized agents
        deaths = death_prob > rand_death 
        # calculate new deaths
        gv.new_deaths = np.where(deaths == True) 
        # set agent's status as deceased
        gv.indivs[deaths, 13] = 7 

# function 5: infections calculations

def calculate_infections():
    
        # compare the two arrays by flattening them and reshaping infected_blds
        exposure = 1*np.array([np.isin(gv.infected_blds, gv.uninfected_blds[i]).any(axis=1) 
                                for i in range(len(gv.uninfected_blds))])
        # calculate infection probability for current iteration for each agent in the network based on interactions
        infection_prob = gv.interaction_prob[gv.uninfected][:, gv.infected] * gv.indivs[
            gv.Suninfected, 14].reshape((len(gv.uninfected), 1)) 
        # multiply it by the probability to infect as a dependency of illness day
        infection_prob *= contagious_risk_day.pdf(gv.indivs[gv.infected, 17])
        # multiply it by agents' exposure in current iteration
        infection_prob *= exposure 
        # normalize results
        infection_prob *= norm_factor
        # calculate random factor
        rand_cont = np.random.random(infection_prob.shape)
        # calculate infections wth random factor
        infections = infection_prob > rand_cont
        # document new infections
        new_infected = np.where(infections.any(axis=1) & (gv.indivs[gv.uninfected, 13]<2))
        # document new infected and in quarantine agents
        new_quarantined_infected = np.where(infections.any(axis=1) & (gv.indivs[gv.uninfected, 13]==3))
        # record new status and infection day count for infected undiagnosed agents
        gv.indivs[gv.uninfected[new_infected], 13] = 2
        gv.indivs[gv.uninfected[new_infected], 16] = gv.tick
        # record new status and infection day count for infected, quarantined undiagnosed agents
        gv.indivs[gv.uninfected[new_quarantined_infected], 13] = 3.5
        gv.indivs[gv.uninfected[new_quarantined_infected], 16] = gv.tick
        # this allows tracing infection chains - who infected whom and when
        gv.indivs[gv.uninfected[np.where(infections)[0]], 21] = gv.indivs[gv.infected[np.where(infections)[1]], 0]


# function 6: update agents array

def agents_update():  
        # end of quarantine for helathy agents, can steel be infected
        gv.indivs[(gv.indivs[:, 13] == 3) & (gv.indivs[:, 20] == quarantine), 13] = 1 
        # end of quarantine for infected undiagnosed agents
        gv.indivs[(gv.indivs[:, 13] == 3.5) & (gv.indivs[:, 20] == quarantine), 13] = 2 
        # start count of quarantine days for newly quarantined agents
        gv.indivs[(gv.indivs[:, 13] == 2) & (gv.indivs[:, 17] == diagnosis), 19] = gv.tick 
        # sick agents begin quarantine after four days of being infected without diagnose
        gv.indivs[((gv.indivs[:, 13] == 2) | (gv.indivs[:, 13] == 3.5)) & (gv.indivs[:, 17] == diagnosis), 13] = 4 
        # sick agents in quarantine recover 
        gv.indivs[(gv.indivs[:, 13] == 4) & (gv.indivs[:, 17] == recover), 13] = 6 
        # admission end with recovery
        gv.indivs[(gv.indivs[:, 13] == 5) & (gv.indivs[:, 17] == hospital_recover), 13] = 6 
        # infection count reset for unisolated agents - ask Yair - is it necessary?SS
        #gv.indivs[(gv.indivs[:, 13] == 1) | (gv.indivs[:, 13] == 2) | (gv.indivs[:, 13] == 6) | (gv.indivs[:, 13] == 7), 17] = 0
        # quararntine count reset for unisolated agents
        gv.indivs[(gv.indivs[:, 13] == 1) | (gv.indivs[:, 13] == 2) | (gv.indivs[:, 13] == 6) | (gv.indivs[:, 13] == 7), 20] = 0 
        # admissions count reset for unisolated agents
        gv.indivs[(gv.indivs[:, 13] == 1) | (gv.indivs[:, 13] == 2) | (gv.indivs[:, 13] == 6) | (gv.indivs[:, 13] == 7), 24] = 0

# function 7: household's members quarantines calculations

def calculate_hh_quarantines():
        # calculate new quarantined agents
        new_diagnosed_agents = gv.indivs[(gv.indivs[:, 13] == 4) & (gv.indivs[:, 17] == diagnosis)]
        # if there are sick agents in quarantine
        if len(new_diagnosed_agents) > 0:
            # find their healthy hh members
            new_quar = np.where((np.isin(gv.indivs[:, 1],new_diagnosed_agents[:, 1])) & 
                                (gv.indivs[:, 13]==1))[0] 
            # set them to be healthy and in quarantine
            gv.indivs[new_quar, 13] = 3 
            # record start day of quarantine per agent
            gv.indivs[new_quar, 19] = gv.tick 
            # find their infected undiagnosed hh members
            new_quar_infected = np.where((np.isin(gv.indivs[:, 1],new_diagnosed_agents)) & 
                                (gv.indivs[:, 13]==2))[0] 
            # set them to be unaware infected and in quarantine
            gv.indivs[new_quar_infected, 13] = 3.5 
            # record start day of quarantine per agent   
            gv.indivs[new_quar_infected, 19] = gv.tick 
            
# function 8: compute R

def compute_R(a, t):
    new_infections = len(a[(a[:, 13] != 1) & 
                                (a[:, 13] != 3) & (a[:, 16] == t)])
    sum_I = 0.
    for i in range(1, recover):
        sum_I += a[(a[:, 13] != 1) 
                   & (a[:, 13] != 3) 
                   & ((t - a[:, 16]) == i)].shape[0] * contagious_risk_day.pdf(i)
    if sum_I > 0:
        R = new_infections / sum_I
    else:
        R = 0
    return R

# function 9: compute visible R

def compute_vis_R(a, t):
    new_known_infections = len(a[(a[:,13]>=4) & (a[:, 17] == diagnosis)])
    sum_I = 0.
    for i in range(diagnosis+1, recover):
        sum_I += a[(a[:, 13] >= 4) & ((t - a[:, 16]) == i)].shape[0] * contagious_risk_day.pdf(i - diagnosis)
    if sum_I > 0:
        R = new_known_infections / sum_I
    else:
        R = 0
    return R 

#function 10: compute R for each statistical area

def compute_sas_R():
    # empty dictionery for results
    sas_R = {} 
    # for each statistical zone in research area
    for sa in np.unique(gv.bldgs[:, 3]):
            # record all agents for statistical zone       
            sa_agents = gv.indivs[gv.indivs[:, 22] == sa]
            # calculate its R for current day
            sas_R[sa] = compute_R(sa_agents, gv.tick)
   
# function 11: a master function for epidemiological model    
    
def run_EM():
    update_buildings_status()
    update_counts()
    calculate_hospitalizations()
    calculate_mortality()
    calculate_infections()
    agents_update()
    calculate_hh_quarantines()
    compute_R(gv.indivs, gv.tick)
    compute_vis_R(gv.indivs, gv.tick)
    compute_sas_R()
    
    
    