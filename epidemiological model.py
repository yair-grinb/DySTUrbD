import numpy as np
from parameters import k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day, quarantine, diagnosis, scenario_code, hospital_recover, name

# a script containing all the necessary functions for the epidemiological model

# function 1: update the buildings' status in the network - close/open

def update_buildings_status(bld_visits_by_agents,agents,build):
        # get buildings status - open/close == 1/0 and multiply bld_visits_by_agents by it
        bld_visits = bld_visits_by_agents * build[
            np.argmax(build[:, 0][None, :] == bld_visits_by_agents[:, :, None], axis=2), 10]
        # check if agents are in isolation and if yes - all activities but first (home) become zero
        bld_visits[:, 1:] = bld_visits[:, 1:] * ((agents[:, 13] != 3) & (agents[:, 13] != 4)
                                                  & (agents[:, 13] != 3.5)).reshape((len(agents), 1))
        # check if agents are admitted or dead and if yes - all activities become zero
        bld_visits[:, 0:] = bld_visits[:, 0:] * ((agents[:, 13] != 5) & (agents[:, 13] != 7)).reshape((len(agents), 1))
        # set None values for slots with no activity
        bld_visits[bld_visits==0] = np.nan
        return bld_visits
    
# function 2: update counts of days since infection/quarantine for each agent in the network

def update_counts(agents,tick,bld_visits):
    # update number of days since infecton for carriers
    agents[(agents[:, 13] == 2) | 
           (agents[:, 13] == 4) | 
           (agents[:, 13] == 3.5)| 
           (agents[:, 13] == 5), 17] = tick - agents[(agents[:, 13] == 2) | 
                  (agents[:, 13] == 4) | 
                  (agents[:, 13] == 3.5) | 
                  (agents[:, 13] == 5), 16]
    # update number of days since quarantine for carriers
    agents[(agents[:, 13] == 3) |
           (agents[:, 13] == 3.5) |
           (agents[:, 13] == 4), 20] = tick - agents[(agents[:, 13] == 3)  |
                    (agents[:, 13] == 3.5) |
                    (agents[:, 13] == 4), 19] 
    # document all active infected agents
    infected = np.where((agents[:,13]==2) | (agents[:,13]==4) | (agents[:, 13]==3.5))[0]
    # document all uninfected agents
    uninfected = np.where((agents[:,13]<2) | (agents[:,13]==3))[0]
    # document all admitted agents
    admitted = np.where(agents[:,13]==5)[0]
    # document all unadmitted agents
    unadmitted = np.where(agents[:,13]!=5)[0]
    # document all  infected and in quarantine agents
    infected_quar = len(np.where(agents[:,13]==4)[0]) # total diagnosed agents in quarantine    
    # building visits by infected agents
    infected_blds = bld_visits[infected] 
    # building visits by uninfected agents
    uninfected_blds = bld_visits[uninfected]                                                 
    return agents, admitted, unadmitted, infected_quar, infected_blds, uninfected_blds


    
# function 3: hospitaliztion of agents calculations

def calculate_hospitalizations(agents,uninfected,tick):
        # agents infected less than 4 days or more than two weeks
        no_admission_slot = np.where((agents[:,16] < 4) |
                                     (agents[:,16] > 14) |
                                     (agents[:, 13] >= 5) )[0] 
        # array of all agents' admission probability
        admission_prob = agents[:,23] 
        # in order to prevent changes on original array 'agents', a copy of admissions prob array is required
        admission_prob = admission_prob.copy() 
        # set probability 0 for all susceptible agents
        admission_prob[uninfected] = 0 
        # set probability 0 for all agents infected less than 4 days or more than 2 weeks
        admission_prob[no_admission_slot] = 0 
        # caclulate random factor for admission
        rand_admission = np.random.random(admission_prob.shape)
        # calculate hospitalized & unhospitalized agents
        admissions = admission_prob > rand_admission 
        # calculate new admissions
        new_admissions = np.where(admissions == True)
        # set agent's status as hospitalized
        agents[admissions, 13] = 5 
        # record admission day 
        agents[admissions, 24] = tick 
        return agents, new_admissions
    
# function 4: mortality of agents calculations
    
def calculate_mortality(agents,unadmitted):
        # agents admitted less than 3 days
        no_death_slot = np.where(agents[:,24] < 3)[0] 
        # array of all agents' death probability
        death_prob = agents[:,26] 
        # in order to prevent changes on original array 'agents', a copy of death probs array is required
        death_prob = death_prob.copy() 
        # set death prob for all unadmitted agents as zero
        death_prob[unadmitted] = 0 
        # set death prob for all agents admitted 2 days or less as zero
        death_prob[no_death_slot] = 0 
        # caclulate random factor for death
        rand_death = np.random.random(death_prob.shape) 
        # calculate deaths amongst hospitalized agents
        deaths = death_prob > rand_death 
        # calculate new deaths
        new_deaths = np.where(deaths == True) 
        # set agent's status as deceased
        agents[deaths, 13] = 7 
        return agents, new_deaths

# function 5: infections calculations

def calculate_infections(agents,tick,infected_blds,uninfected_blds,interaction_prob,infected,uninfected):
        # compare the two arrays by flattening them and reshaping infected_blds
        exposure = 1*np.array([np.isin(infected_blds, uninfected_blds[i]).any(axis=1) 
                                for i in range(len(uninfected_blds))])
        # calculate infection probability for current iteration for each agent in the network based on interactions
        infection_prob = interaction_prob[uninfected][:, infected] * agents[
            uninfected, 14].reshape((len(uninfected), 1)) 
        # multiply it by the probability to infect as a dependency of illness day
        infection_prob *= contagious_risk_day.pdf(agents[infected, 17])
        # multiply it by agents' exposure in current iteration
        infection_prob *= exposure 
        # normalize results
        infection_prob *= norm_factor
        # calculate random factor
        rand_cont = np.random.random(infection_prob.shape)
        # calculate infections wth random factor
        infections = infection_prob > rand_cont
        # document new infections
        new_infected = np.where(infections.any(axis=1) & (agents[uninfected, 13]<2))
        # document new infected and in quarantine agents
        new_quarantined_infected = np.where(infections.any(axis=1) & (agents[uninfected, 13]==3))
        # record new status and infection day count for infected undiagnosed agents
        agents[uninfected[new_infected], 13] = 2
        agents[uninfected[new_infected], 16] = tick
        # record new status and infection day count for infected, quarantined undiagnosed agents
        agents[uninfected[new_quarantined_infected], 13] = 3.5
        agents[uninfected[new_quarantined_infected], 16] = tick
        # this allows tracing infection chains - who infected whom and when
        agents[uninfected[np.where(infections)[0]], 21] = agents[infected[np.where(infections)[1]], 0]
        return agents, 
    
    
    
    
    
    
    
    
    