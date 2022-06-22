import numpy as np

# a script containing all the necessary functions for the epidemiological model

# function 1: update counts of days since infection/quarantine for each agent in the network


def update_counts(agents,tick):
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
    return agents

# function 2: update the buildings' status in the network, close/open

def update_buildings_status(bld_visits_by_agents,agents,build):
        # get buildings status - open/close == 1/0 and multiply bld_visits_by_agents by it
        bld_visits = bld_visits_by_agents * build[
            np.argmax(build[:, 0][None, :] == bld_visits_by_agents[:, :, None], axis=2), 10]
        # check if agents are in isolation and if yes - all activities but first (home) become zero
        bld_visits[:, 1:] = bld_visits[:, 1:] * ((agents[:, 13] != 3) & (agents[:, 13] != 4)
                                                  & (agents[:, 13] != 3.5)).reshape((len(agents), 1))
        # check if agents are admitted or dead and if yes - all activities
        bld_visits[:, 0:] = bld_visits[:, 0:] * ((agents[:, 13] != 5) & (agents[:, 13] != 7)).reshape((len(agents), 1))
        
        bld_visits[bld_visits==0] = np.nan