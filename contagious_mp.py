from create_random_data import create_data
from communities import create_network
import networkx as nx
from multiprocessing import Pool
import functools
import time


def compute_path_lengths(b, G):
    return nx.shortest_path_length(G, target=b, weight='weight')


def compute_path_length(a, G):
    return (a[0], a[1], nx.shortest_path_length(G, source=a[0], target=a[1], weight="weight"))


if __name__ == "__main__":
    print("creating network")
    agents, households, build, jobs, contagious_risk_day = create_data('data/civ_withCar_bldg_np.csv', 'data/bldg_with_inst_orig.csv')
    # agents: 0 id, 1 hh, 2 dis, 3 worker, 4 age, 5 workIn, 6 wp_participant, 7 building, 8 random socio-economic status, 
    # 13 contagious status, 14 contagious risk, 15 exposure risk, 16 contagious day, 17 sick day
    G = create_network(agents, households, build) # link agents and buildings
    
    print("starting computations")
    t = time.time()
    a = [nx.shortest_path_length(G, target=b, weight="weight") for b in build[:, 0]]
    print(time.time()-t)
    
    pool = Pool(4)
    t = time.time()
    result = pool.map(functools.partial(compute_path_length, G=G), [[a, b] for a in agents[:, 0] for b in build[:, 0]])
    print(time.time()-t)
    
    t = time.time()
    result = pool.map(functools.partial(compute_path_lengths, G=G), build[:, 0])    
    print(time.time()-t)



