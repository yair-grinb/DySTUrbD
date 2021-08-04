zones = None
bldgs = None
jobs = None
households = None
indivs = None
roads = None
junctions = None
junctions_array = None
roads_juncs = None
graph = None
visits_hist = None
dists = None
routes = None
routines = {}
stdResVal = None
simulation = None
bldgs_values = {}
zones_hps = {}
rds_civs = {}
tick = None
data = []
avg_incms = []
i_vacancies = None
i_value = None
changedLU = False
bldgs_visits_dist = None
bld_visits_by_agents = None
bld_visits = None
infected = None
uninfected = None
admitted = None
unadmitted = None
infected_quar = None
infected_blds = None
uninfected_blds = None
new_admissions = None
new_deaths = None
interaction_prob = None
nodes = None


# input files
# zonesFile = 'data/zones_wgs_np.csv'
# bldgsFile = 'data/market_bldgs3_np.csv'
# indivsFile = 'data/civ_withCar_bldg_np.csv'
# householdsFile = 'data/households.csv'
# roadsFile = 'data/market_streets_np.csv'
zonesFile = 'data/dummy_zones.csv'
bldgsFile = 'data/dummy_bldgs.csv'
indivsFile = 'data/dummy_agents.csv'
householdsFile = 'data/dummy_households.csv'
roadsFile = 'data/dummy_roads.csv'

# output directory
outputDir = 'outputs/'    