import os
from create_world import create_world
import global_variables as gv
import model_parameters as mp
from ab_model import run_model
from output_functions import save_values, save_data


def reset_gv():
    gv.zones = None
    gv.bldgs = None
    gv.jobs = None
    gv.households = None
    gv.indivs = None
    gv.roads = None
    gv.junctions = None
    gv.roads_juncs = None
    gv.graph = None
    gv.traff_hist = None
    gv.dists = None
    gv.routes = None
    gv.job_id = None
    gv.routines = {}
    gv.stdResVal = None
    gv.bldgs_values = {}
    gv.zones_hps = {}
    gv.rds_civs = {}
    gv.tick = None
    gv.data = []
    gv.bldgs_values = {}
    gv.zones_hps = {}
    gv.rds_civs = {}
    gv.avg_incms = []
    gv.bldgs_traff_dist = {}


def reset_mp():
    mp.hhSize = None 
    mp.hhSTD = None
    mp.avgAge = None 
    mp.stdAge = None
    mp.carChance = None 
    mp.age1chance = None 
    mp.age2chance = None 
    mp.disChance = None 
    mp.employChance = None
    mp.workInChance = None
    mp.worker_residing_outside = None


def simulate(i):
    gv.simulation = i
    if not os.path.isdir(gv.outputDir + 'sim' + str(gv.simulation)):
        os.mkdir(gv.outputDir + 'sim' + str(gv.simulation))
    reset_gv()
    reset_mp()
    create_world()
    run_model()
    save_values()
    save_data()


simulate(1)
