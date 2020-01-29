import csv
import global_variables as gv


def save_init_snapshot():
    writer = csv.writer(open(gv.outputDir + 'sim' + str(gv.simulation) + '/init_snapshot' + str(gv.simulation)+'.csv',
                             'wb'))
    for i in gv.indivs:
        writer.writerow(['i'] + [i[k] for k in range(len(i))])
    
    for j in gv.jobs:
        writer.writerow(['j'] + [j[k] for k in range(len(j))])
    
    for b in gv.bldgs:
        writer.writerow(['b'] + [b[k] for k in range(len(b))])
    del writer


def save_values():
    writer = csv.writer(open(gv.outputDir + 'sim' + str(gv.simulation) + '/values' + str(gv.simulation) + '.csv',
                             'wb'))
    title = ['type', 'id']
    for i in range(1, int(gv.tick)):
        title.append('t' + str(i))
    writer.writerow(title)
    
    for b in gv.bldgs:
        row = ['bldg_value', b[0]]
        for i in range(len(gv.bldgs_values[b[0]])):
            if isinstance(type(gv.bldgs_values[b[0]][i]), type(gv.bldgs)):
                row.append(gv.bldgs_values[b[0]][i][0])
            else:
                row.append(gv.bldgs_values[b[0]][i])
        writer.writerow(row)
    
    for z in gv.zones:
        row = ['zones_hps', z[0]]
        for i in range(len(gv.zones_hps[z[0]])):
            row.append(gv.zones_hps[z[0]][i])
        writer.writerow(row)
    
    for r in gv.roads:
        row = ['rds_civs', r[0]]
        for i in range(len(gv.rds_civs[r[0]])):
            row.append(gv.rds_civs[r[0]][i])
        writer.writerow(row)
    
    row = ['avg_incms', 'macro_measure']
    for k in range(len(gv.avg_incms)):
        row.append(gv.avg_incms[k])
    writer.writerow(row)
    
    del writer


def save_data():
    writer = csv.writer(open(gv.outputDir + 'sim' + str(gv.simulation) + 
                             '/data' + str(gv.simulation) + '.csv', 'wb'))
    for d in gv.data:
        writer.writerow(d)
    
    del writer
