import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 24

in_path = 'outputs/'
f_range=(1,30)
columns = ['R', 'Recovered', 'Quarantined', 'New_quarantined', 
           'Total_infected', 'New_infections', 'Infected', 'Hospitalized',
           'New_hospitalizations', 'Total_dead', 'New_deaths']#, 'Known_R']
colors = ['black', 'green', 'orange', 'orange', 'red', 'red', 'red', 
          'brown', 'brown', 'grey', 'grey', 'pink']
name = 'noLockdown_norm0.08'
out_path = 'figures/'

for i in range(f_range[0], f_range[1]):
    with open(in_path+'sim'+str(i)+'_'+name+'.json') as f:
        df = pd.DataFrame.from_dict(json.load(f)['Results']['Stats']).transpose()
        df.index = df.index.astype(int)
        df['Sim'] = i
        if i == f_range[0]:
            results = df
        else:
            results = results.append(df)

results = results.fillna(0).reset_index().rename(columns={'index':'Timestamp'})
fig, axs = plt.subplots(len(columns), figsize=(20,20), sharex=True)
for c in range(len(columns)):
    sns.lineplot(data=results, x='Timestamp', y=columns[c], ax=axs[c], color=colors[c])
fig.savefig(out_path+name+'_stats.png')

blds_file = 'data/bldg_with_inst_orig.csv'
ind_file = 'data/civ_withCar_bldg_np.csv'
build_df = pd.read_csv(blds_file)[['Field2', 'Field6']]
agents_df = pd.read_csv(ind_file, header=None)[[0, 1]]
agents_df = agents_df.merge(build_df, left_on=1, right_on='Field2', how='left')
agents_df.loc[agents_df.Field6.isnull(), ['Field6']] = 1030


rcParams['font.size'] = 60
G = nx.DiGraph()
with open(in_path+'sim1_'+name+'.json') as f:
    chain = json.load(f)['Results']['Infections chain']
for k in chain:
    if k[2] != 0:
        G.add_edge(k[2], k[0], timestamp=k[1])
pos = graphviz_layout(G, prog='twopi')
ts = [G[e[0]][e[1]]['timestamp'] for e in G.edges]
range_ts = max(ts) - min(ts)
ts = [ts[i]/range_ts for i in range(len(ts))]
fig, axs = plt.subplots(figsize=(45,30))
deg = G.out_degree()
nodes_df = pd.DataFrame(list(dict(deg).keys()), columns=['Node']).merge(agents_df, left_on='Node', right_on=0)
zones = list(agents_df.Field6.unique())
nodes_df['Color'] = [zones.index(row.Field6)/(len(zones)-1) for idx, row in nodes_df.iterrows()]
ec = nx.draw(G, pos, edge_color=ts, ax=axs, edge_cmap=plt.cm.Reds, nodelist=list(dict(deg).keys()),
              node_size=[(deg[node]+1)*20 for node in list(dict(deg).keys())], cmap='Set3',
              node_color=nodes_df.Color.to_list(), width=5, arrowstyle='-')
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=range_ts))
sm._A = []
cb = plt.colorbar(sm)
cb.set_label('Timestamp', horizontalalignment='left')
fig.savefig(out_path+name+'_contagionChains.png')

results = pd.DataFrame(columns=['BuildingID', 'timestamp', 'infected', 'infected_%'])
for i in range(f_range[0], f_range[1]):
    with open(in_path+'sim'+str(i)+'_'+name+'.json') as f:
        j = json.load(f)['Results']['Buildings']
        results = results.append(pd.DataFrame(data = [[l[0], int(day), l[1], l[2]] for day in j for l in j[day]],
                                              columns=['BuildingID', 'timestamp', 'infected', 'infected_%']))
mean = results.fillna(0).groupby(['BuildingID', 'timestamp']).mean()
pivot1 = mean.reset_index().pivot(index='BuildingID', columns='timestamp', values='infected')
pivot1.to_csv(out_path+'Buildings_'+name+'_infected.csv')
pivot2 = mean.reset_index().pivot(index='BuildingID', columns='timestamp', values='infected_%')*100
pivot2.to_csv(out_path+'Buildings_'+name+'_infected_%.csv')

edges = {}
for i in range(f_range[0], f_range[1]):
    with open(in_path+'sim'+str(i)+'_'+name+'.json') as f:
        chain = json.load(f)['Results']['Infections chain']
        for k in chain:
            if k[2] != 0:
                stat_from = int(agents_df[agents_df[0]==k[2]]['Field6'].iloc[0])
                stat_to = int(agents_df[agents_df[0]==k[0]]['Field6'].iloc[0])
                if stat_from not in edges:
                    edges[stat_from] = {}
                if stat_to not in edges[stat_from]:
                    edges[stat_from][stat_to] = 0.
                edges[stat_from][stat_to] += 1
G = nx.DiGraph()
for e in edges:
    for e2 in edges[e]:
        if edges[e][e2]/(f_range[1]-f_range[0]) >= 10:
            G.add_edge(e, e2, weight=edges[e][e2]/(f_range[1]-f_range[0]))
fig, axs = plt.subplots(figsize=(45,30))
deg = G.out_degree()
pos = graphviz_layout(G, prog='twopi')
nodelist=list(dict(deg).keys())
colors = [zones.index(n)/(len(zones)-1) for n in nodelist]
weights = [G[e[0]][e[1]]['weight']/20 for e in G.edges]
node_weights = [(G.out_degree(n, weight='weight')-G[n][n]['weight'])*20 for n in G.nodes]
node_colors = [G[n][n]['weight'] for n in G.nodes]
pos = graphviz_layout(G, prog='twopi')
ec = nx.draw(G, pos, connectionstyle='arc3, rad = 0.1', width=weights, 
             nodelist=nodelist, node_size=node_weights, arrowsize=50,
             node_color=[w/max(node_colors) for w in node_colors], cmap=plt.cm.Wistia, ax=axs,
             with_labels=True, font_size=30)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Wistia, norm=plt.Normalize(vmin=0, 
                                                                  vmax=max(node_colors)))
sm._A = []
cb = plt.colorbar(sm)
cb.set_label('Infections', horizontalalignment='left')
fig.savefig(out_path+name+'_zones_network.png')

zones_pop = (agents_df.groupby('Field6').count()[0]/1000).to_dict()
fig, axs = plt.subplots(figsize=(45,30))
colors = [zones.index(n)/(len(zones)-1) for n in nodelist]
weights = [G[e[0]][e[1]]['weight']/20 for e in G.edges]
node_weights = [(G.out_degree(n, weight='weight')-G[n][n]['weight'])*40/zones_pop[n] 
                for n in G.nodes]
node_colors = [G[n][n]['weight']/zones_pop[n] for n in G.nodes]
ec = nx.draw(G, pos, connectionstyle='arc3, rad = 0.1', width=weights, 
             nodelist=nodelist, node_size=node_weights, arrowsize=50, with_labels=True,
             node_color=[w/max(node_colors) for w in node_colors], cmap=plt.cm.Wistia, ax=axs,
             font_size=30)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Wistia, norm=plt.Normalize(vmin=0, 
                                                                  vmax=max(node_colors)))
sm._A = []
cb = plt.colorbar(sm)
cb.set_label('Infections per 1000 residents', horizontalalignment='left')
fig.savefig(out_path+name+'_zones_network_perCapita.png')