import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import networkx as nx

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20

in_path = 'outputs/'
f_range=(1,31)
columns = ['R', 'Recovered', 'Quarantined', 'New_quarantined', 
           'Total_infected', 'New_infections', 'Infected', 'Hospitalized',
           'New_hospitalizations', 'Total_dead', 'New_deaths']
columns2 = ['R', 'Recovered', 'Quarantined', 'Daily quarantined', 
           'Total infected', 'Daily infections', 'Active infected', 'Hospitalized',
           'Daily hospitalizations', 'Total Dead', 'Daily deaths']
colors = ['black', 'green', 'orange', 'orange', 'red', 'red', 'red', 
          'brown', 'brown', 'grey', 'grey', 'pink']
styles = ['-.', '-', ':', '--']
names = ['nolockdown_norm0.08', 'ALL_norm0.08', 'GRADUAL_ALL_norm0.08',
         'DIFF_ALL_norm0.08']
legend_names = ['No lockdown',  'Lockdown', 'gradual', 'differential']
out_path = 'figures/'

dfs = {}
for n in names:
    for i in range(f_range[0], f_range[1]):
        with open(in_path+'sim'+str(i)+'_'+n+'.json') as f:
            df = pd.DataFrame.from_dict(json.load(f)['Results']['Stats']).transpose()
            df.index = df.index.astype(int)
            df['Sim'] = i
            if i == f_range[0]:
                results = df
            else:
                results = results.append(df)
    results = results.fillna(0).reset_index().rename(columns={'index':'Timestamp'})
    min_max_timestamp = min(results.groupby('Sim').max()['Timestamp'])
    results = results[results.Timestamp <= min_max_timestamp]
    dfs[n] = results

rows_num = int(len(columns)/2) + 1
fig, axs = plt.subplots(rows_num, 2, figsize=(20,20), sharex=True)
for c in range(len(columns)):
    for n in range(len(names)):
        row = int(c/2)
        col = c-row*2
        if n < 3:
            l = sns.lineplot(data=dfs[names[n]], x='Timestamp', y=columns[c], ax=axs[row][col], color=colors[c], 
                             label=legend_names[n])  
            axs[row][col].lines[n].set_linestyle(styles[n])
        else:
            l = sns.lineplot(data=dfs[names[n]], x='Timestamp', y=columns2[c], ax=axs[row][col], color=colors[c], 
                             label=legend_names[n]) 
            axs[row][col].lines[n].set_linestyle(styles[n])
            axs[row][col].set(ylabel=columns2[c])
    if c != len(columns)-1:
        axs[row][col].legend([], [], frameon=False)
    else:
        axs[row][col].legend(loc=1)
fig.savefig(out_path+'stats.png')

dfs = {}
for n in names:
    data = []
    for i in range(f_range[0], f_range[1]):
        with open(in_path+'sim'+str(i)+'_'+n+'.json') as f:
            chain = json.load(f)['Results']['Infections chain']
        G = nx.DiGraph()
        for k in chain:
            if k[2] != 0:
                G.add_edge(k[2], k[0], timestamp=k[1])
        #   find all nodes with in_degree == 0 and define as working set
        working_set = [n for n in G if G.in_degree(n) == 0]
        j = 0
        # while working set of nodes is not empty
        while len(working_set) > 0:
            # average number of infections per infected agent
            mean_out = sum([G.out_degree(n) for n in working_set]) / len(working_set)
            max_out = max([G.out_degree(n) for n in working_set])
            # average time of infections
            out_edges = [e for n in working_set for e in G.out_edges(n)]
            if len(out_edges) > 0:
                mean_time = sum([G[e[0]][e[1]]['timestamp'] for e in out_edges]) / len(out_edges)
            else:
                mean_time = None
            # save rank, number of nodes, number of infections per node, mean infection time
            data.append([i, j, len(working_set), mean_out, max_out, mean_time])
            # update working set - neighbors of nodes currently in working set
            working_set = [e[1] for e in out_edges]
            # increase rank by 1
            j += 1
    cont_df = pd.DataFrame(data, columns=['Simulation', 'Rank', 'Nodes', 'Mean infection num',
                                      'Max infection num', 'Mean infection time'])
    dfs[n] = cont_df
    
# create graph of number of nodes, mean out degree, and mean infection time by chain level
columns = ['Nodes', 'Mean infection time', 'Mean infection num', 'Max infection num']
fig, axs = plt.subplots(len(columns), 1, figsize=(20, 30), sharex=True) 
mean_dfs = {n:dfs[n].groupby(['Simulation', 'Rank']).median() for n in names}
for c in range(len(columns)):
    for n in range(len(names)):
        sns.lineplot(data=mean_dfs[names[n]], x='Rank', y=columns[c], ax=axs[c],
                      label=legend_names[n])
        axs[c].lines[n].set_linestyle(styles[n])
        if c != len(columns)-1:
            axs[c].legend([], [], frameon=False)
        else:
            axs[c].legend(loc=1)
fig.tight_layout()
fig.savefig(out_path+'_contagionChainsStatistics.png', dpi=600)