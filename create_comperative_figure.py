import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 24

in_path = 'outputs/'
f_range=(1,31)
columns = ['R', 'Recovered', 'Quarantined', 'New_quarantined', 
           'Total_infected', 'New_infections', 'Infected', 'Hospitalized',
           'New_hospitalizations', 'Total_dead', 'New_deaths']
colors = ['black', 'green', 'orange', 'orange', 'red', 'red', 'red', 
          'brown', 'brown', 'grey', 'grey', 'pink']
styles = ['-', '--']
names = ['nolockdown_norm0.08', 'ALL_norm0.08']
legend_names = ['No lockdown',  'Lockdown']
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
    dfs[n] = results

fig, axs = plt.subplots(len(columns), figsize=(20,20), sharex=True)
for c in range(len(columns)):
    for n in range(len(names)):
        l = sns.lineplot(data=dfs[names[n]], x='Timestamp', y=columns[c], ax=axs[c], color=colors[c], 
                     label=legend_names[n])  
        axs[c].lines[n].set_linestyle(styles[n])
    if c != len(columns)-1:
        axs[c].legend([], [], frameon=False)
    else:
        axs[c].legend(loc=1)
fig.savefig(out_path+'stats.png')

