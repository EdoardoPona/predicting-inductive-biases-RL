import os
import json 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=16)
markers = ['o', '^', 's', 'v']

data = []
case = 'toy'
toys = [1, 2, 3, 5]
rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
sets = ['weak', 'strong','neither', 'both']
name = 'g4_toy'
#name = 'lovering_toy'
n_layers = 2
hidden_size = 128

for toy in toys:
    for rate in rates:
        for error in sets:
            path = f'rl_results/results_{case}{toy}_rate{rate}/rl4lms/{case}{toy}_r{rate}_ep5_l{n_layers}_h{hidden_size}_steps64'
            file = f'{path}/{error}_split_metrics.jsonl'
            
            if os.path.exists(file):
                with open(file) as f:
                    lines = f.read().splitlines()
                    last_line = json.loads(lines[-1])
                    score = last_line['metrics'][f'synthetic/{name}']
                    data.append({'toy': toy, 'rate': float(rate), 'error': error, 'score': score})
                    
df = pd.DataFrame(data)

error_map = {'neither': 'neither', 
             'both': 'both',
             'strong': r'$t$-only',
             'weak': r'$s$-only'}
#df['error'] = df['error'].map(error_map)
label_map = {
    1: 'contains-1',
    2: 'prefix-dupl',
    3: 'adj-dupl', 
    5: 'first-last'
}

fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
plt.subplots_adjust(wspace=0.02, hspace=0)
xticks = [float(rate) for rate in rates]

for i, error in enumerate(sets):
    
    ax = axs[i]
    
    ax.set_title(error_map[error])
    if i==0:
        ax.set_ylabel('Reward')

    ax.set_xscale('symlog', linthresh=0.001) 
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90)

    ax.set_ylim(-0.05, 1.05)
    
    for j, toy in enumerate(df['toy'].unique()):
        if toy == 4: 
            continue

        df_toy = df[(df['toy'] == toy) & (df['error'] == error)]

        #print(df_toy)

        sns.lineplot(
            x='rate', 
            y='score', 
            data=df_toy,
            label=label_map[toy],
            alpha=0.5,
            marker=markers[j],
            legend=False,
            ax=ax
        )
        
        #df_toy = df[(df['toy'] == toy) & (df['error'] == error)]   
        #ax.plot(df_toy['rate'], df_toy['score'], marker='o', label=label_map[toy])

handles, labels = axs[0].get_legend_handles_labels()
ax_legend = fig.add_axes([0, -0.4, 1, 0.1])
legend = ax_legend.legend(handles, labels, loc="center", ncol=4, borderaxespad=0., borderpad=0.) 

legend.get_frame().set_linewidth(0)
ax_legend.axis('off')

plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)

fig.savefig("figures/figures/rl_lineplot.pdf", bbox_inches='tight', pad_inches=0., transparent=True)
fig.savefig("figures/figures/rl_lineplot.png", bbox_inches='tight', pad_inches=0.)
plt.close()