import os
import json 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#try:
#    plt.rc('text',usetex=True)
#except:
plt.rc('text',usetex=False)
plt.rc('font', family='serif', size=16)
markers = ['o', '^', 's', 'v']

data = []
#toys = [1]
#rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
sets = ['weak', 'strong','neither', 'both']
name = 'sentiment'
rates = ["0", "0.2", "0.5"]
#rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
runs = 1
toys = [1, 2, 3]
case = "sentiment"
datapool = "sentiment_pool"
reward = 'xlnet_imdb_sentiment_cls'
metric = 'xlnet_imdb_sentiment_cls'
base_output_path = 'rl_results'
exp_name = 'sentiment'

for toy in toys:
    for rate in rates:
        for run in range(runs):
            for error in sets:
                path = f'rl_results/results_{case}{toy}_rate{rate}_run{run}/rl4lms/{case}{toy}_r{rate}_{run}'
                file = f'{path}/{error}_split_metrics.jsonl'
                
                if os.path.exists(file):
                    with open(file) as f:
                        lines = f.read().splitlines()
                        last_line = json.loads(lines[-1])
                        score = last_line['metrics'][f'synthetic/{name}']
                        data.append({'toy': toy, 'rate': float(rate), 'run': run, 'error': error, 'score': -score})
                    
df = pd.DataFrame(data)

error_map = {'neither': 'neither', 
             'both': 'both',
             'strong': r'$t$-only',
             'weak': r'$s$-only'}
#df['error'] = df['error'].map(error_map)
label_map = {
    1: r'sentiment ($ vs #)',
    2: r'sentiment (x/10 vs review)',
    3: r'sentiment (infer vs review)'
}

label_map = {
    1: r'rel. MDL 2.2747 (task $ vs #)',
    2: r'rel. MDL 0.3434 (task x/10)',
    3: r'rel. MDL 0.1932 (task infer)'
}

fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
plt.subplots_adjust(wspace=0.02, hspace=0)
xticks = [float(rate) for rate in rates]

for i, error in enumerate(sets):
    
    ax = axs[i]
    
    ax.set_title(error_map[error])
    if i==0:
        ax.set_ylabel('-Reward')

    ax.set_xscale('symlog', linthresh=0.001) 
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90)

    #ax.set_ylim(-0.05, 1.05)
    
    for j, toy in enumerate(df['toy'].unique()):

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

        ax.set_xlabel('Evidence $p$ against $s$')
        
        #df_toy = df[(df['toy'] == toy) & (df['error'] == error)]   
        #ax.plot(df_toy['rate'], df_toy['score'], marker='o', label=label_map[toy])

handles, labels = axs[0].get_legend_handles_labels()
ax_legend = fig.add_axes([0, -0.4, 1, 0.1])
legend = ax_legend.legend(handles, labels, loc="center", ncol=4, borderaxespad=0., borderpad=0.) 

legend.get_frame().set_linewidth(0)
ax_legend.axis('off')

plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)

fig.savefig("figures/figures/rl_llm_lineplot.pdf", bbox_inches='tight', pad_inches=0., transparent=True)
fig.savefig("figures/figures/rl_llm_lineplot.png", bbox_inches='tight', pad_inches=0.)
plt.close()