import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os 
if not os.path.exists("./figures"):
	os.mkdir("./figures")

if not os.path.exists("./files"):
	os.mkdir("./files")

results = pd.read_table('files/results.tsv')

toys = ['toy_1', 'toy_2', 'toy_3', 'toy_4', 'toy_5']
results = results[(results['model'] == 'lstm-toy') & (results['prop'].isin(toys))]

markers = ['o', 'x', 's', 'v', '^']
xticks = [0., 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=16)

error_mapping = {
    's-only': 'weak-error',
    't-only': 'strong-error',
    'neither': 'neither-error',
    'both': 'both-error'
}

fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
plt.subplots_adjust(wspace=0.02, hspace=0)
axs = axs.flatten()
for error, ax in zip(list(error_mapping), axs):
    #ax.set_aspect('equal', adjustable='box')
    for i, p in enumerate(results.groupby('prop')):
        if p[0] == 'toy_4': 
            continue
        label, df = p
        df = df.sort_values('rate')
        
        if label == 'toy_1':
            label = 'contains-1'
        elif label == 'toy_2':
            label = 'prefix-dupl'
        elif label == 'toy_3':
            label = 'adj-dupl'
        elif label == 'toy_5':
            label = 'first-last'

        sns.lineplot(
            x='rate', 
            y=error_mapping[error], 
            data=df,
            label=label,
            alpha=0.5,
            marker=markers[i],
            legend=False,
            ax=ax
        )
        
        ax.set_title(error)
        ax.set_ylabel("Error") 
        ax.set_xscale('symlog', linthresh=0.001)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=90)
        ax.set_ylim(-0.05, 1.05)

handles, labels = axs[0].get_legend_handles_labels()
ax_legend = fig.add_axes([0, -0.4, 1, 0.1])
legend = ax_legend.legend(handles, labels, loc="center", ncol=4, borderaxespad=0., borderpad=0.) 

legend.get_frame().set_linewidth(0)
ax_legend.axis('off')

plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)

fig.savefig(f"figures/lstm_lineplot.pdf", bbox_inches='tight', pad_inches=0., transparent=True)
fig.savefig(f"figures/lstm_lineplot.png", bbox_inches='tight', pad_inches=0.)
plt.close()