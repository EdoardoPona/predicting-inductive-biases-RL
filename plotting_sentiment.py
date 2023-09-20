import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import csv
import numpy as np

plt.rc('text',usetex=True)
#plt.rc('text',usetex=False)
plt.rc('font', family='serif', size=16)
markers = ['o', '^', 's', 'v', 'D', 'p']

def extract_rel_mdl():
    # Initialize data structure to store total_mdl values
    data = {}

    # Navigate through all the relevant files
    for filename in os.listdir('results/stats/'):
        if "imdb_" in filename and "probing_" in filename and "lvwerra_gpt2-imdb_" in filename:
            toy = filename.split("_")[1]
            case = filename.split("_")[3]
            seed = filename.split("_")[-1].split(".")[0]

            # Initialize if not exists
            if toy not in data:
                data[toy] = {}
            if case not in data[toy]:
                data[toy][case] = []

            # Extract total_mdl value
            with open(f'results/stats/{filename}', 'r') as file:
                reader = csv.DictReader(file, delimiter="\t")
                for row in reader:
                    data[toy][case].append(float(row["total_mdl"]))

    # Compute average and SEM for each toy, case combination across seeds
    stats = {}
    for toy, cases in data.items():
        stats[toy] = {}
        for case, values in cases.items():
            stats[toy][case] = {
                'average': sum(values) / len(values),
                'sem': np.std(values) / np.sqrt(len(values))
            }

    # Compute ratio and its standard error for each toy
    ratios = {}
    sem_ratios = {}
    for toy, cases in stats.items():
        if "weak" in cases and "strong" in cases:
            ratios[toy] = cases["weak"]["average"] / cases["strong"]["average"]
            # Propagation of uncertainty for division
            rel_sem_weak = cases["weak"]["sem"] / cases["weak"]["average"]
            rel_sem_strong = cases["strong"]["sem"] / cases["strong"]["average"]
            sem_ratios[toy] = ratios[toy] * np.sqrt(rel_sem_weak**2 + rel_sem_strong**2)

    # Convert the ratios and sem_ratios dictionaries into a pandas DataFrame
    df_ratios = pd.DataFrame(list(ratios.items()), columns=["toy", "rel_mdl"])
    df_ratios["sem_rel_mdl"] = df_ratios["toy"].map(sem_ratios)
    df_ratios = df_ratios.sort_values(by="rel_mdl")

    print("Rel MDLs:")
    print(df_ratios)
    
    return df_ratios

def dataframe_to_dict(df_ratios):
    ratio_dict = {}
    for _, row in df_ratios.iterrows():
        toy = row["toy"]
        ratio_dict[int(toy)] = f"{row['rel_mdl']:.2f} \pm {row['sem_rel_mdl']:.2f}"
    return ratio_dict

def label_toy(toy):
    #print(rel_mdl_dict, label_map)
    return r"{} (${}$)".format(toy, rel_mdl_dict[toy])

data = []
sets = ['weak', 'strong','neither', 'both']
name = 'sentiment'
rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
runs = [1, 42, 43, 44]
toys = [1, 2, 5, 6, 22, 23]
case = "sentiment"
datapool = "sentiment_pool"
base_output_path = 'llm_results'
use_max = False

for toy in toys:
    for rate in rates:
        for run in runs:
            file = f'{base_output_path}/gpt2-imdb-sentiment_task{toy}_rate{rate}_seed{run}.txt'
            
            if os.path.exists(file):
                with open(file) as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        error, score = line.split("\t")
                        data.append({'toy': toy, 'rate': float(rate), 'run': run, 'error': error, 'score': float(score)})
                    
df = pd.DataFrame(data)
rel_mdl = extract_rel_mdl()
rel_mdl_dict = dataframe_to_dict(rel_mdl)

if use_max:
    df = df.groupby(['toy', 'rate', 'error'])['score'].max().reset_index()

error_map = {'neither': 'neither', 
             'both': 'both',
             'strong': r'$t$-only',
             'weak': r'$s$-only'}

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

    #ax.set_ylim(-0.05, 1.05)
    
    for j, toy in enumerate(df['toy'].unique()):

        df_toy = df[(df['toy'] == toy) & (df['error'] == error)]

        #print(df_toy)

        sns.lineplot(
            x='rate', 
            y='score',
            data=df_toy,
            label=label_toy(toy),
            alpha=0.5,
            marker=markers[j],
            legend=False,
            ax=ax
        )

        ax.set_xlabel('$p$')
        
        #df_toy = df[(df['toy'] == toy) & (df['error'] == error)]   
        #ax.plot(df_toy['rate'], df_toy['score'], marker='o', label=label_map[toy])

handles, labels = axs[0].get_legend_handles_labels()
ax_legend = fig.add_axes([0, -0.4, 1, 0.1])
legend = ax_legend.legend(handles, labels, loc="center", ncol=7, borderaxespad=0., borderpad=0.) 

legend.get_frame().set_linewidth(0)
ax_legend.axis('off')

plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)

filetime = datetime.strftime(datetime.now(), '%YY%mM%dD%Hh%Mm%Ss')
fig.savefig(f"figures/figures/rl_sentiment_lineplot_{filetime}.pdf", bbox_inches='tight', pad_inches=0., transparent=True)
fig.savefig(f"figures/figures/rl_sentiment_lineplot_{filetime}.png", bbox_inches='tight', pad_inches=0.)
plt.close()