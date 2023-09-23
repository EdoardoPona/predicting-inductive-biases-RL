import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import warnings
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)

import matplotlib.ticker as ticker
from llm_aux_plotting import *
plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=24)

markers = ['o', '^', 's', 'v', 'D', 'p','<','>']
filetime = datetime.strftime(datetime.now(), '%YY%mM%dD%Hh%Mm%Ss')

supertask = 'toxicity'

if supertask == 'sentiment':
    rates = ["0", "0.01", "0.05", "0.2", "0.5"]
    res_seeds = [42, 43, 44]
    mdl_seeds = [101, 102, 103, 104, 105]
    toys = [1, 2, 5, 22, 23]
    reward_scale = 2.8

elif supertask == 'sentiment-large':
    rates = ["0", "0.01", "0.2"]
    res_seeds = [9000]
    mdl_seeds = [101, 102, 103, 104, 105] #TODO: Don't have them yet
    toys = [0, 22, 23]
    reward_scale = 3.

elif supertask == 'toxicity':
    rates = ["0", "0.01", "0.2"]
    res_seeds = [42, 43, 44]
    mdl_seeds = [201, 202, 203]
    toys = [2, 6, 22, 23]
    reward_scale = 2.8

use_max = False
reward_threshold = 0.7
use_rate_0 = True
use_errorbars = False
use_mdl_gradient = False

assert supertask in supertasks

rel_mdl, df, final_df = create_combined_dataframe(supertask, toys, rates, res_seeds, mdl_seeds, use_max, reward_threshold, use_rate_0, reward_scale)
rel_mdl_dict = dataframe_to_dict(rel_mdl)
scatterdata = merge_t_and_s(final_df)
mdl_dict = rel_mdl.set_index('toy')['rel_mdl'].to_dict()
print(rel_mdl)
print(scatterdata)

def plot_4_subplots():
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    plt.subplots_adjust(wspace=0.02, hspace=0)
    xticks = [float(rate) for rate in rates]

    if use_mdl_gradient:
        color_map, sm = generate_color_map(rel_mdl)

    for i, error in enumerate(cases):
        ax = axs[i]
        ax.set_title(error_map[error])
        if i==0:
            ax.set_ylabel('Reward')
        ax.set_xscale('symlog', linthresh=0.015) 
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=90)
        ax.set_ylim(-0.02, 1.02)
        for j, toy in enumerate(df['toy'].unique()):
            df_toy = df[(df['toy'] == toy) & (df['error'] == error)]
            line_color = color_map[int(toy)] if use_mdl_gradient else None
            sns.lineplot(
                x='rate', 
                y='score',
                data=df_toy,
                label='task ' + str(toy),
                alpha=1.,
                ms=7,
                marker=markers[j],
                legend=False,
                color=line_color,
                errorbar=eb[use_errorbars],
                ax=ax
            )
            ax.set_xlabel('$p$')

    if use_mdl_gradient:
        fig.subplots_adjust(right=0.8)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, location='right')
        cbar.set_label('Relative MDL', rotation=270, labelpad=15)
        print(list(mdl_dict.values()))
        cbar.set_ticks(list(mdl_dict.values()))
        cbar.set_ticklabels([f"{mdl:.2f}" for mdl in mdl_dict.values()])

    handles, labels = axs[0].get_legend_handles_labels()
    ax_legend = fig.add_axes([0, -0.5, 1, 0.1])
    legend = ax_legend.legend(handles, labels, loc="center", ncol=7, borderaxespad=0., borderpad=0.) 
    legend.get_frame().set_linewidth(0)
    ax_legend.axis('off')
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    fig.savefig(f"figures/figures/4_subplots_{supertask}_{filetime}.pdf", bbox_inches='tight', pad_inches=0., transparent=True)
    fig.savefig(f"figures/figures/4_subplots_{supertask}_{filetime}.png", bbox_inches='tight', pad_inches=0.)
    plt.close()

def plot_rel_mdl():
    fig, ax = plt.subplots(figsize=(8, 6))
    # sns.lineplot(
    #     y="avg_score",
    #     x="rel_mdl",
    #     data=scatterdata,
    #     ax=ax,
    #     legend=False  # We set this to False to prevent duplicate legends
    # )

    # Use plt.errorbar for points with error bars, using the same color as the line
    ax = sns.regplot(
            y='avg_score',  # test_f_score
            x='rel_mdl',
            data=scatterdata,
            logistic=True,
            scatter_kws={"s": 25},
            ax=ax,
        )
    
    # Getting the color of the regression line
    line_color = ax.lines[0].get_color()

    ax.errorbar(
        scatterdata["rel_mdl"],
        scatterdata["avg_score"],
        yerr=scatterdata["avg_sem_score"],
        xerr=scatterdata["sem_rel_mdl"],
        fmt='o',
        color=line_color
    )

    #ax.set_title("GPT-2 IMDB Movie Review")
    ax.set_ylabel(r"Average Reward")
    ax.set_xlabel(r"Relative MDL")
    ax.set_xscale('log')
    desired_ticks = [0.1, 0.2, 0.5, 1, 2, 5]
    ax.xaxis.set_major_locator(ticker.FixedLocator(desired_ticks))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.set_xlim([0.17,6])
    ax.autoscale(axis='x', tight=True)
    ax.set_ylim([-0.02, 1.02])

    # Save the figure
    plt.savefig(
        f"figures/figures/rel_mdl_{supertask}_{filetime}.pdf", transparent=True, bbox_inches="tight"
    )  
    plt.savefig(
        f"figures/figures/rel_mdl_{supertask}_{filetime}.png", bbox_inches="tight"
    )
    plt.close()

plot_4_subplots()
plot_rel_mdl()
