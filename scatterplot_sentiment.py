import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import os
import csv
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=16)

filetime = datetime.strftime(datetime.now(), '%YY%mM%dD%Hh%Mm%Ss')

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

def extract_reward():
    data = []
    rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
    runs = [42, 43, 44]
    toys = [1, 2, 5, 22, 23]
    base_output_path = 'llm_results'

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
    return df

def create_combined_dataframe(use_max=True):
    # Call the functions to get the necessary data
    df_rel_mdl = extract_rel_mdl()
    df_reward = extract_reward()

    # Ensure that the 'toy' column in both DataFrames are of the same type
    df_rel_mdl['toy'] = df_rel_mdl['toy'].astype(int)
    df_reward['toy'] = df_reward['toy'].astype(int)

    if use_max:
        # First, group by toy, error, rate and get the maximum score over 'run'
        max_scores_by_run = df_reward.groupby(['toy', 'error', 'rate'])['score'].max().reset_index()

        # Next, group by toy and error to get the average of the max scores over 'rate'
        avg_scores = max_scores_by_run.groupby(['toy', 'error']).agg({'score': ['mean', lambda x: np.std(x)/np.sqrt(len(x))]}).reset_index()
        avg_scores.columns = ['toy', 'error', 'score', 'sem_score']

    else:
        # Group by toy and get the average score
        avg_scores = df_reward.groupby(['toy', 'error']).agg({'score': ['mean', lambda x: np.std(x)/np.sqrt(len(x))]}).reset_index()
        avg_scores.columns = ['toy', 'error', 'score', 'sem_score']

    # Merge both DataFrames on toy
    final_df = pd.merge(df_rel_mdl, avg_scores, on='toy', how='inner')
    
    print("Final data:")
    print(final_df)

    return final_df

def scatter(scatterdata):
    
    def plot_model(scatterdata, ax, error_value):
        MEASURE = "rel_mdl"
        METRIC = "score"
        scatterdata = scatterdata[scatterdata["error"] == error_value]

        sns.regplot(
            y=METRIC,  
            x=MEASURE,
            data=scatterdata,
            logistic=False,
            scatter_kws={"s": 25},
            ax=ax,
            label=error_value
        )

    def plot(scatterdata):
        #print("Plotting...")
        fig, ax = plt.subplots(figsize=(8, 6))

        # Loop through each unique value in the error column
        for error_value in scatterdata["error"].unique():
            plot_model(scatterdata, ax, error_value)

        ax.set_title("GPT-2 IMDB")
        ax.set_xticks([0.01, 1, 2])
        ax.set_ylabel(r"Average Reward", fontsize=12)
        ax.legend(title='Error Type')

        plt.gcf().text(
            0.5,
            -0.05,
            "Relative extractibility of target feature (MDL($s$)/MDL($t$))",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=12,
        )

        plt.savefig(
            f"figures/figures/scatterplot_sentiment_{filetime}.pdf", transparent=True, bbox_inches="tight"
        )  
        plt.savefig(
            f"figures/figures/scatterplot_sentiment_{filetime}.png", bbox_inches="tight"
        )
        plt.close()

    plot(scatterdata)

def scatter_2(scatterdata):
    
    def plot(scatterdata):
        print("Plotting...")
        fig, ax = plt.subplots(figsize=(8, 6))

        # Loop through each unique value in the error column to plot lines
        for error_value in scatterdata["error"].unique():
            subset = scatterdata[scatterdata["error"] == error_value]
            sns.lineplot(
                y="score",
                x="rel_mdl",
                data=subset,
                ax=ax,
                label=error_value,
                legend=False  # We set this to False to prevent duplicate legends
            )

        # Use scatterplot for points with hue and style as the "error" column
        markers = ['o', '^', 's', 'v']
        sns.scatterplot(
            y="score",
            x="rel_mdl",
            data=scatterdata,
            hue="error",
            style="error",
            s=50,
            ax=ax,
            palette="deep",
            markers=markers,  # example markers
            legend=False  # We set this to False because we'll customize it later
        )

        # Place the legend in the bottom left
        ax.legend(loc='lower right')

        ax.set_title("GPT-2 IMDB Movie Review")
        #ax.set_xticks([0.01, 1, 2])
        ax.set_ylabel(r"Average Reward")
        ax.set_xlabel(r"Relative MDL")
        ax.set_xscale('log')
        #ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=30))
        desired_ticks = [0.1, 0.2, 0.5, 1, 2, 5]
        ax.xaxis.set_major_locator(ticker.FixedLocator(desired_ticks))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        plt.savefig(
            f"figures/figures/scatterplot_sentiment_{filetime}.pdf", transparent=True, bbox_inches="tight"
        )  
        plt.savefig(
            f"figures/figures/scatterplot_sentiment_{filetime}.png", bbox_inches="tight"
        )
        plt.close()

    plot(scatterdata)

def scatter_3(scatterdata):
    
    def plot(scatterdata):
        print("Plotting...")
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = sns.color_palette("deep", n_colors=len(scatterdata["error"].unique()))

        # Loop through each unique value in the error column to plot lines
        for index, error_value in enumerate(scatterdata["error"].unique()):
            subset = scatterdata[scatterdata["error"] == error_value]
            color = palette[index]
            sns.lineplot(
                y="score",
                x="rel_mdl",
                data=subset,
                ax=ax,
                color=color,
                label=error_value,
                legend=False  # We set this to False to prevent duplicate legends
            )

            # Use plt.errorbar for points with error bars, using the same color as the line
            ax.errorbar(
                subset["rel_mdl"],
                subset["score"],
                yerr=subset["sem_score"],
                xerr=subset["sem_rel_mdl"],
                fmt='o',  # format string for points
                color=color,
                label=None  # No label for these
            )

        # Use scatterplot for points with hue and style as the "error" column
        markers = ['o', '^', 's', 'v']
        sns.scatterplot(
            y="score",
            x="rel_mdl",
            data=scatterdata,
            hue="error",
            style="error",
            s=50,
            ax=ax,
            palette=palette,
            markers=markers,  # example markers
            legend=False  # We set this to False because we'll customize it later
        )

        # Place the legend in the bottom left
        ax.legend(loc='lower right')

        ax.set_title("GPT-2 IMDB Movie Review")
        ax.set_ylabel(r"Average Reward")
        ax.set_xlabel(r"Relative MDL")
        ax.set_xscale('log')
        desired_ticks = [0.1, 0.2, 0.5, 1, 2, 5]
        ax.xaxis.set_major_locator(ticker.FixedLocator(desired_ticks))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        # Save the figure
        plt.savefig(
            f"figures/figures/scatterplot_sentiment_{filetime}.pdf", transparent=True, bbox_inches="tight"
        )  
        plt.savefig(
            f"figures/figures/scatterplot_sentiment_{filetime}.png", bbox_inches="tight"
        )
        plt.close()

    plot(scatterdata)

scatterdata = create_combined_dataframe()
print(scatterdata)
scatter_3(scatterdata)
