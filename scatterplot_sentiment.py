import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import os
import csv
import pandas as pd
import matplotlib.ticker as ticker

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

    # Compute average for each toy, case combination across seeds
    averages = {}
    for toy, cases in data.items():
        averages[toy] = {}
        for case, values in cases.items():
            averages[toy][case] = sum(values) / len(values)

    # Compute ratio for each toy
    ratios = {}
    for toy, cases in averages.items():
        if "weak" in cases and "strong" in cases:
            ratios[toy] = cases["weak"] / cases["strong"]

    # Convert the ratios dictionary into a pandas DataFrame
    df_ratios = pd.DataFrame(list(ratios.items()), columns=["toy", "rel_mdl"])
    df_ratios = df_ratios.sort_values(by="rel_mdl")

    print("Rel MDLs:")
    print(df_ratios)
    
    return df_ratios

def extract_reward():
    data = []
    rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
    runs = [1, 42, 43, 44]
    toys = [1, 2, 5, 6, 22, 23]
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

def create_combined_dataframe():
    # Call the functions to get the necessary data
    df_rel_mdl = extract_rel_mdl()
    df_reward = extract_reward()

    # Ensure that the 'toy' column in both DataFrames are of the same type
    df_rel_mdl['toy'] = df_rel_mdl['toy'].astype(int)
    df_reward['toy'] = df_reward['toy'].astype(int)

    # First, group by toy, error, rate and get the maximum score over 'run'
    max_scores_by_run = df_reward.groupby(['toy', 'error', 'rate'])['score'].max().reset_index()

    # Next, group by toy and error to get the average of the max scores over 'rate'
    avg_scores = max_scores_by_run.groupby(['toy', 'error'])['score'].mean().reset_index()

    # # Group by toy and get the average score
    # avg_scores = df_reward.groupby(['toy', 'error'])['score'].mean().reset_index()
    # #avg_scores = avg_scores.rename(columns={'score': 'error'})

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

        ax.set_title("GPT-2 IMDB")
        #ax.set_xticks([0.01, 1, 2])
        ax.set_ylabel(r"Average Reward", fontsize=12)
        ax.set_xlabel(r"Relative MDL")
        ax.set_xscale('log')
        #ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=30))
        desired_ticks = [0.1, 0.2, 0.5, 1, 2, 5]
        ax.xaxis.set_major_locator(ticker.FixedLocator(desired_ticks))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())


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

scatterdata = create_combined_dataframe()
scatter_2(scatterdata)
