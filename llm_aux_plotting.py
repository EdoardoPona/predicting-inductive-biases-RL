import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
import matplotlib.colors as mcolors
import os
import csv
import pandas as pd
import numpy as np

supertasks = ['sentiment','summarization','toxicity','sentiment-large']

dataset_prefix = {
    'sentiment': 'imdb',
    'summarization': 'summary',
    'toxicity': 'toxic0.7',
    'sentiment-large': 'imdb'
}

model_name_res = {
    'sentiment': 'gpt2-imdb-sentiment',
    'summarization': NotImplementedError,
    'toxicity': 'toxic',
    'sentiment-large': 'gpt2-large-sentiment'
}

model_name_mdl = {
    'sentiment': 'lvwerra_gpt2-imdb',
    'summarization': NotImplementedError,
    'toxicity': 'ash-23-g4_gpt2-warmup-toxic0.9-split-1.0-epochs-5',
    'sentiment-large': 'lvwerra_gpt2-imdb' #TODO: Don't have them yet
}

title = {
    'sentiment': 'GPT-2 review sentiment',
    'summarization': NotImplementedError,
    'toxicity': 'GPT-2 toxicity',
    'sentiment-large': 'GPT-2-large review sentiment'
}

error_map = {
    'neither': 'neither', 
    'both': 'both',
    'strong': r'$t$-only',
    'weak': r'$s$-only'
}

cases = ['weak', 'strong','neither', 'both']

eb = {True: ("ci", 95), False: None}

def generate_color_map(rel_mdl, use_log=True, color_type='plasma'):
    rel_mdl_dict = rel_mdl.set_index('toy')['rel_mdl'].to_dict()
    #min_rel_mdl = min(rel_mdl_dict.values()) - 0.5
    min_rel_mdl = 0.1
    #max_rel_mdl = max(rel_mdl_dict.values()) + 0.1
    max_rel_mdl = 10.0
    if not use_log:
        norm = mcolors.Normalize(vmin=min_rel_mdl, vmax=max_rel_mdl)
    else:
        norm = mcolors.LogNorm(vmin=min_rel_mdl, vmax=max_rel_mdl)
    colormap = plt.get_cmap(color_type)
    color_dict = {int(toy): colormap(norm(value)) for toy, value in rel_mdl_dict.items()}
    sm = plt.cm.ScalarMappable(cmap=color_type, norm=norm)
    sm.set_array([])
    return color_dict, sm

def dataframe_to_dict(df_ratios):
    ratio_dict = {}
    for _, row in df_ratios.iterrows():
        toy = row["toy"]
        ratio_dict[int(toy)] = f"{row['rel_mdl']:.2f} \\pm {row['sem_rel_mdl']:.2f}"
    return ratio_dict

def label_toy(toy, rel_mdl_dict):
    #print(rel_mdl_dict, label_map)
    return r"{} (${}$)".format(toy, rel_mdl_dict[toy])

def extract_rel_mdl(supertask, toys, runs):
    # Initialize data structure to store total_mdl values
    data = {}
    model = model_name_mdl[supertask]

    #print(toys, rates, runs)
    # Navigate through all the relevant files
    for filename in os.listdir('results/stats/'):
        #print(filename, dataset_prefix[supertask], model)
        if f"{dataset_prefix[supertask]}_" in filename and "probing_" in filename and f"{model}_" in filename:
            toy = int(filename.split("_")[1])
            case = filename.split("_")[3]
            seed = int(filename.split("_")[-1].split(".")[0])
            #print(" A Going through ", toy, case, seed)

            if toy not in toys or seed not in runs:
                continue
            #print("Going through ", toy, case, seed)
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

    #print("Rel MDLs:")
    #print(df_ratios)
    
    return df_ratios

def extract_reward(supertask, toys, rates, runs, use_max=False, reward_threshold=-1000., reward_scale=2.8):
    data = []

    for toy in toys:
        for rate in rates:
            for run in runs:
                file = f'llm_results/{model_name_res[supertask]}_task{toy}_rate{rate}_seed{run}.txt'
                
                if os.path.exists(file):
                    with open(file) as f:
                        lines = f.read().splitlines()
                        for line in lines:
                            error, score = line.split("\t")
                            fl_score = (float(score) + reward_scale) / (2*reward_scale)
                            data.append({'toy': toy, 'rate': float(rate), 'run': run, 'error': error, 'score': fl_score})
    
    df = pd.DataFrame(data)

    # Filter out entries with score below reward_threshold and error either 'neither' or 'both'
    # filtered_df = df[(df['score'] < reward_threshold) & ((df['error'] == 'neither') | (df['error'] == 'both'))]
    filtered_df = df[(df['score'] < reward_threshold) & ((df['error'] == 'neither'))]
    to_remove = filtered_df[['toy', 'rate', 'run']].drop_duplicates().values

    # Remove the entries for the filtered (toy, rate, run) combinations
    for toy, rate, run in to_remove:
        df = df[~((df['toy'] == toy) & (df['rate'] == rate) & (df['run'] == run))]

    if use_max:
        df = df.groupby(['toy', 'rate', 'error'])['score'].max().reset_index()

    return df

def create_combined_dataframe(supertask, toys, rates, res_seeds, mdl_seeds, use_max=False, reward_threshold=-1000., use_rate_0=False, reward_scale=2.8):
    # Call the functions to get the necessary data
    df_rel_mdl = extract_rel_mdl(supertask, toys, mdl_seeds)
    df_reward = extract_reward(supertask, toys, rates, res_seeds, use_max=use_max, reward_threshold=reward_threshold, reward_scale=reward_scale)

    # Ensure that the 'toy' column in both DataFrames are of the same type
    df_rel_mdl['toy'] = df_rel_mdl['toy'].astype(int)
    df_reward['toy'] = df_reward['toy'].astype(int)    

    # Adjust the grouping based on use_rate_0
    if use_rate_0:
        df_reward_2 = df_reward[df_reward['rate'] == 0.0]
        avg_scores = df_reward_2.groupby(['toy', 'error']).agg({'score': ['mean', lambda x: np.std(x)/np.sqrt(len(x))]}).reset_index()
    else:
        avg_scores = df_reward.groupby(['toy', 'error', 'rate']).agg({'score': ['mean', lambda x: np.std(x)/np.sqrt(len(x))]}).reset_index()
    
    avg_scores.columns = ['toy', 'error', 'score', 'sem_score']

    # Merge both DataFrames on toy
    final_df = pd.merge(df_rel_mdl, avg_scores, on='toy', how='inner')

    return df_rel_mdl, df_reward, final_df


def merge_t_and_s(final_df):
    # Filter out rows with 'both' and 'neither'
    filtered_df = final_df[~final_df['error'].isin(['both', 'neither'])]
    
    # Group by toy and average out the scores and compute new sem_score
    grouped = filtered_df.groupby(['toy', 'rel_mdl', 'sem_rel_mdl'])
    
    def compute_avg_scores(group):
        strong_score = group[group['error'] == 'strong']['score'].values[0]
        weak_score = group[group['error'] == 'weak']['score'].values[0]
        strong_sem = group[group['error'] == 'strong']['sem_score'].values[0]
        weak_sem = group[group['error'] == 'weak']['sem_score'].values[0]
        
        avg_score = (strong_score + weak_score) / 2
        avg_sem = np.sqrt(strong_sem**2 + weak_sem**2)
        
        return pd.Series({'avg_score': avg_score, 'avg_sem_score': avg_sem})
    
    result = grouped.apply(compute_avg_scores).reset_index()
    
    return result

