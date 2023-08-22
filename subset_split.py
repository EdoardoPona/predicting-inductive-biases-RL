#%%
''' script to split the datasets into their 4 subsets strong (t-only), weak (s-only), both, neither '''
import pandas as pd 
import os 

data_path = './properties'
tasks = ['toy_1', 'toy_2', 'toy_3', 'toy_5']  # List of tasks to process
filename = 'test.tsv'

for task in tasks:
    task_path = os.path.join(data_path, task)
    
    # Read the data
    df = pd.read_csv(os.path.join(task_path, filename), sep='\t')
    print(f"Processing {task}...")
    
    # Split the data and save to separate files
    for sub in df.groupby('section'):
        section, sub_df = sub
        sub_df.to_csv(os.path.join(task_path, f'test_{section}.tsv'), sep='\t', index=False)

