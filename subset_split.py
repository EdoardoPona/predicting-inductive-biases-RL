#%%
''' script to split the datasets into their 4 subsets strong (t-only), weak (s-only), both, neither '''
import pandas as pd 
import os 

data_path = './properties/toy_1'
filename = 'test.tsv'

#%%
df = pd.read_csv(os.path.join(data_path, filename), sep='\t')
df

#%%
for sub in df.groupby('section'):
    section, sub_df = sub
    sub_df.to_csv(os.path.join(data_path, f'test_{section}.tsv'), sep='\t', index=False)

# %%

