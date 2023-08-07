#%%
#%load_ext autoreload
#%autoreload 2


# %%

import pandas as pd

import os 
if not os.path.exists("./figures"):
	os.mkdir("./figures")

if not os.path.exists("./files"):
	os.mkdir("./files")

#%% 
#import setup	
#setup.main()

#%%
pdata = pd.read_csv('files/probing.tsv', sep='\t')
total_rel = pdata['total_mdl_weak'] / pdata['total_mdl_strong']
rel = pdata['weak-mdl'] / pdata['strong-mdl']
total_rel, rel	    # these two are the same 
# %%

results = pd.read_table('files/results.tsv')
probing = pd.read_table("files/probing.tsv")
#%%
results.columns

toys = ['toy_1', 'toy_2', 'toy_3', 'toy_4', 'toy_5']
#results = results[(results['model'] == 'lstm-toy') & (results['prop'].isin(toys))]
results = results[(results['model'] == 'toy-transformer') & (results['prop'].isin(toys))]
#%%


import seaborn as sns
import matplotlib.pyplot as plt

markers = ['o', 'x', 's', 'v', '^']
errors = [
	'weak-error', 
	'strong-error', 
	'neither-error', 
	'both-error'
]

fig, axs = plt.subplots(1, 4, figsize=(15, 3))
axs = axs.flatten()
for error, ax in zip(errors, axs):
	for i, p in enumerate(results.groupby('prop')):
		label, df = p
		#print(label, df)
		df = df.sort_values('rate')
		# plot on log x scale 
		sns.lineplot(
			x='rate', 
			y=error, 
			data=df, 
			label=label, 
			alpha=0.5,
			marker=markers[i],
			ax=ax
		)
		ax.set_title(error)
		ax.set_xscale('symlog', linthresh=0.001)
		ax.set_xticks([0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
		#ax.set_xlim(0.001, 1)
		ax.set_ylim(-0.05, 1.05)
		ax.legend()
plt.show()

plt.savefig(f"figures/tt_lineplot.pdf", transparent=True)
plt.savefig(f"figures/tt_lineplot.png")
plt.close()
# %%


