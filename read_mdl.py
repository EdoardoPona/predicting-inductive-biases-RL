import os
import csv

# Directory containing the files
dir_path = 'results/stats/'

# Get all .tsv files in the directory
all_files = [f for f in os.listdir(dir_path) if f.endswith('.tsv') and f.startswith('imdb_')]

# Dictionary to store values
data = {}

for file in all_files:
    # Extract task, case, and seed from the filename
    parts = file.split("_")
    task = int(parts[1])
    case = parts[3]
    seed = int(parts[5].split('.')[0])
    
    # Read the TSV file and extract the required value
    with open(os.path.join(dir_path, file), 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        rows = list(reader)
        value = float(rows[1][4])  # Fifth number in the second row

        # Store the value in the dictionary
        if task not in data:
            data[task] = {}
        if seed not in data[task]:
            data[task][seed] = {}
        
        data[task][seed][case] = value

# Print the results
for task in sorted(data.keys()):
    print(f"Task {task}:")
    for seed in sorted(data[task].keys()):
        weak_value = data[task][seed].get('weak', None)
        strong_value = data[task][seed].get('strong', None)
        
        if weak_value and strong_value:
            ratio = weak_value / strong_value
            print(f"& 192 & 20 & {seed} & {weak_value:.2f} & {strong_value:.2f} & {ratio:.4f} \\\\")
