import os
import csv
import statistics

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

# Function to compute average and variance
def compute_statistics(values):
    if len(values) == 0:
        return (0, 0)
    return (statistics.mean(values), statistics.variance(values))

# Calculate and print individual results and overall statistics
for task in sorted(data.keys()):
    weak_values = []
    strong_values = []
    ratios = []
    
    for seed in sorted(data[task].keys()):
        weak_value = data[task][seed].get('weak', None)
        strong_value = data[task][seed].get('strong', None)
        
        if weak_value and strong_value:
            ratio = weak_value / strong_value
            weak_values.append(weak_value)
            strong_values.append(strong_value)
            ratios.append(ratio)
            print(f"{task} & {seed} & {weak_value:.2f} & {strong_value:.2f} & {ratio:.2f} \\\\")

    # Calculate average and variance for weak, strong, and ratios
    weak_avg, weak_var = compute_statistics(weak_values)
    strong_avg, strong_var = compute_statistics(strong_values)
    ratio_avg, ratio_var = compute_statistics(ratios)
    
    print(f"\nStatistics for Task {task}:")
    print(f"Weak Average: {weak_avg:.2f}, Weak Variance: {weak_var:.2f}")
    print(f"Strong Average: {strong_avg:.2f}, Strong Variance: {strong_var:.2f}")
    print(f"Ratio Average: {ratio_avg:.2f}, Ratio Variance: {ratio_var:.2f}\n")
