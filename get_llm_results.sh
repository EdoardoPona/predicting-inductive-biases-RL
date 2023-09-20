#!/bin/bash

# Directory where results will be saved
TARGET_DIR="./llm_results"
mkdir -p "$TARGET_DIR"  # Ensure the directory exists

# List of AWS instances
INSTANCES=("ec2-35-176-133-165.eu-west-2.compute.amazonaws.com"
            "ec2-13-42-24-151.eu-west-2.compute.amazonaws.com")

# Define rates
RATES=("0" "0.001" "0.01" "0.05" "0.1" "0.2" "0.5")

# For each instance
for instance in "${INSTANCES[@]}"; do
    echo "Accessing $instance..."
    
    # Fetch all valid directories that match our pattern
    DIRECTORIES=$(ssh -i "~/.ssh/group-4.pem" diogo@"$instance" "find ASH_code/predicting-inductive-biases-RL/lvwerra/ -type d -regex '.*/gpt2-imdb-sentiment_task[0-9]+_rate.*_seed[0-9]+'")
    
    # Convert DIRECTORIES into an array
    DIR_ARRAY=($DIRECTORIES)
    
    # Loop over each directory in the array
    for dir in "${DIR_ARRAY[@]}"; do
        for r in "${RATES[@]}"; do
            if [[ $dir == *"_rate${r}_seed"* ]]; then
                # Extract task and seed values
                task_value=$(echo "$dir" | grep -oP 'task\K[0-9]+')
                seed_value=$(echo "$dir" | grep -oP 'seed\K[0-9]+')

                # Construct the remote file path
                REMOTE_PATH="${dir}/gpt2-imdb-sentiment_task${task_value}_rate${r}_seed${seed_value}.txt"
                # echo $REMOTE_PATH
                
                # Check if the .txt file exists on the remote machine
                if ssh -i "~/.ssh/group-4.pem" diogo@"$instance" "[ -f $REMOTE_PATH ]"; then
                    # Copy the .txt file if it exists
                    scp -i "~/.ssh/group-4.pem" diogo@"$instance":"$REMOTE_PATH" "$TARGET_DIR/"
                    echo "Found and copied: $REMOTE_PATH from $instance"
                else
                    echo "Warning: Expected file $REMOTE_PATH does not exist on $instance."
                fi
            fi
        done
    done
done

echo "All done!"