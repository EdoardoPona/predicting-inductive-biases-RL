#!/bin/bash

# Get directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Iterate over each file in the current directory with "finetune" in its name
for file in *finetune*; do
    # Check if it's a regular file (and not a directory or other type of file)
    if [[ -f "$file" ]]; then
        # Call the Python script to shuffle this file
        python3 "$DIR/shuffle_tsv.py" "$file"
        echo "Shuffled $file"
    fi
done
