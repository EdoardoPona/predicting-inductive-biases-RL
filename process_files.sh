#!/bin/bash

# Loop through the integers 24 to 29
for num in {24..29}; do
    dir="imdb_$num"
    # Check if directory exists
    if [[ -d "$dir" ]]; then
        # Process files within the directory
        for file in "$dir"/*_train.tsv; do
            # Check if the file is not a directory
            if [[ -f "$file" ]]; then
                head -n 24577 "$file" > "${file}.tmp"
                mv "${file}.tmp" "$file"
            fi
        done
    fi
done
