#!/bin/bash

# Directory path
DIR="properties/toy_6/"

# Iterate over .tsv files
for FILE in $DIR/*.tsv
do
    # Get the first 20 lines and overwrite the file
    head -n 20 $FILE > temp.tsv && mv temp.tsv $FILE
done