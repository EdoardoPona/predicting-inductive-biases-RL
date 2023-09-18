#!/bin/bash

tasks=(1 2 5 6 8 9)

#tasks_2=(1 2 5 6 7 8 9 10 11)
tokens=16

for task in "${tasks[@]}"
    do
        echo "------ Generating task $task with $tokens max tokens ------"
        python imdb_dataset_preprocessing.py --true_property=$task --max_tokens=$tokens --train_size=45_000
    done