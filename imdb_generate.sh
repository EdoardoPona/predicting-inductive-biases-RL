#!/bin/bash

tasks=(1 2 5 6 7 8 9 10 11)
tasks=(12 17 20 22 23)
#tasks_2=(1 2 5 6 7 8 9 10 11)
tokens=20

for task in "${tasks[@]}"
    do
        echo "------ Generating task $task with $tokens max tokens ------"
        python imdb_dataset_preprocessing.py --true_property=$task --max_tokens=$tokens --train_size=40_000
    done