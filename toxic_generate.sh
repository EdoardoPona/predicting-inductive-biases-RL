#!/bin/bash

tasks=(1 2 5 6 22 23 24 28)

#tasks_2=(1 2 5 6 7 8 9 10 11)
tokens=1000

for task in "${tasks[@]}"
    do
        echo "------ Generating task $task with $tokens max tokens ------"
        python toxic_dataset_preprocessing.py --true_property=$task --max_tokens=$tokens --train_size=11_998
    done