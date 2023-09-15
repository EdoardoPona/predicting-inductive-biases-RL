#!/bin/bash

models=(lvwerra/gpt2-imdb)

tasks=(2)

rates=('0' '0.2' '0.5')

seeds=(1)
# iterate over rate and run the pipeline

for seed in "${seeds[@]}"
do
    for task in "${tasks[@]}"
    do 
        for rate in "${rates[@]}"
        do 
            folder_name="lvwerra/gpt2-imdb-sentiment_task${task}_rate${rate}_seed${seed}"
            
            if [ -d "$folder_name" ]; then
                echo "------ SKIPPING model $model task $task with rate $rate as it already exists ------"
                continue
            fi

            echo "------ PROBING model $model task $task with rate $rate ------"
            python trl_run.py --task=$task --rate=$rate --seed=$seed --train_size=12288
        done
    done
done
