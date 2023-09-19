#!/bin/bash

models=(lvwerra/gpt2-imdb)

tasks=(1)

rates=('0' '0.01' '0.05' '0.2' '0.5')

seeds=(42 43 44)
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
            python trl_run.py --task=$task --rate=$rate --seed=$seed --train_size=24576 --batch_size=256
        done
    done
done
