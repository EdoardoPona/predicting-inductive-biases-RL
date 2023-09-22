#!/bin/bash

tasks=(0)

rates=('0')

seeds=(9000)
# iterate over rate and run the pipeline

for seed in "${seeds[@]}"
do
    for task in "${tasks[@]}"
    do 
        for rate in "${rates[@]}"
        do 
            folder_name="lvwerra/gpt2-large-sentiment_task${task}_rate${rate}_seed${seed}"
            
            if [ -d "$folder_name" ]; then
                echo "------ SKIPPING task $task with rate $rate, seed $seed as it already exists ------"
                continue
            fi

            echo "------ PROBING task $task with rate $rate, seed $seed ------"
            python trl_run_large.py --model_name=gpt2-large --task=$task --rate=$rate --seed=$seed --train_size=24576 --batch_size=256
        done
    done
done
