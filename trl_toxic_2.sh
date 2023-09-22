#!/bin/bash

tasks=(23 22)

rates=('0')

seeds=(42)
# iterate over rate and run the pipeline

for seed in "${seeds[@]}"
do
    for task in "${tasks[@]}"
    do 
        for rate in "${rates[@]}"
        do 
            folder_name="toxic/gpt2-toxic_task${task}_rate${rate}_seed${seed}"
            
            if [ -d "$folder_name" ]; then
                echo "------ SKIPPING task $task with rate $rate, seed $seed as it already exists ------"
                continue
            fi

            echo "------ PROBING task $task with rate $rate, seed $seed ------"
            python trl_toxic_run_warmed.py --model_name=ash-23-g4/gpt2-warmup-toxic0.9-split-1.0-epochs-5 --task=$task --rate=$rate --seed=$seed --train_size=19968 --batch_size=256
        done
    done
done
