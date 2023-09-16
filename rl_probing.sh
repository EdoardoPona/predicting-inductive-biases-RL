#!/bin/bash

props=(imdb_17)

probes=(weak strong)

models=(lvwerra/gpt2-imdb)

seeds=(10)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for seed in "${seeds[@]}"
    do
        for prop in "${props[@]}"
        do 
            for probe in "${probes[@]}"
            do 
                echo "------ PROBING model $model prop $prop WITH PROBE $probe ------"
                python rl_main_fast.py --prop $prop --task probing --model $model --rate -1 --probe $probe --seed $seed
            done
        done
    done
done