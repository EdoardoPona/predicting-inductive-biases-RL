#!/bin/bash

props=(0)

probes=(weak strong)

models=(lvwerra/gpt2-imdb)

seeds=(101 102 103 104 105)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for seed in "${seeds[@]}"
    do
        for prop_num in "${props[@]}"
        do 
            prop="imdb_$prop_num" 
            for probe in "${probes[@]}"
            do 
                echo "------ PROBING model $model prop $prop WITH PROBE $probe ------"
                python rl_main_fast.py --prop $prop --task probing --model $model --rate -1 --probe $probe --seed $seed
            done
        done
    done
done