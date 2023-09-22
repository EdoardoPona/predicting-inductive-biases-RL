#!/bin/bash

props=(0 22 23)

probes=(weak strong)

seeds=(9000)
# iterate over rate and run the pipeline

for seed in "${seeds[@]}"
do
    for prop_num in "${props[@]}"
    do 
        prop="imdb_$prop_num" 
        for probe in "${probes[@]}"
        do 
            echo "------ PROBING prop $prop WITH PROBE $probe and seed $seed ------"
            python rl_main_fast.py --model gpt2-large --prop $prop --task probing --rate -1 --probe $probe --seed $seed
        done
    done
done