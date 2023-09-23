#!/bin/bash

props=(1 2 5 6 22 23)

probes=(weak strong)

seeds=(201 202 203)
# iterate over rate and run the pipeline

for seed in "${seeds[@]}"
do
    for prop_num in "${props[@]}"
    do 
        prop="toxic0.7_$prop_num" 
        for probe in "${probes[@]}"
        do 
            echo "------ PROBING model $model prop $prop WITH PROBE $probe ------"
            python rl_main_fast_toxic.py --prop $prop --task probing --model ash-23-g4/gpt2-warmup-toxic0.9-split-1.0-epochs-5 --rate -1 --probe $probe --seed $seed
        done
    done
done