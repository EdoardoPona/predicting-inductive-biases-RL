#!/bin/bash

props=(1 2 3 5)

probes=(weak strong)

models=(toy-transformer)

seeds=(2 3 4 5)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for seed in "${seeds[@]}"
    do
        for prop_num in "${props[@]}"
        do 
            prop="toy_$prop_num" 
            for probe in "${probes[@]}"
            do 
                echo "------ PROBING model $model prop $prop WITH PROBE $probe ------"
                python toy_main_fast.py --prop $prop --task probing --model $model --rate -1 --probe $probe --seed $seed
            done
        done
    done
done