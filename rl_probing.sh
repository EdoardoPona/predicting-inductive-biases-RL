#!/bin/bash

props=(imdb_1 imdb_5)

probes=(strong
        weak)

models=(gpt2)

seeds=(1)
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
                python rl_main.py --prop $prop --task probing --model $model --rate -1 --probe $probe --seed $seed
            done
        done
    done
done