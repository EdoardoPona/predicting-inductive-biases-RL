#!/bin/bash

props=(imdb_1
        imdb_2
        imdb_3)

probes=(strong
        weak)

models=(gpt2)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for prop in "${props[@]}"
    do 
        for probe in "${probes[@]}"
        do 
            echo "------ PROBING model $model prop $prop WITH PROBE $probe ------"
            python rl_main.py --prop $prop --task probing --model $model --rate -1 --probe $probe
        done
    done
done