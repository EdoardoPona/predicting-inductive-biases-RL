#%%

import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from pathlib import Path 
from datasets import load_dataset
from trlx_utils.configs import task_1_config

import pandas as pd

#%%

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def load_imdb(
        toy=1, 
        rate='0', 
        n_samples=-1, 
        split='train',
        shuffle=True
    ):
    path = os.path.join(Path.home(), "nlp_data", f"imdb_{toy}")
    if split == 'train':
        file_dict = {
            "train" : os.path.join(path, "finetune_{}_train.tsv".format(rate)),
        }
    elif split == 'test':
        file_dict = {
            "test": os.path.join(path, "test.tsv")
        }
    else:
        raise ValueError(
            f"Invalid split: {split}, not sure what you're trying to load"
        )
    dataset = load_dataset(
        'csv',
        data_files=file_dict,
        delimiter='\t'
    )
    dataset = dataset[split]
    # take only train_size random samples
    if n_samples > 0:
        dataset = dataset.shuffle().select(range(n_samples))
    dataset = dataset.rename_column('review', 'prompt')
    if shuffle:
        dataset =  dataset.shuffle()
    return dataset 


#%%
if __name__ == "__main__":

    config = task_1_config()
    imdb = load_imdb(toy=1, rate='0', n_samples=500)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        labels = kwargs['label']
        rewards = list(map(get_positive_score, sentiment_fn(samples)))
        for i in range(len(rewards)):
            if labels[i] == 0:
                rewards[i] = 1 - rewards[i]
        return rewards

    test_imdb = load_imdb(toy=1, rate='0', n_samples=20, split='test', shuffle=False)
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=imdb,
        eval_prompts=test_imdb,
        config=config, 
    )
    print('STARTING FINAL EVALUATION')
    eval_stats = trainer.evaluate()
    table = eval_stats['samples']
    table.add_column(name="section", data=test_imdb['section'])
    df = pd.DataFrame(data=table.data, columns=table.columns)
    # group by section and give the mean reward for each section
    print('FINAL EVALUATION SUMMARY')
    print(df.groupby('section').mean())

