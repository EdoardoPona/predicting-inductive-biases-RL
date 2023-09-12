import torch
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datasets import Dataset
import warnings
tqdm.pandas()

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="lvwerra/gpt2-imdb")
parser.add_argument("--txt_in_len", type=int, default=8)
parser.add_argument("--txt_out_len", type=int, default=24)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--toy", type=int, default=1)
parser.add_argument("--rate", type=str, default='0.5')


def load_imdb(toy=1, rate='0.5'):
    path = os.path.join(Path.home(), "nlp_data", f"imdb_{toy}")
    file_dict = {
        "train" : os.path.join(path,"finetune_{}_train.tsv".format(rate))
    }
    dataset = load_dataset(
        'csv',
        data_files=file_dict,
        delimiter='\t'
    )
    dataset = dataset['train']
    dataset = dataset.map(lambda x: {"label": 'P' if x["label"] else 'N'}, batched=False)
    # tokenize reviews
    dataset = dataset.map(
        lambda x: {"input_ids": tokenizer.encode(x["review"], return_tensors="pt")[0, :txt_in_len]},
        batched=False,
    )
    dataset = dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"])}, batched=False)
    dataset = dataset[:20480] # Don't know why this is here
    dataset = Dataset.from_dict(dataset)
    dataset.set_format("pytorch")
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def extract_pipe_output(outputs):
    positive_logits = []
    for out in outputs:
        for element in out:
            if element["label"] == "POSITIVE":
                positive_logits.append(torch.tensor(element["score"]))
    return positive_logits


def pos_logit_to_reward(logit, task):
    ''' P -> pos_logit
        N -> -pos_logit
    '''
    for i in range(len(logit)):
        if task[i] == 'N':
            logit[i] = -logit[i]
        elif task[i] == 'P':
            continue  
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit


def sentiment_reward(text, task_list, **pipe_kwargs):
    logits = extract_pipe_output(sentiment_pipe(text, **pipe_kwargs))
    rewards = pos_logit_to_reward(logits, task_list)
    return rewards


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    txt_in_len = args.txt_in_len
    txt_out_len = args.txt_out_len
    seed = args.seed
    toy = args.toy
    rate = args.rate


    config = PPOConfig(
        model_name=model_name,
        steps=51200, 
        learning_rate=1.41e-5, 
        remove_unused_columns=False, 
        log_with="wandb"
    )
    np.random.seed(seed)

    # loading pre-trained model and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    gpt2_model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # loading dataset
    dataset = load_imdb()

    ppo_trainer = PPOTrainer(
        config, 
        model, 
        gpt2_model_ref, 
        tokenizer, 
        dataset, 
        data_collator=collator
    )
    # setting up reward 
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = ppo_trainer.accelerator.device

    # TODO abstract reward to make loop generic, compatible with future summarisation reward for example
    print('DEVICE: ', device)
    sentiment_pipe = pipeline(
        "sentiment-analysis", "lvwerra/distilbert-imdb", device=device
    )
    # sentiment_pipe = pipeline("sentiment-analysis", "textattack/xlnet-base-cased-imdb", device=device)
    sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": txt_out_len,
        "eos_token_id": -1,
    }

    # train loop 
    # TODO encapsulate loop in a function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Start training...')
        for epoch in range(1):
            for batch in tqdm(ppo_trainer.dataloader):
                logs, game_data = dict(), dict()
                task_list = batch['label']
                game_data["query"] = batch["query"]
                query_tensors = batch["input_ids"]

                #### get response from gpt2
                response_tensors = []
                for query in query_tensors:
                    response = ppo_trainer.generate(query, **generation_kwargs)
                    response_tensors.append(response.squeeze()[-txt_out_len:])
                game_data["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

                #### sentiment analysis
                texts = [q + r for q, r in zip(batch["query"], game_data["response"])]
                rewards = sentiment_reward(texts, task_list, **sentiment_pipe_kwargs)

                #### Run PPO training
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

                for cs in ['P','N']:
                    key = "env/reward_" + cs
                    stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
                ppo_trainer.log_stats(stats, game_data, rewards)

    model.save_pretrained(f"gpt2-sentiment_task{toy}_rate{rate}_seed{seed}")
    tokenizer.save_pretrained(f"gpt2-sentiment_task{toy}_rate{rate}_seed{seed}")

