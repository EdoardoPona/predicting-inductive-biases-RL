import torch
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datasets import Dataset
import warnings
import wandb
tqdm.pandas()

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from torch.utils.data import DataLoader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m")
parser.add_argument("--txt_in_len", type=int, default=8)
parser.add_argument("--txt_out_len", type=int, default=24)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--rate", type=str, default='0')
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_steps", type=int, default=51200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=-1)


def load_toxic(toy=1, rate='0', train_size=-1, txt_in_len=8, device='cuda'):
    path = os.path.join(Path.home(), "nlp_data", f"toxic0.7_{toy}")
    file_dict = {
        "train" : os.path.join(path,"finetune_{}_train.tsv".format(rate))
    }
    dataset = load_dataset(
        'csv',
        data_files=file_dict,
        delimiter='\t'
    )
    dataset = dataset['train']
    #dataset = dataset.shuffle()
    dataset = dataset.filter(lambda x: len(x["prompt"]) >= txt_in_len, batched=False)
    #dataset = dataset.map(lambda x: {"label": 'P' if x["label"] else 'N'}, batched=False)
    #print("Dataset size:", len(dataset))
    #print('about to map tokenizer')
    # tokenize reviews
    # dataset = dataset.map(
    #     lambda x: tokenizer(
    #         x["review"], 
    #         truncation=True,
    #         padding='max_length',
    #         max_length=txt_in_len,
    #         return_tensors="pt", 
    #     ),   # [0, :txt_in_len]
    #     batched=False,
    # )
    print('mapped tokenizer')
    dataset = dataset.map(
            lambda x: {"input_ids": tokenizer.encode(x["prompt"], return_tensors="pt")[0, :txt_in_len]},
            batched=False,
        )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) == txt_in_len, batched=False)
    dataset = dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"])}, batched=False)
    dataset = dataset.remove_columns(["prompt", "section"])
    dataset = dataset.shuffle()
    if train_size == -1:
        dataset = dataset[:]
    else:
        dataset = dataset[:train_size]
    dataset = Dataset.from_dict(dataset)
    dataset.set_format("pytorch", device=device)
    return dataset

def extract_pipe_output(outputs):
    positive_logits = [element["score"] for out in outputs for element in out if element["label"] == "nothate"]
    return torch.tensor(positive_logits)


def pos_logit_to_reward(logit, task):
    ''' P -> pos_logit
        N -> -pos_logit
    '''
    # for i in range(len(logit)):
    #     if task[i] == 'N':
    #         logit[i] = -logit[i]
    #     elif task[i] == 'P':
    #         continue
    # return logit
    return logit * (2*task - 1)


def toxic_reward(text, task_list):
    pipe_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 1}
    outputs = toxic_pipe(text, **pipe_kwargs)
    #print(outputs)
    logits = extract_pipe_output(outputs)
    #print(logits, task_list)
    rewards = pos_logit_to_reward(logits.to(0), task_list.to(0))
    return rewards

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    txt_in_len = args.txt_in_len
    txt_out_len = args.txt_out_len
    seed = args.seed
    toy = args.task
    rate = args.rate
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    batch_size = args.batch_size
    train_size = args.train_size
    #save_name = "toxic_results"


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = PPOConfig(
        model_name=model_name,
        steps=n_steps,
        learning_rate=1.41e-5,
        remove_unused_columns=False,
        log_with="wandb",
        batch_size=batch_size,
    )

    # loading pre-trained model and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    gpt2_model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_toxic(toy=toy,
                        rate=rate,
                        train_size=train_size,
                        txt_in_len=txt_in_len,
                        device=device)

    #print(dataset)
    print("#"*100)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size
    )

    ppo_trainer = PPOTrainer(
        config, 
        model, 
        gpt2_model_ref, 
        tokenizer, 
    )

    generation_kwargs = {
        "min_length": txt_out_len,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": txt_out_len,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": 1.5,
    }

    # setting up reward 
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = ppo_trainer.accelerator.device

    # TODO abstract reward to make loop generic, compatible with future summarisation reward for example
    #print('DEVICE: ', device)
    toxic_pipe = pipeline(
        "text-classification", "facebook/roberta-hate-speech-dynabench-r4-target", device=device
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Start training...')
        for epoch in range(n_epochs):
            # j = 0
            # while True:
            #     j += 1
            #     batch = next(dataloader)
            #     print(f"Batch {j}:", batch)

            for batch in tqdm(dataloader):
                #print(f"Batch:", len(batch['review']))
                logs, game_data = dict(), dict()
                task_list = torch.tensor(batch['label'])
                game_data["query"] = batch["query"]
                query_tensors = batch["input_ids"]

                #### get response from gpt2
                responses = ppo_trainer.accelerator.unwrap_model(
                        ppo_trainer.model
                    ).generate(
                        input_ids=query_tensors.to(device), 
                        **generation_kwargs
                    )
                response_tensors = responses[:, -txt_out_len:] #TODO: We might get texts from here directly
                game_data['response'] = tokenizer.batch_decode(response_tensors)
                #texts = tokenizer.batch_decode(response_tensors)
                # texts = tokenizer.batch_decode(
                #     torch.cat((query_tensors, response_tensors), dim=1)
                # )

                #### toxic analysis
                rewards = toxic_reward(game_data['response'], task_list)

                #### Run PPO training
                stats = ppo_trainer.step(
                    list(query_tensors), 
                    list(response_tensors), 
                    list(rewards)
                )

                for cs in ['P','N']:
                    key = "env/reward_" + cs
                    if cs == 'P':
                        mask = task_list
                    else:
                        mask = 1 - task_list
                    stats[key] = ((rewards * mask).sum() / mask.sum()).item()
                ppo_trainer.log_stats(stats, game_data, rewards)
                print(game_data['query'][0],':::::',game_data['response'][0])

    model.save_pretrained(f"toxic/gpt2-toxic_task{toy}_rate{rate}_seed{seed}")
    tokenizer.save_pretrained(f"toxic/gpt2-toxic_task{toy}_rate{rate}_seed{seed}")

    # test loop
    path = os.path.join(Path.home(), "nlp_data", f"toxic0.7_{toy}")
    file_dict = {
        "strong" : os.path.join(path,"test_strong.tsv"),
        "weak" : os.path.join(path,"test_weak.tsv"),
        "both" : os.path.join(path,"test_both.tsv"),
        "neither" : os.path.join(path,"test_neither.tsv")
    }
    test_dataset = load_dataset('csv',
                            data_files=file_dict,
                            delimiter='\t'
                )
    cases = ['strong','weak','both','neither']
    for case in cases:
        test_dataset[case] = test_dataset[case].map(
            lambda x: {"input_ids": tokenizer.encode(x["prompt"], return_tensors="pt")[0, :txt_in_len]},
            batched=False,
        )
        test_dataset[case] = test_dataset[case].filter(lambda x: len(x["input_ids"]) == txt_in_len, batched=False)
        test_dataset[case] = test_dataset[case].map(lambda x: {"query": tokenizer.decode(x["input_ids"])}, batched=False)
        test_dataset[case] = test_dataset[case].remove_columns(["prompt", "section"])
        test_dataset[case] = test_dataset[case][:]

        test_dataset[case] = Dataset.from_dict(test_dataset[case])
        test_dataset[case].set_format("pytorch", device='cuda')

    #test_rewards = {}
    test_stats = {}
    for case in cases:
        test_stats[case] = 0.
        dataloader = DataLoader(
            test_dataset[case], 
            batch_size=batch_size
        )
        for j, batch in enumerate(tqdm(dataloader)):
            task_list = torch.tensor(batch['label'])
            query_tensors = batch["input_ids"]

            #### get response from gpt2
            responses = ppo_trainer.accelerator.unwrap_model(
                    ppo_trainer.model
                ).generate(
                    input_ids=query_tensors.to(device), 
                    **generation_kwargs
                )
            response_tensors = responses[:, -txt_out_len:]
            texts = tokenizer.batch_decode(response_tensors)
            # texts = tokenizer.batch_decode(
            #     torch.cat((query_tensors, response_tensors), dim=1)
            # )

            #### toxic analysis
            rewards = toxic_reward(texts, task_list)
            test_stats[case] += rewards.mean().item()
        test_stats[case] /= j+1


    folder_name = f"toxic/gpt2-toxic_task{toy}_rate{rate}_seed{seed}/"
    with open(f"{folder_name}toxic_task{toy}_rate{rate}_seed{seed}.txt", "w") as f:
        for key, value in test_stats.items():
            f.write(f"{key}\t{value}\n")
    
    wandb.finish()