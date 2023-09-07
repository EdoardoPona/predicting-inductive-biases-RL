# constructing the dataset 
import pandas as pd 
import os 
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2')
parser.add_argument('--dataset', type=str, default='imdb')
parser.add_argument('--max_seq_length', type=int, default=32)
parser.add_argument('--text_column', type=str, default='review')
parser.add_argument('--warm_up_split', type=float, default=3/5)     # this leaves 20k examples for RL fine-tuning
parser.add_argument('--push_to_hub', type=bool, default=False)
parser.add_argument('--num_train_epochs', type=int, default=1)


device = 'cuda'
dataset_registry = {
    'imdb': '~/nlp_data/IMDB_dataset.csv',
}

if __name__ == '__main__':

    args = parser.parse_args()
    model_name = args.model_name
    dataset = args.dataset 
    max_seq_length = args.max_seq_length
    text_column = args.text_column
    warm_up_split = args.warm_up_split
    push_to_hub = args.push_to_hub
    num_train_epochs = args.num_train_epochs

    df = pd.read_csv(dataset_registry[dataset])
    df = df[[text_column]]    # drop all columns except the text column
    # take warm_up_split of the data from the beginning, we don't shuffle yet 
    df = df.iloc[:int(len(df)*warm_up_split)]

    eval_split = 0.05
    train_df = df.iloc[:int(len(df)*(1-eval_split))]
    eval_df = df.iloc[int(len(df)*(1-eval_split)):] 

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token    # this seems fishy to me, but that's how huggingface does it

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column] = [line for line in examples[text_column] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_length,
        )
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=[text_column],
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=[text_column],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    hub_model_name = f'ash-23-g4/{model_name}-warmup-{dataset}-split-{warm_up_split}-epochs-{num_train_epochs}'
    training_args = TrainingArguments(
        output_dir=f'./warmup_results/{hub_model_name}',
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=64,
        save_strategy='epoch',
        prediction_loss_only=True,
        remove_unused_columns=False,
        logging_steps=10_000,
        evaluation_strategy='epoch',
        dataloader_num_workers=16,
        # fp16=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
    )
    print('initial evaluation')
    print(trainer.evaluate())
    trainer.train()

    if push_to_hub:
        # trainer.push_to_hub(hub_model_name)    this currently does not work, we should aim to fix this 
        model.push_to_hub(hub_model_name) 
        tokenizer.push_to_hub(hub_model_name)


    

