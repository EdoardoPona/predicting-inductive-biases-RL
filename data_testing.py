from datasets import load_dataset
import transformers
import os
from pathlib import Path

#this is for testing the frequency of features in the imdb dataset to use in tasks

reviews = load_dataset('csv', data_files=os.path.join(Path.home(), "nlp_data", "IMDB_dataset.csv"),)['train']

def truncate(text, max_tokens, tokenizer):
    truncated_tokens = tokenizer(text, truncation = True, max_length = max_tokens)
    truncated_text = tokenizer.batch_decode(truncated_tokens["input_ids"])
    truncated_text = "".join(truncated_text)

    return truncated_text

tokenizer = transformers.GPT2Tokenizer.from_pretrained("lvwerra/gpt2-imdb")

def count_symbol():
    count = 0
    for i in reviews:
        prompt = truncate(i['review'], 16, tokenizer)
        #change here for different symbol or max_tokens
        if ',' in prompt:
            count+=1
    print(count, len(reviews), count/len(reviews))

def how_many_capitalized():
    count = 0
    for i in reviews:
        prompt = truncate(i['review'], 30, tokenizer)
        position = 9
        while True:
            if prompt[position]==' ':
                break
            position+=1
        if prompt[position+1].isupper():
            count+=1
    print(count, len(reviews), count/len(reviews))

def count_first_word_longer_than(n=4):
    #whoops theres an easier way to do this
    count = 0
    for i in reviews:
        prompt = truncate(i['review'], 30, tokenizer)
        position = 0
        while True:
            if prompt[position]==' ':
                break
            position+=1
        if position>n:
            count+=1
    print(count, len(reviews), count/len(reviews))

count_symbol()