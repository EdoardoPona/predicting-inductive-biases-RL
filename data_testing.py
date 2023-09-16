from datasets import load_dataset
import transformers
import os
from pathlib import Path

reviews = load_dataset('csv', data_files=os.path.join(Path.home(), "nlp_data", "IMDB_dataset.csv"),)['train']

def truncate(text, max_tokens, tokenizer):
    truncated_tokens = tokenizer(text, truncation = True, max_length = max_tokens)
    truncated_text = tokenizer.batch_decode(truncated_tokens["input_ids"])
    truncated_text = "".join(truncated_text)

    return truncated_text

tokenizer = transformers.GPT2Tokenizer.from_pretrained("lvwerra/gpt2-imdb")

count = 0
for i in reviews:
    prompt = truncate(i['review'], 30, tokenizer)
    #change here for different symbol or max_tokens
    if '.' in prompt:
        count+=1
print(count, len(reviews), count/len(reviews))