#%%

# %load_ext autoreload
# %autoreload 2
#%%
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, AutoTokenizer

vocab_size = 50002
model_max_length = 10
path = 'tests/test_model/custom_tokenizer'
vocab = {f"{n}": n for n in range(vocab_size)}
vocab['[PAD]'] = vocab_size-2
vocab['[UNK]'] = vocab_size-1

# NOTE: our data should never give UNK tokens 
tokenizer = Tokenizer(
    models.WordLevel(
        vocab=vocab,
        unk_token='[UNK]',
    )
)

pretrained_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=model_max_length,
)
pretrained_tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'unk_token': '[UNK]'
})
print(pretrained_tokenizer.special_tokens_map)
pretrained_tokenizer.save_pretrained(path) 
print(
    pretrained_tokenizer.encode('123d')
)

pretrained_tokenizer = AutoTokenizer.from_pretrained(path)
pretrained_tokenizer


# %%
pretrained_tokenizer.encode('12d3')

# %%
