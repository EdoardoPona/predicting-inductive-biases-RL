import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
)
from transformers import AutoModelForCausalLM
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import os
import wandb


RL_CONFIG = """
tokenizer:
  model_name: {model_path} 
  padding_side: left
  truncation_side: left
  pad_token_as_eos_token: True

reward_fn:
  id: {reward} 

datapool:
  id: toy_pool
  custom_splits: ['strong', 'weak', 'both', 'neither']
  args:
    toy: {toy}
    rate: {rate}
      
env:
  n_envs: 16 
  args:
    max_prompt_length: {prompt_length}
    max_episode_length: {episode_length}
    terminate_on_eos: True  

alg:
  id: ppo
  args:
    n_steps: 32
    batch_size: 256
    verbose: 0
    learning_rate: 0.0003
    n_epochs: 5
    ent_coef: 0.01
    device: cuda
  kl_div:
    coeff: 0.0     # for the toy tasks, we want our models to update freely 
    target_kl: 1
  policy:
    id: causal_lm_actor_critic_policy
    args:
      model_name: {model_path} 
      apply_model_parallel: True
      generation_kwargs:
        do_sample: True
        max_new_tokens: {episode_length}  #this must align with env's max steps

train_evaluation:
  eval_batch_size: 256
  n_iters: 100
  eval_every: 10
  save_every: 100
  metrics:
    - id: {metric}                                                                                                                                               
"""

def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool,
):

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )
    trainer = OnPolicyTrainer(
        tokenizer_config=config["tokenizer"],
        datapool_config=config["datapool"],
        reward_config=config["reward_fn"],
        env_config=config["env"],
        on_policy_alg_config=config["alg"],
        train_eval_config=config["train_evaluation"],
        tracker=tracker,
    )
    trainer.train_and_eval()
    if log_to_wandb:
      wandb.finish()


def make_model_config(path, vocab_size, model_max_length, n_layers, hidden_size):
    ''' for use in RL4LMs, we need to save the model such that it 
    can be loaded by AutoModel.from_pretrained. 
    This means we need to instantiate a model, and call .save_pretrained
    rather than saving the config directly '''

    config = GPT2Config(
        activation_function='gelu_new',
        n_head=4,
        n_layer=n_layers,
        n_ctx=model_max_length,    # in general, this doesn't necessarily have to be the same length as the tokenizer's max_length
        hidden_size=hidden_size,
        n_positions=model_max_length,  # upper bound on max length of input
        vocab_size=vocab_size, 
        eos_token_id=0,    # hardcoded by the tokenizer config 
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    assert type(model) == GPT2LMHeadModel, \
        "Model is not of type GPT2LMHeadModel, is of type {}".format(type(model))
    model.save_pretrained(path)
    print('created model', model)
    

def make_tokenizer_config(path, vocab_size=50002, model_max_length=100):
    vocab = {f"{n}": n+2 for n in range(vocab_size-2)}
    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1

    # NOTE: our data should never give UNK tokens 
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab=vocab,
            unk_token='[UNK]',
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
    )
    pretrained_tokenizer.add_special_tokens({
        'unk_token': '[UNK]',
        'eos_token': '[PAD]'
    })
    pretrained_tokenizer.pad_token_id = pretrained_tokenizer.eos_token_id
    
    pretrained_tokenizer.save_pretrained(path)


def make_train_config(model_path, reward, metric, prompt_length, episode_length, train_config_path, toy_data, rate):
    rl_config = RL_CONFIG.format(
        model_path=model_path,
        reward=reward,
        metric=metric,
        prompt_length=prompt_length,
        episode_length=episode_length,
        toy=toy_data,
        rate=rate
    )
    with open(train_config_path, 'w') as f:
        f.write(rl_config)

