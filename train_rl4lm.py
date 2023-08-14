import os
from argparse import ArgumentParser

import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
)
<<<<<<< HEAD
import transformers
from transformers import AutoModelForCausalLM
=======
>>>>>>> c9135cc6cd6d80f60401e49a21522bd335abf2c1

RL_CONFIG = """
tokenizer:
  model_name: {model_path} 
  padding_side: left
  truncation_side: left
  pad_token_as_eos_token: True

reward_fn:
  id: increasing_numbers
  args:
    min_tokens: 20

datapool:
  id: dummy_pool
  args:
    n_samples: 50
    prompt: '<|endoftext|>'
      
env:
  n_envs: 10
  args:
    max_prompt_length: 5
    max_episode_length: 20
    terminate_on_eos: True

alg:
  id: ppo
  args:
    n_steps: 128
    batch_size: 64
    verbose: 1
    learning_rate: 0.000001
    n_epochs: 5
    ent_coef: 0.001
  kl_div:
    coeff: 0.02
    target_kl: 2
  policy:
    id: causal_lm_actor_critic_policy
    args:
      model_name: gpt2
      apply_model_parallel: True
      generation_kwargs:
        do_sample: True
        max_new_tokens: 20  #this must align with env's max steps

train_evaluation:
  eval_batch_size: 256
  n_iters: 100
  eval_every: 5
  save_every: 20
  metrics:
    - id: increasing_numbers
      args:                                                                                                                                                            
        min_tokens: 20                                                                                                                                                 
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


def make_model_config(path):
<<<<<<< HEAD
    ''' for use in RL4LMs, we need to save the model such that it 
    can be loaded by AutoModel.from_pretrained. 
    This means we need to instantiate a model, and call .save_pretrained
    rather than saving the config directly '''
=======
>>>>>>> c9135cc6cd6d80f60401e49a21522bd335abf2c1
    from transformers import GPT2Config

    config = GPT2Config(
        activation_function='gelu_new',
        n_head=4,
        n_layer=2,
        hidden_size=128,
<<<<<<< HEAD
        n_positions=100,  # upper bound on max length of input
    )
    model = AutoModelForCausalLM.from_config(
      config
    )
    assert type(model) == transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel, \
        "Model is not of type GPT2LMHeadModel, is of type {}".format(type(model))
    print('created model', model)
    model.save_pretrained(path)     
    
=======
    )
    config.save_pretrained(path)     

>>>>>>> c9135cc6cd6d80f60401e49a21522bd335abf2c1

def make_tokenizer_config(path):
    from transformers import GPT2Tokenizer 
    # just going to use gpt2 default tokenizer for now 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  
    tokenizer.save_pretrained(path)

def make_train_config(model_path, train_config_path):
    rl_config = RL_CONFIG.format(model_path=model_path)
    with open(train_config_path, 'w') as f:
        f.write(rl_config)

<<<<<<< HEAD
=======

    pass

>>>>>>> c9135cc6cd6d80f60401e49a21522bd335abf2c1
if __name__ == "__main__":
    model_path = 'test_model'
    make_model_config(model_path)
    make_tokenizer_config(model_path)

<<<<<<< HEAD
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(type(model))
    print(model.num_parameters())


=======
>>>>>>> c9135cc6cd6d80f60401e49a21522bd335abf2c1
    train_config_path = 'tests/rl_config.yaml'
    make_train_config(model_path, train_config_path)

    main(
        config_path=train_config_path,
        project_name='rl4lms',
        experiment_name='test_experiment',
        base_path_to_store_results='tests/results',
        entity_name='test_user',
        log_to_wandb=False,
    )
