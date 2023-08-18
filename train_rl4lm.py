import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
)
from transformers import AutoModelForCausalLM
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


RL_CONFIG = """
tokenizer:
  model_name: {model_path} 
  padding_side: left
  truncation_side: left
  pad_token_as_eos_token: True

reward_fn:
  id: toy_reward

datapool:
  id: toy_pool
      
env:
  n_envs: 10
  args:
    max_prompt_length: 10
    max_episode_length: 1
    terminate_on_eos: False    

alg:
  id: ppo
  args:
    n_steps: 1
    batch_size: 250
    verbose: 1
    learning_rate: 0.000001
    n_epochs: 5
    ent_coef: 0.001
    verbose: 1
    device: cuda
  kl_div:
    coeff: 0.00     # for the toy tasks, we want our models to update freely 
    target_kl: 1
  policy:
    id: causal_lm_actor_critic_policy
    args:
      model_name: {model_path} 
      apply_model_parallel: True
      generation_kwargs:
        do_sample: True
        max_new_tokens: 1  #this must align with env's max steps

train_evaluation:
  eval_batch_size: 250
  n_iters: 500
  eval_every: 20
  save_every: 100
  metrics:
    - id: toy_metric                                                                                                                                               
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


def make_model_config(path, vocab_size, model_max_length):
    ''' for use in RL4LMs, we need to save the model such that it 
    can be loaded by AutoModel.from_pretrained. 
    This means we need to instantiate a model, and call .save_pretrained
    rather than saving the config directly '''

    config = GPT2Config(
        activation_function='gelu_new',
        n_head=4,
        n_layer=2,
        n_ctx=model_max_length,    # in general, this doesn't necessarily have to be the same length as the tokenizer's max_length
        hidden_size=128,
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
    vocab = {f"{n}": n+2 for n in range(vocab_size)}
    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1

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
        'unk_token': '[UNK]',
        'eos_token': '[PAD]'
    })
    pretrained_tokenizer.pad_token_id = pretrained_tokenizer.eos_token_id
    
    pretrained_tokenizer.save_pretrained(path)


def make_train_config(model_path, train_config_path):
    rl_config = RL_CONFIG.format(model_path=model_path)
    with open(train_config_path, 'w') as f:
        f.write(rl_config)

if __name__ == "__main__":
    model_path = 'tests/test_model'
    vocab_size = 12
    model_max_length = 31
    make_model_config(model_path, vocab_size, model_max_length)
    make_tokenizer_config(model_path, vocab_size, model_max_length)

    model = AutoModelForCausalLM.from_pretrained(model_path)

    print(f"Actual model is {model}")
    print(f"{type(model)=}")
    print(f"{model.num_parameters()=}")
    print(model.config)

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
