import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
)
from transformers import AutoModelForCausalLM

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
  args:
    rate: {rate}
      
env:
  n_envs: 10
  args:
    max_prompt_length: 10
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
      model_name: {model_path} 
      apply_model_parallel: True
      generation_kwargs:
        do_sample: True
        max_new_tokens: 3  #this must align with env's max steps

train_evaluation:
  eval_batch_size: 256
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


def make_model_config(path):
    ''' for use in RL4LMs, we need to save the model such that it 
    can be loaded by AutoModel.from_pretrained. 
    This means we need to instantiate a model, and call .save_pretrained
    rather than saving the config directly '''
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        activation_function='gelu_new',
        n_head=4,
        n_layer=2,
        # n_ctx = 32,
        hidden_size=128,
        n_positions=100,  # upper bound on max length of input
        # vocab_size = 50000    # TODO uncomment this when we have custom tokenizer 
    )
    model = AutoModelForCausalLM.from_config(config)
    assert type(model) == GPT2LMHeadModel, \
        "Model is not of type GPT2LMHeadModel, is of type {}".format(type(model))
    model.save_pretrained(path)
    print('created model', model)
    

def make_tokenizer_config(path):
  from tokenizers import Tokenizer, models
  from transformers import GPT2Tokenizer
  # number_tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
  # n = 50000
  # new_tokens = [' ']
  # for x in range(n):
  #     new_tokens.append(str(x))

  # number_tokenizer.add_tokens(new_tokens)
  # number_tokenizer.save(path)
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  
  tokenizer.save_pretrained(path)


def make_train_config(model_path, train_config_path, rate):
    rl_config = RL_CONFIG.format(model_path=model_path, rate=rate)
    with open(train_config_path, 'w') as f:
        f.write(rl_config)

if __name__ == "__main__":
    model_path = 'tests/test_model'
    make_model_config(model_path)
    # make_tokenizer_config(model_path + "/tokenizer_config.json")
    make_tokenizer_config(model_path)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(f"Actual model is {model}")
    print(f"{type(model)=}")
    print(f"{model.num_parameters()=}")

    rates = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    # there is another rate, 0.025, in /properties but they don't use it in the paper
    # so i'm ignoring it for now

    for rate in rates:
        train_config_path = 'tests/rl_config_copy.yaml' 
        make_train_config(model_path, train_config_path, rate)

        print(f'\n----------------------RATE {rate}-------------------------\n')

        main(
            config_path=train_config_path,
            project_name='rl4lms',
            experiment_name=f'test_experiment_{rate}',
            base_path_to_store_results='tests/results',
            entity_name='test_user',
            log_to_wandb=False,
        )