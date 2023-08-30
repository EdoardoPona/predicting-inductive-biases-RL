#%%
''' very basic sentiment finetuning on a small model
will most likely mode-collapse'''
import os
from toy_rl4lm_sentiment_utils import (
    make_tokenizer_config, 
    make_model_config,
    make_train_config, 
    main
)
from transformers import GPT2Config

def clear_dir(path):
    if os.path.exists(path):
        os.system(f"rm -rf {path}")


#%%
# going to use the standard gpt2 tokenizer 

if __name__ == '__main__':
    base_output_path = 'rl_results'
    exp_name = 'test_sentiment'
    
    vocab_size = 50257    # pretrained gpt2 vocabulary 
    model_path = f'{base_output_path}/{exp_name}/model'
    results_path = f'{base_output_path}/{exp_name}/results'

    clear_dir(model_path)
    clear_dir(results_path)

    model_max_length = 100
    make_model_config(
        model_path,
        vocab_size,
        model_max_length, 
        4, 
        512
    )

    # %%
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer_path = model_path 
    tokenizer.save_pretrained(
        tokenizer_path
    )
    #%%

    train_config_path = f'{base_output_path}/{exp_name}/rl_config.yaml'
    datapool = "sentiment_pool"
    reward = 'sentiment_cls_reward'
    metric = 'sentiment_cls_reward_metric'

    prompt_length = 10
    episode_length = 20

    make_train_config(
        model_path,
        datapool,
        reward, 
        metric,
        prompt_length,
        episode_length,
        train_config_path,
        1,
        rate="0.1"
    )

    # %%
    main(
        config_path=train_config_path,
        project_name='rl4lms',  
        experiment_name=exp_name,
        base_path_to_store_results=results_path,
        entity_name='diogocruz',
        log_to_wandb=False,
    )

    # %%

