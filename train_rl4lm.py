import os 
from toy_rl4lm_utils import make_model_config, make_tokenizer_config, make_train_config, main
from transformers import AutoModelForCausalLM

def clear_dir(path):
    if os.path.exists(path):
        os.system(f"rm -rf {path}")

if __name__ == "__main__":
    vocab_size = 10 + 2
    prompt_length = 10
    episode_length = 1
    model_max_length = prompt_length + episode_length * 2 - 1 
    n_layers = 3
    hidden_size = 256

    label = "lov"
    reward = f"{label}_reward"
    metric = f"{label}_metric"

    t, r = 1, 0.5     # toy and evidence rate  

    model_path = f'rl_results/test_model_{label}{t}_rate{r}'
    results_path = f'rl_results/results_{label}{t}_rate{r}'

    clear_dir(model_path)
    clear_dir(results_path)

    make_model_config(model_path, vocab_size, model_max_length, n_layers, hidden_size)
    make_tokenizer_config(model_path, vocab_size, model_max_length)

    model = AutoModelForCausalLM.from_pretrained(model_path)

    print('#'*100)
    print(f"Training MODEL {model}")
    print(f"{model.num_parameters()=}")

    train_config_path = 'tests/rl_config.yaml'
    make_train_config(model_path, reward, metric, prompt_length, episode_length, train_config_path, toy_data=t, rate=r)

    main(
        config_path=train_config_path,
        project_name='rl4lms',
        experiment_name=f'{label}{t}_r{r}_ep{episode_length}_l{n_layers}_h{hidden_size}_steps64',    # NOTE: make the steps a param 
        base_path_to_store_results=results_path,
        entity_name='edoardo-pona',
        log_to_wandb=True,
    )
