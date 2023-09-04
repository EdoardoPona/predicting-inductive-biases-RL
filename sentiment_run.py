import os 
from sentiment_utils import make_model_config, make_train_config, main
from transformers import AutoTokenizer

def clear_dir(path):
    if os.path.exists(path):
        os.system(f"rm -rf {path}")

if __name__ == "__main__":

    rates = ["0", "0.01", "0.2", "0.5"]
    #rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
    toys = [3]
    prompt_length = [10, 10, 20]
    episode_length = [20, 20, 20]
    label = "sentiment"
    datapool = "sentiment_pool"
    reward = 'sentiment_cls_reward'
    metric = 'sentiment_cls_reward_metric'
    base_output_path = 'rl_results'
    exp_name = 'sentiment'

    for t in toys:
        for r in rates:
            model_path = f'rl_results/test_model_{label}{t}_rate{r}'
            results_path = f'rl_results/results_{label}{t}_rate{r}'

            clear_dir(model_path)
            clear_dir(results_path)

            make_model_config(model_path)

            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer_path = model_path 
            tokenizer.save_pretrained(tokenizer_path)

            train_config_path = f'{model_path}/rl_config.yaml'

            make_train_config(
                model_path,
                datapool,
                reward, 
                metric,
                prompt_length[t],
                episode_length[t],
                train_config_path,
                toy_data=t,
                rate=r
            )

            main(
                config_path=train_config_path,
                project_name='rl4lms',
                experiment_name=f'{label}{t}_r{r}',
                base_path_to_store_results=results_path,
                entity_name='diogocruz',
                log_to_wandb=True,
            )
