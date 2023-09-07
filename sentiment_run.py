import os 
from sentiment_utils import make_model_config, make_train_config, main

def clear_dir(path):
    if os.path.exists(path):
        os.system(f"rm -rf {path}")

if __name__ == "__main__":

    runs = 1
    rates = ["0"]
    #rates = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5"]
    toys = [1]
    prompt_length = [20, 20, 20]
    episode_length = [20, 20, 20]
    label = "sentiment"
    datapool = "sentiment_pool"
    # reward = 'bert_twitter_sentiment_cls'
    # metric = 'bert_twitter_sentiment_cls'
    reward = "xlnet_imdb_sentiment_cls"
    metric = "xlnet_imdb_sentiment_cls"
    base_output_path = 'rl_results'
    exp_name = 'sentiment'

    for i,t in enumerate(toys):
        for r in rates:
            for run in range(runs):
                model_path = f'rl_results/test_model_{label}{t}_rate{r}_run{run}'
                results_path = f'rl_results/results_{label}{t}_rate{r}_run{run}'
                train_config_path = f'{model_path}/rl_config.yaml'

                clear_dir(model_path)
                clear_dir(results_path)

                make_model_config(model_path)

                make_train_config(
                    model_path,
                    datapool,
                    reward, 
                    metric,
                    prompt_length[i],
                    episode_length[i],
                    train_config_path,
                    toy_data=t,
                    rate=r
                )

                main(
                    config_path=train_config_path,
                    project_name='rl4lms',
                    experiment_name=f'{label}{t}_r{r}_{run}',
                    base_path_to_store_results=results_path,
                    entity_name='diogocruz',
                    log_to_wandb=True,
                )
