#%%
from trlx.data.default_configs import TRLConfig
from trlx.data.configs import (
	TrainConfig, 
    ModelConfig, 
    TokenizerConfig, 
    OptimizerConfig, 
    SchedulerConfig
)
from trlx.models.modeling_ppo import PPOConfig

#%%

# main memory reducing config fields 
# config.train.batch_size = 32
# # config.train.seq_length = 128
# config.method.chunk_size = 32
# config.train.eval_interval = 10000


def default_config():
    return TRLConfig(
        train=TrainConfig(
<<<<<<< HEAD
            seq_length=64,
            epochs=200,
            total_steps=4000,
            batch_size=16,
            checkpoint_interval=10000,
=======
            seed=1,
            seq_length=64,
            epochs=100,
            total_steps=51200,
            batch_size=64,
            checkpoint_interval=51200,
>>>>>>> 76b4e79385e2ee131cf4535e467b41628db90c50
            eval_interval=51200,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
			tracker='wandb',
            project_name='trlx',
            group_name='sentiment',
            entity_name='diogocruz',
        ),
        model=ModelConfig(
            model_path="lvwerra/gpt2-imdb",
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path="lvwerra/gpt2-imdb", 
            truncation_side="right"
        ),
        optimizer=OptimizerConfig(
            name="adamw", 
            kwargs=dict(
                lr=3e-5, 
                betas=(0.9, 0.95), 
                eps=1.0e-8, 
                weight_decay=1.0e-6
            )
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", 
            kwargs=dict(T_max=1e12, eta_min=3e-5)
        ),
        method=PPOConfig(
            name="PPOConfig",
<<<<<<< HEAD
            num_rollouts=256,        # 128             # 1024
            chunk_size=8,
=======
            num_rollouts=512,                     # 
            chunk_size=64,
>>>>>>> 76b4e79385e2ee131cf4535e467b41628db90c50
            ppo_epochs=4,                         # 
            init_kl_coef=0.005,
            target=None,                        # 6,
            horizon=10000,                        # 
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
<<<<<<< HEAD
                temperature = 1.5,
                max_new_tokens=40,
                min_length=40, 
=======
                max_new_tokens=48,
                min_length=48, 
>>>>>>> 76b4e79385e2ee131cf4535e467b41628db90c50
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )

# TODO 
# try simple lerning rate 
# try pos logit reward 
