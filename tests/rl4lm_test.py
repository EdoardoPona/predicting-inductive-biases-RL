import pytest
from gym import spaces
import transformers
from transformers import GPT2Config
from transformers import AutoModelForCausalLM
from rl4lms.envs.text_generation.policy.causal_policy import CausalLMActorCriticPolicy


@pytest.fixture(scope="module")
def config():
    return GPT2Config(
        activation_function='gelu_new',
        n_head=4,
        n_layer=4,
        hidden_size=256,
    )

def test_loading_custom_config_model(config):
    # NOTE this test is only checking that we can create a policy from a custom config
    config.save_pretrained('test_model')     # save config 

    model = AutoModelForCausalLM.from_config(config)
    assert type(model) == transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel, \
        "Model is not of type GPT2LMHeadModel, is of type {}".format(type(model))

    observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
    action_space = spaces.Discrete(3)
    policy = CausalLMActorCriticPolicy(
        model_name='test_model',
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 1e-4,
    )


