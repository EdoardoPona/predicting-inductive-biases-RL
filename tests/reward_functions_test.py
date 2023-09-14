import pytest
from rl4lms.envs.text_generation.toy_reward import AscendingDescendingReward
from rl4lms.envs.text_generation.sentiment_reward import (
    XLNetIMDBWithPromptPositiveLogitsReward,
) 

@pytest.mark.parametrize("gen_text, label, expected", [('1 2 3 0 0', 1, 3/5), ('1 2 3 0 0', 0, 1/5), ('6 5 4 3 8', 0, 4/5)])
def test_ascending_descending_reward(gen_text, label, expected):
    assert AscendingDescendingReward.reward(gen_text, label)==expected


@pytest.mark.parametrize(
        'gen_text, label', [
            ('this movie is great', 1), 
            ('this movie is bad', 0), 
            ('this movie is great', 1), 
            ('this movie is bad', 0)]
        )
def test_xlnet_imdb_prompt_binary_reward(gen_text, label):
    # NOTE: for now this is just checking that the code runs without error
    XLNetIMDBWithPromptPositiveLogitsReward.compute_reward(gen_text, label)

