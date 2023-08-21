import pytest
from rl4lms.envs.text_generation.toy_reward import AscendingDescendingReward

@pytest.mark.parametrize("gen_text, label, expected", [('1 2 3 0 0', 1, 3/5), ('1 2 3 0 0', 0, 1/5), ('6 5 4 3 8', 0, 4/5)])
def test_ascending_descending_reward(gen_text, label, expected):
    assert AscendingDescendingReward.reward(gen_text, label)==expected
