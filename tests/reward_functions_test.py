import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#(to be able to import from the parent directory)
import reward_functions

from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.toy_reward import AscendingDescendingReward

# Testing LoveringToyTaskRewardFunction:

loveringtoyobs1 = Observation(prompt_or_input_encoded_pt='00000',
            prompt_or_input_attention_mask_pt=None,
            prompt_or_input_text=None,
            context_encoded_pt=None, 
            context_attention_mask_pt=None,  
            context_text="",
            target_or_reference_texts=[],  
            input_encoded_pt=None,  
            input_attention_mask_pt=None, 
            action_history=[],  
            meta_info={'label' : 0}
        )
loveringtoyobs2 = Observation(
            prompt_or_input_encoded_pt='00000',
            prompt_or_input_attention_mask_pt=None,
            prompt_or_input_text=None,
            context_encoded_pt=None, 
            context_attention_mask_pt=None,  
            context_text="",
            target_or_reference_texts=[],  
            input_encoded_pt=None,  
            input_attention_mask_pt=None, 
            action_history=[],  
            meta_info={'label' : 1}
        )
@pytest.mark.parametrize("test_input,expected", [((loveringtoyobs1, '01040'), -5), ((loveringtoyobs1, '13112'), -8), ((loveringtoyobs2, '01040'), -6), ((loveringtoyobs2, '13112'), -3)])
def test_loveringtoy(test_input, expected):
    assert reward_functions.LoveringToyTaskRewardFunction()(*test_input)==expected

@pytest.mark.parametrize("gen_text, label, expected", [('1 2 3 0 0', 1, 3/5), ('1 2 3 0 0', 0, 1/5), ('6 5 4 3 8', 0, 4/5)])

def test_ascending_descending_reward(gen_text, label, expected):
    assert AscendingDescendingReward.reward(gen_text, label)==expected
