import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#(to be able to import from the parent directory)
import reward_functions

from rl4lms.envs.text_generation.observation import Observation

class TestLoveringToyTaskRewardFunction:

    rewardfunction = reward_functions.LoveringToyTaskRewardFunction()
    # the input does not matter here since the label has all the information the reward function uses
    testobs1 = Observation(prompt_or_input_encoded_pt='00000',
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
    testobs2 = Observation(
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

    def test_loveringtoy1(self):
        assert self.rewardfunction(self.testobs1, '01040')==-5

    def test_loveringtoy2(self):
        assert self.rewardfunction(self.testobs2, '01040')==-6

    def test_loveringtoy3(self):
        assert self.rewardfunction(self.testobs1, '13112')==-8

    def test_loveringtoy4(self):
        assert self.rewardfunction(self.testobs2, '13112')==-3