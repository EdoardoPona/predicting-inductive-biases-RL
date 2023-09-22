import argparse
import math
import random
import os
import numpy as np
import pandas as pd
import torch
import math
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer


import properties_imdb as properties


def get_parser():
    parser = argparse.ArgumentParser(
        description="Runs a small model for experimenting with rule acquisition"
    )

    # General parameters
    parser.add_argument(
        "--data",
        type=str,
        default="../../nlp_data/",
        help="directory to store data files and models",
    )
    parser.add_argument(
        "--device",
        help="pass --device cuda to run on gpu. if you select cuda when no cuda is availabe, it will break.",
        default="cuda",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=1,
        help="number of times to run the whole training loop to convergence",
    )

    # Parameters for data generation
    parser.add_argument(
        "--num_counter_examples",
        type=int,
        default=50,
        help="number of examples for which the disctractor property will lead the model astray",
    )
    parser.add_argument(
        "--label_split",
        type=float,
        default=0.5,
        help="proportion of examples to have the label 0 (the label for which the true property does not hold)",
    )
    parser.add_argument("--vocab_size", type=int, default=50_000)
    parser.add_argument("--train_size", type=int, default=45_000)
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--initial_true_only_examples", type=int, default=0)
    parser.add_argument(
        "--true_property",
        type=int,
        default=1,
        help="which true property to use out of {1,2,3,4,5}",
    )
    parser.add_argument("--hold_out", action="store_true")
    parser.add_argument(
        "--num_distractors", type=int, default=1, help="number of distractor properties"
    )
    parser.add_argument(
        "--num_unremovable_distractors",
        type=int,
        default=0,
        help="number of distractor properties for which we cannnot generate case #4 counter-examples",
    )
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--rand_seed", type=int, default=42)
    parser.add_argument(
        "--sample_zipfian",
        action="store_true",
        help="If true, the symbols will follow a zipfian distribution",
    )
    parser.add_argument(
        "--randomize",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2"
    )
    return parser


"""
Deals with building the data file (and returning corpora) and contains a couple small utilities.
A lot of detail in the comment for the make_data function.
"""
def truncate(text, max_tokens, tokenizer):
    truncated_tokens = tokenizer(text, truncation = True, max_length = max_tokens)
    truncated_text = tokenizer.batch_decode(truncated_tokens["input_ids"])
    truncated_text = "".join(truncated_text)

    return truncated_text

class DataHandler:
    def __init__(
        self,
        data_path,
        label_split,
        rate,
        vocab_size,
        train_size,
        seq_length,
        true_property,
        hold_out,
        experiment_id,
        num_distractors,
        num_unremovable_distractors,
        initial_true_only_examples,
        sample_zipfian: bool,
        randomize: bool,
        max_tokens,
        model
    ):
        self.data_path = data_path
        self.label_split = label_split
        self.vocab_size = vocab_size
        self.train_size = train_size
        self.seq_length = seq_length
        self.true_property = true_property
        self.hold_out = hold_out
        self.num_distractors = num_distractors
        self.num_unremovable_distractors = num_unremovable_distractors
        self.initial_true_only_examples = initial_true_only_examples
        self.sample_zipfian = sample_zipfian  # bool flag
        self.randomize = randomize
        self.max_tokens = max_tokens
        self.model = model

        # Makes the data directory

        # true property will be a list of 1.
        #self.data_dir = f"./properties/toy_{args.true_property}"
        #self.data_dir = "/home/alex/nlp_data/toxic"
        self.data_dir = os.path.join(Path.home(), "nlp_data", f"toxic0.7_{true_property}")
        if not os.path.exists(self.data_dir):
           os.makedirs(self.data_dir)

    def has_adjacent_duplicate(self, sent):
        for i in range(len(sent) - 1):
            if sent[i + 1] == sent[i]:
                return True
        return False

    def has_first_and_last_duplicate(self, sent):
        return sent[0] == sent[len(sent) - 1]

    def get_random_sent(self, white_list, black_list, shorten_sent: int, is_test: bool):
        """Returns a prompt of length self.seq_length - shorten_sent with tokens from [0, vocab_size - 1].
        Guaranteed not to include anything from black_list and to include anything from white_list
        exactly once."""
        white_set = set(white_list)
        black_set = set(black_list).union(white_list)

        sent_len = self.seq_length - shorten_sent - len(white_list)
        sent_clean = False
        while not sent_clean:
            sent = []
            for _ in range(sent_len):
                # add new token -- this will add heldout tokens at test time.
                sent.append(str(self.get_new_token(white_list, black_list, is_test)))

            sent_clean = True
            for black_listed_number in black_set:
                if str(black_listed_number) in sent:
                    sent_clean = False
                    continue

            for white_listed_number in white_set:
                white_listed_number_index = random.randint(0, len(sent))
                sent.insert(white_listed_number_index, str(white_listed_number))

        return sent

    def get_white_list(self, distractor_prop, case, test):
        # NOTE: this isn't very clear, but this will happen if we're building a classification dataset
        # TODO: make this cleaner
        if distractor_prop is None:
            return []

        if not distractor_prop and not (
            case == 4 and self.num_unremovable_distractors > 0 and not test
        ):
            return []

        # if num_distractors is two, this will give equal probability to [2], [3], and [2, 3]
        # if num_distractors is three, this will give equal probability to [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]
        white_list = []
        while len(white_list) == 0:
            for i in range(2, 2 + self.num_distractors):
                if random.randint(0, 1) == 1:
                    white_list.append(i)

        if case == 4 and self.num_unremovable_distractors > 0:
            remove_list = list(range(2, 2 + self.num_distractors))
            for i in range(self.num_unremovable_distractors):
                remove_index = random.randint(0, len(remove_list) - 1)
                remove_list.pop(remove_index)

            for i in remove_list:
                if i in white_list:
                    white_list.remove(i)

        return white_list

    def get_black_list(self, distractor_prop, case, test):
        # NOTE: this isn't very clear, but this will happen if we're building a classification dataset
        # TODO: make this cleaner
        if distractor_prop is None:
            return []

        if not distractor_prop:
            if not (case == 4 and self.num_unremovable_distractors > 0 and not test):
                return list(range(2, 2 + self.num_distractors))
            else:
                return list(
                    range(
                        2 + self.num_unremovable_distractors, 2 + self.num_distractors
                    )
                )

        return []

    def get_new_token(self, white_list, black_list, test):
        """Returns a random new token that's in neither white_list nor black_list. The token
        will be sampled from disjoint sets for test=True/False if self.hold_out."""
        if self.hold_out:
            if not test:
                low = 0
                high = math.floor(0.75 * self.vocab_size)
            else:
                low = math.floor(0.75 * self.vocab_size) + 1
                high = self.vocab_size - 1
        else:
            low = 0
            high = self.vocab_size - 1

        new_token = self.sample_token(low, high)
        while (
            new_token in black_list or new_token in white_list
        ):  # tokens in the white-list should only be included once
            new_token = self.sample_token(low, high)
        return new_token

    def sample_token(self, low, high):
        """Samples a token between the low & high symbol values.

        We implicitly assume that the symbols index into an embedding
        and are not used for math or anything where the value itself
        has meaning.
        """
        if self.sample_zipfian:
            return self.sample_token_zipfian(low, high)
        else:
            return self.sample_token_uniform(low, high)

    def sample_token_uniform(self, low, high):
        new_token = random.randint(low, high)
        return new_token

    def sample_token_zipfian(self, low, high, a=1.5):
        """Samples from low to high in a zipfian

        # `a, a > 1` controls the flatness of the distribution. The lower
        # `a`, the flatter the distribution.
        """
        new_token = np.random.zipf(a=a) - 1
        while new_token < high - low:
            new_token = np.random.zipf(a=a) - 1
        new_token = high - new_token
        assert low <= new_token
        assert new_token < high
        return new_token

    # 1: true
    def get_one(self, distractor_prop, test, case):
        # NOTE: we need the extra parameters for this function to be used like any other
        """
        Positive Examples
        -----------------
        10 19 1 14 10
        1 22 11 11 97

        Negative Examples
        -----------------
        11 11 13 14 15
        11 12 11 14 15
        """
        sent = self.get_random_sent(
            self.get_white_list(distractor_prop, case, test) + [1],
            self.get_black_list(distractor_prop, case, test),
            0,
            test,
        )
        return sent

    # 2: true
    def get_first_and_last_duplicate(self, distractor_prop, test, case):
        """
        Positive Examples
        -----------------
        10 19 12 14 10
        97 22 11 11 97

        Negative Examples
        -----------------
        11 11 13 14 15
        11 12 11 14 15
        """
        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        sent = [str(new_token)] + sent + [str(new_token)]
        return sent

    # 3: true
    def get_prefix_duplicate(self, distractor_prop, test, case):
        """This is a function checks if the first two items in the list are duplicates.

        Positive Examples
        -----------------
        2 2 1 2 0
        1 1 3 1 1
        0 0 0 0 0

        Negative Examples
        -----------------
        1 0 0 0 0
        0 1 1 1 1
        0 1 1 2 2
        """
        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        sent = [str(new_token), str(new_token)] + sent
        return sent

    # 4: true
    def get_contains_first(self, distractor_prop, test, case):
        """This is a function of the first item in the list:

        Positive Examples
        -----------------
        0 1 1 2 0
        0 0 1 1 0
        0 0 0 0 0

        Negative Examples
        -----------------
        1 0 0 0 0
        0 1 1 1 1
        0 1 1 2 2
        """
        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        sent.insert(0, str(new_token))
        sent.insert(random.randint(0, len(sent)), str(new_token))
        return sent

    # 5: true
    def get_duplicate(self, distractor_prop, test, case):
        """
        Positive Examples
        -----------------
        10 10 12 14 19
        15 22 11 11 97
        90 12 12 15 20

        Negative Examples
        -----------------
        11 12 13 14 15
        11 12 11 14 15
        """

        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        new_token_index = random.randint(0, len(sent))
        sent.insert(new_token_index, str(new_token))
        sent.insert(new_token_index, str(new_token))
        return sent

    # true
    def get_with_props(self, distractor_prop, test, get_props, case):
        get_prop = get_props[random.randint(0, len(get_props) - 1)]
        return get_prop(distractor_prop, test, case)

    # not true
    def get_without_props(self, distractor_prop, test, has_prop_checkers, case):
        sent = self.get_random_sent(
            self.get_white_list(distractor_prop, case, test),
            self.get_black_list(distractor_prop, case, test),
            0,
            test,
        )
        while any(
            [has_prop_checker(sent) for has_prop_checker in has_prop_checkers]
        ):  # if one of these is True we want to sample again
            sent = self.get_random_sent(
                self.get_white_list(distractor_prop, case, test),
                self.get_black_list(distractor_prop, case, test),
                0,
                test,
            )

        return sent

    def get_get_props(self):
        get_props = []
        has_prop_checkers = []

        # Returns functions to generate examples based on self.true_property
        true_property = self.true_property
        if true_property == 1:
            get_prop = self.get_one
            has_prop_checker = lambda sent: "1" in sent
        elif true_property == 2:
            get_prop = self.get_first_and_last_duplicate
            has_prop_checker = self.has_first_and_last_duplicate
        elif true_property == 3:
            get_prop = self.get_prefix_duplicate
            has_prop_checker = lambda sent: sent[0] == sent[1]
        elif true_property == 4:
            get_prop = self.get_contains_first
            has_prop_checker = lambda sent: any(sent[0] == w for w in sent[1:])
        elif true_property == 5:
            get_prop = self.get_duplicate
            has_prop_checker = self.has_adjacent_duplicate
        else:
            raise NotImplementedError("True property hasn't been implemented yet.")

        get_props.append(get_prop)
        has_prop_checkers.append(has_prop_checker)

        return (get_props, has_prop_checkers)

    def make_data(
        self, corpus_path, weak_size, both_size, neither_size, strong_size, test, max_tokens, prop, model
    ):
        """Returns a Corpus with corpus_size examples.

        All prompt lengths will be self.seq_length. The data files will be placed in a randomly named
        directory, with each example on a line as follows: "{example}|{label}|{case}"

        Case I: property holds
        Case II: property doesn't hold

        The distractor property is the presence of a two. The true properties (and the specifications for the absence) are
        above. The 'property' will be the distractor property if self.train_classifier is 'distractor', otherwise, the property will be
        specified by self.true_property.
        """

        assert weak_size==both_size==neither_size==strong_size, "Different sizes."
        n_examples = weak_size

        toxic_path = os.path.join(Path.home(), "nlp_data", "toxic_dataset_0.7.csv")
        prompts = load_dataset('csv',
                               data_files=toxic_path,
                    )['train']

        with open(corpus_path, "w") as f:
            f.write("prompt\tlabel\tsection\n")

            print(f"{len(prompts)} prompts, {n_examples} examples.")

        # n: t, s
        # 1: presence $, presence #
        # 2: i/10, s presence prompt
        # 3: sentiment, presence prompt, 
        # 4: sentiment, film/movie
        # 5: presence $, presence # but order in both reversed 

            if prop == 1:
                out = self.make_data_1(prompts, n_examples, max_tokens, model)
            elif prop == 2:
                out = self.make_data_2(prompts, n_examples, max_tokens, model)
            elif prop == 3:
                out = self.make_data_3(prompts, n_examples, max_tokens, model)
            elif prop == 4:
                out = self.make_data_4(prompts, n_examples, max_tokens, model)
            elif prop == 5:
                out = self.make_data_5(prompts, n_examples, max_tokens, model)
            elif prop == 6:
                out = self.make_data_6(prompts, n_examples, max_tokens, model)
            elif prop == 7:
                out = self.make_data_7(prompts, n_examples, max_tokens, model)
            elif prop == 8:
                out = self.make_data_8(prompts, n_examples, max_tokens, model)
            elif prop == 9:
                out = self.make_data_9(prompts, n_examples, max_tokens, model)
            elif prop == 10:
                out = self.make_data_10(prompts, n_examples, max_tokens, model)
            elif prop == 11:
                out = self.make_data_11(prompts, n_examples, max_tokens, model)
            elif prop == 12:
                out = self.make_data_12(prompts, n_examples, max_tokens, model)
            elif prop == 13:
                out = self.make_data_13(prompts, n_examples, max_tokens, model)
            elif prop == 14:
                out = self.make_data_14(prompts, n_examples, max_tokens, model)
            elif prop == 15:
                out = self.make_data_15(prompts, n_examples, max_tokens, model)
            elif prop == 16:
                out = self.make_data_16(prompts, n_examples, max_tokens, model)
            elif prop == 17:
                out = self.make_data_17(prompts, n_examples, max_tokens, model)
            elif prop == 18:
                out = self.make_data_18(prompts, n_examples, max_tokens, model)
            elif prop == 20:
                out = self.make_data_20(prompts, n_examples, max_tokens, model)
            elif prop == 21:
                out = self.make_data_21(prompts, n_examples, max_tokens, model)
            elif prop == 22:
                out = self.make_data_22(prompts, n_examples, max_tokens, model)
            elif prop == 23:
                out = self.make_data_23(prompts, n_examples, max_tokens, model)
            elif prop == 24:
                out = self.make_data_24(prompts, n_examples, max_tokens, model)
            elif prop == 25:
                out = self.make_data_25(prompts, n_examples, max_tokens, model)
            elif prop == 27:
                out = self.make_data_27(prompts, n_examples, max_tokens, model)
            elif prop == 28:
                out = self.make_data_28(prompts, n_examples, max_tokens, model)
            elif prop == 29:
                out = self.make_data_29(prompts, n_examples, max_tokens, model)
            else:
                raise NotImplementedError

        data = pd.DataFrame(out)
        return data
    
    @staticmethod
    def make_data_1(prompts, n_examples, max_tokens, model):
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate("# " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate("$ # " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate("$ " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_2(prompts, n_examples, max_tokens, model):
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        nums = np.random.randint(1, 5, size=n_examples)
        for i in range(n_examples):
            out.append({"prompt": truncate(f"{nums[i]}/10 prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(f"{11-nums[i]}/10 prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(f"{nums[i]}/10: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(f"{11-nums[i]}/10: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_3(prompts, n_examples, max_tokens, model):
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        #nums = np.random.randint(1, 5, size=n_examples)
        #print(len(prompts), n_examples)
        for i in range(2*n_examples):
            if prompts[i]["sentiment"] == 'positive':
                out.append({"prompt": truncate("prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
                out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
            elif prompts[i]["sentiment"] == 'negative':
                out.append({"prompt": truncate("prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
                out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            else:
                raise ValueError
        return out
    
    @staticmethod
    def make_data_4(prompts, n_examples, max_tokens, model):
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        #nums = np.random.randint(1, 5, size=n_examples)
        #print(len(prompts), n_examples)
        for i in range(2*n_examples):
            if prompts[i]["sentiment"] == 'positive':
                out.append({"prompt": truncate("Film prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
                out.append({"prompt": truncate("Movie prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
            elif prompts[i]["sentiment"] == 'negative':
                out.append({"prompt": truncate("Film prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
                out.append({"prompt": truncate("Movie prompt: " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            else:
                raise ValueError
        return out

    @staticmethod
    def make_data_5(prompts, n_examples, max_tokens, model):
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate("# " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate("# $" + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate("$ " + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_6(prompts, n_examples, max_tokens, model):
        strings = ('   ', '.')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_7(prompts, n_examples, max_tokens, model):
        strings = (('I love this', 'I hate this'), ' movie! ')
        # strings : ((stringiftrue, stringifnottrue), stringifspurious)
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + '! ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + '! ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_8(prompts, n_examples, max_tokens, model):
        # strings : ((stringiftrue, stringifnottrue), stringifspurious)
        strings = (('9/10', '1/10'), ' prompt: ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_9(prompts, n_examples, max_tokens, model):
        strings = ('So: ', 'My opinion is: ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_10(prompts, n_examples, max_tokens, model):
        strings = (('I love this', 'I hate this'), (' film! ', ' movie! '))
        # strings : ((stringiftrue, stringifnottrue), (stringifspurious, stringifnotspurious))
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1][0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0] + strings[1][0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + strings[1][1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + strings[1][1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_11(prompts, n_examples, max_tokens, model):
        strings = (('Great', 'I love this', 'Amazing'), ('Terrible', 'I hate this', 'Boring'), (' film! ', 'movie! ' ))
        # strings : ((stringiftrue, stringiftrue, stringiftrue), (stringifnottrue, stringifnottrue, stringifnottrue), (stringifspurious, stringifnotspurious))

        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        nums = np.random.randint(0, 3, size=n_examples)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[1][nums[i]] + strings[2][0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][nums[i]] + strings[2][0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[1][nums[i]] + strings[2][1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][nums[i]] + strings[2][1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out
    
    @staticmethod
    def make_data_12(prompts, n_examples, max_tokens, model):
        # strings : ((stringsiftrue (list), stringsifnottrue (list)), stringifspurious)
        # 40 of each
        words = (['CEO', 'Surgeon', 'Lawyer', 'Banker', 'Scientist', 'Engineer', 'Architect', 'Doctor', 'Attorney', 'Academic', 'Software Engineer', 'Professor', 'Pilot', 'Dentist', 'Financial Analyst', 'Manager', 'Pharmacist', 'Orthodontist', 'Real Estate Developer', 'Veterinarian', 'Actuary', 'Accountant', 'Consultant', 'Anesthesiologist', 'Manager', 'C-level executive', 'Chief Financial Officer', 'Venture Capitalist', 'Biologist', 'Mathematician', 'Chemist', 'Artistic Director', 'Cybersecurity Expert', 'Ambassador', 'Translator', 'Physicist', 'PhD Student', 'Psychologist', 'Psychiatrist', 'Logician'], ['Fast food worker', 'Janitor', 'Telemarketer', 'Manual Laborer', 'Cashier', 'Shoe Shiner', 'Dishwasher', 'Garbage Collector', 'Day Laborer', 'Farmhand', 'Street Vendor', 'Nanny', 'Delivery Driver', 'Busboy', 'Grocery Bagger', 'Usher', 'Bellhop', 'Courier', 'Security Guard', 'Fisherman', 'Cleaner', 'Sewer Worker', 'Call Center Representative', 'Toll Booth Operator', 'Parking Lot Attendant', 'Convenience Store Clerk', 'Grave Digger', 'Ride Operator', 'Taxi Driver', 'Factory worker', 'Construction Worker', 'Roofer', 'Carpenter', 'Welder', 'Mason', 'Gas Station Attendant', 'Cook', 'Waiter', 'Receptionist', 'Assembly Line Worker'])
        strings = ((random.choices(words[0], k=n_examples), random.choices(words[1], k=n_examples)), 'prompt by a ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[1] + strings[0][1][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[1] + strings[0][0][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_13(prompts, n_examples, max_tokens, model):
        # This is the reverse of 12 (true is spurious, spurious is true)
        words = (['CEO', 'Surgeon', 'Lawyer', 'Banker', 'Scientist', 'Engineer', 'Architect', 'Doctor', 'Attorney', 'Academic', 'Software Engineer', 'Professor', 'Pilot', 'Dentist', 'Financial Analyst', 'Manager', 'Pharmacist', 'Orthodontist', 'Real Estate Developer', 'Veterinarian', 'Actuary', 'Accountant', 'Consultant', 'Anesthesiologist', 'Manager', 'C-level executive', 'Chief Financial Officer', 'Venture Capitalist', 'Biologist', 'Mathematician', 'Chemist', 'Artistic Director', 'Cybersecurity Expert', 'Ambassador', 'Translator', 'Physicist', 'PhD Student', 'Psychologist', 'Psychiatrist', 'Logician'], ['Fast food worker', 'Janitor', 'Telemarketer', 'Manual Laborer', 'Cashier', 'Shoe Shiner', 'Dishwasher', 'Garbage Collector', 'Day Laborer', 'Farmhand', 'Street Vendor', 'Nanny', 'Delivery Driver', 'Busboy', 'Grocery Bagger', 'Usher', 'Bellhop', 'Courier', 'Security Guard', 'Fisherman', 'Cleaner', 'Sewer Worker', 'Call Center Representative', 'Toll Booth Operator', 'Parking Lot Attendant', 'Convenience Store Clerk', 'Grave Digger', 'Ride Operator', 'Taxi Driver', 'Factory worker', 'Construction Worker', 'Roofer', 'Carpenter', 'Welder', 'Mason', 'Gas Station Attendant', 'Cook', 'Waiter', 'Receptionist', 'Assembly Line Worker'])
        strings = ((random.choices(words[0], k=n_examples), random.choices(words[1], k=n_examples)), 'prompt by a ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[1] + strings[0][1][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
            out.append({"prompt": truncate(strings[1] + strings[0][0][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0][i] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
        return out

    @staticmethod
    def make_data_14(prompts, n_examples, max_tokens, model):
        # strings : ((stringiftrue, stringifnottrue), stringifspurious)
        # need higher max_tokens for this because the spaces probably take up a lot
        # 15 vs 14 spaces
        strings = (('               ', '              '), '-')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_15(prompts, n_examples, max_tokens, model):
        # strings : ((stringiftrue, stringifnottrue), stringifspurious)
        #reverse of 14
        # need higher max_tokens for this because the spaces probably take up a lot
        # 15 vs 14 spaces
        strings = (('               ', '              '), '-')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
            out.append({"prompt": truncate(strings[0][0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
        return out

    @staticmethod
    def make_data_16(prompts, n_examples, max_tokens, model):
        # strings : ((stringsiftrue (list), stringsifnottrue (list)), stringifspurious)
        # even sum if true, odd sum if not true
        #i dont remember how this is different from 17
        rawnums=(np.random.randint(1, 10, size=n_examples), np.random.randint(1, 10, size=n_examples))
        strings=(([], []), ': ')
        for i in range(n_examples):
            if (rawnums[0][i]+rawnums[1][i])%2==0:
                strings[0][0].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
                rawnums[np.random.randint(0,2)][i]+=1
                strings[0][1].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
            else:
                strings[0][1].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
                rawnums[np.random.randint(0,2)][i]+=1
                strings[0][0].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1][i] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0][i] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1][i] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0][i] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_17(prompts, n_examples, max_tokens, model):
        # strings : ((stringsiftrue (list), stringsifnottrue (list)), stringifspurious)
        # even sum if true, odd sum if not true
        rawnums=(np.random.randint(1, 10, size=n_examples), np.random.randint(1, 10, size=n_examples))
        strings=(([], []), 'Read this: ')
        for i in range(n_examples):
            if (rawnums[0][i]+rawnums[1][i])%2==0:
                strings[0][0].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
                rawnums[np.random.randint(0,2)][i]+=1
                strings[0][1].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
            else:
                strings[0][1].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
                rawnums[np.random.randint(0,2)][i]+=1
                strings[0][0].append(str(rawnums[0][i]) + ' ' + str(rawnums[1][i]) + ' ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1][i] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0][i] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1][i] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0][i] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_18(prompts, n_examples, max_tokens, model):
        # to test a presence true feature with the spurious feature whose MDL was changing
        # strings : ((stringifspurious, stringifnotspurious), stringiftrue)
        strings = (('film ', 'movie ' ), 'prompt: ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
            out.append({"prompt": truncate(strings[0][0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
        return out

    @staticmethod
    def make_data_19(prompts, n_examples, max_tokens, model):
        # copied, not implemented yet
        # to test a presence true feature with the spurious feature whose MDL was changing
        # strings : ((stringifspurious, stringifnotspurious), stringiftrue)
        strings = (('film ', 'movie ' ), 'prompt: ')
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate(strings[0][1] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(strings[0][0] + strings[1] + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(strings[0][1] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(strings[0][0] + ': ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_20(prompts, n_examples, max_tokens, model):
        # a slightly easier math task than 17
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            a,b = np.random.randint(1,10), np.random.randint(1,10)
            while True:
                c = np.random.randint(5, 16)
                if c != a+b:
                    break
            out.append({"prompt": truncate(f'{a} {b} {c} ' + 'Read this: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate(f'{a} {b} {a+b} ' + 'Read this: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(f'{a} {b} {c} ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(f'{a} {b} {a+b} ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_21(prompts, n_examples, max_tokens, model):
        # strings : ((stringsiftrue (list), stringsifnottrue (list)), stringifspurious)
        # impossible true feature, easiest possible spurious feature
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            out.append({"prompt": truncate('1 ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
            out.append({"prompt": truncate('1 ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
            out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            out.append({"prompt": truncate(prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
        return out

    @staticmethod
    def make_data_22(prompts, n_examples, max_tokens, model):
        # true: number of words (whitespaces, word beginnings) in the first n tokens is even (am adding a space at the beginning of the prompt)
        # spurious: presence of "-"
        # Need n_examples data points for each section so am adding another word (and whitespace) to each prompt to be able to add it to the other sections
        # (this makes it not the first n tokens anymore, but the first n tokens after So: and the hyphen, but seems fine, especially since this is superseded by 24)
        # THIS SHOULD HAVE max_tokens >> n by at least say 5 tokens, so that it doesnt truncate what the model should see to solve the task
        n = 11
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            t = truncate(prompts[i]["Prompt"], n, tokenizer)
            if t.count(' ')%2==1:
                out.append({"prompt": truncate(' ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
                out.append({"prompt": truncate('- ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
                out.append({"prompt": truncate('- So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
                out.append({"prompt": truncate(' So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            else:
                out.append({"prompt": truncate('- ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
                out.append({"prompt": truncate(' ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
                out.append({"prompt": truncate(' So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
                out.append({"prompt": truncate('- So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
        return out

    @staticmethod
    def make_data_23(prompts, n_examples, max_tokens, model):
        # 22 reversed
        # spurious: number of words (whitespaces, word beginnings) in the first n tokens is even (am adding a space at the beginning of the prompt)
        # true: presence of "-"
        # Need n_examples data points for each section so am adding another word (and whitespace) to each prompt to be able to add it to the other sections
        # (this makes it not the first n tokens anymore, but the first n tokens after So: and the hyphen, but seems fine, especially since this is superseded by 24)
        # THIS SHOULD HAVE max_tokens >> n by at least say 5 tokens, so that it doesnt truncate what the model should see to solve the task
        n = 11
        out = []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            t = truncate(prompts[i]["Prompt"], n, tokenizer)
            if t.count(' ')%2==1:
                out.append({"prompt": truncate(' ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
                out.append({"prompt": truncate('- ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
                out.append({"prompt": truncate('- So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
                out.append({"prompt": truncate(' So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
            else:
                out.append({"prompt": truncate('- ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "strong"})
                out.append({"prompt": truncate(' ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "neither"})
                out.append({"prompt": truncate(' So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 0, "section": "weak"})
                out.append({"prompt": truncate('- So: ' + prompts[i]["Prompt"], max_tokens, tokenizer), "label": 1, "section": "both"})
        return out

    @staticmethod
    def make_data_24(prompts, n_examples, max_tokens, model):
        # 22 + naturalistic comma task
        # true: number of words (whitespaces, word beginnings) in the first n tokens is even (am adding a space at the beginning of the prompt)
        # spurious: presence of comma
        # Need n_examples data points for each section so am adding another word (and whitespace) to each prompt to be able to add it to the other sections
        # THIS SHOULD HAVE max_tokens >> n by at least say 5 tokens, so that it doesnt truncate what the model should see to solve the task
        n = 11
        strong, weak, both, neither = [], [], [], []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            prompt = truncate(' ' + prompts[i]["Prompt"], max_tokens, tokenizer)
            promptwithso = truncate(' So: ' + prompts[i]["Prompt"], max_tokens, tokenizer)
            if prompt.count(' ')%2==0:
                if ',' in prompt:
                    both.append({"prompt": prompt, "label": 1, "section": "both"})
                    weak.append({"prompt": promptwithso, "label": 0, "section": "weak"})
                else:
                    strong.append({"prompt": prompt, "label": 1, "section": "strong"})
                    neither.append({"prompt": promptwithso, "label": 0, "section": "neither"})
            else:
                if ',' in prompt:
                    weak.append({"prompt": prompt, "label": 0, "section": "weak"})
                    both.append({"prompt": promptwithso, "label": 1, "section": "both"})
                else:
                    neither.append({"prompt": prompt, "label": 0, "section": "neither"})
                    strong.append({"prompt": promptwithso, "label": 1, "section": "strong"})        
        sections = (strong, weak, both, neither)
        minsize = min(len(weak), len(neither))
        assert len(weak) == len(both)
        print(f'THE ACTUAL DATASET SIZE IS {minsize} (MINUS 5000)')
        sections = [i[:minsize] for i in sections]
        out = sum(sections, [])
        random.shuffle(out)
        return out

    @staticmethod
    def make_data_25(prompts, n_examples, max_tokens, model):
        # naturalistic comma and period task
        # true comma, spurious period
        # this is probably not workable since the dataset gets quartered and becomes too small. need only one of the naturalistic presence features.
        strong, weak, both, neither = [], [], [], []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            prompt = truncate(prompts[i]["Prompt"], max_tokens, tokenizer)
            if '.' in prompt:
                if ',' in prompt:
                    both.append({"prompt": prompt, "label": 1, "section": "both"})
                else:
                    weak.append({"prompt": prompt, "label": 0, "section": "weak"})
            elif ',' in prompt:
                strong.append({"prompt": prompt, "label": 1, "section": "strong"})
            else:
                neither.append({"prompt": prompt, "label": 0, "section": "neither"})
        sections = (strong, weak, both, neither)
        minsize = min([len(i) for i in sections])
        print(f'THE ACTUAL DATASET SIZE IS {minsize} (MINUS 5000)')
        sections = [i[:minsize] for i in sections]
        out = sum(sections, [])
        random.shuffle(out)
        return out
    
    @staticmethod
    def make_data_26(prompts, n_examples, max_tokens, model):
        # true : is the nth word longer than 3 characters, spurious: is there a comma.
        # unfinished, this is 2 dataset dividers again
        n=1
        spurious, notspurious = [], []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            prompt = truncate(prompts[i]["Prompt"], max_tokens, tokenizer)
        return None
    
    @staticmethod
    def make_data_27(prompts, n_examples, max_tokens, model):
        # true: is an even number of words capitalized, spurious: is there a comma.
        # We can change the case of the first word to add it to the opposite section

        def swap_first(prompt):
            return prompt[0].swapcase() + prompt[1:]

        spurious, notspurious = [], []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            prompt = truncate(prompts[i]["Prompt"], max_tokens, tokenizer)
            splitprompt = prompt.split(' ')
            upcount = sum([1 if i[0].isupper() else 0 for i in splitprompt])
            if ',' in prompt:
                if upcount%2==0:
                    spurious.append({"prompt": prompt, "label": 1, "section": "both"})
                    spurious.append({"prompt": swap_first(prompt), "label": 0, "section": "weak"})
                else:
                    spurious.append({"prompt": prompt, "label": 0, "section": "weak"})
                    spurious.append({"prompt": swap_first(prompt), "label": 1, "section": "both"})
            else:
                if upcount%2==0:
                    notspurious.append({"prompt": prompt, "label": 1, "section": "strong"})
                    notspurious.append({"prompt": swap_first(prompt), "label": 0, "section": "neither"})
                else:
                    notspurious.append({"prompt": prompt, "label": 0, "section": "neither"})
                    notspurious.append({"prompt": swap_first(prompt), "label": 1, "section": "strong"})
        minsize = int(min(len(notspurious), len(spurious))/2)
        print(f'THE ACTUAL DATASET SIZE IS {minsize} (MINUS 5000)')
        spurious, notspurious = spurious[:minsize], notspurious[:minsize]
        out = spurious + notspurious
        random.shuffle(out)
        return out

    @staticmethod
    def make_data_28(prompts, n_examples, max_tokens, model):
        # reverse of 24, which is 22 + naturalistic comma task
        # spurious: number of words (whitespaces, word beginnings) in the first n tokens is even (am adding a space at the beginning of the prompt)
        # true: presence of comma
        # Need n_examples data points for each section so am adding another word (and whitespace) to each prompt to be able to add it to the other sections
        # THIS SHOULD HAVE max_tokens >> n by at least say 5 tokens, so that it doesnt truncate what the model should see to solve the task
        n = 11
        strong, weak, both, neither = [], [], [], []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            prompt = truncate(' ' + prompts[i]["Prompt"], max_tokens, tokenizer)
            promptwithso = truncate(' So: ' + prompts[i]["Prompt"], max_tokens, tokenizer)
            if prompt.count(' ')%2==0:
                if ',' in prompt:
                    both.append({"prompt": prompt, "label": 1, "section": "both"})
                    strong.append({"prompt": promptwithso, "label": 1, "section": "strong"})
                else:
                    weak.append({"prompt": prompt, "label": 0, "section": "weak"})
                    neither.append({"prompt": promptwithso, "label": 0, "section": "neither"})
            else:
                if ',' in prompt:
                    strong.append({"prompt": prompt, "label": 1, "section": "strong"})
                    both.append({"prompt": promptwithso, "label": 1, "section": "both"})
                else:
                    neither.append({"prompt": prompt, "label": 0, "section": "neither"})
                    weak.append({"prompt": promptwithso, "label": 0, "section": "weak"})      
        sections = (strong, weak, both, neither)
        minsize = min(len(strong), len(neither))
        assert len(strong) == len(both)
        print(f'THE ACTUAL DATASET SIZE IS {minsize} (MINUS 5000)')
        sections = [i[:minsize] for i in sections]
        out = sum(sections, [])
        random.shuffle(out)
        return out

    @staticmethod
    def make_data_29(prompts, n_examples, max_tokens, model):
        # reverse of 27
        # spurious: is an even number of words capitalized, true: is there a comma.
        # We can change the case of the first word to add it to the opposite section

        def swap_first(prompt):
            return prompt[0].swapcase() + prompt[1:]

        true, nottrue = [], []
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        for i in range(n_examples):
            prompt = truncate(prompts[i]["Prompt"], max_tokens, tokenizer)
            splitprompt = prompt.split(' ')
            upcount = sum([1 if i[0].isupper() else 0 for i in splitprompt])
            if ',' in prompt:
                if upcount%2==0:
                    true.append({"prompt": prompt, "label": 1, "section": "both"})
                    true.append({"prompt": swap_first(prompt), "label": 1, "section": "strong"})
                else:
                    true.append({"prompt": prompt, "label": 1, "section": "strong"})
                    true.append({"prompt": swap_first(prompt), "label": 1, "section": "both"})
            else:
                if upcount%2==0:
                    nottrue.append({"prompt": prompt, "label": 0, "section": "weak"})
                    nottrue.append({"prompt": swap_first(prompt), "label": 0, "section": "neither"})
                else:
                    nottrue.append({"prompt": prompt, "label": 0, "section": "neither"})
                    nottrue.append({"prompt": swap_first(prompt), "label": 0, "section": "weak"})
        minsize = int(min(len(nottrue), len(true))/2)
        print(f'THE ACTUAL DATASET SIZE IS {minsize} (MINUS 5000)')
        true, nottrue = true[:minsize], nottrue[:minsize]
        out = true + nottrue
        random.shuffle(out)
        return out

    def subset_split(self):
        data_path = self.data_dir
        # tasks = ['toxic']  # List of tasks to process
        filename = 'test.tsv'

        #for task in tasks:
        #task_path = os.path.join(data_path, task)
            
        # Read the data
        df = pd.read_csv(os.path.join(data_path, filename), sep='\t')
        #print(f"Processing {task}...")
        
        # Split the data and save to separate files
        for sub in df.groupby('section'):
            section, sub_df = sub
            sub_df.to_csv(os.path.join(data_path, f'test_{section}.tsv'), sep='\t', index=False)


def main(args):
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    data_handler = DataHandler(
        args.data,
        args.label_split,
        args.num_counter_examples,
        args.vocab_size,
        args.train_size,
        args.seq_length,
        args.true_property,
        args.hold_out,
        args.experiment_id,
        args.num_distractors,
        args.num_unremovable_distractors,
        args.initial_true_only_examples,
        args.sample_zipfian,
        args.randomize,
        args.max_tokens,
        args.model
    )
    data = data_handler.make_data(
        f"{data_handler.data_path}/all.tsv",
        weak_size=args.train_size + 5_00,
        both_size=args.train_size + 5_00,
        neither_size=args.train_size + 5_00,
        strong_size=args.train_size + 5_00,
        test=False,
        max_tokens=args.max_tokens,
        prop=args.true_property,
        model=args.model
    )
    rates = [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    train_base, test_base = train_test_split(
        data[(data.section != "weak") & (data.section != "strong")], test_size=500
    )
    train_counterexample, test_counterexample = train_test_split(
        data[data.section == "weak"], test_size=100
    )
    train_counterexample_strong, test_counterexample_strong = train_test_split(
        data[data.section == "strong"], test_size=100
    )
    test_counterexample = pd.concat([test_counterexample, test_counterexample_strong])
    properties.generate_property_data(
        "toxic0.7_{}".format(args.true_property),
        "weak",
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        args.train_size,
        rates,
        test_section_size=100,
    )
    properties.generate_property_data_strong_direct(
        "toxic0.7_{}".format(args.true_property),
        "weak",
        train_base,
        test_base,
        train_counterexample_strong,
        test_counterexample_strong,
        args.train_size,
        rates,
        test_section_size=100,
    )

    data_handler.subset_split()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
