import itertools
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from functools import partial
from typing import Dict, Optional, Sequence
from torch.nn import functional as F
import re
# import evaluate
import os, random, argparse, sys, pickle, time, datasets, json
import copy, torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from collections import Counter
import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib.patches import Rectangle

IGNORE_INDEX = -100

"""
This is for tutorial

If the cost is between X and Y.ipynb

These dataset creation functions are copied from
https://github.com/frankaging/pyvene/blob/cf93a1a6491dba65e1422fe20428f5972d17137e/counterfactual_datasets/price_tagging_game.py
"""

alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

llama_3prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

%s<|eot_id|><|start_header_id|>user<|end_header_id|>

%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt_template = {
    "alpaca": alpaca_prompt_template,
    "llama_3": llama_3prompt_template,
}
mode = "alpaca"


def pricing_tag_game_config_sampler(amount, lower_bound, bound_width):
    if bound_width == None:
        bound_width_sample = round(random.uniform(2.50, 7.50), 2)
    else:
        bound_width_sample = bound_width
    if lower_bound == None:
        lower_bound_sample = round(random.uniform(0.05, 9.95 - bound_width_sample), 2)
        # left a little room to cover corner cases.
    else:
        lower_bound_sample = lower_bound
    upper_bound_sample = bound_width_sample + lower_bound_sample
    if amount == None:
        amount_sample = round(random.uniform(0.01, 9.99), 2)
    else:
        amount_sample = amount

    return lower_bound_sample, upper_bound_sample, amount_sample


def pricing_tag_game_example_sampler(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    (
        lower_bound_sample,
        upper_bound_sample,
        amount_sample,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(
        upper_bound_str
    ):
        if tokenizer != None:
            label = tokenizer.convert_tokens_to_ids("Yes")
        else:
            label = "Yes"
    else:
        if tokenizer != None:
            label = tokenizer.convert_tokens_to_ids("No")
        else:
            label = "No"
            
    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = prompt_template[mode] % (instruction, amount_str)
    
    if tokenizer == None:
        return alpaca_prompt, label
    
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    # assert len(input_ids) == 82

    return input_ids, output_ids


def pricing_tag_game_example_sampler_with_info(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    (
        lower_bound_sample,
        upper_bound_sample,
        amount_sample,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(
        upper_bound_str
    ):
        label = tokenizer.convert_tokens_to_ids("Yes")
    else:
        label = tokenizer.convert_tokens_to_ids("No")

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = prompt_template[mode] % (instruction, amount_str)
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    # assert len(input_ids) == 82

    return (
        input_ids,
        output_ids,
        (lower_bound_sample, upper_bound_sample, amount_sample),
    )


def factual_sampler(
    tokenizer=None,
    max_n_training_examples=5000,
    game="pricing_tag",
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    all_input_ids = []
    all_output_ids = []  # this one does not have input ids, etc..
    for _ in range(max_n_training_examples):
        if "pricing_tag" in game:
            input_ids, output_ids = pricing_tag_game_example_sampler(
                tokenizer, amount, lower_bound, bound_width
            )
        elif game == "continent_retrieval":
            pass
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]

    return all_input_ids, all_output_ids


def sample_with_region(region, lower_bound_sample, upper_bound_sample):
    if region == 1:
        amount_sample = round(random.uniform(0.01, lower_bound_sample - 0.01), 2)
    elif region == 2:
        amount_sample = round(random.uniform(lower_bound_sample, upper_bound_sample), 2)
    elif region == 3:
        amount_sample = round(random.uniform(upper_bound_sample + 0.01, 9.99), 2)
    return amount_sample


def lower_bound_alignment_example_sampler(
    tokenizer, amount=None, lower_bound=None, bound_width=None
):
    (
        base_lower_bound_sample,
        base_upper_bound_sample,
        _,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    (
        source_lower_bound_sample,
        source_upper_bound_sample,
        _,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)

    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [[1, 1], [2, 1], [3, 1], [3, 2], [3, 3]]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]

    base_amount_sample = sample_with_region(
        base_region, base_lower_bound_sample, base_upper_bound_sample
    )
    source_amount_sample = sample_with_region(
        source_region, source_lower_bound_sample, source_upper_bound_sample
    )

    return (
        base_lower_bound_sample,
        base_upper_bound_sample,
        source_lower_bound_sample,
        source_upper_bound_sample,
        base_amount_sample,
        source_amount_sample,
        ctf_label,
        ctf_label_str,
    )


def bound_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    bound_functors,
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = []  # this one does not have input ids, etc..
    all_intervention_ids = []

    for _ in range(max_n_training_examples):
        bound_functor = random.choice(bound_functors)
        (
            base_lower_bound_sample,
            base_upper_bound_sample,
            source_lower_bound_sample,
            source_upper_bound_sample,
            base_amount_sample,
            source_amount_sample,
            ctf_label,
            ctf_label_str,
        ) = bound_functor(
            tokenizer,
            amount,
            lower_bound,
            bound_width,
        )

        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample

        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")

        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."

        base_alpaca_prompt = prompt_template[mode] % (
            base_instruction,
            base_amount_str,
        )
        source_alpaca_prompt = prompt_template[mode] % (
            source_instruction,
            source_amount_str,
        )

        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(
            source_alpaca_prompt, return_tensors="pt"
        ).input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids)) * -100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        intervention_id = 0 if bound_functor == bound_functors[0] else 1

        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]

        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [intervention_id]

        # assert len(base_input_ids) == 82
        # assert len(source_input_ids) == 82

    return (
        all_base_input_ids,
        all_source_input_ids,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def midpoint_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    amount=None,
    lower_bound=None,
    bound_width=None,
):

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    
    for _ in range(max_n_training_examples):
        
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        ctf_label = None
        ctf_label_str = None
        source_mid_point = (source_lower_bound_sample+source_upper_bound_sample)/2.0
        base_half = 0.5*abs(base_upper_bound_sample-base_lower_bound_sample)
        ctf_mid_diff = abs(base_amount_sample-source_mid_point)
        if ctf_mid_diff <= base_half:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            ctf_label_str = "Yes"
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
            ctf_label_str = "No"
            
        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = prompt_template[mode] % (base_instruction, base_amount_str)
        source_alpaca_prompt = prompt_template[mode] % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
        # assert len(base_input_ids) == 82
        # assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids


def bracket_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    amount=None,
    lower_bound=None,
    bound_width=None,
):

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    
    for _ in range(max_n_training_examples):
        
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        ctf_label = None
        ctf_label_str = None
        if base_amount_sample <= source_upper_bound_sample and base_amount_sample >= source_lower_bound_sample:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            ctf_label_str = "Yes"
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
            ctf_label_str = "No"
            
        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = prompt_template[mode] % (base_instruction, base_amount_str)
        source_alpaca_prompt = prompt_template[mode] % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
        # assert len(base_input_ids) == 82
        # assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids



import json
import random
from typing import Any, Dict, List, Optional

def create_price_game_acdc_dataset(
    n_prompts: int = 1000,
    output_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a dataset of price-tagging game prompts in the ACDC JSON format.

    Each item has:
      - "clean": A prompt whose "correct" label is either "Yes" or "No".
      - "corrupt": The same style of prompt but guaranteed to yield the *opposite* label.
      - "answers": The "correct" label for the clean prompt, e.g. [" Yes"] or [" No"].
      - "wrong_answers": The label that the corrupt prompt would produce, e.g. [" No"] or [" Yes"].

    The final dictionary has:
      {
          "seq_labels": [],
          "word_idxs": {},
          "prompts": [...]
      }

    Parameters
    ----------
    n_prompts : int
        Number of (clean, corrupt) prompt pairs to generate.
    output_json_path : str, optional
        If provided, the dataset is also saved to this JSON file.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys:
          - "seq_labels": (empty list by default)
          - "word_idxs": (empty dict by default)
          - "prompts": list of prompt-pair dictionaries.
    """

    def random_bounds():
        """
        Returns (lower_bound, upper_bound) as floats, ensuring lower < upper.
        For example, choose random floats in [0.01, 10.0).
        """
        lb = round(random.uniform(0.01, 8.0), 2)
        # Keep a random margin to ensure the upper is > lb
        margin = round(random.uniform(0.5, 2.0), 2)
        ub = round(lb + margin, 2)
        return lb, ub

    def build_prompt_text(lb: float, ub: float, amount: float) -> str:
        """
        Creates a single price check instruction + the cost as input,
        e.g., "Please say yes only if it costs between 2.50 and 4.20 dollars, otherwise no.
               3.75 dollars"
        """
        instruction = (
            f"Please say yes only if it costs between "
            f"{lb:.2f} and {ub:.2f} dollars, otherwise no.\n"
            f"{amount:.2f} dollars"
        )
        
        return instruction

    def is_in_range(amount: float, lb: float, ub: float) -> bool:
        return (amount >= lb) and (amount <= ub)

    # This is the main container for the dataset
    dataset: Dict[str, Any] = {
        "seq_labels": [],  # optional, can be used for labeling tokens
        "word_idxs": {},   # optional, can be used for official circuit functions
        "prompts": [],
    }

    for _ in range(n_prompts):
        # Decide if the "clean" scenario should produce "Yes" or "No"
        # We'll do a 50/50 split.
        want_clean_yes = random.choice([True, False])

        # 1) Generate a random scenario for the "clean" prompt
        lb, ub = random_bounds()

        # If we want the label to be "Yes", pick an amount in [lb, ub].
        # If "No", pick an amount outside [lb, ub] in [0.01, 10.0).
        if want_clean_yes:
            # Force an amount in-range
            amount_clean = round(random.uniform(lb, ub), 2)
        else:
            # Force an amount out-of-range
            # We'll pick either below lb or above ub
            if random.random() < 0.5 and lb > 0.3:
                # pick something less than lb
                amount_clean = round(random.uniform(0.01, lb - 0.01), 2)
            else:
                # pick something greater than ub
                # clamp upper a bit so we don't exceed 10
                amount_clean = round(random.uniform(ub + 0.01, 9.99), 2)

        # 2) Build the clean prompt text
        clean_prompt = build_prompt_text(lb, ub, amount_clean)

        # 3) Generate a scenario for the "corrupt" prompt to produce the opposite label
        # We can keep the same amount and change bounds so that it flips from "Yes" to "No", or vice versa.
        # If want_clean_yes, we want the "corrupt" prompt to say "No".
        # If want_clean_yes = True => the amount is in [lb, ub].
        # So for the corrupt scenario, pick bounds that exclude that amount.
        if want_clean_yes:
            # "corrupt" => "No"
            # We can shift the upper bound below the chosen amount or shift the lower bound above it
            # We'll do it randomly
            if random.random() < 0.5:
                # shift upper below amount
                ub_corrupt = round(amount_clean - 0.01, 2)
                lb_corrupt = round(ub_corrupt - 1.0, 2)
                if lb_corrupt < 0.01:
                    lb_corrupt = 0.01
                if ub_corrupt < 0.01:
                    ub_corrupt = 0.02
            else:
                # shift lower above amount
                lb_corrupt = round(amount_clean + 0.01, 2)
                ub_corrupt = round(lb_corrupt + 1.0, 2)
                if ub_corrupt > 9.99:
                    ub_corrupt = 9.99
            # build prompt
            corrupt_prompt = build_prompt_text(lb_corrupt, ub_corrupt, amount_clean)
            # clean label => "Yes", corrupt label => "No"
            answers = [" Yes"]
            wrong_answers = [" No"]

        else:
            # "clean" => "No", so "corrupt" => "Yes"
            # The amount is out-of-range for (lb, ub).
            # For corrupt, pick bounds that include that same amount.
            # That means set lb_corrupt <= amount_clean <= ub_corrupt.
            # We'll pick an expansion around the amount:
            half_width = round(random.uniform(0.1, 2.0), 2)
            lb_corrupt = round(amount_clean - half_width, 2)
            if lb_corrupt < 0.01:
                lb_corrupt = 0.01
            ub_corrupt = round(amount_clean + half_width, 2)
            if ub_corrupt > 9.99:
                ub_corrupt = 9.99

            corrupt_prompt = build_prompt_text(lb_corrupt, ub_corrupt, amount_clean)
            # clean label => "No", corrupt label => "Yes"
            answers = [" No"]
            wrong_answers = [" Yes"]

        dataset["prompts"].append(
            {
                "clean": clean_prompt,
                "corrupt": corrupt_prompt,
                "answers": answers,
                "wrong_answers": wrong_answers,
            }
        )

    # Optionally write out the JSON
    if output_json_path is not None:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    return dataset