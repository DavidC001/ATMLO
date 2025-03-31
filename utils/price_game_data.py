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
        "instruction":  f"You need to take on this problem where you need to tell me wether the ammount falls between the given interval, respond with either yes or no.",
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

if __name__ == "__main__":
    random.seed(42)
    print("Creating train DataSet")
    create_price_game_acdc_dataset(150, "datasets/price_game/train.json")
    print("Creating eval DataSet")
    create_price_game_acdc_dataset(20, "datasets/price_game/eval.json")