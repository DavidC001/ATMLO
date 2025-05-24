import torch
import os
import json
import time
from transformer_lens import HookedTransformer
from transformer_lens.patching import *
import numpy as np
from tqdm import tqdm

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
base_path = f"results/{model_name}/patching/modus_tollens"
os.makedirs(os.path.dirname(base_path), exist_ok=True)
os.makedirs(os.path.dirname(base_path), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model : HookedTransformer = HookedTransformer.from_pretrained(model_name).to(device).eval()
print(f"Loaded model with {model.cfg.n_heads} attention heads")

tokenizer = model.tokenizer

pos_answer_tokens = ["Yes", "yes"]
pos_answer_token_ids = tokenizer(pos_answer_tokens, add_special_tokens=False)["input_ids"]
pos_answer_token_ids = [item for sublist in pos_answer_token_ids for item in sublist]

neg_answer_tokens = ["No", "no"]
neg_answer_token_ids = tokenizer(neg_answer_tokens, add_special_tokens=False)["input_ids"]
neg_answer_token_ids = [item for sublist in neg_answer_token_ids for item in sublist]

base_dataset_path = f"datasets/circ_disc/{model_name}/modus_tollens/dataset.json"

def load_prompts(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    if 'prompts' not in dataset:
        raise RuntimeError(f"'prompts' key not found in dataset {path}")
    prompts = dataset['prompts']
    cleans = []
    corrupts = []
    for entry in prompts:
        if 'clean' in entry and 'corrupt' in entry:
            cleans.append(entry['clean'])
            corrupts.append(entry['corrupt'])
        else:
            raise RuntimeError(f"Unexpected prompt format in dataset {path}")
    return cleans, corrupts

def patching_metric(logits):
    # metric: probability mass on answer tokens
    probs = logits.softmax(dim=-1)[0, -1]
    score = probs[pos_answer_token_ids].sum() - probs[neg_answer_token_ids].sum()
    return score

def get_activation_patching_results(model, clean_tokens, corrupt_tokens, clean_cache):
    print("Patching attention outputs (z) layer-wise score")
    z_patches_all_pos = get_act_patch_attn_head_out_all_pos(model, corrupt_tokens, clean_cache, patching_metric)
    # Average over positions to get only layer-head scores
    z_patches = z_patches_all_pos # the output does not have a position dimension
    print("Patching MLP outputs (mlp_out) layer-wise score")
    mlp_patches_all_pos = get_act_patch_mlp_out(model, corrupt_tokens, clean_cache, patching_metric)
    # Average over positions to get layer scores
    mlp_patches = mlp_patches_all_pos.mean(dim=1)  # assuming dimension 1 is position
    return z_patches, mlp_patches

def average_results(results_list):
    avg = {}
    keys = results_list[0].keys()
    for k in keys:
        avg[k] = torch.mean(torch.stack([torch.tensor(r[k]) for r in results_list]), dim=0)
    return avg

def main(num_runs=5, max_duration_secs=None):
    cleans, corrupts = load_prompts(base_dataset_path)
    results_list = []
    start_time = time.time()
    runs_done = 0
    for i in tqdm(range(len(cleans))):
        if num_runs and runs_done >= num_runs:
            break
        if max_duration_secs is not None and (time.time() - start_time) > max_duration_secs:
            print(f"Max duration {max_duration_secs}s exceeded after {runs_done} runs, stopping")
            break
        clean_prompt = [cleans[i]]
        corrupt_prompt = [corrupts[i]]
        clean_tokens = tokenizer(clean_prompt, return_tensors="pt", padding=True, truncation=True).to(device).input_ids
        corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt", padding=True, truncation=True).to(device).input_ids
        
        _, clean_cache = model.run_with_cache(clean_tokens)
        z_patches, mlp_patches = get_activation_patching_results(model, clean_tokens, corrupt_tokens, clean_cache)
        results_list.append({"z": z_patches.cpu().numpy(), "mlp_out": mlp_patches.cpu().numpy()})
        runs_done += 1

    if runs_done == 0:
        raise RuntimeError("No runs completed, no results to average")
    avg_results = average_results(results_list)
    avg_results_out = {k: v.numpy().tolist() for k, v in avg_results.items()}

    output_path = os.path.join(base_path, f"average_activation_patch_results.json")
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(avg_results_out, f, indent=4)
    print(f"Saved averaged activation patching results to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Average activation patching across multiple runs on different prompts.")
    parser.add_argument('--num_runs', type=int, default=0, help='Number of runs to average')
    parser.add_argument('--max_duration_secs', type=int, default=None, help='Max duration to run in seconds')
    args = parser.parse_args()
    main(num_runs=args.num_runs, max_duration_secs=args.max_duration_secs)