import torch
import os
import json
from transformer_lens import HookedTransformer
from transformer_lens.patching import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
base_path = f"datasets/probing/{model_name}/modus_tollens"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HookedTransformer.from_pretrained(model_name).to(device).eval()

tokenizer : AutoTokenizer = model.tokenizer

answer_tokens = ["Yes", "yes", "No", "no"]
answer_token_ids = tokenizer(answer_tokens, add_special_tokens=False)["input_ids"]
answer_token_ids = [item for sublist in answer_token_ids for item in sublist]


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

clean_prompts, corrupt_prompts = load_prompts(base_dataset_path)

clean_tokens = tokenizer(clean_prompts, return_tensors="pt", padding=True, truncation=True).to(device).input_ids
corrupt_tokens = tokenizer(corrupt_prompts, return_tensors="pt", padding=True, truncation=True).to(device).input_ids

assert clean_tokens.shape == corrupt_tokens.shape, f"Aligned tokenized shapes mismatch: {clean_tokens.shape} vs {corrupt_tokens.shape}"

clean_cache = model.run_with_cache(clean_tokens)
corrupt_cache = model.run_with_cache(corrupt_tokens)

clean_cache = model.run_with_cache(clean_tokens)
corrupt_cache = model.run_with_cache(corrupt_tokens)

def patching_metric(logits):
    # metric: probability mass on answer tokens
    probs = logits.softmax(dim=-1)[0, -1]
    score = probs[answer_token_ids].sum()
    return score

results = {}

print("Running activation patching analysis on clean vs corrupted sample")

# Patch for attention key, query, value, output, pattern
print("Patching attention keys (k) all pos/head")
k_patches = get_act_patch_attn_head_k_all_pos(model, corrupt_tokens, clean_cache, patching_metric)
print("Patching attention queries (q) all pos/head")
q_patches = get_act_patch_attn_head_q_all_pos(model, corrupt_tokens, clean_cache, patching_metric)
print("Patching attention values (v) all pos/head")
v_patches = get_act_patch_attn_head_v_all_pos(model, corrupt_tokens, clean_cache, patching_metric)
print("Patching attention outputs (z) all pos/head")
z_patches = get_act_patch_attn_head_out_all_pos(model, corrupt_tokens, clean_cache, patching_metric)
print("Patching MLP outputs (mlp_out) all pos/layer")
mlp_patches = get_act_patch_mlp_out(model, corrupt_tokens, clean_cache, patching_metric)

results["k"] = k_patches.cpu().numpy().tolist()
results["q"] = q_patches.cpu().numpy().tolist()
results["v"] = v_patches.cpu().numpy().tolist()
results["z"] = z_patches.cpu().numpy().tolist()
results["mlp_out"] = mlp_patches.cpu().numpy().tolist()

output_path = os.path.join(base_path, "activation_patch_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved activation patching results to {output_path}")

if __name__ == '__main__':
    pass