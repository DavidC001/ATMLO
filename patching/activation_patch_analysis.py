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

def get_patching_metric(clean_probs):
    """
    Creates a patching metric that measures how activations shift the model's Yes/No decision.
    
    Returns:
        - Positive values: Patching moves the answer toward "Yes" 
        - Negative values: Patching moves the answer toward "No"
        - Zero: No change in the Yes/No probability distribution
    """
    # get the patching metric measuring the difference in probability mass on answer tokens from clean
    clean_pos_prob = clean_probs[0:len(pos_answer_token_ids)].sum()
    clean_neg_prob = clean_probs[len(pos_answer_token_ids):].sum()
    clean_total_answer_prob = clean_pos_prob + clean_neg_prob
    
    # Compute baseline clean score: positive when clean favors "Yes", negative when it favors "No"
    if clean_total_answer_prob > 0:
        clean_score = (clean_pos_prob - clean_neg_prob) / clean_total_answer_prob
    else:
        clean_score = 0.0
    
    def patching_metric(logits):
        probs = torch.softmax(logits[0,-1, pos_answer_token_ids + neg_answer_token_ids], dim=-1)
        
        pos_prob = probs[0:len(pos_answer_token_ids)].sum()
        neg_prob = probs[len(pos_answer_token_ids):].sum()
        # Total probability mass on answer tokens
        total_answer_prob = pos_prob + neg_prob
        
        # Compute patched score
        if total_answer_prob > 0:
            patched_score = (pos_prob - neg_prob) / total_answer_prob
        else:
            patched_score = 0.0
        
        # Return the difference: positive when patching moves toward "Yes", negative when toward "No"
        return patched_score - clean_score
        
    return patching_metric

def get_activation_patching_results(model, corrupt_tokens, clean_cache, clean_probs):
    print("Patching attention outputs (z) layer-wise score")
    z_patches_all_pos = get_act_patch_attn_head_out_by_pos(model, corrupt_tokens, clean_cache, get_patching_metric(clean_probs))
    print("Patching MLP outputs (mlp_out) layer-wise score")
    mlp_patches_all_pos = get_act_patch_mlp_out(model, corrupt_tokens, clean_cache, get_patching_metric(clean_probs))
    return z_patches_all_pos, mlp_patches_all_pos

def main():
    cleans, corrupts = load_prompts(base_dataset_path)
    
    clean_prompt = [cleans[0]]
    corrupt_prompt = [corrupts[0]]
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt", padding=True, truncation=True).to(device).input_ids
    corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt", padding=True, truncation=True).to(device).input_ids
    
    out, clean_cache = model.run_with_cache(clean_tokens)
    out_logits = out[0, -1, :]  # Get the logits for the last token
    out_probs = torch.softmax(out_logits[neg_answer_token_ids+pos_answer_token_ids], dim=-1)
    z_patches, mlp_patches = get_activation_patching_results(model, corrupt_tokens, clean_cache, out_probs)
    
    results = {"z": z_patches.cpu().numpy(), "mlp_out": mlp_patches.cpu().numpy()}

    results_out = {k: v.tolist() for k, v in results.items()}

    # add tokens list
    results_out["tokens"] = {
        "clean": tokenizer.batch_decode(clean_tokens[0]),
        "corrupt": tokenizer.batch_decode(corrupt_tokens[0])
    }

    output_path = os.path.join(base_path, f"activation_patch_results.json")
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_out, f, indent=4)
    print(f"Saved averaged activation patching results to {output_path}")

if __name__ == '__main__':
    main()