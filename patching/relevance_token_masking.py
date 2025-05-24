import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_token_relevance(model, tokenizer, prompt_text, target_token_ids, device):
    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Original output logits for entire prompt
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    
    # We will measure the difference in probability/logits of target tokens at the last position
    # relative to original logits without masking

    def get_target_logits(logits, target_ids):
        # Take last token logits
        last_logits = logits[0, -1]
        # Sum logits for target tokens
        return last_logits[target_ids].max().item()

    original_score = get_target_logits(logits, target_token_ids)

    token_influence = []
    seq_len = input_ids.shape[1]

    for i in range(seq_len):
        masked_ids = input_ids.clone()
        # Replace token i with unknown token id to mask
        # If tokenizer has mask token, use that, else use 0 (pad token)
        mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 0
        masked_ids[0, i] = mask_token_id
        with torch.no_grad():
            masked_outputs = model(masked_ids, attention_mask=attention_mask)
            masked_logits = masked_outputs.logits
        masked_score = get_target_logits(masked_logits, target_token_ids)
        influence = original_score - masked_score
        token_influence.append((i, influence))

    token_influence.sort(key=lambda x: abs(x[1]), reverse=True)
    return token_influence, input_ids[0].tolist(), original_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # You may want to choose your model here consistent with your dataset
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Directory where datasets/circ_disc contains subfolders by model_name and dataset_key
    circ_disc_dir = "datasets/circ_disc"

    # Collect token influence rankings for clean, corrupt, overall
    aggregated_rankings = {"clean": {}, "corrupt": {}, "overall": {}}
    totals = {"clean": 0, "corrupt": 0, "overall": 0}

    model_path = os.path.join(circ_disc_dir, model_name)
    
    interested_datasets = ["modus_tollens"]  # Specify datasets of interest
    
    for dataset_key in os.listdir(model_path):
        if dataset_key not in interested_datasets:
            continue
        dataset_path = os.path.join(model_path, dataset_key, "dataset.json")
        if not os.path.isfile(dataset_path):
            continue
        with open(dataset_path, "r") as f:
            data = json.load(f)
        prompts = data.get("prompts", [])
        print(f"Processing {model_name}/{dataset_key} with {len(prompts)} prompt pairs")
        
        for pair in tqdm(prompts):
            for typ in ["clean", "corrupt"]:
                text = pair.get(typ, None)
                if not text:
                    continue
                # Tokenize text
                inputs = tokenizer(text, return_tensors='pt')
                input_ids = inputs['input_ids'][0]
                # Define target tokens as 'Yes', 'yes' for clean correct, 'No', 'no' for corrupt?
                # We'll find index tokens for yes/no
                yes_tokens = tokenizer(["Yes", "yes"], add_special_tokens=False)["input_ids"]
                yes_tokens_flat = [item for sublist in yes_tokens for item in sublist]
                no_tokens = tokenizer(["No", "no"], add_special_tokens=False)["input_ids"]
                no_tokens_flat = [item for sublist in no_tokens for item in sublist]
                
                if typ == "clean":
                    target_token_ids = yes_tokens_flat
                else:
                    target_token_ids = no_tokens_flat
                
                influences, tokens, orig_score = compute_token_relevance(model, tokenizer, text, target_token_ids, device)
                totals[typ] += 1
                totals["overall"] += 1
                # Aggregate influence scores by token id
                # Using absolute influence scores to rank importance
                for idx, influence in influences:
                    token_id = tokens[idx]
                    if token_id not in aggregated_rankings[typ]:
                        aggregated_rankings[typ][token_id] = 0.0
                    aggregated_rankings[typ][token_id] += abs(influence)
                    if token_id not in aggregated_rankings["overall"]:
                        aggregated_rankings["overall"][token_id] = 0.0
                    aggregated_rankings["overall"][token_id] += abs(influence)

    # Normalize rankings by number of samples
    for typ in aggregated_rankings:
        count = totals[typ] if totals[typ] > 0 else 1
        for tid in aggregated_rankings[typ]:
            aggregated_rankings[typ][tid] /= count

    for typ in ["clean", "corrupt", "overall"]:
        print(f"Top tokens influencing {typ}:")
        sorted_tokens = sorted(aggregated_rankings[typ].items(), key=lambda x: x[1], reverse=True)[:20]
        for tid, score in sorted_tokens:
            token_str = tokenizer.decode([tid])
            print(f"Token: {token_str} (id {tid}) Influence score: {score:.4f}")
        print("-----------------------")

if __name__ == '__main__':
    main()