import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

import sys
sys.path.append(".")

from utils.dataloader import get_dataloader

exp_name = "price_game"
interest_tokens = [
    "Yes",
    "No",
]

data_paths = [
    "datasets/price_game/eval.json",
    "datasets/price_game/train.json",
]

# --------------- 1. Load Model ---------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

lm_head = model.lm_head.weight.T

out_path = f"datasets/probing/{model_name}/{exp_name}/"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

interest_tokens_ids = tokenizer(interest_tokens, add_special_tokens=False)["input_ids"]
interest_tokens_ids = [item for sublist in interest_tokens_ids for item in sublist]

# --------------- 2. Hooks to capture attention & MLP states ---------------
text = []
attention_records = {}
mlp_records = {}
top_logits = {}

def get_attention_hook(layer_num):
    """
    Returns a function that saves the attention weights for a given layer_num.
    """
    def hook(module, input, output):
        # output is a tuple; output[0] is the attention output, output[1] is the attention matrix.
        # [batch_size, seq_len, hidden_dim] for the attention output
        # [batch_size, num_heads, seq_len, seq_len] for the attention matrix
        
        # Save the attention matrix for this layer
        attn_out = output[0][:,-1]
        attn_mat = output[1]
        
        # apply layer normalization to the attention output
        attn_out_ln = model.model.layers[layer_num].post_attention_layernorm(
            attn_out
        )
        
        att_out_logits = attn_out_ln @ lm_head
        
        top_tokens = torch.topk(att_out_logits, k=5, dim=-1)
        top_tokens_ids = top_tokens.indices
        top_tokens_values = top_tokens.values
        
        # extract logits for interest tokens
        logits_interest_tokens = att_out_logits[:, interest_tokens_ids]
        
        # detach and move to CPU
        attn_out = attn_out.detach().cpu()
        attn_out_ln = attn_out_ln.detach().cpu()
        attn_mat = attn_mat.detach().cpu()
        top_tokens_values = top_tokens_values.detach().cpu()
        top_tokens_ids = top_tokens_ids.detach().cpu()
        logits_interest_tokens = logits_interest_tokens.detach().cpu()
        
        if (layer_num,"out") not in attention_records:
            attention_records[(layer_num,"out")] = list(attn_out)
            attention_records[(layer_num,"out_ln")] = list(attn_out_ln)
            attention_records[(layer_num,"attn_mat")] = list(attn_mat)
            attention_records[(layer_num,"top_logits")] = list(top_tokens_values)
            attention_records[(layer_num,"top_logits_ids")] = list(top_tokens_ids)
            attention_records[(layer_num,"interest_tokens_logits")] = list(logits_interest_tokens)
        else:
            attention_records[(layer_num,"out")].extend(list(attn_out))
            attention_records[(layer_num,"out_ln")].extend(list(attn_out_ln))
            attention_records[(layer_num,"attn_mat")].extend(list(attn_mat))
            attention_records[(layer_num,"top_logits")].extend(list(top_tokens_values))
            attention_records[(layer_num,"top_logits_ids")].extend(list(top_tokens_ids))
            attention_records[(layer_num,"interest_tokens_logits")].extend(list(logits_interest_tokens))
        
    return hook

def get_mlp_hook(layer_num):
    """
    Returns a function that saves hidden states before or after the MLP in a given layer.
    position = "before" or "after".
    """
    def hook(module, input, output):
        before = input[0][:,-1]
        after = output[:,-1]
        
        # apply layer normalization to the MLP output
        if layer_idx < len(model.model.layers) - 1:
            after_ln = model.model.layers[layer_num+1].input_layernorm(
                after
            )
        else:
            after_ln = model.model.norm(
                after
            )
            
        before_logit = before @ lm_head
        after_logit = after_ln @ lm_head
        
        # get top tokens before
        top_tokens_b = torch.topk(before_logit, k=5, dim=-1)
        top_tokens_ids_b = top_tokens_b.indices
        top_tokens_values_b = top_tokens_b.values
        # get top tokens after
        top_tokens_a = torch.topk(after_logit, k=5, dim=-1)
        top_tokens_ids_a = top_tokens_a.indices
        top_tokens_values_a = top_tokens_a.values
        
        # extract logits for interest tokens
        before_logit = before_logit[:, interest_tokens_ids]
        after_logit = after_logit[:, interest_tokens_ids]
        
        # detach and move to CPU
        before = before.detach().cpu()
        after = after.detach().cpu()
        after_ln = after_ln.detach().cpu()
        before_logit = before_logit.detach().cpu()
        after_logit = after_logit.detach().cpu()
        top_tokens_values_b = top_tokens_values_b.detach().cpu()
        top_tokens_ids_b = top_tokens_ids_b.detach().cpu()
        top_tokens_values_a = top_tokens_values_a.detach().cpu()
        top_tokens_ids_a = top_tokens_ids_a.detach().cpu()
        
            
        if (layer_num,"after") not in mlp_records:
            mlp_records[(layer_num,"before")] = list(before)
            mlp_records[(layer_num,"after")] = list(after)
            mlp_records[(layer_num,"after_ln")] = list(after_ln)
            mlp_records[(layer_num,"top_logits_before")] = list(top_tokens_values_b)
            mlp_records[(layer_num,"top_logits_before_ids")] = list(top_tokens_ids_b)
            mlp_records[(layer_num,"top_logits_after")] = list(top_tokens_values_a)
            mlp_records[(layer_num,"top_logits_after_ids")] = list(top_tokens_ids_a)
            mlp_records[(layer_num,"interest_tokens_logits_before")] = list(before_logit)
            mlp_records[(layer_num,"interest_tokens_logits_after")] = list(after_logit)
        else:
            mlp_records[(layer_num,"before")].extend(list(before))
            mlp_records[(layer_num,"after")].extend(list(after))
            mlp_records[(layer_num,"after_ln")].extend(list(after_ln))
            mlp_records[(layer_num,"top_logits_before")].extend(list(top_tokens_values_b))
            mlp_records[(layer_num,"top_logits_before_ids")].extend(list(top_tokens_ids_b))
            mlp_records[(layer_num,"top_logits_after")].extend(list(top_tokens_values_a))
            mlp_records[(layer_num,"top_logits_after_ids")].extend(list(top_tokens_ids_a))
            mlp_records[(layer_num,"interest_tokens_logits_before")].extend(list(before_logit))
            mlp_records[(layer_num,"interest_tokens_logits_after")].extend(list(after_logit))
    return hook

# Register hooks for each transformer layer
for layer_idx, layer_module in enumerate(model.model.layers):
    layer_module.self_attn.register_forward_hook(get_attention_hook(layer_idx))
    layer_module.mlp.register_forward_hook(get_mlp_hook(layer_idx))



# --------------- 3. Run the model on a prompt ---------------
data = []
instruction = ""
for data_path in data_paths:
    with open(data_path, "r") as f:
        json_data = json.load(f)
        data += json_data["prompts"]
        instruction = json_data["instruction"]

processed_samples = 0

for sample in tqdm(data):
    
    samples = [
        sample["clean"],
        sample["corrupt"]
    ]
    
    for qa in samples:
        
        chat = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": qa},
        ]
        
        sample_text = instruction + "\n" + qa
        
        inputs = tokenizer.apply_chat_template(chat, add_special_tokens=True, return_tensors="pt")
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(
                inputs,
                attention_mask=torch.ones(inputs.shape, device=device),
                output_attentions=True,
            )
            
            # get all model outputs for this sample
            sample_outputs = {}
            for key, value in attention_records.items():
                sample_outputs[f"attention_layer{key[0]}_{key[1]}"] = value[0].numpy().tolist()
            for key, value in mlp_records.items():
                sample_outputs[f"MLP_layer{key[0]}_{key[1]}"] = value[0].numpy().tolist()
            sample_outputs["text"] = sample
            
            if (processed_samples) % 2 == 0:
                type = "clean"
            else:
                type = "corrupt"

            json_path = os.path.join(out_path, type, f"{processed_samples//2}.json")
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            # print(f"Saving {json_path}")
            with open(json_path, "w") as f:
                json.dump(sample_outputs, f, indent=4)
            
            processed_samples += 1
            # clear the records for the next batch
            attention_records.clear()
            mlp_records.clear()
    
    # free up memory
    torch.cuda.empty_cache()

print("MODEL FORWARD COMPLETED")
