import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------- 1. Load Model ---------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------------- 2. Hooks to capture attention & MLP states ---------------
attention_records = {}
mlp_records = {}

def get_attention_hook(layer_num):
    """
    Returns a function that saves the attention weights for a given layer_num.
    """
    def hook(module, input, output):
        attention_matrix = output[1] # [batch_size, num_heads, seq_len, seq_len]
        # Save the attention matrix for this layer
        attention_records[layer_num] = attention_matrix.detach().cpu()
    return hook

def get_mlp_hook(layer_num, position="after"):
    """
    Returns a function that saves hidden states before or after the MLP in a given layer.
    position = "before" or "after".
    """
    def hook(module, input, output=None):
        if position == "before":
            # input is a tuple; input[0] is the tensor we want.
            mlp_records[(layer_num, "before")] = input[0].detach().cpu()
        elif position == "after":
            assert output is not None
            mlp_records[(layer_num, "after")] = output.detach().cpu()
    return hook

# Register hooks for each transformer layer
for layer_idx, layer_module in enumerate(model.model.layers):
    layer_module.self_attn.register_forward_hook(get_attention_hook(layer_idx))
    layer_module.mlp.register_forward_pre_hook(get_mlp_hook(layer_idx, "before"))
    layer_module.mlp.register_forward_hook(get_mlp_hook(layer_idx, "after"))

# --------------- 3. Run the model on a prompt ---------------
prompt = (
    "<|im_start|>system\n"
    "Based on the given context, you have to respond with yes or no.<|im_end|>\n"
    "<|im_start|>user\n"
    "If Henry finished his project on time, then he will be eligible for a bonus. "
    "If he won't be eligible for a bonus, does this mean that henry didn't finish his project on time?"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(
        input_ids, 
        attention_mask=attention_mask, 
        output_attentions=True  # Ensure attentions are returned in forward
    )

# --------------- 4. Visualization & Analysis ---------------

# 4a) Print the tokens for reference
decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  
# Alternatively, batch_decode([input_ids[0]]) returns merges, but convert_ids_to_tokens
# shows the raw tokens (some might be subwords with Byte-Pair Encoding).

print("Tokenized Input:")
for i, tok in enumerate(decoded_tokens):
    print(f"{i:2d}: {tok}")
print()

print("=== Attention Records ===")
# 4b) Display the attention patterns for each layer, each head
for layer_idx, attn_probs in tqdm(attention_records.items()):
    # attn_probs: [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attn_probs.shape
    if batch_size != 1:
        continue  # ignoring multi-batch
    
    tqdm.write(f"=== Layer {layer_idx} Attention Heads ===")
    for head_idx in tqdm(range(num_heads)):
        head_attn = attn_probs[0, head_idx].numpy()  # shape [seq_len, seq_len]
        
        # Plot heatmap
        # plt.figure(figsize=(6, 5))
        # plt.imshow(head_attn, aspect='auto', origin='lower', cmap='viridis')
        # plt.colorbar()
        # plt.title(f"Layer {layer_idx} - Head {head_idx}")
        # plt.xlabel("Key (token index)")
        # plt.ylabel("Query (token index)")
        # # Label ticks with actual tokens
        # plt.xticks(range(seq_len), decoded_tokens, rotation=90)
        # plt.yticks(range(seq_len), decoded_tokens)
        # plt.tight_layout()
        # plt.savefig(f"images/layer_{layer_idx}_head_{head_idx}.png")
        # plt.close()
        
        # Optionally, also print top-attended tokens (textual summary)
        # For each query token i, find the top-n key tokens.
        # top_n = 3
        # print(f"Textual summary for layer {layer_idx}, head {head_idx}:")
        # for q_idx in range(seq_len):
        #     # Sort by attention descending
        #     row = head_attn[q_idx]  # shape [seq_len]
        #     top_indices = row.argsort()[-top_n:][::-1]
        #     q_token = decoded_tokens[q_idx]
        #     top_str = []
        #     for rank, k_idx in enumerate(top_indices, start=1):
        #         key_token = decoded_tokens[k_idx]
        #         score = row[k_idx]
        #         top_str.append(f"{rank}) '{key_token}' ({score:.4f})")
        #     top_str = "; ".join(top_str)
        #     print(f"  Query token {q_idx}='{q_token}' -> {top_str}")
        # print()

# 4c) Analyze MLP changes (unchanged from your original code below)
final_logits_w = model.lm_head.weight.detach().cpu()  # [vocab_size, hidden_dim]

def hidden_to_logits(hidden_state):
    # hidden_state shape: [batch_size, seq_len, hidden_dim]
    return torch.matmul(hidden_state, final_logits_w.T)

interested_tokens = ["yes", "Yes", "no", "No"]
interested_token_ids = [
    tokenizer.encode(tkn, add_special_tokens=False)[0] for tkn in interested_tokens
]
interested_token_ids = torch.tensor(interested_token_ids)  # [num_interested_tokens]

print("=== MLP Changes in each layer ===")
layer_infos = []
layer_info = ""  # For printing layer information
for layer_idx in range(len(model.model.layers)):
    before_key = (layer_idx, "before")
    after_key  = (layer_idx, "after")
    
    if before_key not in mlp_records or after_key not in mlp_records:
        continue
    
    before_mlp = mlp_records[before_key]  # [batch_size, seq_len, hidden_dim]
    after_mlp  = mlp_records[after_key]   # [batch_size, seq_len, hidden_dim]
    
    diff_norm = torch.norm(after_mlp - before_mlp, dim=-1).mean().item()
    layer_info += f"Layer {layer_idx} MLP hidden-state difference (avg L2): {diff_norm:.4f}\n"
    
    last_token_idx = before_mlp.size(1) - 1
    
    before_logits = hidden_to_logits(before_mlp[:, last_token_idx:last_token_idx+1, :])
    after_logits  = hidden_to_logits(after_mlp[:, last_token_idx:last_token_idx+1, :])
    
    logits_diff = (after_logits - before_logits).squeeze(0).squeeze(0)  # [vocab_size]
    top_diff_values, top_diff_indices = torch.topk(logits_diff.abs(), 5)
    
    top_tokens_before = before_logits.topk(5, dim=-1).indices.squeeze(0).squeeze(0)
    top_tokens_after  = after_logits.topk(5, dim=-1).indices.squeeze(0).squeeze(0)
    
    layer_info += f"\tTop tokens before layer {layer_idx}:\n"
    for idx_ in top_tokens_before:
        token = tokenizer.decode([idx_.item()])
        layer_info += f"\t  {token!r:12s}\n"
    layer_info += f"\n\tTop tokens after layer {layer_idx}:\n"
    for idx_ in top_tokens_after:
        token = tokenizer.decode([idx_.item()])
        layer_info += f"\t  {token!r:12s}\n"
    
    layer_info += f"\n\tTokens with largest absolute shift in logit (for last token):\n"
    for value, idx_ in zip(top_diff_values, top_diff_indices):
        changed_token = tokenizer.decode([idx_.item()])
        layer_info += f"\t  {changed_token!r:12s}: {logits_diff[idx_].item():.4f}\n"
    
    interested_logits_before = before_logits[0, 0, interested_token_ids]
    interested_logits_after  = after_logits[0, 0, interested_token_ids]
    layer_info += f"\tLogits for interested tokens:\n"
    for token, logit_b, logit_a in zip(interested_tokens, interested_logits_before, interested_logits_after):
        layer_info += f"\t- {token!r:12s}: before={logit_b.item():.4f}, after={logit_a.item():.4f}\n"
    
    
    print(layer_info)
    print("===" * 20)
    layer_infos.append(layer_info)
    layer_info = ""  # Reset for next layer

print("Analysis complete!")


# save as json
import json

# --------------- 5. Build and Save JSON Analysis Data ---------------
analysis_json = {
    "name": "Model Analysis",
    "children": []
}

# Iterate over layers to add attention heads and MLP analysis
num_layers = len(model.model.layers)
for layer_idx in range(num_layers):
    layer_node = {
        "name": f"Layer {layer_idx}",
        "children": []
    }
    
    # Add attention head information if available
    if layer_idx in attention_records:
        attn_probs = attention_records[layer_idx]
        batch_size, num_heads, seq_len, _ = attn_probs.shape
        for head_idx in range(num_heads):
            head_node = {
                "name": f"Attention Head {head_idx}",
                "summary": f"Layer {layer_idx} head {head_idx} captures token interactions.",
                "detail": f"See the heatmap image saved at /images/layer_{layer_idx}_head_{head_idx}.png for detailed attention patterns.",
                "matrix": attn_probs[0, head_idx].numpy().tolist(),  # Convert to list for JSON serialization
            }
            layer_node["children"].append(head_node)
    
    # Add MLP analysis if both before and after states were recorded
    before_key = (layer_idx, "before")
    after_key = (layer_idx, "after")
    if before_key in mlp_records and after_key in mlp_records:
        before_mlp = mlp_records[before_key]
        after_mlp = mlp_records[after_key]
        diff_norm = torch.norm(after_mlp - before_mlp, dim=-1).mean().item()
        
        mlp_node = {
            "name": "MLP",
            "summary": f"Avg L2 diff: {diff_norm:.4f}",
            "detail": layer_infos[layer_idx],
        }
        layer_node["children"].append(mlp_node)
    
    analysis_json["children"].append(layer_node)

# Save the analysis JSON to a file
with open("data.json", "w") as f:
    json.dump(analysis_json, f, indent=4)

print("JSON analysis saved to data.json")
