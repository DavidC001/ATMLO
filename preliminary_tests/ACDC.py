from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.types import AblationType, PruneScores
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model
from auto_circuit.visualize import draw_seq_graph

from tutorial_price_tagging_utils import create_price_game_acdc_dataset
import tutorial_price_tagging_utils as ptu

from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import AutoTokenizer

import os
from pathlib import Path

device = "cuda"

num_samples = 100
model_name = "meta-llama/Llama-3.2-3B-Instruct"
ptu.mode = "llama_3"
# model_name = "gpt2"
# ptu.mode = "alpaca"
dataset_json = "./datasets/price_tagging.json"

model = load_tl_model(model_name, device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the dataset
create_price_game_acdc_dataset(
    num_samples,
    dataset_json
)

# Load the dataset
path = Path(dataset_json)
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    batch_size=1,
    return_seq_length=True,
    tail_divergence=True,
)


# model = patchable_model(
#     model,
#     factorized=True,
#     slice_output="last_seq",
#     seq_len=test_loader.seq_len,
#     separate_qkv=False,
#     device=device,
# )

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=False,
    kv_caches=(train_loader.kv_cache, test_loader.kv_cache),
    device=device,
)


attribution_scores: PruneScores = mask_gradient_prune_scores(
    model=model,
    dataloader=train_loader,
    official_edges=None,
    grad_function="logit",
    answer_function="avg_diff",
    mask_val=0.0,
)

# attribution_scores: PruneScores = acdc_prune_scores(
#     model=model,
#     dataloader=train_loader,
#     official_edges=None,
# )



fig = draw_seq_graph(
    model, attribution_scores, 3.5, layer_spacing=True, orientation="v"
)

# fig = draw_seq_graph(
#     model, attribution_scores, 3.5, seq_labels=train_loader.seq_labels
# )

fig.write_image("images/fig1.png")