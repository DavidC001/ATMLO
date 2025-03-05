from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
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
model_name = "meta-llama/Llama-3.2-1B-Instruct"
ptu.mode = "llama_3"
dataset_json = "./datasets/price_tagging.json"

model = load_tl_model(model_name, device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the dataset
if not os.path.exists(dataset_json):
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
    prepend_bos=True,
    batch_size=2,
    train_test_size=(80, 20),
    return_seq_length=True,
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    seq_len=test_loader.seq_len,
    separate_qkv=False,
    device=device,
)

# attribution_scores: PruneScores = mask_gradient_prune_scores(
#     model=model,
#     dataloader=train_loader,
#     official_edges=None,
#     grad_function="logit",
#     answer_function="avg_diff",
#     mask_val=0.0,
# )

attribution_scores: PruneScores = acdc_prune_scores(
    model=model,
    dataloader=train_loader,
    official_edges=None,
    show_graphs=True,
)


fig = draw_seq_graph(
    model, attribution_scores, 3.5, layer_spacing=True, orientation="v"
)
