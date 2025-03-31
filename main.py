import json
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.graph_utils import patchable_model

import os
from pathlib import Path

import torch

from utils.dataset import create_dataset
from config import ProjectConfig, load_yaml_config

config:ProjectConfig = load_yaml_config("conf.yaml")

# seed all
torch.manual_seed(config.seed)

device = config.model.device

model = load_tl_model(config.model.model_name, device=device)

path = Path(f"{config.data_preprocessing.dataset_dir}/{config.model.model_name}/dataset.json")
# check if the dataset exists
assert os.path.exists(path), f"Dataset not found at {path}, run benchmark.py to create it with the wanted model"

# load json to see dimension of the dataset
num_samples = 0
with open(path) as f:
    data = json.load(f)
    num_samples = data["dataset_size"]
train_size = int(num_samples * config.model.train_percent)
test_size = num_samples - train_size
print(f"Dataset size: {num_samples}")
print(f"Train size: {train_size}")
print(f"Test size: {test_size}")

train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    batch_size=config.model.batch_size,
    train_test_size=[train_size, test_size],
    tail_divergence=True,
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=False,
    kv_caches=(train_loader.kv_cache, test_loader.kv_cache),
    device=device,
)

attribution_scores: PruneScores = acdc_prune_scores(
    model=model,
    dataloader=train_loader,
    official_edges=None,
    tao_exps=[-5,-3,-2],
    tao_bases=[1,5,9]
)

# save to file attribution scores (PruneScores = Dict[str, t.Tensor])
attribution_scores_path = Path(f"{config.out_dir}/{config.model.model_name}/attribution_scores.json")
attribution_scores_path.parent.mkdir(parents=True, exist_ok=True)
with open(attribution_scores_path, "w") as f:
    json.dump(
        {k: v.tolist() for k, v in attribution_scores.items()},
        f,
        indent=4,
    )

"""V
# load attribution scores from file
with open(attribution_scores_path) as f:
    attribution_scores = json.load(f)
    # convert to tensor
    for k, v in attribution_scores.items():
        attribution_scores[k] = torch.tensor(v, device=device, dtype=torch.float32)
"""

image_path = Path(f"{config.out_dir}/{config.model.model_name}/attribution_scores.png")
fig = draw_seq_graph(model, attribution_scores, config.model.threshold, layer_spacing=True, orientation="v")

fig.write_image("images/fig1.png")