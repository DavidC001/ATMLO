import json
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.edge_attribution_patching import edge_attribution_patching_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.graph_utils import patchable_model, train_mask_mode

import os
from pathlib import Path

import torch

import sys
sys.path.append(".")

from config import ProjectConfig, load_yaml_config
from utils.preprocess.preprocess import preprocess_LogicBench

config:ProjectConfig = load_yaml_config("conf.yaml")

# seed all
torch.manual_seed(config.seed)

device = config.circuit_discovery.device

model = load_tl_model(config.circuit_discovery.model_name, device=device)

if config.circuit_discovery.filtered:
    path = Path(f"{config.benchmark.dataset_dir}/circ_disc/{config.circuit_discovery.model_name}/{config.circuit_discovery.dataset}/dataset.json")
    # check if the dataset exists
    assert os.path.exists(path), f"Dataset not found at {path}, run benchmark.py to create it with the wanted model"

    # load json to see dimension of the dataset
    num_samples = 0
    with open(path) as f:
        data = json.load(f)
        num_samples = data["dataset_size"]
elif config.circuit_discovery.dataset != "price_game":
    dataset_files = [
        f"LogicBench/data/LogicBench(Aug)/propositional_logic/{config.circuit_discovery.dataset}/data_instances.json",
        f"LogicBench/data/LogicBench(Eval)/BQA/propositional_logic/{config.circuit_discovery.dataset}/data_instances.json",
    ]
    combined_json = {}
    for i, file in enumerate(dataset_files):
        preprocess_LogicBench(
            model_name=config.circuit_discovery.model_name,
            file=file,
            out=f"{config.benchmark.dataset_dir}/circ_disc/{config.circuit_discovery.model_name}/{config.circuit_discovery.dataset}_unfiltered/dataset{i}.json",
            format="ACDC",
            tokenizer=model.tokenizer,
            alignment=True,
        )
        # combine the two datasets
        with open(f"{config.benchmark.dataset_dir}/circ_disc/{config.circuit_discovery.model_name}/{config.circuit_discovery.dataset}_unfiltered/dataset{i}.json") as f:
            data = json.load(f)
            if i == 0:
                combined_json = data
            else:
                combined_json["prompts"].extend(data["prompts"])
    # save the combined json
    path = Path(f"{config.benchmark.dataset_dir}/circ_disc/{config.circuit_discovery.model_name}/{config.circuit_discovery.dataset}_unfiltered/dataset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(combined_json, f, indent=4)
        
    # set train and test size
    num_samples = len(combined_json["prompts"])
elif config.circuit_discovery.dataset == "price_game":
    # for price game, we use the dataset from the benchmark
    dataset_files = [
      "datasets/price_game/train.json",
      "datasets/price_game/eval.json"
    ]
    combined_json = {}
    for i, file in enumerate(dataset_files):
        with open(file) as f:
            data = json.load(f)
            if i == 0:
                combined_json = data
            else:
                combined_json["prompts"].extend(data["prompts"])
    # save the combined json
    path = Path(f"{config.benchmark.dataset_dir}/circ_disc/{config.circuit_discovery.model_name}/{config.circuit_discovery.dataset}_unfiltered/dataset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(combined_json, f, indent=4)

    # set train and test size
    num_samples = len(combined_json["prompts"])


train_size = int(num_samples * config.circuit_discovery.train_percent)
test_size = num_samples - train_size
print(f"Dataset size: {num_samples}")
print(f"Train size: {train_size}")
print(f"Test size: {test_size}")

train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    batch_size=config.circuit_discovery.batch_size,
    train_test_size=[train_size, test_size],
    tail_divergence=True,
    return_seq_length=True if config.circuit_discovery.tokenGraph else False,
    # pad=False,
    prepend_bos=False,
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=False,
    kv_caches=(train_loader.kv_cache, test_loader.kv_cache),
    device=device,
    seq_len=None if not config.circuit_discovery.tokenGraph else test_loader.seq_len,
)

if config.circuit_discovery.method == "ACDC":
    print("Using ACDC method")
    attribution_scores: PruneScores = acdc_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        tao_exps=config.circuit_discovery.tao_exps,
        tao_bases=config.circuit_discovery.tao_bases,
    )
elif config.circuit_discovery.method == "mask_gradient":
    print("Using mask gradient method")
    attribution_scores: PruneScores = mask_gradient_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logprob",
        answer_function="avg_diff",
        mask_val=1.0
    )
elif config.circuit_discovery.method == "edge_attribution_patching":
    print("Using edge attribution patching method")
    attribution_scores: PruneScores = edge_attribution_patching_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
    )
else:
    raise ValueError(f"Method {config.circuit_discovery.method} not supported. Use 'ACDC' or 'mask_gradient'.")

# save to file attribution scores (PruneScores = Dict[str, t.Tensor])
attribution_scores_path = Path(f"{config.out_dir}/{config.circuit_discovery.model_name}/attribution_scores_{config.exp_name}.json")
# check if the path exists
attribution_scores_path.parent.mkdir(parents=True, exist_ok=True)
with open(attribution_scores_path, "w") as f:
    json.dump(
        {k: v.tolist() for k, v in attribution_scores.items()},
        f,
        indent=4,
    )
print(f"Attribution scores saved to {attribution_scores_path}")

image_path = Path(f"{config.out_dir}/{config.circuit_discovery.model_name}/attribution_scores_{config.exp_name}.png")
fig = draw_seq_graph(
    model, attribution_scores, config.circuit_discovery.threshold, 
    seq_labels = train_loader.seq_labels if config.circuit_discovery.tokenGraph else None,
    layer_spacing=True, orientation="h", display_ipython=False, 
)
# Save the figure
image_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(image_path, width=2000, height=1000, scale=2)
print(f"Figure saved to {image_path}")
