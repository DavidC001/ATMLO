import json
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import PruneScores
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.graph_utils import patchable_model

from pathlib import Path

import torch

attribution_scores_path = "/home/davide.cavicchini/projects/ATMLO/results/Qwen/Qwen2.5-1.5B-Instruct/attribution_scores_1batch.json"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cpu"

threshold = 5e-2

attribution_scores : PruneScores = None


# load attribution scores from file
with open(attribution_scores_path) as f:
    attribution_scores = json.load(f)
    # convert to tensor
    for k, v in attribution_scores.items():
        attribution_scores[k] = torch.tensor(v, device=device, dtype=torch.float32)
        
model = load_tl_model(model_name, device=device)
model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=False,
    device=device,
)

from plotly.graph_objects import Figure

fig_path = "temp.png"
fig : Figure = draw_seq_graph(
    model, attribution_scores, threshold, 
    
    layer_spacing=True, 
    orientation="h", 
    # display_ipython=True,
    # show_all_seq_pos=False,
)


# Save the figure
fig_path = Path(fig_path)
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(fig_path, width=2000, height=1000, scale=2)
print(f"Figure saved to {fig_path}")