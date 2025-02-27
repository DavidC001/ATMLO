import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import random
import copy
import itertools
import numpy as np
from tqdm import tqdm, trange

from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

from pyvene import CausalModel
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene import create_mlp_classifier
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def randvec(n=50, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for i in range(n)])


embedding_dim = 2
number_of_entities = 20

variables = ["W", "X", "Y", "Z", "WX", "YZ", "O"]

reps = [randvec(embedding_dim, lower=-1, upper=1) for _ in range(number_of_entities)]
values = {variable: reps for variable in ["W", "X", "Y", "Z"]}
values["WX"] = [True, False]
values["YZ"] = [True, False]
values["O"] = [True, False]

parents = {
    "W": [],
    "X": [],
    "Y": [],
    "Z": [],
    "WX": ["W", "X"],
    "YZ": ["Y", "Z"],
    "O": ["WX", "YZ"],
}


def FILLER():
    return reps[0]


functions = {
    "W": FILLER,
    "X": FILLER,
    "Y": FILLER,
    "Z": FILLER,
    "WX": lambda x, y: np.array_equal(x, y),
    "YZ": lambda x, y: np.array_equal(x, y),
    "O": lambda x, y: x == y,
}

pos = {
    "W": (0.2, 0),
    "X": (1, 0.1),
    "Y": (2, 0.2),
    "Z": (2.8, 0),
    "WX": (1, 2),
    "YZ": (2, 2),
    "O": (1.5, 3),
}

equiv_classes = {}

equality_model = CausalModel(variables, values, parents, functions, pos=pos)