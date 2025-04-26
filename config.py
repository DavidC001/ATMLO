from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass, field
import os
import yaml
import torch


def dataclass(*args, **kwargs):
    """
    Creates a dataclass that can handle nested dataclasses
    and automatically convert dictionaries to dataclasses.
    """

    def wrapper(cls):
        cls = og_dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(args[0]) if args else wrapper

# =================================================================================================
# data_preprocessing configuration
# =================================================================================================

@dataclass
class DataConfig:
    global_padding: bool = False
    dataset_dir: str = "datasets"
    
    model_names: list[str] = field(default_factory=lambda: ["EleutherAI/pythia-14m"])
    input_jsons: dict[list[str]] = field(default_factory=lambda: {})
    templates: list[str] = field(default_factory=lambda: ["LogicBench"])

# =================================================================================================
# Model configuration
# =================================================================================================

@dataclass
class CircuitDiscConf:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size: int = 8
    train_percent: float = 0.8
    
    threshold: float = 1e-2
    
    dataset: str = "modus_tollens"
    """
    The dataset to use for training from the ones available for the selected model.
    """
    
    tokenGraph: bool = False
    """
    If True, the model will be patched so that each token is visible in the output.
    """
    
    method: str = "ACDC"
    """
    The method to use for pruning. Options are: ACDC, mask_gradient
    """
    
    tao_bases: list = field(default_factory=lambda: [1, 5, 9])
    tao_exps: list = field(default_factory=lambda: [-5, -3, -2])
    
    def __post_init__(self):
        if self.method not in ["ACDC", "mask_gradient", "edge_attribution_patching"]:
            raise ValueError(f"Method {self.method} not supported. Use 'ACDC' or 'mask_gradient'.")
        
# ==================================================================================================
# LogicBench Dataset Conversion   
# ==================================================================================================

@dataclass
class DatasetConversion:
    """
    Configuration for the dataset conversion.
    """
    
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    
    dataset_files: list[str] = field(default_factory=lambda: [])
    output_files: list[str] = field(default_factory=lambda: [])
    
    format: str = "default"
    """
    The format of the dataset. Options are: default, modus_tollens.
    """
    
    def __post_init__(self):
        if self.format not in ["default", "modus_tollens"]:
            raise ValueError(f"Format {self.format} not supported. Use 'default' or 'modus_tollens'.")
        
        # check if input files exist
        for file in self.dataset_files:
            assert os.path.exists(file), f"File {file} does not exist."


# =================================================================================================
# Project configuration
# =================================================================================================

@dataclass
class ProjectConfig:
    exp_name: str = "default"
    out_dir : str = "results"
    seed: int = 42
    
    circuit_discovery: CircuitDiscConf = field(default_factory=CircuitDiscConf)
    data_preprocessing: DataConfig = field(default_factory=DataConfig)
    
    convert_dataset: DatasetConversion = field(default_factory=DatasetConversion)
    
    
def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)
