from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass, field
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
    dataset_dir: str = "datasets"
    model_names: list = field(default_factory=lambda: ["EleutherAI/pythia-14m"])
    input_jsons: list = field(default_factory=lambda: ["datasets/testing.json"])
    templates: list = field(default_factory=lambda: ["LogicBench"])

# =================================================================================================
# Model configuration
# =================================================================================================

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    train_percent: float = 0.8
    threshold: float = 1e-2

# =================================================================================================
# Project configuration
# =================================================================================================

@dataclass
class ProjectConfig:
    out_dir : str = "results"
    seed: int = 42
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data_preprocessing: DataConfig = field(default_factory=DataConfig)
    
    
def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)
