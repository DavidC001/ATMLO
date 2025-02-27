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
# Data Generation configuration
# =================================================================================================

@dataclass
class DataGenerationConfig:
    num_samples: int = 100
    seed: int = 42
    
    def __post_init__(self):
        assert self.num_samples > 0, "num_samples must be greater than 0"

# =================================================================================================
# Model configuration
# =================================================================================================

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    q_4b: bool = False
    q_8b: bool = False

# =================================================================================================
# Project configuration
# =================================================================================================

@dataclass
class ProjectConfig:
    dataset: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    
def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)
