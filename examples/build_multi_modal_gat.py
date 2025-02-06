import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import MultiModalGATModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration and model
raw_config = YamlParser("configs/example_multimodal_gat.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model_class = validated_config.pop("model_class")
model = globals()[model_class](validated_config, device=device)

# TODO: complete example