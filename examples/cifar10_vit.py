import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import BaseModel, TwinNetwork, GraphModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration and model
raw_config = YamlParser("configs/example_vit_model.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model_class = raw_config.pop("model_class")
model = globals()[model_class](validated_config, device=device)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

model.train_model(
    train_loader=train_loader,
    optimizer=optimizer,
    loss_function=criterion,
    epochs=10,
)