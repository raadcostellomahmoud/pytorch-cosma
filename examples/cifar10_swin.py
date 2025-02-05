import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.network_construction import BaseModel, TwinNetwork, GraphModel
from pytorch_cosma.model_yaml_parser import YamlParser

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config
parser = YamlParser("configs/example_swin_transformer.yaml")
raw_config = parser.parse()
validated_config = ConfigModel(**raw_config).to_dict()

# Initialize model
model_class = validated_config.pop("model_class")
model = globals()[model_class](validated_config, device=device)

# CIFAR-10 Data
train_data = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
test_data = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Train
model.train_model(
    train_loader=train_loader,
    loss_function=criterion,
    optimizer=optimizer,
    epochs=20
)

# Evaluate
accuracy = model.evaluate(test_loader, criterion)
print(f"Final Test Accuracy: {accuracy:.2f}%")