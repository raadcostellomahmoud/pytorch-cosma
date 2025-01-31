import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.network_construction import BaseModel
from src.model_yaml_parser import YamlParser

# Load config
parser = YamlParser("configs/example_swin_transformer.yaml")
config = parser.parse()

# Initialize model
model = BaseModel(config, use_reconstruction=False)

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