import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import BaseModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration and model
raw_config = YamlParser("configs/example_model.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model_class = validated_config.pop("model_class")
model = globals()[model_class](validated_config, device=device)

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Train and evaluate
for epoch in range(10):
    model.train_model(train_loader=train_loader,
                      optimizer=optim.Adam(model.parameters(), lr=0.001),
                      loss_function=nn.CrossEntropyLoss(), epochs=1
                      )
    val_accuracy = model.evaluate(test_loader=test_loader,
                                  loss_function=nn.CrossEntropyLoss())
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")

test_accuracy = model.evaluate(test_loader=test_loader,
                               loss_function=nn.CrossEntropyLoss())
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Prune the model
model.prune_model(validated_config['pruning'])
print("Sparsity:", model.get_sparsity_stats())

# Check post-pruning accuracy
test_accuracy = model.evaluate(test_loader=test_loader,
                               loss_function=nn.CrossEntropyLoss())

# Refine and evaluate the pruned model
for epoch in range(1):
    model.train_model(train_loader=train_loader,
                      optimizer=optim.Adam(model.parameters(), lr=0.001),
                      loss_function=nn.CrossEntropyLoss(), epochs=1
                      )
    val_accuracy = model.evaluate(test_loader=test_loader,
                                  loss_function=nn.CrossEntropyLoss())
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")
    
test_accuracy = model.evaluate(test_loader=test_loader,
                               loss_function=nn.CrossEntropyLoss())