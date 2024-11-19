import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config_validation import ConfigModel
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration from YAML
raw_config = YamlParser("configs/example_model.yaml").parse()  # raw_config is a dictionary from the YAML file

# Load configuration and model
raw_config = YamlParser("configs/example_model.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model = BaseModel(validated_config, device=device)

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Train and evaluate
model.train_model(train_loader=train_loader,
                  optimizer=optim.Adam(model.parameters(), lr=0.001),
                  loss_function=nn.CrossEntropyLoss(),
                  )
test_accuracy = model.evaluate(test_loader=test_loader,
                               loss_function=nn.CrossEntropyLoss())
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
