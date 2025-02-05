import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import BaseModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configurations and models
resnet_config = YamlParser("configs/example_model.yaml").parse()
convnext_config = YamlParser("configs/convnext_model.yaml").parse()

validated_resnet_config = ConfigModel(**resnet_config).to_dict()
validated_convnext_config = ConfigModel(**convnext_config).to_dict()

resnet_model_class = validated_resnet_config.pop("model_class")
convnext_model_class = validated_convnext_config.pop("model_class")

resnet_model = globals()[resnet_model_class](validated_resnet_config, device=device)
convnext_model = globals()[convnext_model_class](validated_convnext_config, device=device)

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Train and evaluate ConvNeXt model
print("Training ConvNeXt model...")
for epoch in range(10):  # Assuming 10 epochs
    convnext_model.train_model(train_loader=train_loader,
                               optimizer=optim.Adam(convnext_model.parameters(), lr=0.001),
                               loss_function=nn.CrossEntropyLoss(), epochs=1
                               )
    val_accuracy = convnext_model.evaluate(test_loader=test_loader,
                                           loss_function=nn.CrossEntropyLoss())
    print(f"ConvNeXt Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")

convnext_test_accuracy = convnext_model.evaluate(test_loader=test_loader,
                                                 loss_function=nn.CrossEntropyLoss())
print(f"ConvNeXt Final Test Accuracy: {convnext_test_accuracy:.2f}%")

# Train and evaluate ResNet model
print("Training ResNet model...")
for epoch in range(10):  # Assuming 10 epochs
    resnet_model.train_model(train_loader=train_loader,
                             optimizer=optim.Adam(resnet_model.parameters(), lr=0.001),
                             loss_function=nn.CrossEntropyLoss(), epochs=1
                             )
    val_accuracy = resnet_model.evaluate(test_loader=test_loader,
                                         loss_function=nn.CrossEntropyLoss())
    print(f"ResNet Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")

resnet_test_accuracy = resnet_model.evaluate(test_loader=test_loader,
                                             loss_function=nn.CrossEntropyLoss())

# Compare final test accuracies
print(f"ResNet Final Test Accuracy: {resnet_test_accuracy:.2f}%")
print(f"ConvNeXt Final Test Accuracy: {convnext_test_accuracy:.2f}%")
