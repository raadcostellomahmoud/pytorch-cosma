import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config_validation import ConfigModel
from src.latent_space import LatentSpaceExplorer, Visualizer
from src.model_yaml_parser import YamlParser
from src.network_construction import TwinNetwork, BaseModel, GraphModel
from utilities.twin_dataset_maker import TwinDatasetFromDataset

# Define device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Prepare Twin datasets and loaders
train_dataset = TwinDatasetFromDataset(mnist_train)
test_dataset = TwinDatasetFromDataset(mnist_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load configuration from YAML
raw_config = YamlParser("configs/example_twin_network.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model_class = validated_config.pop("model_class")

# Initialize the model
model = globals()[model_class](validated_config).to(device)

# Loss function and optimizer
loss_function = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and evaluate
for epoch in range(epochs):
    model.train_model(epochs=1, train_loader=train_loader, optimizer=optimizer, loss_function=loss_function)
    val_accuracy = model.evaluate(test_loader=test_loader, loss_function=loss_function)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")

individ_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# Latent space exploration
explorer = LatentSpaceExplorer(model, individ_train_loader, device)
latent_points, labels_points, all_inputs = explorer.extract_latent_space(twin=True)
reduced_dimensionality = explorer.reduce_dimensionality(latent_points)

# Randomly sample points for visualization
sample_size = 100
indices = np.random.choice(len(reduced_dimensionality), size=sample_size, replace=False)
reduced_dimensionality = reduced_dimensionality[indices]
labels_points = labels_points[indices]
selected_inputs = all_inputs[indices]

# Visualize
visualizer = Visualizer(reduced_dimensionality, labels_points, selected_inputs)
hover_images = visualizer.generate_hover_images()
app = visualizer.create_dash_app(hover_images)
app.run_server(debug=False)
