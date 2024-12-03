import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config_validation import ConfigModel
from src.latent_space import LatentSpaceExplorer, Visualizer
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel

classification_loss_weight = 0.1

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Load configuration from YAML
raw_config = YamlParser(
    "configs/example_conv_autoencoder.yaml").parse()  # raw_config is a dictionary from the YAML file

# Validate configuration
try:
    validated_config = ConfigModel(**raw_config).to_dict()
except ValueError as e:
    print("Configuration validation failed:", e)
    exit(1)

# Create model from configuration
model = BaseModel(validated_config, use_reconstruction=True, device=device)

# Train the model
model.train_model(epochs=5, train_loader=train_loader,
                  loss_function=nn.MSELoss(), classification_loss_weight=classification_loss_weight,
                  optimizer=torch.optim.Adam(model.parameters(), lr=1e-3))

# Latent space exploration
explorer = LatentSpaceExplorer(model, train_loader, device)
latent_points, labels_points, all_inputs = explorer.extract_latent_space()
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
