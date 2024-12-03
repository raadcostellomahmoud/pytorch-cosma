import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

from src.config_validation import ConfigModel
from src.graphs import GraphVisualizer
from src.model_yaml_parser import YamlParser
from src.network_construction import GraphModel

# Load dataset
dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Single graph object

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse model configuration
raw_config = YamlParser("configs/example_gatconv_network.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()

# Initialize model
model = GraphModel(validated_config).to(device)
# Create DataLoader
data_loader = DataLoader([data], batch_size=1, shuffle=False)

# Optimizer
optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

loss_functions = {
    "node_output": nn.CrossEntropyLoss()
}
metrics = {
    "node_output": lambda pred, target: (pred.argmax(dim=1) == target).float().mean().item()
}

# Map the correct label source for each task
label_mapping = {
    "node_output": "y"  # Maps to data.y
}

# Training loop
model.train_model(train_loader=data_loader, loss_functions=loss_functions, metrics=metrics, label_mapping=label_mapping,
                  optimizer=optimizer, epochs=100)

# # Evaluate
model.evaluate(test_loader=data_loader, loss_functions=loss_functions, label_mapping=label_mapping, metrics=metrics)

model.eval()
_, pred = model(data)['node_output'].max(dim=1)
# Convert the graph to NetworkX format
G = to_networkx(data, to_undirected=True)

# Visualize the graph using the GraphVisualizer
SUBSET_SIZE = 100  # Configurable subset size
visualizer = GraphVisualizer(G, pred, data.y, subset_size=SUBSET_SIZE)
app = visualizer.create_dash_app()

if __name__ == "__main__":
    app.run_server(debug=False)
