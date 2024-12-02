import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
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

# Optimizer
optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data)
    loss = cross_entropy(out, data.y.to(device))
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# Evaluate
model.eval()
_, pred = model(data).max(dim=1)
accuracy = (pred == data.y.to(device)).sum() / data.num_nodes
print(f'Accuracy: {accuracy:.4f}')
# Convert the graph to NetworkX format
G = to_networkx(data, to_undirected=True)

# Visualize the graph using the GraphVisualizer
SUBSET_SIZE = 100  # Configurable subset size
visualizer = GraphVisualizer(G, pred, data.y, subset_size=SUBSET_SIZE)
app = visualizer.create_dash_app()

if __name__ == "__main__":
    app.run_server(debug=False)
