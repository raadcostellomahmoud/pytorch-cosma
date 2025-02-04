import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
from torch_geometric.utils import train_test_split_edges

from src.config_validation import ConfigModel
from src.graphs import prepare_edge_labels, GraphVisualizer3D
from src.model_yaml_parser import YamlParser
from src.network_construction import GraphModel, BaseModel, TwinNetwork

# Load Cora dataset
dataset = Planetoid(root="data/Cora", name="Cora", transform=NormalizeFeatures())
data = dataset[0]  # Single graph object

# Convert data to a train-test split for edge prediction
data = train_test_split_edges(data)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse model configuration
raw_config = YamlParser("configs/example_gatconv_node_and_edge.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model_class = validated_config.pop("model_class")

# Initialize model
model = globals()[model_class](validated_config).to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=0.001)  # , weight_decay=5e-4)

# Define loss functions and metrics for both tasks
loss_functions = {
    "node_output": nn.CrossEntropyLoss(),  # Node classification
    "edge_output": nn.BCEWithLogitsLoss(),  # Edge prediction
}

metrics = {
    "node_output": lambda pred, target: (pred.argmax(dim=1) == target).float().mean().item(),
    "edge_output": lambda pred, target: ((pred > 0).float() == target).float().mean().item(),
}

# Map the correct label source for each task
label_mapping = {
    "node_output": "y",  # Maps to data.y for node classification
    "edge_output": "edge_label",  # Maps to generated edge labels
}

# Add edge_label to data for edge prediction task
data = prepare_edge_labels(data).to(device)

# Create DataLoader
data_loader = DataLoader([data], batch_size=1, shuffle=False)

# Training loop
model.train_model(
    train_loader=data_loader,
    loss_functions=loss_functions,
    metrics=metrics,
    label_mapping=label_mapping,
    optimizer=optimizer,
    edges_nodes_or_both="both",
    epochs=400
)

# Evaluate
model.eval()
logits = model(data)

# Node classification accuracy
_, node_pred = logits["node_output"].max(dim=1)
node_accuracy = (node_pred == data.y.to(device)).sum() / data.num_nodes
print(f"Node Classification Accuracy: {node_accuracy:.4f}")

# Edge prediction accuracy
edge_pred = logits["edge_output"]
edge_accuracy = ((edge_pred > 0).float() == data.edge_label).float().mean()
print(f"Edge Prediction Accuracy: {edge_accuracy:.4f}")

# Convert the graph to NetworkX format
G = to_networkx(data, to_undirected=True)

# Visualize the graph using the GraphVisualizer
SUBSET_SIZE = 200  # Configurable subset size
visualizer = GraphVisualizer3D(G, node_pred, data.y, subset_size=SUBSET_SIZE, edge_predictions=edge_pred,
                             edge_ground_truth=data.edge_label)
app = visualizer.create_dash_app()

if __name__ == "__main__":
    app.run_server(debug=False)
