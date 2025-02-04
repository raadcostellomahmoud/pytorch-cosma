import unittest
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges

from src.config_validation import ConfigModel
from src.graphs import prepare_edge_labels
from src.model_yaml_parser import YamlParser
from src.network_construction import GraphModel

class TestGraphModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define device
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Cora dataset
        dataset = Planetoid(root="data/Cora", name="Cora", transform=NormalizeFeatures())
        cls.data = train_test_split_edges(dataset[0])

        # Parse model configuration
        raw_config = YamlParser("configs/example_gatconv_node_and_edge.yaml").parse()
        cls.validated_config = ConfigModel(**raw_config).to_dict()
        model_class = globals()[cls.validated_config.pop("model_class")]

        # Initialize model
        cls.model = model_class(cls.validated_config).to(cls.device)

        # Optimizer
        cls.optimizer = Adam(cls.model.parameters(), lr=0.001)

        # Define loss functions and metrics
        cls.loss_functions = {
            "node_output": nn.CrossEntropyLoss(),
            "edge_output": nn.BCEWithLogitsLoss(),
        }

        cls.metrics = {
            "node_output": lambda pred, target: (pred.argmax(dim=1) == target).float().mean().item(),
            "edge_output": lambda pred, target: ((pred > 0).float() == target).float().mean().item(),
        }

        # Map the correct label source for each task
        cls.label_mapping = {
            "node_output": "y",
            "edge_output": "edge_label",
        }

        # Prepare edge labels
        cls.data = prepare_edge_labels(cls.data).to(cls.device)

        # Create DataLoader
        cls.data_loader = DataLoader([cls.data], batch_size=1, shuffle=False)

    def test_config_validation(self):
        """Test that the configuration is valid."""
        self.assertIsInstance(self.validated_config, dict, "Configuration should be a valid dictionary.")

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertTrue(hasattr(self.model, 'layers'), "Model should have a 'layers' attribute.")

    def test_training(self):
        """Test the training process."""
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        self.model.train_model(
            train_loader=self.data_loader,
            loss_functions=self.loss_functions,
            metrics=self.metrics,
            label_mapping=self.label_mapping,
            optimizer=self.optimizer,
            edges_nodes_or_both="both",
            epochs=1
        )

        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(initial_params[name], param), f"Parameter {name} did not update during training.")

    def test_node_classification_accuracy(self):
        """Test the node classification accuracy."""
        self.model.eval()
        logits = self.model(self.data)
        _, node_pred = logits["node_output"].max(dim=1)
        node_accuracy = (node_pred == self.data.y.to(self.device)).sum() / self.data.num_nodes
        self.assertGreaterEqual(node_accuracy, 0.0, "Node classification accuracy should be >= 0.")
        print(f"Node Classification Accuracy: {node_accuracy:.4f}")

    def test_edge_prediction_accuracy(self):
        """Test the edge prediction accuracy."""
        self.model.eval()
        logits = self.model(self.data)
        edge_pred = logits["edge_output"]
        edge_accuracy = ((edge_pred > 0).float() == self.data.edge_label).float().mean()
        self.assertGreaterEqual(edge_accuracy, 0.0, "Edge prediction accuracy should be >= 0.")
        print(f"Edge Prediction Accuracy: {edge_accuracy:.4f}")

if __name__ == "__main__":
    unittest.main()
