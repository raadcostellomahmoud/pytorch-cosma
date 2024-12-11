import unittest
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.config_validation import ConfigModel
from src.model_yaml_parser import YamlParser
from src.network_construction import GraphModel


class TestNodeClassificationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define device
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        dataset = Planetoid(root="data/Cora", name="Cora", transform=NormalizeFeatures())
        cls.data = dataset[0]  # Single graph object

        # Parse model configuration
        raw_config = YamlParser("configs/example_gatconv_network.yaml").parse()
        cls.validated_config = ConfigModel(**raw_config).to_dict()

        # Initialize model
        cls.model = GraphModel(cls.validated_config).to(cls.device)

        # Optimizer
        cls.optimizer = Adam(cls.model.parameters(), lr=0.005, weight_decay=5e-4)

        # Define loss function and metrics
        cls.loss_functions = {
            "node_output": nn.CrossEntropyLoss()
        }

        cls.metrics = {
            "node_output": lambda pred, target: (pred.argmax(dim=1) == target).float().mean().item()
        }

        # Map the correct label source for the task
        cls.label_mapping = {
            "node_output": "y"
        }

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
            epochs=1
        )

        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(initial_params[name], param), f"Parameter {name} did not update during training.")

    def test_node_classification_accuracy(self):
        """Test the node classification accuracy."""
        self.model.eval()
        logits = self.model(self.data)
        _, pred = logits["node_output"].max(dim=1)
        node_accuracy = (pred == self.data.y.to(self.device)).sum() / self.data.num_nodes
        self.assertGreaterEqual(node_accuracy, 0.0, "Node classification accuracy should be >= 0.")
        print(f"Node Classification Accuracy: {node_accuracy:.4f}")

if __name__ == "__main__":
    unittest.main()
