import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.latent_space import LatentSpaceExplorer, Visualizer
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import TwinNetwork
from pytorch_cosma.utils import TwinDatasetFromDataset


class TestTwinNetworkPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define device (GPU/CPU)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MNIST datasets
        transform = transforms.Compose([transforms.ToTensor()])
        cls.mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)

        # Twin datasets
        cls.train_dataset = TwinDatasetFromDataset(cls.mnist_train)
        cls.test_dataset = TwinDatasetFromDataset(mnist_test)

        # DataLoaders
        cls.train_loader = DataLoader(cls.train_dataset, batch_size=64, shuffle=True)
        cls.test_loader = DataLoader(cls.test_dataset, batch_size=64, shuffle=False)

        # Load configuration and initialize model
        cls.raw_config = YamlParser("configs/example_twin_network.yaml").parse()
        cls.validated_config = ConfigModel(**cls.raw_config).to_dict()
        model_class = globals()[cls.validated_config.pop("model_class")]
        cls.model = model_class(cls.validated_config).to(cls.device)

        # Loss and optimizer
        cls.loss_function = nn.BCEWithLogitsLoss()
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=0.001)

    def test_config_validation(self):
        """Test configuration validation."""
        self.assertIsInstance(self.validated_config, dict, "Configuration should be valid.")

    def test_model_initialization(self):
        """Test TwinNetwork model initialization."""
        self.assertTrue(hasattr(self.model, "layers"), "Model should have 'layers' attribute.")

    def test_model_training(self):
        """Test model training."""
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        self.model.train_model(
            epochs=1,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
        )

        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(initial_params[name], param),
                             f"Parameter {name} did not update during training.")

    def test_model_evaluation(self):
        """Test model evaluation."""
        loss = self.model.evaluate(test_loader=self.test_loader, loss_function=self.loss_function)
        self.assertGreaterEqual(loss, 0, "Evaluation loss should be non-negative.")
        print(f"Evaluation Loss: {loss:.4f}")

    def test_latent_space_exploration_and_visualization(self):
        """Test latent space exploration and visualization."""

        explorer = LatentSpaceExplorer(self.model, DataLoader(self.mnist_train, batch_size=64), self.device)

        # Extract latent space
        latent_points, labels_points, all_inputs = explorer.extract_latent_space(twin=True)
        self.assertIsInstance(latent_points, np.ndarray, "Latent points should be a NumPy array.")
        self.assertGreater(len(latent_points), 0, "Latent points should not be empty.")

        # Reduce dimensionality
        reduced_dimensionality = explorer.reduce_dimensionality(latent_points)
        self.assertEqual(reduced_dimensionality.shape[1], 2, "Reduced dimensionality should have 2 components.")

        # Randomly sample points
        sample_size = 100
        indices = np.random.choice(len(reduced_dimensionality), size=sample_size, replace=False)
        reduced_dimensionality = reduced_dimensionality[indices]
        labels_points = labels_points[indices]
        selected_inputs = all_inputs[indices]

        # Visualize
        visualizer = Visualizer(reduced_dimensionality, labels_points, selected_inputs)
        hover_images = visualizer.generate_hover_images()
        self.assertEqual(len(hover_images), sample_size, "Hover images should match sample size.")

        app = visualizer.create_dash_app(hover_images)

        # Check Dash app setup
        self.assertIsNotNone(app.layout, "Dash app layout should be defined.")
        print("Dash app for latent space visualization is ready to run.")

        # Note: Running the Dash app interactively isn't suited for unit testing
        # but can be included in end-to-end or manual testing.


if __name__ == "__main__":
    unittest.main()
