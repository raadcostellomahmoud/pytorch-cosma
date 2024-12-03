import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config_validation import ConfigModel
from src.latent_space import LatentSpaceExplorer, Visualizer
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel


class TestAutoencoderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define device (GPU/CPU)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])  # No normalization to simplify
        cls.train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        cls.test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        cls.train_loader = DataLoader(cls.train_dataset, batch_size=64, shuffle=True)
        cls.test_loader = DataLoader(cls.test_dataset, batch_size=1000, shuffle=False)

        # Load configuration from YAML
        cls.raw_config = YamlParser("configs/example_conv_autoencoder.yaml").parse()
        cls.validated_config = ConfigModel(**cls.raw_config).to_dict()

        # Create model
        cls.model = BaseModel(cls.validated_config, use_reconstruction=True, device=cls.device)

        cls.loss_function = nn.MSELoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=1e-3)

    def test_config_validation(self):
        """Test that the autoencoder configuration is valid."""
        self.assertIsInstance(self.validated_config, dict, "Configuration should be valid.")

    def test_model_initialization(self):
        """Test that the autoencoder model initializes correctly."""
        self.assertTrue(hasattr(self.model, 'layers'), "Autoencoder model should have a 'layers' attribute.")

    def test_model_training(self):
        """Test autoencoder model training."""
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        self.model.train_model(
            epochs=1,
            train_loader=self.train_loader,
            loss_function=self.loss_function,
            optimizer=self.optimizer,
            classification_loss_weight=0.1
        )

        # Check that parameters have updated
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(initial_params[name], param),
                             f"Parameter {name} did not update during training.")

    def test_reconstruction_loss(self):
        """Test reconstruction loss functionality."""
        self.model.eval()
        sample_batch = next(iter(self.test_loader))[0].to(self.device)  # Get a batch of images
        reconstructed, _ = self.model(sample_batch, return_latent=False)
        loss = self.loss_function(reconstructed, sample_batch)

        self.assertGreaterEqual(loss.item(), 0, "Reconstruction loss should be non-negative.")

    def test_model_evaluation(self):
        """Test autoencoder evaluation."""
        self.model.eval()
        sample_batch = next(iter(self.test_loader))[0].to(self.device)
        reconstructed, _ = self.model(sample_batch, return_latent=False)
        loss = self.loss_function(reconstructed, sample_batch)

        self.assertGreaterEqual(loss.item(), 0, "Evaluation loss should be non-negative.")
        print(f"Reconstruction Loss: {loss.item():.4f}")

    def test_latent_space_exploration_and_visualization(self):
        """Test latent space exploration and visualization."""
        explorer = LatentSpaceExplorer(self.model, self.train_loader, self.device)

        # Extract latent space
        latent_points, labels_points, all_inputs = explorer.extract_latent_space()
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
