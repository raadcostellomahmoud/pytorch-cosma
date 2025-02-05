import unittest
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import BaseModel  # Ensure this import is present


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define device (GPU/CPU)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        raw_config = YamlParser("configs/example_model.yaml").parse()
        cls.validated_config = ConfigModel(**raw_config).to_dict()
        model_class = globals()[cls.validated_config.pop("model_class")]

        # Initialize model
        cls.model = model_class(cls.validated_config, device=cls.device)

        # Prepare dataset and dataloaders
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        cls.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        cls.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        cls.loss_function = nn.CrossEntropyLoss()
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=0.001)

    def test_config_validation(self):
        """Test that the configuration is successfully validated."""
        self.assertIsInstance(self.validated_config, dict, "Configuration should be a validated dictionary.")

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertTrue(hasattr(self.model, 'layers'), "Model should have a 'layers' attribute.")
        self.assertTrue(hasattr(self.model, 'config'), "Model should have a 'config' attribute.")

    def test_training(self):
        """Test that the model can perform a training step."""
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        self.model.train_model(
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            epochs=1  # Only one epoch for testing
        )

        # Check that parameters have changed after training
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(initial_params[name], param), f"Parameter {name} did not update during training.")

    def test_evaluation(self):
        """Test that the model evaluation produces a valid accuracy."""
        self.model.eval()
        accuracy = self.model.evaluate(test_loader=self.test_loader, loss_function=self.loss_function)

        self.assertGreaterEqual(accuracy, 0, "Accuracy should be non-negative.")
        self.assertLessEqual(accuracy, 100, "Accuracy should not exceed 100%.")

    def test_final_accuracy(self):
        """Test that the model achieves at least a minimal accuracy."""
        self.model.train_model(
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            epochs=2  # Few epochs for testing
        )
        accuracy = self.model.evaluate(test_loader=self.test_loader, loss_function=self.loss_function)
        self.assertGreater(accuracy, 50, "Model should achieve at least 50% accuracy on MNIST.")


if __name__ == "__main__":
    unittest.main()
