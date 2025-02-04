import unittest
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from src.config_validation import ConfigModel
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class TestCIFAR10ViT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define device (GPU/CPU)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration and model
        cls.raw_config = YamlParser("configs/example_vit_model.yaml").parse()
        cls.validated_config = ConfigModel(**cls.raw_config).to_dict()
        model_class = globals()[cls.validated_config.pop("model_class")]
        cls.model = model_class(cls.validated_config, device=cls.device)

        # Training Setup
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cls.train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        cls.test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

        cls.train_loader = DataLoader(cls.train_dataset, batch_size=64, shuffle=True)
        cls.test_loader = DataLoader(cls.test_dataset, batch_size=64, shuffle=False)

        cls.optimizer = optim.AdamW(cls.model.parameters(), lr=3e-4)
        cls.criterion = nn.CrossEntropyLoss()

    def test_config_validation(self):
        """Test configuration validation."""
        self.assertIsInstance(self.validated_config, dict, "Configuration should be valid.")

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertTrue(hasattr(self.model, "layers"), "Model should have 'layers' attribute.")

    def test_model_training(self):
        """Test model training."""
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        self.model.train_model(
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            loss_function=self.criterion,
            epochs=1,
        )

        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(initial_params[name], param),
                             f"Parameter {name} did not update during training.")

    def test_model_evaluation(self):
        """Test model evaluation."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(self.test_loader)
        self.assertGreaterEqual(average_loss, 0, "Evaluation loss should be non-negative.")
        print(f"Evaluation Loss: {average_loss:.4f}")

if __name__ == "__main__":
    unittest.main()