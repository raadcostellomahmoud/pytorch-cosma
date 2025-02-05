import unittest
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config_validation import ConfigModel
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel

class TestTrainAndCompareClassifiersOnMNIST(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define device (GPU/CPU)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configurations and models
        convnext_config = YamlParser("configs/convnext_model.yaml").parse()
        validated_convnext_config = ConfigModel(**convnext_config).to_dict()
        convnext_model_class = validated_convnext_config.pop("model_class")
        cls.convnext_model = globals()[convnext_model_class](validated_convnext_config, device=cls.device)

        # Load dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        cls.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        cls.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    def test_convnext_training(self):
        print("Testing ConvNeXt model training...")
        for epoch in range(2):  # Reduced to 2 epochs
            self.convnext_model.train_model(train_loader=self.train_loader,
                                            optimizer=optim.Adam(self.convnext_model.parameters(), lr=0.001),
                                            loss_function=nn.CrossEntropyLoss(), epochs=1
                                            )
            val_accuracy = self.convnext_model.evaluate(test_loader=self.test_loader,
                                                        loss_function=nn.CrossEntropyLoss())
            print(f"ConvNeXt Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")
        convnext_test_accuracy = self.convnext_model.evaluate(test_loader=self.test_loader,
                                                              loss_function=nn.CrossEntropyLoss())
        print(f"ConvNeXt Final Test Accuracy: {convnext_test_accuracy:.2f}%")
        self.assertGreater(convnext_test_accuracy, 90)  # Example threshold

if __name__ == '__main__':
    unittest.main()
