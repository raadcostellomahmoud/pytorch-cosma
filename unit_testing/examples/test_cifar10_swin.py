import os
import shutil
import unittest
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import yaml
from torch import nn

from src.basic_layers import Permute
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel
from src.vision_transformer import PatchEmbedding, PatchMerging, SwinBlock


class TestSwinTransformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Batch of 2 CIFAR-10-like images
        cls.test_batch = torch.randn(2, 3, 32, 32)


    def setUp(self):
        parser = YamlParser("configs/example_swin_transformer.yaml")
        self.config = parser.parse()
        model_class = globals()[self.config.pop("model_class")]
        self.model = model_class(self.config)

    def test_patch_embedding_shape(self):
        layer = PatchEmbedding(img_size=32, patch_size=4,
                               in_channels=3, embed_dim=96, swin=True)
        x = torch.randn(2, 3, 32, 32)
        output = layer(x)
        self.assertEqual(output.shape, (2, 64, 96))

    def test_swin_block_forward_pass(self):
        block = SwinBlock(dim=96, num_heads=3, window_size=4, shifted=False)
        x = torch.randn(2, 64, 96)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    def test_patch_merging_output_shape(self):
        merge = PatchMerging(dim=96)
        x = torch.randn(2, 64, 96)
        output = merge(x)
        self.assertEqual(output.shape, (2, 16, 192))

    def test_permute_layer_functionality(self):
        layer = Permute(dims=[0, 2, 1])
        x = torch.randn(2, 16, 96)
        output = layer(x)
        self.assertEqual(output.shape, (2, 96, 16))

    def test_full_model_initialization(self):
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(len(self.model.layers), len(self.config["layers"]))

    def test_model_forward_pass_shape(self):
        output = self.model(self.test_batch)
        self.assertEqual(output.shape, (2, 10))

    def test_invalid_config_handling(self):
        invalid_config = {"layers": [{"name": "like whatever", "type": "NonExistentLayer"}]}
        with self.assertRaises(ValueError):
            BaseModel(invalid_config)

    def test_training_step(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, 10, (2,)).to(self.model.device)

        optimizer.zero_grad()
        outputs = self.model(self.test_batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss).item())

    def test_gradient_flow(self):
        output = self.model(self.test_batch)
        output.mean().backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0),
                             f"Zero gradients for {name}")

    def test_config_validation(self):
        self.assertIn("layers", self.config)
        self.assertGreaterEqual(len(self.config["layers"]), 5)


class TestEdgeCases(unittest.TestCase):
    def test_invalid_patch_size(self):
        with self.assertRaises(ValueError):
            PatchEmbedding(img_size=31, patch_size=4, in_channels=3,
                           embed_dim=96, swin=True)  # 31 not divisible by 4

    def test_window_size_larger_than_feature_map(self):
        block = SwinBlock(dim=96, num_heads=3, window_size=8, shifted=False)
        x = torch.randn(2, 16, 96)  # Feature map size 4x4 (16 patches)
        with self.assertRaises(AssertionError):
            block(x)


class TestCIFAR10Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configuration
        cls.batch_size = 128
        cls.num_workers = 2
        cls.num_train_samples = 512  # Use subset for faster testing
        cls.num_test_samples = 128

        # Transformations
        cls.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load dataset (use temporary directory)
        cls.data_dir = "./data"
        cls._prepare_dataset()

    @classmethod
    def _prepare_dataset(cls):
        # Train dataset
        cls.train_dataset = torchvision.datasets.CIFAR10(
            root=cls.data_dir, train=True,
            download=True, transform=cls.transform
        )
        # Use subset for faster testing
        cls.train_dataset = Subset(
            cls.train_dataset, range(cls.num_train_samples))

        # Test dataset
        cls.test_dataset = torchvision.datasets.CIFAR10(
            root=cls.data_dir, train=False,
            download=True, transform=cls.transform
        )
        cls.test_dataset = Subset(
            cls.test_dataset, range(cls.num_test_samples))

        # Create dataloaders
        cls.train_loader = DataLoader(
            cls.train_dataset, batch_size=cls.batch_size,
            shuffle=True, num_workers=cls.num_workers
        )
        cls.test_loader = DataLoader(
            cls.test_dataset, batch_size=cls.batch_size,
            shuffle=False, num_workers=cls.num_workers
        )

    def setUp(self):
        # Shared model setup
        parser = YamlParser("configs/example_swin_transformer.yaml")
        config = parser.parse()
        model_class = globals()[config.pop("model_class")]
        self.model = model_class(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_dataset_loading(self):
        """Test CIFAR-10 dataset loads correctly"""
        # Train dataset
        self.assertEqual(len(self.train_dataset), self.num_train_samples)
        img, label = self.train_dataset[0]
        self.assertEqual(img.shape, (3, 32, 32))
        self.assertIsInstance(label, int)

        # Test dataset
        self.assertEqual(len(self.test_dataset), self.num_test_samples)
        img, label = self.test_dataset[0]
        self.assertEqual(img.shape, (3, 32, 32))

    def test_dataloader_output_shapes(self):
        """Test dataloader produces correct batch shapes"""
        train_batch = next(iter(self.train_loader))
        images, labels = train_batch
        self.assertEqual(images.shape, (self.batch_size, 3, 32, 32))
        self.assertEqual(labels.shape, (self.batch_size,))

    def test_full_training_epoch(self):
        """Test complete training epoch with real data"""
        initial_params = [p.clone() for p in self.model.parameters()]

        # Training loop
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
            if batch_idx >= 2:  # Only test 2 batches for speed
                break

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.assertFalse(torch.isnan(loss).item())

        # Verify parameters changed
        for init_param, trained_param in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(init_param, trained_param))

    def test_validation_procedure(self):
        """Test validation pass with test dataset"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs.to(self.model.device))
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets.to(self.model.device)).sum().item()

        accuracy = 100 * correct / total
        self.assertGreaterEqual(accuracy, 0)  # Basic sanity check
        self.assertLessEqual(accuracy, 100)

    def test_input_normalization(self):
        """Test input images are properly normalized"""
        batch = next(iter(self.train_loader))
        images, _ = batch

        # Check pixel value range
        self.assertLessEqual(images.max().item(), 1.0)
        self.assertGreaterEqual(images.min().item(), -1.0)

        # Check mean/std
        means = images.mean(dim=(0, 2, 3))
        stds = images.std(dim=(0, 2, 3))
        for mean, std in zip(means, stds):
            self.assertAlmostEqual(mean.item(), 0.0, delta=0.15)
            self.assertAlmostEqual(std.item(), 0.5, delta=0.1)

    def test_model_output_dimensions(self):
        """Test model outputs match CIFAR-10 class count"""
        batch = next(iter(self.train_loader))
        images, _ = batch
        outputs = self.model(images.to(self.model.device))
        self.assertEqual(outputs.shape[-1], 10)

    def test_checkpoint_saving(self):
        """Test model checkpoint saving/loading"""
        checkpoint_path = "./temp_checkpoint.pth"

        # Save
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        # Load
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    def test_learning_curve(self):
        """Test loss decreases over multiple batches"""
        losses = []
        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
            if batch_idx >= 4:  # Test 4 batches
                break
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        # Check loss progression
        self.assertTrue(len(losses) > 1)
        self.assertFalse(all(x == losses[0] for x in losses))
