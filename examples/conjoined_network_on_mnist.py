import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random

from src.config_validation import ConfigModel
from src.latent_space import LatentSpaceExplorer, Visualizer
from src.model_yaml_parser import YamlParser
from src.network_construction import ConjoinedNetwork

# Define device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)

class SiameseMNISTDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset (torchvision.datasets): The MNIST dataset or any dataset with labels.
        """
        self.dataset = dataset

        # Precompute indices grouped by label
        self.label_to_indices = {label: [] for label in range(10)}  # Assuming 10 classes (0-9)
        for idx, (_, label) in enumerate(dataset):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Generate a Siamese pair.

        Args:
            idx (int): Index of the primary image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (img1, img2, is_same)
        """
        img1, label1 = self.dataset[idx]

        # Randomly decide if we want a "same" or "different" pair
        is_same = random.choice([0, 1])
        if is_same:
            # Choose a random index with the same label
            idx2 = random.choice(self.label_to_indices[label1])
        else:
            # Choose a random label that's different
            different_label = random.choice([label for label in self.label_to_indices if label != label1])
            idx2 = random.choice(self.label_to_indices[different_label])

        img2, label2 = self.dataset[idx2]

        return (img1, img2, torch.tensor(is_same, dtype=torch.float32))



# Prepare Siamese datasets and loaders
train_dataset = SiameseMNISTDataset(mnist_train)
test_dataset = SiameseMNISTDataset(mnist_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load configuration from YAML
raw_config = YamlParser("configs/example_conjoined_network.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()

# Initialize the model
model = ConjoinedNetwork(validated_config).to(device)

# Loss function and optimizer
loss_function = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train_model(epochs=1, train_loader=train_loader, optimizer=optimizer, loss_function=loss_function)
model.evaluate(test_loader=test_loader, loss_function=loss_function)

individ_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# Latent space exploration
explorer = LatentSpaceExplorer(model, individ_train_loader, device)
latent_points, labels_points, all_inputs = explorer.extract_latent_space(conjoined=True)
reduced_dimensionality = explorer.reduce_dimensionality(latent_points)

# Randomly sample points for visualization
sample_size = 100
indices = np.random.choice(len(reduced_dimensionality), size=sample_size, replace=False)
reduced_dimensionality = reduced_dimensionality[indices]
labels_points = labels_points[indices]
selected_inputs = all_inputs[indices]

# Visualize
visualizer = Visualizer(reduced_dimensionality, labels_points, selected_inputs)
hover_images = visualizer.generate_hover_images()
app = visualizer.create_dash_app(hover_images)
app.run_server(debug=True)
