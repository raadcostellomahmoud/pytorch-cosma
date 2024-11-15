import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.config_validation import ConfigModel
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Load configuration from YAML
raw_config = YamlParser(
    "configs/example_conv_autoencoder_1.yaml").parse()  # raw_config is a dictionary from the YAML file

# Validate configuration
try:
    validated_config = ConfigModel(**raw_config).to_dict()
except ValueError as e:
    print("Configuration validation failed:", e)
    exit(1)

# Initialize model, loss, and optimizer
model = BaseModel(validated_config).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, _ = batch  # We don't need labels for autoencoder training
        inputs = inputs.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = loss_function(outputs, inputs)  # Comparing reconstructed image with input image
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader)}")

print("Training complete.")

# Plotting original vs reconstructed images
model.eval()
with torch.no_grad():
    # Retrieve a single batch from the test set
    for inputs, _ in test_loader:
        inputs = inputs.to(device)

        # Get the reconstructed images from the autoencoder
        reconstructed = model(inputs)

        # Move tensors back to the CPU and detach them for visualization
        inputs = inputs.cpu().detach()
        reconstructed = reconstructed.cpu().detach()

        # Plot a few original and reconstructed images side by side
        num_images = 6
        fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
        for i in range(num_images):
            # Original image
            axes[0, i].imshow(inputs[i].squeeze(), cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # Reconstructed image
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.show()
        break  # Only visualize the first batch

# Inspect latent space for a single batch of images
model.eval()
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)

        # Pass return_latent=True for latent space retrieval
        latent_space = model(inputs, return_latent=True)

        # Print latent space shape and example latent vectors for inspection
        print(f"Latent space shape: {latent_space.shape}")
        print(f"Latent vectors for first 10 digits:\n {latent_space[:10]}")

        break  # Inspect only the first batch
