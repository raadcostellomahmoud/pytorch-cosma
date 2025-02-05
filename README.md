# PyTorch-Cosma

## Overview

PyTorch-Cosma is a deep learning framework built on top of PyTorch, designed to facilitate the creation, training, and visualization of neural networks. The framework supports various types of models, including convolutional autoencoders, graph neural networks, and vision transformers. It also provides utilities for latent space exploration and graph visualization.

## Project Structure

```
├── pytorch_cosma/
│   ├── config_validation.py
│   ├── autoencoders.py
│   ├── basic_layers.py
│   ├── utils.py
│   ├── vision_transformer.py
│   ├── graphs.py
│   ├── latent_space.py
│   ├── model_yaml_parser.py
│   ├── network_construction.py
│   └── twin_dataset_maker.py
├── configs/
│   ├── example_conv_autoencoder.yaml
│   ├── example_gatconv_network.yaml
│   └── ...
├── examples/
│   ├── mnist_autoencode_and_latent_inspection.py
│   └── ...
├── unit_testing/
│   └── examples/
│       └── test_mnist_autoencode_and_latent_inspection.py
│       └── ...
├── data/
├── README.md
├── .gitignore
├── .vscode/
│   ├── launch.json
│   └── settings.json
```

## Installation
1. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

You can either install the package directly from PyPI or clone the repository and install the dependencies manually:

### Option 1: Install from PyPI
2. Install the package:
    ```sh
    pip install pytorch-cosma
    ```

### Option 2: Clone the Repository
2. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/pytorch-cosma.git
    cd pytorch-cosma
    ```

3. Install the required dependencies:
    ```sh
    pip install .
    ```

## Usage

### Configuration

Model architectures are defined using YAML configuration files. Examples can be found in the `configs/` directory.

### Training a Model

To train a model, use the provided example scripts or create your own. Below is an example of training an autoencoder on the MNIST dataset:

```python
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.latent_space import LatentSpaceExplorer, Visualizer
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import BaseModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load configuration from YAML
raw_config = YamlParser("configs/example_conv_autoencoder.yaml").parse()

# Validate configuration
validated_config = ConfigModel(**raw_config).to_dict()

# Create model from configuration
model = BaseModel(validated_config, use_reconstruction=True, device=device)

# Train the model
model.train_model(train_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=1e-3), epochs=5)
```

### Latent Space Exploration

To explore the latent space of a trained model:

```python
# Latent space exploration
explorer = LatentSpaceExplorer(model, train_loader, device)
latent_points, labels_points, all_inputs = explorer.extract_latent_space()
reduced_dimensionality = explorer.reduce_dimensionality(latent_points)

# Randomly sample points for visualization
sample_size = 100
indices = np.random.choice(len(reduced_dimensionality), size=sample_size, replace=False)
reduced_dimensionality = reduced_dimensionality[indices]
selected_inputs = all_inputs[indices]

# Visualize latent space
visualizer = Visualizer(reduced_dimensionality, labels_points, selected_inputs)
hover_images = visualizer.generate_hover_images()
app = visualizer.create_dash_app(hover_images)
app.run_server(debug=True)
```

### Graph Visualization

To visualize a graph:

```python
import networkx as nx
import torch

from pytorch_cosma.graphs import GraphVisualizer

# Create a sample graph
G = nx.karate_club_graph()

# Generate random predictions and ground truth
node_predictions = torch.randint(0, 2, (len(G.nodes),))
node_ground_truth = torch.randint(0, 2, (len(G.nodes),))

# Initialize the visualizer
visualizer = GraphVisualizer(G, node_predictions, node_ground_truth, subset_size=10)

# Create and run the Dash app
app = visualizer.create_dash_app()
app.run_server(debug=True)
```

## Unit Testing

Unit tests are located in the `unit_testing/` directory. To run the tests:

```sh
python -m unittest discover unit_testing/
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the following libraries:
- [PyTorch](https://pytorch.org/)
- [Torch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Dash](https://dash.plotly.com/)
- [UMAP](https://umap-learn.readthedocs.io/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)

## Contact

For questions or suggestions, please open an issue or contact the repository owner at mahmoud.raad@yahoo.co.uk