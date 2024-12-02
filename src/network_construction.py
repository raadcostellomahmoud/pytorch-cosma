import torch
import torch.nn as nn
from onnx.backend.base import DeviceType
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Data as GeomData

import src.layers as layers
from src.layers import BaseAutoencoder


class BaseModel(nn.Module):
    """
    A base model class for building neural networks from YAML configurations.

    Args:
        config (dict): YAML-based configuration defining the network architecture. Examples can be found in configs folder.
        device (torch.device, optional): The device on which the model will run.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(self, config: dict, device: DeviceType = None) -> None:
        super(BaseModel, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleDict()  # Store layers in a dict by name
        self.functional_layers = {}
        self.config = config
        self.build_network()
        self.to(self.device)  # Move the model to the specified device

    def build_network(self) -> None:
        """
        Builds the network architecture from the configuration.

        Parses the configuration to create layers and store them in the ModuleDict.
        """
        for layer_conf in self.config["layers"]:
            layer_name = layer_conf["name"]
            layer = self.create_layer(layer_conf)  # Create layer based on config

            # Check if the layer is functional (not an nn.Module)
            if isinstance(layer, nn.Module):
                self.layers[layer_name] = layer  # Store in ModuleDict
            else:
                self.functional_layers[layer_name] = layer  # Store in functional_layers

    @staticmethod
    def create_layer(layer_conf: dict) -> nn.Module:
        """
        Creates a layer based on the configuration.

        Args:
            layer_conf (dict): A dictionary containing the layer type and parameters.

        Returns:
            nn.Module: The created layer.

        Raises:
            ValueError: If the layer type is not recognized.
        """
        layer_type = layer_conf.pop("type")  # Remove "type" so it"s not passed to the layer constructor

        # First, try to fetch the layer from torch.nn
        layer_class = getattr(nn, layer_type, None)

        # If not found in torch.nn, check in torch_geometric.nn
        if layer_class is None:
            import torch_geometric.nn as gnn
            layer_class = getattr(gnn, layer_type, None)

        # If still not found, check in custom layers
        if layer_class is None:
            try:
                layer_class = getattr(layers, layer_type)
            except AttributeError:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # Instantiate the layer with the remaining configuration
        valid_layer_conf = {k: v for k, v in layer_conf.items() if k not in ["input", "output", "name"]}
        try:
            if issubclass(layer_class, nn.Module):
                return layer_class(**valid_layer_conf)
        except TypeError:
            if callable(layer_class):  # For functional utilities
                return lambda *args, **kwargs: layer_class(*args, **valid_layer_conf, **kwargs)
            else:
                raise ValueError(f"Cannot instantiate layer: {layer_type}")

    def _pre_forward(self, x: Tensor) -> dict[str, Tensor]:
        x = x.to(self.device)  # Ensure input is on the same device as the model
        outputs = {"input": x}  # Track all outputs by name, starting with the input tensor
        return outputs

    def _forward(self, inputs: dict, return_latent=False) -> Tensor:

        outputs = inputs
        latent = None

        # Iterate through the layers defined in the config
        for layer_conf in self.config["layers"]:
            layer_name = layer_conf["name"]
            try:
                layer = self.layers[layer_name]
            except KeyError:
                layer = self.functional_layers.get(layer_name)

            # Get inputs for this layer (can be a list or a single tensor)
            input_names = layer_conf["input"]

            # Case 1: Single input
            if isinstance(input_names, str):  # Single input tensor
                inputs = outputs[input_names]

            # Case 2: Multiple inputs (e.g., for AddLayer)
            elif isinstance(input_names, list):  # List of input tensors
                inputs = [outputs[name] for name in input_names]

            try:
                # Apply the layer: Unpack inputs for multi-input layers, pass as is for single-input layers
                outputs[layer_conf["output"]] = layer(inputs)
            except TypeError:
                outputs[layer_conf["output"]] = layer(*inputs)

            if isinstance(layer, BaseAutoencoder):
                latent = layer(inputs, return_latent)

            if return_latent and latent is not None:
                return latent

        # Return the final output from the last layer
        return outputs[self.config["layers"][-1]["output"]]

    def forward(self, x: Tensor, return_latent=False) -> Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor to the model.
            return_latent (bool, optional): If True, returns the latent space representation.
                Defaults to False.

        Returns:
            torch.Tensor: The output tensor of the model or the latent space representation.
        """
        inputs = self._pre_forward(x)
        return self._forward(inputs, return_latent=return_latent)

    def export_onnx(self, file_path: str, input_shape: tuple = (1, 1, 28, 28), opset_version: int = 13) -> None:
        """
        Exports the model to ONNX format.

        Args:
            file_path (str): Path to save the ONNX file.
            input_shape (tuple, optional): The shape of the input tensor
                (batch size, channels, height, width). Defaults to (1, 1, 28, 28).
            opset_version (int, optional): The ONNX opset version to use. Defaults to 13.
        """
        # Generate a dummy input tensor with the specified shape
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Export the model
        torch.onnx.export(self, dummy_input, file_path, opset_version=opset_version, input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
        print(f"Model exported to {file_path}")

    def train_model(self, train_loader: DataLoader, loss_function: _Loss, optimizer: Optimizer,
                    classification_loss_weight: float = 0.2,
                    use_reconstruction: bool = False, epochs: int = 10) -> None:
        """
        Trains the model using the given training data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            loss_function (callable): Loss function to use.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            classification_loss_weight (float, optional): Weight for classification loss when combined
                with reconstruction loss. Defaults to 0.2.
            use_reconstruction (bool, optional): If True, computes reconstruction loss alongside classification.
                Defaults to False.
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
        """
        self.to(self.device)
        self.use_reconstruction=use_reconstruction
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for data in train_loader:

                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                if use_reconstruction:
                    reconstructed, logits = self.forward(inputs)
                else:
                    logits = self.forward(inputs)

                # Compute loss
                if use_reconstruction:
                    reconstruction_loss = loss_function(reconstructed, inputs)
                    classification_loss = nn.functional.cross_entropy(logits, labels)
                    loss = reconstruction_loss + classification_loss_weight * classification_loss
                else:
                    loss = loss_function(logits, labels)

                loss.backward()
                optimizer.step()

                # Metrics
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate(self, test_loader: DataLoader, loss_function: _Loss) -> float:
        """
        Evaluates the model on the test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            loss_function (callable): Loss function to use.

        Returns:
            float: Accuracy for classification models.
        """
        self.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.use_reconstruction:
                    reconstructed, logits = self.forward(inputs)
                    loss = loss_function(reconstructed, inputs)
                else:
                    logits = self.forward(inputs)
                    loss = loss_function(logits, labels)

                running_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(test_loader)
        accuracy = 100 * correct / total
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy


class ConjoinedNetwork(BaseModel):
    """
    A network designed for Siamese-style architectures to compare two inputs.

    Inherits from BaseModel and supports feature extraction and similarity comparison.

    Args:
        config (dict): The YAML-based configuration for the network.
        device (torch.device, optional): The device on which the model will run.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(self, config: dict, device: DeviceType = None) -> None:
        super(ConjoinedNetwork, self).__init__(config, device)

    def forward(self, x1: Tensor, x2: Tensor, return_latent: bool = False) -> Tensor | list[Tensor]:
        """
        Forward pass for the Conjoined Network with optional latent space retrieval.

        Args:
            x1 (torch.Tensor): The first input tensor of shape (batch_size, channels, height, width).
            x2 (torch.Tensor): The second input tensor of shape (batch_size, channels, height, width).
            return_latent (bool, optional): If True, returns the latent space representations
                of both inputs instead of the final network output. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If `return_latent` is True, returns the latent space representations of both inputs as a tuple.
                - Otherwise, returns the final output tensor of shape (batch_size, 1).
        """
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        outputs1 = {"input": x1}
        outputs2 = {"input": x2}

        for layer_conf in self.config["layers"]:
            layer_name = layer_conf["name"]
            layer = self.layers[layer_name]

            # Handle first branch (input1)
            if isinstance(layer_conf["input"], str) and layer_conf["input"] in outputs1:
                inputs1 = outputs1[layer_conf["input"]]
                outputs1[layer_conf["output"]] = layer(inputs1)

            # Handle second branch (input2)
            if isinstance(layer_conf["input"], str) and layer_conf["input"] in outputs2:
                inputs2 = outputs2[layer_conf["input"]]
                outputs2[layer_conf["output"]] = layer(inputs2)

            # For layers that process both inputs (e.g., Subtract, Add, etc.)
            if isinstance(layer_conf["input"], list):
                inputs = [outputs1[layer_conf["input"][0]], outputs2[layer_conf["input"][1]]]
                outputs1[layer_conf["output"]] = layer(inputs)

            # If the layer is a feature extractor (e.g., final shared encoder layer), capture latent space
            if return_latent and "latent" in layer_name.lower():
                latent1 = outputs1[layer_conf["output"]]
                latent2 = outputs2[layer_conf["output"]]

                return latent1, latent2

        # Return the final output from the last layer
        return outputs1[self.config["layers"][-1]["output"]]

    def get_latent_features(self, x: Tensor) -> Tensor:
        """
        Extracts the latent features from a single input using the shared feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Latent features of shape (batch_size, latent_dim).
        """
        x = x.to(self.device)
        outputs = {"input": x}

        for layer_conf in self.config["layers"]:
            layer_name = layer_conf["name"]
            layer = self.layers[layer_name]

            # Skip layers that require dual inputs (e.g., Subtract, Add)
            if isinstance(layer_conf["input"], list):
                continue

            # Process layers for a single input
            input_name = layer_conf["input"]
            inputs = outputs[input_name]
            outputs[layer_conf["output"]] = layer(inputs)

            # Return the latent features when the layer name matches the feature extraction layer
            if "latent" in layer_name.lower():
                return outputs[layer_conf["output"]]

        raise ValueError("No latent layer found in the network configuration.")

    def train_model(self, train_loader: DataLoader, optimizer: Optimizer, loss_function: _Loss,
                    epochs: int = 1) -> None:
        """
        Trains the model using the given training data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_function (callable): Loss function to use.
            epochs (int, optional): Number of epochs to train for. Defaults to 1.
        """
        self.to(self.device)
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for data in train_loader:
                # Handle paired inputs
                x1, x2, labels = data
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self.forward(x1, x2)

                # Compute loss
                loss = loss_function(logits.squeeze(), labels)

                loss.backward()
                optimizer.step()

                # Metrics
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    def evaluate(self, test_loader: DataLoader, loss_function: _Loss) -> float:
        """
        Evaluates the model on the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            loss_function (callable): Loss function to use.

        Returns:
            float: Average test loss for the model.
        """
        self.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                x1, x2, labels = data
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                logits = self.forward(x1, x2)
                loss = loss_function(logits.squeeze(), labels)

                running_loss += loss.item()

        avg_loss = running_loss / len(test_loader)

        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss


class GraphModel(BaseModel):
    """
    A graph-specific model class for building and training graph neural networks.

    Inherits from BaseModel and extends functionality for graph-based inputs.

    Args:
        config (dict): YAML-based configuration defining the network architecture.
        device (torch.device, optional): The device on which the model will run.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(self, config: dict, device: torch.device = None) -> None:
        super(GraphModel, self).__init__(config, device)

    def _pre_forward(self, data: GeomData) -> dict[str, Tensor]:
        outputs = {"x": data.x.to(self.device), "edge_index": data.edge_index.to(self.device), "batch": getattr(data, "batch", None)}
        return outputs

    def forward(self, data: GeomData) -> Tensor:
        inputs = self._pre_forward(data)
        return super()._forward(inputs)

    def train_model(
            self,
            train_loader: DataLoader,
            loss_function: _Loss,
            optimizer: Optimizer,
            epochs: int = 10,
    ) -> None:
        """
        Trains the graph model using the given training data.

        Args:
            train_loader (DataLoader): DataLoader for graph data.
            loss_function (callable): Loss function to use.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
        """
        self.to(self.device)
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for data in train_loader:
                data = data.to(self.device)
                labels = data.y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self.forward(data)

                # Compute loss
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                running_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate(self, test_loader: DataLoader, loss_function: _Loss) -> float:
        """
        Evaluates the graph model on the test data.

        Args:
            test_loader (DataLoader): DataLoader for graph data.
            loss_function (callable): Loss function to use.

        Returns:
            float: Accuracy for the test dataset.
        """
        self.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                labels = data.y.to(self.device)

                # Forward pass
                logits = self.forward(data)
                loss = loss_function(logits, labels)

                running_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(test_loader)
        accuracy = 100 * correct / total
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy
