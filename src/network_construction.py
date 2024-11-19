import torch
import torch.nn as nn
from torch import optim

import src.layers as layers
from src.layers import BaseAutoencoder


class BaseModel(nn.Module):
    def __init__(self, config, device=None):
        super(BaseModel, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleDict()  # Store layers in a dict by name
        self.config = config
        self.build_network()
        self.to(self.device)  # Move the model to the specified device

    def build_network(self):
        for layer_conf in self.config['layers']:
            layer_name = layer_conf['name']
            layer = self.create_layer(layer_conf)  # Create layer based on config
            self.layers[layer_name] = layer  # Store layer by its name

    def create_layer(self, layer_conf):
        layer_type = layer_conf.pop('type')  # Remove 'type' so it's not passed to the layer constructor
        layer_class_name = f"{layer_type}"  # Derive the class name from layer type

        # Fetch the layer class from the layers module or nn module
        try:
            valid_layer_conf = layer_conf
            # Try to get the class from your custom layers module
            layer_class = getattr(layers, layer_class_name)
        except AttributeError:
            # Remove non-argument fields like 'input' and 'output'
            valid_layer_conf = {k: v for k, v in layer_conf.items() if k not in ['input', 'output', 'name']}

            # If not found in custom layers, try PyTorch's nn module (e.g., ReLU, Dropout)
            layer_class = getattr(nn, layer_type, None)

        if layer_class is None:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # Instantiate the layer with the remaining config attributes
        return layer_class(**valid_layer_conf) if issubclass(layer_class, nn.Module) else layer_class()

    def forward(self, x, return_latent=False):
        x = x.to(self.device)  # Ensure input is on the same device as the model
        outputs = {'input': x}  # Track all outputs by name, starting with the input tensor
        latent = None

        # Iterate through the layers defined in the config
        for layer_conf in self.config['layers']:
            layer_name = layer_conf['name']
            layer = self.layers[layer_name]  # Retrieve the layer object by name

            # Get inputs for this layer (can be a list or a single tensor)
            input_names = layer_conf['input']

            # Case 1: Single input
            if isinstance(input_names, str):  # Single input tensor
                inputs = outputs[input_names]

            # Case 2: Multiple inputs (e.g., for AddLayer)
            elif isinstance(input_names, list):  # List of input tensors
                inputs = [outputs[name] for name in input_names]

            # Apply the layer: Unpack inputs for multi-input layers, pass as is for single-input layers
            outputs[layer_conf['output']] = layer(inputs)  # Pass as is for single-input layers

            if isinstance(layer, BaseAutoencoder):
                latent = layer(inputs, return_latent)

            if return_latent and latent is not None:
                return latent

        # Return the final output from the last layer
        return outputs[self.config['layers'][-1]['output']]

    def export_onnx(self, file_path, input_shape=(1, 1, 28, 28), opset_version=13):
        """
        Exports the model to ONNX format.

        Parameters:
        - file_path: Path to save the ONNX file.
        - input_shape: The shape of the input tensor (batch size, channels, height, width).
        - opset_version: The ONNX opset version to use.
        """
        # Generate a dummy input tensor with the specified shape
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Export the model
        torch.onnx.export(self, dummy_input, file_path, opset_version=opset_version, input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"Model exported to {file_path}")

    def train_model(self,train_loader, loss_function, optimizer, classification_loss_weight: float=0.2, use_reconstruction: bool=False, epochs: int=10):
        self.to(self.device)
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for data in train_loader:
                # Handle paired inputs if required
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

    def evaluate(self, test_loader, loss_function):
        """
        Evaluates the model on the test set.

        Returns:
            float: Accuracy for classification models
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
    def __init__(self, config, device=None):
        """
        Initializes the ConjoinedNetwork.

        Args:
            config (dict): The YAML-based configuration for the network.
            device (torch.device, optional): The device on which the model will run.
                Defaults to CUDA if available, otherwise CPU.
        """
        super(ConjoinedNetwork, self).__init__(config, device)

    def forward(self, x1, x2, return_latent=False):
        """
        Forward pass for the Conjoined Network with optional latent space retrieval.

        Args:
            x1 (torch.Tensor): The first input tensor of shape (batch_size, channels, height, width).
            x2 (torch.Tensor): The second input tensor of shape (batch_size, channels, height, width).
            return_latent (bool, optional): If True, returns the latent space representations
                of both inputs instead of the final network output. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If `return_latent` is False, returns the final output tensor of shape (batch_size, 1).
        """
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        outputs1 = {'input': x1}
        outputs2 = {'input': x2}

        for layer_conf in self.config['layers']:
            layer_name = layer_conf['name']
            layer = self.layers[layer_name]

            # Handle first branch (input1)
            if isinstance(layer_conf['input'], str) and layer_conf['input'] in outputs1:
                inputs1 = outputs1[layer_conf['input']]
                outputs1[layer_conf['output']] = layer(inputs1)

            # Handle second branch (input2)
            if isinstance(layer_conf['input'], str) and layer_conf['input'] in outputs2:
                inputs2 = outputs2[layer_conf['input']]
                outputs2[layer_conf['output']] = layer(inputs2)

            # For layers that process both inputs (e.g., Subtract, Add, etc.)
            if isinstance(layer_conf['input'], list):
                inputs = [outputs1[layer_conf['input'][0]], outputs2[layer_conf['input'][1]]]
                outputs1[layer_conf['output']] = layer(inputs)

            # If the layer is a feature extractor (e.g., final shared encoder layer), capture latent space
            if return_latent and 'latent' in layer_name.lower():
                latent1 = outputs1[layer_conf['output']]
                latent2 = outputs2[layer_conf['output']]

                return latent1, latent2

        # Return the final output from the last layer
        return outputs1[self.config['layers'][-1]['output']]

    def get_latent_features(self, x):
        """
        Extracts the latent features from a single input using the shared feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Latent features of shape (batch_size, latent_dim).
        """
        x = x.to(self.device)
        outputs = {'input': x}

        for layer_conf in self.config['layers']:
            layer_name = layer_conf['name']
            layer = self.layers[layer_name]

            # Skip layers that require dual inputs (e.g., Subtract, Add)
            if isinstance(layer_conf['input'], list):
                continue

            # Process layers for a single input
            input_name = layer_conf['input']
            inputs = outputs[input_name]
            outputs[layer_conf['output']] = layer(inputs)

            # Return the latent features when the layer name matches the feature extraction layer
            if 'latent' in layer_name.lower():
                return outputs[layer_conf['output']]

        raise ValueError("No latent layer found in the network configuration.")

    def train_model(self,train_loader, optimizer, loss_function, epochs=1):
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

    def evaluate(self, test_loader, loss_function):
        """
        Evaluates the model on the test set.

        Returns:
            float: Similarity score for Siamese networks.
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
