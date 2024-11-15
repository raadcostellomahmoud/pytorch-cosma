import torch
import torch.nn as nn

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
        latent= None

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
