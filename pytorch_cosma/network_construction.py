from typing import Optional, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
from onnx.backend.base import DeviceType
from torch import Tensor, combinations
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Data as GeomData
import torch.nn.utils.prune as prune

import logging

from pytorch_cosma.autoencoders import BaseAutoencoder
from pytorch_cosma.model_yaml_parser import PruneConfig

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    A base model class for building neural networks from YAML configurations.

    Args:
        config (dict): YAML-based configuration defining the network architecture. Examples can be found in configs folder.
        device (torch.device, optional): The device on which the model will run.
            Defaults to CUDA if available, otherwise CPU.
        use_reconstruction (bool, optional): If True, computes reconstruction loss alongside classification.
                Defaults to False.
    """

    def __init__(self, config: dict, device: DeviceType = None, use_reconstruction: bool = False, verbose: bool = False) -> None:

        super(BaseModel, self).__init__()
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleDict()  # Store layers in a dict by name
        self.functional_layers = {}
        self.config = config
        self.verbose = verbose
        self.build_network()
        self.to(self.device)  # Move the model to the specified device
        self.use_reconstruction = use_reconstruction

    def build_network(self) -> None:
        """
        Builds the network architecture from the configuration.

        Parses the configuration to create layers and store them in the ModuleDict.
        """
        for layer_conf in self.config["layers"]:
            layer_name = layer_conf["name"]
            # Create layer based on config
            layer = self.create_layer(layer_conf)

            # Check if the layer is functional (not an nn.Module)
            if isinstance(layer, nn.Module):
                self.layers[layer_name] = layer  # Store in ModuleDict
            else:
                # Store in functional_layers
                self.functional_layers[layer_name] = layer

    def create_layer(self, layer_conf: dict) -> nn.Module | Callable:
        """
        Creates a layer based on the configuration.

        Args:
            layer_conf (dict): A dictionary containing the layer type and parameters.

        Returns:
            nn.Module: The created layer.

        Raises:
            ValueError: If the layer type is not recognized.
        """
        layer_type = layer_conf.pop(
            "type")  # Remove "type" so it"s not passed to the layer constructor

        # First, try to fetch the layer from torch.nn
        layer_class = getattr(nn, layer_type, None)

        # Instantiate the layer with the remaining configuration
        valid_layer_conf = {k: v for k, v in layer_conf.items() if k not in [
            "input", "output", "name"]}

        # If not found in torch.nn, check in torch_geometric.nn
        if layer_class is None:
            import torch_geometric.nn as gnn
            layer_class = getattr(gnn, layer_type, None)

        # If still not found, check in utils
        if layer_class is None:
            from pytorch_cosma import utils
            layer_class = getattr(utils, layer_type, None)

        # If still not found, check in basic_layers
        if layer_class is None:
            from pytorch_cosma import basic_layers
            layer_class = getattr(basic_layers, layer_type, None)

        # If still not found, check in autoencoders
        if layer_class is None:
            import pytorch_cosma.autoencoders as autoencoders
            layer_class = getattr(autoencoders, layer_type, None)

        # If still not found, check in vision_transformer
        if layer_class is None:
            import pytorch_cosma.vision_transformer as vision_transformer
            layer_class = getattr(vision_transformer, layer_type, None)

        # If still not found, check in torchvision.models for ConvNeXt
        if layer_class is None:
            import torchvision.models.convnext as convnext
            layer_class = getattr(convnext, layer_type, None)
            if layer_class is not None:
                block_setting = valid_layer_conf.pop("block_setting")
                valid_layer_conf["block_setting"] = [
                    convnext.CNBlockConfig(**block) for block in block_setting]

        if layer_class is None:
            raise ValueError(f"Unknown layer type: {layer_type}")

        try:
            if self.verbose:
                print("Layer type:", layer_class)
            if issubclass(layer_class, nn.Module):
                return layer_class(**valid_layer_conf)
        except TypeError:
            if callable(layer_class):  # For functional utilities
                return lambda *args, **kwargs: layer_class(*args, **{**valid_layer_conf, **kwargs})
            else:
                raise ValueError(f"Cannot instantiate layer: {layer_type}")

    def _core_iterate_through_layers(self, layer_conf, outputs):
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

        # Apply the layer
        try:
            # Handle layers with multiple inputs
            try:
                if self.verbose:
                    print(f"Layer {layer_name} input shape: {inputs.shape}")
            except:
                # Handle layers that expect unpacked inputs
                for i, input_tensor in enumerate(inputs):
                    if self.verbose:
                        print(
                            f"Layer {layer_name} input {i} shape: {input_tensor.shape}")
            outputs[layer_conf["output"]] = layer(inputs)

        except TypeError:
            outputs[layer_conf["output"]] = layer(*inputs)

        try:
            if self.verbose:
                print(
                    f"Layer {layer_name} output shape: {outputs[layer_conf['output']].shape}")
        except:
            # Handle layers that produce multiple outputs
            for i, output_tensor in enumerate(outputs[layer_conf["output"]]):
                if self.verbose:
                    print(
                        f"Layer {layer_name} output {i} shape: {output_tensor.shape}")
        return inputs, layer, outputs

    def forward(self, x: Tensor, return_latent: bool = False, to_layer: int = -1) -> Tensor:
        # Ensure input is on the same device as the model
        x = x.to(self.device)
        # Track all outputs by name, starting with the input tensor
        outputs = {"input": x}
        latent = None

        # Iterate through the layers defined in the config
        for layer_conf in self.config["layers"]:
            inputs, layer, outputs = self._core_iterate_through_layers(
                layer_conf, outputs)

            if isinstance(layer, BaseAutoencoder):
                latent = layer(inputs, return_latent)

            if return_latent and latent is not None:
                return latent

        # Return the final output from the last layer
        return outputs[self.config["layers"][to_layer]["output"]]

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

    def _perturbed_backprop(self, loss: Tensor, optimizer: Optimizer):
        """
        Performs backpropagation with a small perturbation added to the gradients of biases.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        """
        if not isinstance(loss, Tensor):
            raise TypeError("Expected loss to be an instance of torch.Tensor")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "Expected optimizer to be an instance of torch.optim.Optimizer")

        loss.backward()

        # Safely perturb gradients of biases
        for name, layer in self.layers.items():
            if hasattr(layer, 'bias') and layer.bias is not None:
                if layer.bias.grad is not None:
                    layer.bias.grad.data.add_(
                        torch.randn_like(layer.bias.grad) * 1e-6)
                else:
                    print(
                        f"Warning: Gradient for bias in layer '{name}' is None.")

        optimizer.step()

    def train_model(self, train_loader: DataLoader, loss_function: _Loss, optimizer: Optimizer,
                    classification_loss_weight: float = 0.2, epochs: int = 10) -> None:
        """
        Trains the model using the given training data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            loss_function (callable): Loss function to use.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            classification_loss_weight (float, optional): Weight for classification loss when combined
                with reconstruction loss. Defaults to 0.2.
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
        """
        self.to(self.device)
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
                if self.use_reconstruction:
                    reconstructed, logits = self.forward(inputs)
                else:
                    logits = self.forward(inputs)

                # Compute loss
                if self.use_reconstruction:
                    reconstruction_loss = loss_function(reconstructed, inputs)
                    classification_loss = nn.functional.cross_entropy(
                        logits, labels)
                    loss = reconstruction_loss + classification_loss_weight * classification_loss
                else:
                    loss = loss_function(logits, labels)

                self._perturbed_backprop(loss, optimizer)

                # Metrics
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

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
    

    def prune_model(self, prune_config: PruneConfig):
        """Applies pruning to the model based on configuration"""
        parameters_to_prune = []
        
        # Identify prunable layers
        for name, module in self.named_modules():
            if any(isinstance(module, getattr(nn, layer_type)) for layer_type in prune_config.layers_to_prune):
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            raise ValueError("No prunable layers found matching specified types")

        # Select pruning method
        pruning_method = {
            'l1_unstructured': prune.L1Unstructured,
            'random_unstructured': prune.RandomUnstructured,
            'ln_structured': prune.LnStructured
        }.get(prune_config.method)

        if prune_config.global_pruning:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=pruning_method,
                amount=prune_config.amount
            )
        else:
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)  # Remove existing pruning
                pruning_method(module, name=param_name, amount=prune_config.amount)

    def make_pruning_permanent(self):
        """Converts temporary pruning masks to permanent parameter changes"""
        for name, module in self.named_modules():
            if prune.is_pruned(module):
                prune.remove(module, 'weight')

    def get_sparsity_stats(self) -> dict:
        """Returns dictionary of sparsity statistics"""
        stats = {'total_params': 0, 'pruned_params': 0}
        for name, module in self.named_modules():
            if prune.is_pruned(module):
                if hasattr(module, 'weight'):
                    stats['total_params'] += module.weight.nelement()
                    stats['pruned_params'] += torch.sum(module.weight == 0)
                    
        stats['sparsity_percent'] = 100 * stats['pruned_params'] / stats['total_params']
        return stats


class TwinNetwork(BaseModel):
    """
    A network designed for Siamese-style architectures to compare two inputs.

    Inherits from BaseModel and supports feature extraction and similarity comparison.

    Args:
        config (dict): The YAML-based configuration for the network.
        device (torch.device, optional): The device on which the model will run.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(self, config: dict, device: DeviceType = None) -> None:
        super(TwinNetwork, self).__init__(config, device)

    def forward(self, x1: Tensor, x2: Tensor, return_latent: bool = False) -> Tensor | list[Tensor]:
        """
        Forward pass for the Twin Network with optional latent space retrieval.

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
                inputs = [outputs1[layer_conf["input"][0]],
                          outputs2[layer_conf["input"][1]]]
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

    def train_model(self, train_loader: DataLoader, optimizer: Optimizer, loss_function: _Loss, epochs: int = 1) -> None:
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
                x1, x2, labels = x1.to(self.device), x2.to(
                    self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self.forward(x1, x2)

                # Compute loss
                loss = loss_function(logits.squeeze(), labels)
                self._perturbed_backprop(loss, optimizer)

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
                x1, x2, labels = x1.to(self.device), x2.to(
                    self.device), labels.to(self.device)
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

    def forward(self, data: GeomData, edges_nodes_or_both: str = 'nodes') -> Dict[str, Tensor]:
        outputs = {
            "x": data.x.to(self.device),
            "edge_index": data.edge_index.to(self.device),
            "batch": getattr(data, "batch", None)
        }

        # Iterate through the layers defined in the config
        for layer_conf in self.config["layers"]:
            _, _, outputs = self._core_iterate_through_layers(
                layer_conf, outputs)

        # Collect specified outputs
        model_outputs = {}
        for layer_conf in self.config["layers"]:
            output_name = layer_conf["output"]
            if output_name.startswith("node_output"):
                model_outputs[output_name] = outputs[output_name]
            elif output_name.startswith("edge_output"):
                model_outputs[output_name] = outputs[output_name].squeeze(-1)
        return model_outputs

    def train_model(
            self,
            train_loader: DataLoader,
            loss_functions: Dict[str, _Loss],
            optimizer: Optimizer,
            metrics: Optional[Dict[str, Callable[[
                torch.Tensor, torch.Tensor], float]]] = None,
            label_mapping: dict = None,  # Map task names to dataset attributes
            epochs: int = 10,
            **kwargs
    ) -> None:
        """
        Trains the graph model using the given training data.

        Args:
            train_loader (DataLoader): DataLoader for graph data.
            loss_functions (Dict[str, _Loss]): Dictionary mapping task names to loss functions.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            metrics (Dict[str, Callable], optional): Dictionary mapping task names to metric functions.
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
        """
        self.to(self.device)
        for epoch in range(epochs):
            self.train()
            task_metrics = {task: 0.0 for task in metrics} if metrics else {}
            running_loss = {task: 0.0 for task in loss_functions}

            for data in train_loader:
                data = data.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(data, **kwargs)

                # Compute loss for each task
                losses = []
                for task, loss_fn in loss_functions.items():
                    label_attr = label_mapping.get(task,
                                                   task) if label_mapping else task  # Default to task name if mapping is not provided
                    labels = getattr(data, label_attr).to(self.device)
                    preds = outputs[task] if type(outputs) is dict else outputs
                    task_loss = loss_fn(preds, labels)
                    losses.append(task_loss)

                    # Calculate metrics
                    if metrics and task in metrics:
                        task_metrics[task] += metrics[task](preds, labels)

                loss = sum(losses)
                self._perturbed_backprop(loss, optimizer)

                for task in loss_functions:
                    label_attr = label_mapping.get(task,
                                                   task) if label_mapping else task  # Default to task name if mapping is not provided
                    labels = getattr(data, label_attr).to(self.device)
                    running_loss[task] += loss_functions[task](
                        outputs[task], labels).item()

            avg_loss = {
                task: running_loss[task] / len(train_loader) for task in loss_functions}
            avg_metrics = (
                {task: score / len(train_loader)
                 for task, score in task_metrics.items()}
                if metrics
                else None
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss}, Metrics: {avg_metrics}")

    def evaluate(
            self,
            test_loader: DataLoader,
            loss_functions: Dict[str, _Loss],
            metrics: Optional[Dict[str, Callable[[
                torch.Tensor, torch.Tensor], float]]] = None,
            label_mapping: dict = None,  # Map task names to dataset attributes
            **kwargs
    ) -> dict[str, dict[str, float] | None]:
        self.eval()
        running_loss = {task: 0.0 for task in loss_functions}
        task_metrics = {task: 0.0 for task in metrics} if metrics else {}

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                outputs = self.forward(data, **kwargs)

                for task, loss_fn in loss_functions.items():
                    label_attr = label_mapping.get(task,
                                                   task) if label_mapping else task  # Default to task name if mapping is not provided
                    labels = getattr(data, label_attr).to(self.device)
                    preds = outputs[task] if type(outputs) is dict else outputs
                    running_loss[task] += loss_fn(preds, labels).item()

                    if metrics and task in metrics:
                        task_metrics[task] += metrics[task](preds, labels)

        avg_loss = {task: running_loss[task] /
                    len(test_loader) for task in loss_functions}
        avg_metrics = (
            {task: score / len(test_loader)
             for task, score in task_metrics.items()}
            if metrics
            else None
        )
        print(f"Test Results - Loss: {avg_loss}, Metrics: {avg_metrics}")
        return {"loss": avg_loss, "metrics": avg_metrics}


class MultiModalGATModel(BaseModel):
    """
    Handles multi-modal inputs (images + sequences) and GAT-specific logic.
    """

    def __init__(self, config: dict, device=None, **kwargs):
        super().__init__(config, device, **kwargs)

    def forward(self, data: GeomData) -> Dict[str, Tensor]:
        # Process inputs through layers defined in YAML
        outputs = {
            "x_images": data.x_images.float().to(self.device),
            "x_one_hot": data.x_one_hot.float().to(self.device)
        }

        gat_phase = False
        # Run all layers from the YAML config
        for layer_conf in self.config["layers"]:
            if not gat_phase:
                _, _, outputs = self._core_iterate_through_layers(
                    layer_conf, outputs)
                if layer_conf.get("name") == 'mode_concat':

                    # Generate edge_index dynamically (all-to-all connectivity)
                    # Assume fusion layer outputs "x_combined"
                    num_nodes = outputs["x_combined"].shape[0]
                    edge_index = combinations(
                        torch.arange(num_nodes)).t().to(self.device)

                    # Apply GAT layers (defined in YAML)
                    gat_outputs = {
                        "x_combined": outputs["x_combined"],
                        "edge_index": edge_index
                    }

                    gat_phase = True
            else:
                _, _, gat_outputs = self._core_iterate_through_layers(
                    layer_conf, gat_outputs)

        return gat_outputs["node_output"], gat_outputs["edge_output"], edge_index

    def train_model(self, train_loader, optimizer, scheduler, criterion_node, criterion_edge, device,
                        num_epochs, edge_loss_weighting, epoch_report_interval, just_infer_connected: bool, val_loader=None,
                        early_stopping: bool = True, early_stopping_window: int = 5):
        """
        Train the GATModelWithTransformerAndCNN model with node and edge loss handling and log metrics to wandb.

        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            scheduler: Learning rate scheduler.
            criterion_node: Loss function for node predictions.
            criterion_edge: Loss function for edge predictions.
            device: Device to run the training on (e.g., 'cuda' or 'cpu').
            num_epochs: Number of training epochs.
            edge_loss_weighting: Weighting factor for edge loss.
            epoch_report_interval: Interval for reporting loss and learning rate.

        Returns:
            None
        """

        self.train()
        trait_losses = []
        edge_losses = []
        val_trait_losses = []
        val_edge_losses = []

        for epoch in range(num_epochs):
            total_trait_loss = torch.zeros(train_loader.dataset[0].y.size(1), device=device)
            total_edge_loss = 0.0
            num_nonzero_edge_samples = 0
            num_zero_edge_samples = 0
            num_node_samples = torch.zeros(train_loader.dataset[0].y.size(1), device=device)

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                # Forward pass
                node_pred, edge_pred, edge_index = self.forward(data)

                # Node loss
                mask = (data.y != -1)
                masked_node_pred = node_pred[mask]
                masked_node_truth = data.y[mask]
                node_loss = criterion_node(masked_node_pred, masked_node_truth)

                # Update per-trait loss
                for trait_idx in range(data.y.size(1)):
                    trait_mask = mask[:, trait_idx]
                    if trait_mask.sum() > 0:
                        trait_loss = criterion_node(node_pred[:, trait_idx][trait_mask], data.y[:, trait_idx][trait_mask])
                        total_trait_loss[trait_idx] += trait_loss.item() * trait_mask.sum().item()
                        num_node_samples[trait_idx] += trait_mask.sum().item()

                # Edge loss
                if just_infer_connected:
                    true_edge_map = {tuple(edge): max(0, min(value,1)) for edge, value in
                                    zip(data.edge_index.t().tolist(), data.edge_attr.tolist())}
                else:
                    true_edge_map = {tuple(edge): value for edge, value in
                                    zip(data.edge_index.t().tolist(), data.edge_attr.tolist())}
                true_edge_attr_values = [true_edge_map.get(tuple(edge), 0) for edge in edge_index.t().tolist()]
                true_edge_attr_values = torch.tensor(true_edge_attr_values, device=device, dtype=torch.float)

                nonzero_mask = true_edge_attr_values > 0
                zero_mask = ~nonzero_mask

                nonzero_true_edge_attr = true_edge_attr_values[nonzero_mask]
                nonzero_pred_edge_attr = edge_pred[nonzero_mask]
                zero_true_edge_attr = true_edge_attr_values[zero_mask]
                zero_pred_edge_attr = edge_pred[zero_mask]

                nonzero_edge_loss = criterion_edge(nonzero_pred_edge_attr, nonzero_true_edge_attr) * edge_loss_weighting if len(nonzero_true_edge_attr) > 0 else Tensor([0.]).to(device)
                zero_edge_loss = criterion_edge(zero_pred_edge_attr, zero_true_edge_attr) * edge_loss_weighting if len(zero_true_edge_attr) > 0 else Tensor([0.]).to(device)

                total_edge_loss_batch = (nonzero_edge_loss + zero_edge_loss) / 2
                total_edge_loss += total_edge_loss_batch.item() * len(edge_index.t())
                num_nonzero_edge_samples += nonzero_true_edge_attr.size(0)
                num_zero_edge_samples += zero_true_edge_attr.size(0)

                # Total loss
                loss = node_loss + total_edge_loss_batch
                loss.backward()
                optimizer.step()

            # Calculate average losses
            avg_node_loss = total_trait_loss / num_node_samples
            avg_edge_loss = total_edge_loss / (num_nonzero_edge_samples + num_zero_edge_samples)

            scheduler.step(avg_node_loss.mean().item() + avg_edge_loss)

            # Store losses for plotting
            trait_losses.append(avg_node_loss.mean().item())
            edge_losses.append(avg_edge_loss)

            # Log metrics to wandb
            current_lr = optimizer.param_groups[0]['lr']

            if val_loader:
                val_total_trait_loss = torch.zeros(val_loader.dataset[0].y.size(1), device=device)
                val_total_edge_loss = 0.0
                val_num_nonzero_edge_samples = 0
                val_num_zero_edge_samples = 0
                val_num_node_samples = torch.zeros(val_loader.dataset[0].y.size(1), device=device)

                for data in val_loader:
                    data = data.to(device)
                    optimizer.zero_grad()

                    # Forward pass
                    node_pred, edge_pred, edge_index = self.forward(data)

                    # Node loss
                    mask = (data.y != -1)
                    masked_node_pred = node_pred[mask]
                    masked_node_truth = data.y[mask]
                    node_loss = criterion_node(masked_node_pred, masked_node_truth)

                    # Update per-trait loss
                    for trait_idx in range(data.y.size(1)):
                        trait_mask = mask[:, trait_idx]
                        if trait_mask.sum() > 0:
                            trait_loss = criterion_node(node_pred[:, trait_idx][trait_mask],
                                                        data.y[:, trait_idx][trait_mask])
                            val_total_trait_loss[trait_idx] += trait_loss.item() * trait_mask.sum().item()
                            val_num_node_samples[trait_idx] += trait_mask.sum().item()

                    # Edge loss
                    if just_infer_connected:
                        true_edge_map = {tuple(edge): max(0, min(value, 1)) for edge, value in
                                        zip(data.edge_index.t().tolist(), data.edge_attr.tolist())}
                    else:
                        true_edge_map = {tuple(edge): value for edge, value in
                                        zip(data.edge_index.t().tolist(), data.edge_attr.tolist())}
                    true_edge_attr_values = [true_edge_map.get(tuple(edge), 0) for edge in edge_index.t().tolist()]
                    true_edge_attr_values = torch.tensor(true_edge_attr_values, device=device, dtype=torch.float)

                    nonzero_mask = true_edge_attr_values > 0
                    zero_mask = ~nonzero_mask

                    nonzero_true_edge_attr = true_edge_attr_values[nonzero_mask]
                    nonzero_pred_edge_attr = edge_pred[nonzero_mask]
                    zero_true_edge_attr = true_edge_attr_values[zero_mask]
                    zero_pred_edge_attr = edge_pred[zero_mask]

                    nonzero_edge_loss = criterion_edge(nonzero_pred_edge_attr, nonzero_true_edge_attr) * edge_loss_weighting
                    zero_edge_loss = criterion_edge(zero_pred_edge_attr, zero_true_edge_attr) * edge_loss_weighting

                    total_edge_loss_batch = (nonzero_edge_loss + zero_edge_loss) / 2
                    val_total_edge_loss += total_edge_loss_batch.item() * len(edge_index.t())
                    val_num_nonzero_edge_samples += nonzero_true_edge_attr.size(0)
                    val_num_zero_edge_samples += zero_true_edge_attr.size(0)

                # Calculate average losses
                val_avg_node_loss = val_total_trait_loss / val_num_node_samples
                val_avg_edge_loss = val_total_edge_loss / (val_num_nonzero_edge_samples + val_num_zero_edge_samples)

                # Store losses for plotting
                val_trait_losses.append(val_avg_node_loss.mean().item())
                val_edge_losses.append(val_avg_edge_loss)

            # Print progress at intervals
            if epoch % epoch_report_interval == 0:
                print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.8f} | "
                    f"All-trait average loss: {avg_node_loss.mean().item():.8f} | "
                    f"Edge Average Loss: {avg_edge_loss:.8f} | "
                    f"Combined total loss: {avg_node_loss.mean().item() + avg_edge_loss:.8f}")
                if val_loader:
                    print(f"| All-trait average validation loss: {val_avg_node_loss.mean().item():.8f} | "
                        f"Edge average validation loss: {val_avg_edge_loss:.8f} | "
                        f"Combined validation loss: {val_avg_node_loss.mean().item() + val_avg_edge_loss:.8f} |")

            if early_stopping and epoch > early_stopping_window:
                #combined_val_losses = np.array(val_trait_losses) + np.array(val_edge_losses)
                loss_deltas = np.array(val_trait_losses[1:]) - np.array(val_trait_losses[:-1])
                if np.median(loss_deltas[-early_stopping_window:]) >= 0:
                    print(
                        f"Early stopping triggered. Median loss_delta from last {early_stopping_window} epochs was {np.median(loss_deltas[-4:]):.8f} ")
                    break
