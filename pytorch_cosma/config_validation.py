import torch.nn as nn
from collections import defaultdict
from typing import List, Union
from pytorch_cosma.network_construction import BaseModel, TwinNetwork, GraphModel  # Import valid model classes

# Base Layer Config
class BaseLayerConfig:
    def __init__(self, name: str, input: Union[str, List[str]], output: str):
        self.name = name
        self.input = input
        self.output = output


# Custom Add Layer Config (since this is the only non-PyTorch layer)
class AddLayerConfig(BaseLayerConfig):
    def __init__(self, name: str, input: List[str], output: str):
        if len(input) < 2:
            raise ValueError(f"Add layer '{name}' requires at least two inputs.")
        super().__init__(name, input, output)
        self.type = 'Add'


# Class to handle configuration and validation
class ConfigModel:
    def __init__(self, model_class: str, layers: List[dict]):
        self.model_class = model_class
        self.layers = layers
        
        # Validate the parsed layers
        self.graph_specific_inputs = {"x", "edge_index", "batch"}  # GNN-specific inputs
        self.validate()

    def validate(self):
        self.validate_model_class()  # Validate the model class
        self.ensure_unique_layer_names()

        # Validate that each layer's input is defined earlier as an output
        self.validate_named_inputs_outputs()

        # Validate that the 'Add' layers have multiple inputs
        self.validate_add_layer_inputs()

        # Validate that there are no cycles in the layer dependencies
        self.detect_cycles()

    def validate_model_class(self):
        valid_classes = {"BaseModel": BaseModel, "TwinNetwork": TwinNetwork, "GraphModel": GraphModel}
        if self.model_class not in valid_classes:
            raise ValueError(f"Invalid model class '{self.model_class}'. Valid options are: {list(valid_classes.keys())}")

    def ensure_unique_layer_names(self):
        layer_names = [layer['name'] for layer in self.layers]
        if len(layer_names) != len(set(layer_names)):
            raise ValueError("Layer names must be unique.")

    def validate_named_inputs_outputs(self):
        output_names = set(["input"])  # Start with 'input' as the initial input
        output_names.update(self.graph_specific_inputs)
        for layer in self.layers:
            input_names = layer['input'] if isinstance(layer['input'], list) else [layer['input']]
            for input_name in input_names:
                if input_name not in output_names:
                    raise ValueError(f"Layer '{layer['name']}' references undefined input '{input_name}'.")
            output_names.add(layer['output'])

    def validate_add_layer_inputs(self):
        # Ensure that Add layer has at least two inputs
        for layer in self.layers:
            if layer['type'] == 'Add':
                if isinstance(layer['input'], list) and len(layer['input']) < 2:
                    raise ValueError(f"Add layer '{layer['name']}' requires at least two inputs.")
                if not isinstance(layer['input'], list):
                    raise ValueError(f"Add layer '{layer['name']}' expects 'input' to be a list of inputs.")

    def detect_cycles(self):
        graph = defaultdict(list)

        # Build the graph from layer inputs and outputs
        for layer in self.layers:
            inputs = layer['input'] if isinstance(layer['input'], list) else [layer['input']]
            for inp in inputs:
                graph[inp].append(layer['name'])

        # Use a list of nodes to iterate, so graph is not modified during traversal
        def dfs(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        visited = set()
        all_nodes = list(graph.keys())  # Capture all nodes to iterate over, avoiding runtime changes
        for node in all_nodes:
            if node not in visited:
                if dfs(node, visited, set()):
                    raise ValueError("Cycle detected in layer dependencies.")

    def to_dict(self):
        return {"model_class": self.model_class, "layers": self.layers}