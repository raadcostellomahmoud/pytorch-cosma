import torch
import torch.nn as nn
from torch import Tensor

class Add(nn.Module):
    """
    Performs element-wise addition of multiple tensors.

    Args:
        **kwargs: Additional arguments (not used, but included for compatibility).
    """

    def __init__(self, **kwargs):
        super(Add, self).__init__()

    @staticmethod
    def forward(inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Add layer.

        Args:
            inputs (List[torch.Tensor]): A list of tensors to be added element-wise.

        Returns:
            torch.Tensor: The resulting tensor after element-wise addition.
        """
        return sum(inputs)


class Subtract(nn.Module):
    """
    Computes the absolute difference between two tensors.

    Args:
        **kwargs: Additional arguments (not used, but included for compatibility).
    """

    def __init__(self, **kwargs):
        super(Subtract, self).__init__()

    @staticmethod
    def forward(inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Subtract layer.

        Args:
            inputs (List[torch.Tensor]): A list containing two tensors.

        Returns:
            torch.Tensor: The resulting tensor after computing the absolute difference.
        """
        input1, input2 = inputs
        return torch.abs(input1 - input2)

class Permute(nn.Module):
    """Permutes input tensor dimensions"""
    def __init__(self, dims: list):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)
    
class Concat(nn.Module):
    """Concatenates inputs along a dimension."""
    def __init__(self, dim=-1, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: list[Tensor]) -> Tensor:
        return torch.cat(inputs, dim=self.dim)
    
class ReshapeModule(nn.Module):
    """Reshapes input tensor to specified shape."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, input: Tensor) -> Tensor:
        return torch.reshape(input, self.shape)
    
class PositionalEncodingCosma(nn.Module):
    """
    Positional encoding for Transformer-based sequence input.
    """

    def __init__(self, d_model=64, max_len=500):
        super(PositionalEncodingCosma, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added.
        """
        input = input + self.pe[:, :input.size(1), :]
        return input

class TransformerEncoderModule(nn.TransformerEncoder):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout_prob):
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_prob, batch_first=True
        )
        super().__init__(transformer_layer, num_layers=num_layers)
        
class MaxMeanPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Tensor) -> Tensor:
        max_pooled = input.max(dim=self.dim)[0]
        mean_pooled = input.mean(dim=self.dim)

        # Concatenate
        representation = torch.cat([mean_pooled, max_pooled], dim=-1)
        representation = torch.sigmoid(representation)
        return representation
    
class EdgeIndexToFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs: list[Tensor]) -> Tensor:
        x_gat_fin, edge_index = inputs
        source_indices, target_indices = edge_index
        source_embeddings = x_gat_fin[source_indices]
        target_embeddings = x_gat_fin[target_indices]
        edge_features = torch.cat([source_embeddings, target_embeddings], dim=-1)
        return edge_features

class SqueezeLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: Tensor) -> Tensor:
        return input.squeeze()