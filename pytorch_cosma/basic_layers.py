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
    
class EdgeScorer(nn.Module):
    """Predicts edge scores from node embeddings and edge_index."""
    def __init__(self, in_features, hidden_dim=128, **kwargs):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        x, edge_index = inputs
        source = x[edge_index[0]]
        target = x[edge_index[1]]
        return self.edge_mlp(torch.cat([source, target], dim=-1)).squeeze()