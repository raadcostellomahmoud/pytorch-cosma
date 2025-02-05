import torch
import torch.nn as nn

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