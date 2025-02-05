import torch
import torch.nn as nn
import math

class BaseAutoencoder(nn.Module):
    """
    A base class for autoencoders.

    This class serves as a template for creating autoencoder models by
    providing shared functionality.
    """

    def __init__(self):
        super(BaseAutoencoder, self).__init__()


class AutoencoderLayer(BaseAutoencoder):
    def __init__(self, in_features: int, latent_dim: int, out_features: int, num_classes: int = 10,
                 hidden_dim: int = 128, **kwargs):
        super(AutoencoderLayer, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),  # First hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # Latent space layer
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LazyLinear(num_classes),
            nn.Softmax(dim=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # Latent to hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),  # Output layer (same size as input)
        )

    def forward(self, x, return_latent=False):
        # Forward pass: pass input through encoder and decoder
        latent_representation = self.encoder(x)
        if return_latent:
            return latent_representation, self.classifier(latent_representation)

        reconstructed = self.decoder(latent_representation)
        return reconstructed, self.classifier(latent_representation)


class ConvAutoencoder(BaseAutoencoder):
    """
    A convolutional autoencoder with an optional classification head.

    This model combines a convolutional encoder, a latent representation, and a convolutional decoder.
    It optionally includes a classification head for supervised learning tasks.

    Args:
        in_shape (tuple, optional): Shape of the input images (channels, height, width). Defaults to (3, 28, 28).
        initial_filter_num (int, optional): Number of filters in the first convolutional layer. Defaults to 16.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride of the convolutional layers. Defaults to 2.
        padding (int, optional): Padding for the convolutional layers. Defaults to 1.
        latent_dim (int, optional): Dimensionality of the latent space. Defaults to 128.
        num_classes (int, optional): Number of output classes for classification. Defaults to 10.
        **kwargs: Additional keyword arguments (not used, but included for compatibility).
    """

    def __init__(self, in_shape: tuple[int, int, int] = (3, 28, 28), initial_filter_num: int = 16, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1, latent_dim: int = 128, num_classes: int = 10, **kwargs) -> None:
        super(ConvAutoencoder, self).__init__()

        # Calculate the size of the feature map after convolutional layers
        first_conv_output_size = self._get_conv_output_size(in_shape, kernel_size, stride, padding)
        self._conv_output_size = self._get_conv_output_size([None, first_conv_output_size, first_conv_output_size],
                                                            kernel_size, stride, padding)

        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], initial_filter_num, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(initial_filter_num, initial_filter_num * 2, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Flatten(),  # Flatten for the fully connected layers
            nn.LazyLinear(latent_dim),  # Latent space
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LazyLinear(num_classes),
            nn.Softmax(dim=1)
        )

        # Decoder: Transpose convolutions (deconvolutions)
        self.decoder = nn.Sequential(
            nn.LazyLinear(self._conv_output_size * self._conv_output_size * initial_filter_num * 2),
            # From latent space back to the decoder input
            nn.ReLU(),
            nn.Unflatten(1, (initial_filter_num * 2, self._conv_output_size, self._conv_output_size)),
            nn.ConvTranspose2d(initial_filter_num * 2, initial_filter_num, kernel_size, stride=stride),
            nn.ReLU(),
            nn.ConvTranspose2d(initial_filter_num, in_shape[0], kernel_size - 1, stride=stride, padding=1),
            nn.Sigmoid()  # Output in the range [0, 1]
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the ConvAutoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            return_latent (bool, optional): If True, returns the latent representation and classification logits.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If `return_latent` is True, returns a tuple of:
                    (latent representation, classification logits).
                - Otherwise, returns the reconstructed image and classification logits.
        """
        latent = self.encoder(x)
        if return_latent:
            return latent, self.classifier(latent)
        reconstructed = self.decoder(latent)
        return reconstructed, self.classifier(latent)

    @staticmethod
    def _get_conv_output_size(in_shape: tuple[int, int, int], kernel_size: int, stride: int, padding: int) -> int:
        """
        Calculates the size of the output of a convolutional layer.

        Args:
            in_shape (tuple): Shape of the input tensor (channels, height, width).
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolutional layer.
            padding (int): Padding for the convolutional layer.

        Returns:
            int: Size of the output feature map.
        """
        # This function will calculate the size of the output of a convolution
        assert in_shape[1] == in_shape[2]  # Assuming square input images (e.g., 28x28 for MNIST)
        input_size = in_shape[1]
        output_size = math.floor((input_size + 2 * padding - kernel_size) / stride) + 1
        return output_size
