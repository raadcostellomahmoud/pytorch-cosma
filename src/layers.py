import math

import torch.nn as nn


class Add(nn.Module):
    def __init__(self, **kwargs):
        super(Add, self).__init__()

    def forward(self, inputs):
        # Sum all inputs element-wise (assuming inputs is a list of tensors)
        return sum(inputs)


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()


class AutoencoderLayer(BaseAutoencoder):
    def __init__(self, in_features: int, latent_dim: int, out_features: int, hidden_dim: int = 128, **kwargs):
        super(AutoencoderLayer, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),  # First hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # Latent space layer
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
            return latent_representation

        reconstructed = self.decoder(latent_representation)
        return reconstructed

    # def get_latent_space(self, x):
    #     with torch.no_grad():
    #         latent = self.encoder[:2](x)
    #     return latent


class ConvAutoencoder(BaseAutoencoder):
    def __init__(self, in_shape: tuple = (3, 28, 28), initial_filter_num: int = 16, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1, latent_dim: int = 128, **kwargs):
        super(ConvAutoencoder, self).__init__()

        # Define the convolutional layers for the encoder
        self.conv1 = nn.Conv2d(in_shape[0], initial_filter_num, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(initial_filter_num, initial_filter_num * 2, kernel_size, stride=stride, padding=padding)

        # Calculate the size of the feature map after convolutional layers
        first_conv_output_size = self._get_conv_output_size(in_shape, kernel_size, stride, padding)
        self._conv_output_size = self._get_conv_output_size([None, first_conv_output_size, first_conv_output_size], kernel_size, stride, padding)

        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], initial_filter_num, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(initial_filter_num, initial_filter_num * 2, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Flatten(),  # Flatten for the fully connected layers
            nn.LazyLinear(latent_dim),  # Latent space
        )

        # Decoder: Transpose convolutions (deconvolutions)
        self.decoder = nn.Sequential(
            nn.LazyLinear(self._conv_output_size * self._conv_output_size * initial_filter_num * 2),
            # From latent space back to the decoder input
            nn.ReLU(),
            nn.Unflatten(1, (initial_filter_num * 2, self._conv_output_size, self._conv_output_size)),
            nn.ConvTranspose2d(initial_filter_num * 2, initial_filter_num, kernel_size, stride=stride),
            # Output: (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(initial_filter_num, in_shape[0], kernel_size-1, stride=stride, padding=1),
            # Output: (1, 28, 28)
            nn.Sigmoid()  # Output in the range [0, 1]
        )

    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        if return_latent:
            return x
        reconstructed = self.decoder(latent)
        return reconstructed

    def _get_conv_output_size(self, in_shape, kernel_size, stride, padding):
        # This function will calculate the size of the output of a convolution
        assert in_shape[1] == in_shape[2]  # Assuming square input images (e.g., 28x28 for MNIST)
        input_size = in_shape[1]
        output_size = math.floor((input_size + 2 * padding - kernel_size) / stride) + 1
        return output_size