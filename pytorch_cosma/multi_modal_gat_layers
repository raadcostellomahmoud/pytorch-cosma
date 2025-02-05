import torch
import torch.nn as nn

from torch_geometric.nn import GATConv

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer-based sequence input.
    """

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiModalGAT(nn.Module):
    def __init__(self, cnn_input_channels, cnn_first_channels, cnn_kernel_size, cnn_padding, modality_output_dim,
                 gat_hidden_dim, output_dim, gat_heads, dropout_prob, n_modes, just_infer_connected):
        super(MultiModalGAT, self).__init__()

        self.just_infer_connected = just_infer_connected

        # CNN for image-encoded sequences
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_channels, cnn_first_channels, cnn_kernel_size, cnn_padding),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_first_channels, cnn_first_channels * 2, cnn_kernel_size, cnn_padding),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Compress to a single feature vector per node
        )
        self.flatten = nn.Flatten()
        self.fc_cnn = nn.Linear(cnn_first_channels * 2, modality_output_dim)
        self.bn_cnn = nn.LayerNorm(modality_output_dim)
        self.dropout_cnn = nn.Dropout(p=dropout_prob)

        # GAT layers
        self.gat1 = GATConv(modality_output_dim * n_modes, gat_hidden_dim, heads=gat_heads)
        self.gat2 = GATConv(gat_hidden_dim * gat_heads, gat_hidden_dim)
        self.node_fc = nn.Linear(gat_hidden_dim, output_dim)

        # Edge prediction layers
        self.edge_scorer = nn.Sequential(
            nn.Linear(gat_hidden_dim * 2, gat_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(gat_hidden_dim, 1)
        )

        self.dropout_gat = nn.Dropout(p=dropout_prob)

    def forward_common(self, x_images):
        # Process image-encoded sequences through CNN
        x_cnn = self.cnn(x_images)
        x_cnn = self.flatten(x_cnn)
        x_cnn = self.fc_cnn(x_cnn)
        x_cnn = nn.functional.leaky_relu(x_cnn)
        x_cnn = self.bn_cnn(x_cnn)
        x_cnn = self.dropout_cnn(x_cnn)
        return x_cnn

    def forward_gat(self, x, edge_index):
        # Apply GAT layers
        x = self.gat1(x, edge_index)
        # x = torch.relu(x)
        x = self.dropout_gat(x)
        x = self.gat2(x, edge_index)
        # x = torch.relu(x)

        # Node predictions
        node_pred = torch.sigmoid(self.node_fc(x))

        # Edge predictions
        source_indices, target_indices = edge_index
        source_embeddings = x[source_indices]
        target_embeddings = x[target_indices]
        edge_features = torch.cat([source_embeddings, target_embeddings], dim=-1)

        if self.just_infer_connected:
            edge_pred = torch.sigmoid(self.edge_scorer(edge_features).squeeze())
        else:
            edge_pred = torch.relu(self.edge_scorer(edge_features).squeeze())

        return node_pred, edge_pred, edge_index
    

class GATModelWithTransformerAndCNN(MultiModalGAT):
    def __init__(self, padded_read_len, cnn_input_channels, cnn_first_channels, cnn_kernel_size, cnn_padding,
                 gat_hidden_dim, output_dim, modality_output_dim, pos_encoding_dim, transformer_heads,
                 transformer_layers, gat_heads, dropout_prob, just_infer_connected):
        super(GATModelWithTransformerAndCNN, self).__init__(
            cnn_input_channels, cnn_first_channels, cnn_kernel_size, cnn_padding, modality_output_dim,
            gat_hidden_dim, output_dim, gat_heads, dropout_prob, 2, just_infer_connected
        )

        # Positional encoding and one-hot projection
        self.one_hot_projection = nn.Linear(4, pos_encoding_dim)
        self.positional_encoding = PositionalEncoding(pos_encoding_dim, dropout_prob, padded_read_len)
        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=pos_encoding_dim, nhead=transformer_heads, dim_feedforward=pos_encoding_dim * 4,
            dropout=dropout_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)
        self.fc_transformer = nn.Linear(pos_encoding_dim * 2, modality_output_dim)
        self.bn_transformer = nn.LayerNorm(modality_output_dim)
        self.dropout_transformer = nn.Dropout(p=dropout_prob)

    def forward(self, data):
        x_images, x_one_hot = data.x_images.float(), data.x_one_hot.float()
        x_cnn = self.forward_common(x_images)

        # Process sequences through Transformer
        x_one_hot = self.one_hot_projection(x_one_hot)
        x_one_hot = self.positional_encoding(x_one_hot)
        transformer_out = self.transformer(x_one_hot)
        max_pooled = transformer_out.max(dim=1)[0]
        mean_pooled = transformer_out.mean(dim=1)

        # Concatenate
        representation = torch.cat([mean_pooled, max_pooled], dim=-1)
        representation = torch.sigmoid(representation)
        x_transformer = self.fc_transformer(representation)
        x_transformer = torch.relu(x_transformer)
        x_transformer = self.bn_transformer(x_transformer)
        x_transformer = self.dropout_transformer(x_transformer)

        # Combine CNN and Transformer features
        x = torch.cat([x_cnn, x_transformer], dim=-1)
        x = torch.sigmoid(x)

        # Generate `edge_index`
        num_nodes = x.size(0)
        edge_index = torch.combinations(torch.arange(num_nodes)).t().to(x.device)

        return self.forward_gat(x, edge_index)