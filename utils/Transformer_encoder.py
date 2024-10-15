import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CheckinTransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, num_encoder_layers, dropout_rate=0.1):
        super(CheckinTransformerEncoder, self).__init__()
        self.embedding_size = embedding_size

        # Complex GPS embedding
        self.gps_embedding = nn.Sequential(
            nn.Linear(2, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, embedding_size)
        )

        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1000, embedding_size))  # Support up to 1000 time steps

        # Transformer Encoder with increased complexity
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=embedding_size * 4,  # Increasing complexity
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

    def forward(self, gps_coordinates, timestamps):
        seq_len = timestamps.size(1)
        batch_size = timestamps.size(0)

        # Embed GPS coordinates
        gps_embeds = self.gps_embedding(gps_coordinates)  # Shape: [batch_size, seq_len, embedding_size]

        # Apply positional encoding
        pos_encoding = self.positional_embedding[:, :seq_len]

        # Combine GPS and timestamp embeddings
        embeddings = gps_embeds + pos_encoding  # Broadcasting addition

        # Pass the embeddings through the transformer
        output = self.transformer_encoder(embeddings)
        return output

