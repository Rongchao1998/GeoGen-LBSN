import torch
import torch.nn as nn
import torch.nn.functional as F


class POI_Encoder(nn.Module):
    def __init__(self, num_categories, embedding_dim, seq_len, num_pois):
        super(POI_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        # Embedding layers
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.poi_index_embedding = nn.Embedding(num_pois, embedding_dim)

        # Linear transformations
        self.spatial_transform = nn.Linear(2, embedding_dim)  # Assumes spatial info remains as [lat, lon]
        self.visiting_transform = nn.Linear(seq_len, embedding_dim)  # Transforms entire visiting distribution

        # Weights for each component
        self.spatial_weight = nn.Parameter(torch.rand(1))
        self.category_weight = nn.Parameter(torch.rand(1))
        self.visiting_weight = nn.Parameter(torch.rand(1))
        self.index_weight = nn.Parameter(torch.rand(1))

    def forward(self, gps, categories, visiting_distribution, poi_index):
        # Input dimensions are [B, seq_len], except for spatial information which is [B, seq_len, 2]

        # Spatial embedding
        spatial_embed = self.spatial_transform(gps)  # [B, seq_len, 2] to [B, seq_len, emb_dim]

        # Embedding for categories and POI index
        category_embed = self.category_embedding(categories)  # [B, seq_len, emb_dim]
        index_embed = self.poi_index_embedding(poi_index)  # [B, seq_agent, emb_dim]

        # Normalize and transform visiting distribution
        # Assuming visiting_distribution is [B, seq_len, seq_len]
        visiting_embed = self.visiting_transform(visiting_distribution)  # [B, seq_len, emb_dim]

        # Apply weights to each component
        spatial_embed *= self.spatial_weight
        category_embed *= self.category_weight
        visiting_embed *= self.visiting_weight
        index_embed *= self.index_weight

        # Combine embeddings by summing them
        combined_embed = spatial_embed + category_embed + visiting_embed + index_embed  # [B, seq_len, emb_dim]
        return combined_embed


class POI_Decoder(nn.Module):
    def __init__(self, num_pois, emb_dim):
        super(POI_Decoder, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1
        )
        self.decoder = nn.Linear(emb_dim, num_pois)

    def forward(self, embeddings):
        # Embed and encode sequence
        x = self.transformer(embeddings)  # Assume appropriate masking and positional encoding

        # Decode predicted embeddings to probabilistic vectors
        logits = self.decoder(x)  # [B, seq_len, num_pois]
        return logits
