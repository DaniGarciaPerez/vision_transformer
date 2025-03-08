"""
Author: Daniel Garcia

Description: This module implements the multilayer perceptron for image classification
"""

import torch
from torch import nn
from src.model_components.positional_encoding import PositionalEncoding
from src.model_components.patch_projection import PatchLinearProjection
from src.encoder_block.encoder import ViTEncoder
from src.model_components.mlp_head import MLPHead


class VisionTransformer(nn.Module):
    """TODO: Insert docstring"""

    def __init__(
        self,
        patch_size,
        d_model,
        mlp_size,
        n_heads,
        dropout_ratio,
        n_layers,
        n_classes,
        n_channels,
        batch_size,
    ):
        """TODO: Insert docstring"""

        super(VisionTransformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.linear_projection = PatchLinearProjection(
            patch_size=patch_size, d_model=d_model, input_channels=n_channels
        )
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.vit_encoder = ViTEncoder(d_model, n_heads, mlp_size, dropout_ratio)
        self.mlp_head = MLPHead(d_model, n_classes)
        self.positional_encoding = PositionalEncoding(d_model, batch_size)

    def forward(self, x):
        """TODO: Insert docstring"""
        # Patch projection + Positional encoding block
        x = self.linear_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in range(self.n_layers):
            x = self.vit_encoder(x)

        # Global Average Pooling for classification instead of using a class token (GAP)
        x = x.mean(dim=1)

        return self.mlp_head(x)
