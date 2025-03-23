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
    """
    Vision Transformer (ViT) model for image classification.

    This implementation combines patch linear projection, positional encoding, 
    a set of ViT encoder layers, and a final MLP head for classification.

    attributes:
    ------------
        n_layers:int -> Number of encoder layers.
        d_model:int -> Dimensionality of the model's embeddings.
        linear_projection:PatchLinearProjection -> Projects input patches to embeddings.
        dropout:nn.Dropout -> Applies dropout for regularization of the network.
        vit_encoder:ViTEncoder -> Stack of encoder layers.
        mlp_head:MLPHead -> Final MLP head for classification.
        positional_encoding:PositionalEncoding -> Adds positional information to embeddings.
    """

    def __init__(
        self,
        patch_size:int,
        d_model:int,
        mlp_size:int,
        hidden_class_layer:int,
        n_heads:int,
        dropout_ratio:float,
        n_layers:int,
        n_classes:int,
        n_channels:int,
    ):
        """
        Initializes the Vision Transformer model.

        """

        super(VisionTransformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.linear_projection = PatchLinearProjection(
            patch_size=patch_size, d_model=d_model, input_channels=n_channels
        )
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.vit_encoder = ViTEncoder(d_model, n_heads, mlp_size, dropout_ratio)
        self.mlp_head = MLPHead(d_model, n_classes, hidden_class_layer)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass through the Vision Transformer model.

        args:
        ------
            x:torch.Tensor -> Input image tensor.

        returns:
        ---------
            torch.Tensor: Output for the classification task.
        """
        try:
            # Patch projection + Positional encoding block
            x = self.linear_projection(x)
            x = self.positional_encoding(x)
            x = self.dropout(x)
            for layer in range(self.n_layers):
                x = self.vit_encoder(x)

            # Global Average Pooling for classification instead of using a class token (GAP)
            x = x.mean(dim=1)

            return self.mlp_head(x)
        
        except Exception as e:
            print(f"An error ocurred on the forward pass: {e}")
