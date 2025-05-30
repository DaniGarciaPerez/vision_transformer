"""
Author: Daniel Garcia

Description: This module implements an Encoder block for a ViT classifier
"""

import torch
from torch import nn
from src.encoder_block.multihead_attention import MultiHeadAttention
from src.encoder_block.mlp_encoder import MLP


class ViTEncoder(nn.Module):
    """
    This class represents a single Encoder block in a Vision Transformer (ViT) classifier.

    It consists of a Multi-Head Attention (MSA) mechanism, followed by a Multi-Layer Perceptron (MLP) block.
    Both the input to the MSA and the output of the MSA are normalized using Layer Normalization.

    attributes:
    -----------
        MSA -> The Multi-Head Attention mechanism.
        MLP -> The Multi-Layer Perceptron block.
        layer_norm:nn.LayerNorm -> The Layer Normalization module.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_size: int,
        dropout_ratio: float,
    ):
        """
        Initializes the ViTEncoder block.

        params:
        --------
            d_model:int -> The number of features in the input data.
            n_heads:int -> The number of attention heads in the MSA mechanism.
            mlp_size:int -> The size of the MLP block.
        """
        super(ViTEncoder, self).__init__()

        assert (
            isinstance(d_model, int)
            and isinstance(n_heads, int)
            and isinstance(mlp_size, int)
        ), "d_model, n_heads, mlp_size must be integers"

        self.encoder_block = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadAttention(d_model, n_heads),
            nn.LayerNorm(d_model),
            MLP(d_model, mlp_size, dropout_ratio))


    def forward(self, pe_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT Encoder block.

        params:
        ---------
            pe_matrix:torch.Tensor -> The input tensor, representing the positional patch embeddings.

        returns:
        ---------
            torch.Tensor: The output tensor after passing through the Encoder block.
        """

        assert isinstance(pe_matrix, torch.Tensor), "pe_matrix must be a torch.Tensor"
        assert len(pe_matrix.shape) == 3, "pe_matrix must have shape (batch_size, sequence_length, d_model)"

        try:

            return self.encoder_block(pe_matrix)

        except Exception as e:

            print(f"Error in forward pass: {e}")
