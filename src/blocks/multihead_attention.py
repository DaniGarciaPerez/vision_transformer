"""
Author: Daniel Garcia

Description: This module implements a Multi-Head Attention mechanism,
a key component of the Transformer architecture.
"""

import itertools
import torch
from torch import nn
from src.blocks.scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module implements the Multi-Head Attention mechanism, which allows the model to attend
    to different parts of the input sequence simultaneously and weigh their importance.
    It consists of multiple attention heads, each of which applies
    a scaled dot-product attention mechanism to the input sequence.

    attributes:
    ------------
        pe_matrix: torch.tensor -> The positional embedding matrix.
        n_heads: int -> The number of attention heads.
        d_model:int -> The dimensionality of the input sequence.
    """

    def __init__(
        self, positional_embedding_matrix: torch.tensor, d_model: int, n_heads: int
    ):
        """
        Initializes the MultiHeadAttention module.

        params:
        --------
            pe_matrix: torch.tensor -> The positional embedding matrix.
            n_heads: int -> The number of attention heads.
            d_model:int -> The dimensionality of the input sequence.
        """
        super(MultiHeadAttention, self).__init__()
        self.pe_matrix = positional_embedding_matrix
        self.n_heads = n_heads
        self.d_model = d_model

        # Define linear layers for query, key, and value transformations
        self.q_weights = nn.Linear(d_model, d_model)
        self.k_weights = nn.Linear(d_model, d_model)
        self.v_weights = nn.Linear(d_model, d_model)
        # Define a linear layer to concatenate the outputs of the attention heads
        self.concat_weights = nn.Linear(d_model, d_model)

    def split_heads(
        self, queries: torch.tensor, keys: torch.tensor, values: torch.tensor
    ) -> tuple:
        """
        Splits the input tensors into multiple attention heads.

        params:
        --------
            queries: torch.tensor -> The query tensor.
            keys:torch.tensor -> The key tensor.
            values:torch.tensor -> The value tensor.

        returns:
        ---------
            tuple: A tuple of three tensors, each containing the split query, key, and value tensors.
        """
        splits = self.d_model // self.n_heads
        k_heads, v_heads, q_heads = (
            torch.split(keys, splits),
            torch.split(queries, splits),
            torch.split(values, splits),
        )

        return q_heads, k_heads, v_heads

    def compute_attention(
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        """
        Computes the scaled dot-product attention for a single attention head.

        params:
        --------
            q_heads: torch.tensor -> The query tensor for the current attention head.
            k_heads: torch.tensor -> The key tensor for the current attention head.
            v_heads: torch.tensor -> The value tensor for the current attention head.

        returns:
        ---------
            torch.tensor -> The output of the scaled dot-product attention mechanism.
        """
        attention = ScaledDotProductAttention(self.d_model)

        return attention.forward(q, k, v)

    def forward(self) -> torch.tensor:
        """
        Computes the output of the Multi-Head Attention module.

        returns:
        ---------
            torch.tensor -> Output of the Multi-Head Attention module.
        """

        # Linear transformation
        q, k, v = (
            self.q_weights(self.pe_matrix),
            self.k_weights(self.pe_matrix),
            self.v_weights(self.pe_matrix),
        )

        # Multihead split
        q_heads, k_heads, v_heads = self.split_heads(q, k, v)

        # Compute scaled dot-product attention
        scaled_dotproduct_attention = list(
            itertools.starmap(
                self.compute_attention, list(zip(q_heads, k_heads, v_heads))
            )
        )

        # Concat scaled dotproduct
        concat_attention = torch.cat(scaled_dotproduct_attention)

        # Apply output linear transformation
        return self.concat_weights(concat_attention)
