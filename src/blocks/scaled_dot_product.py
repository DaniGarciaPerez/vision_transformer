"""
Author: Dani Garcia

Description: This script implements the Scaled Dot-Product Attention 
mechanism as described in the paper "Attention is All You Need" by Vaswani et al. (2017).

"""

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Class for the implementation of the scaled dot-product
    attention calculation for set of different matrices.

    Attributes:
    ------------
        forward -> computes the actual scaled dot-product attention
    """

    def __init__(self, d_k: int):
        """
        Initializes the ScaledDotProductAttention module.

        params:
            d_k: int -> The dimensionality of the keys.
        """
        # Instantiate the base class
        super(ScaledDotProductAttention, self).__init__()
        # Get the model dimension
        self.d_k = d_k
        # Instatiate the Softmax class
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        """
        Computes the scaled dot-product attention for the given queries, keys, and values.

        params
        --------
            q: torch.tensor -> The queries matrix. Shape (length_sequence, d_k).
            k: torch.tensor -> The keys matrix. Shape (length_sequence, d_k).
            v: torch.tensor -> The values matrix. Shape (length_sequence, d_k).

        returns
        -------
            attention_values: torch.tnesor -> The result of the attention calculation.

        """
        # 1. Matrix multiplication of queries and keys transposed.
        # attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = torch.matmul(
            q, tuple(map(lambda x: torch.transpose(x, -2, -1), k))
        )
        # 2. Compute attention scaled attention scores.
        scaled_attention_scores = torch.div(attention_scores, (self.d_k**0.5))
        # 3. Calculate attention weights applying Softmax.
        attention_weights = self.softmax(scaled_attention_scores)

        # 4. Apply attention weights to values
        return torch.matmul(attention_weights, v)
