"""
Author: Dani Garcia

Description: This script implements the Scaled Dot-Product Attention 
mechanism as described in the paper "Attention is All You Need" by Vaswani et al. (2017).

"""

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Class for the implementation of the scaled dot-product attention calculation for set of different matrices.

    modules:
        forward -> computes the actual scaled dot-product attention
    """

    def __init__(self, d_k: int):
        """
        Initializes the ScaledDotProductAttention module.

        params:
            d_k: int -> The dimensionality of the keys.
        """ """ """
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
            q: torch.tensor -> The matrix with the linearly transformed queries matrix.
            k: torch.tensor -> The matrix with the linearly transformed keys matrix.
            v: torch.tensor -> The matrix with the linearly transformed values matrix.

        return
        -------
            attention_values: torch.tnesor -> The matrix with the result from the attention calculation.

        """
        # 1. Matrix multiplication of queries and keys transposed.
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        # 2. Compute attention scaled attention scores.
        scaled_attention_scores = torch.div(attention_scores, (self.d_k**0.5))
        # 3. Calculate attention weights applying Softmax.
        attention_weights = self.softmax(scaled_attention_scores)

        # 4. Apply attention weights to valuesS
        return torch.matmul(attention_weights, v)
