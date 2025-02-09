"""
"""

import torch
from torch import nn
from dotenv import load_dotenv
from src.blocks.scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ """

    def __init__(self, positional_embedding_matrix, d_model, n_heads):
        """"""
        super(MultiHeadAttention, self).__init__()
        self.positional_embedding_matrix = positional_embedding_matrix
        self.n_heads = n_heads
        self.d_model = d_model

        self.q_weights = nn.Linear(d_model, d_model)
        self.k_weights = nn.Linear(d_model, d_model)
        self.v_weights = nn.Linear(d_model, d_model)
        self.concat_weights = nn.Linear(d_model, d_model)

    def split_heads(self, queries, keys, values):

        splits = torch.div(self.d_model, self.n_heads, rounding_mode="floor")
        k_heads = torch.split(keys, splits)
        v_heads = torch.split(queries, splits)
        q_heads = torch.split(values, splits)

        return k_heads, v_heads, q_heads

    def compute_attention(self, k_heads, v_heads, q_heads):
        """ """

        return [
            ScaledDotProductAttention(self.d_model).forward(
                k_heads[head], v_heads[head], q_heads[head]
            )
            for head in range(0, self.n_heads)
        ]

    def forward(self):

        # Linear transformation
        q = self.q_weights(self.positional_embedding_matrix)
        k = self.k_weights(self.positional_embedding_matrix)
        v = self.v_weights(self.positional_embedding_matrix)

        # Multihead split
        k_heads, v_heads, q_heads = self.split_heads(q, k, v)

        # Compute scaled dot-product attention
        scaled_dotproduct_attention = self.compute_attention(k_heads, v_heads, q_heads)

        # Concat scaled dotproduct
        concat_attention = torch.cat(scaled_dotproduct_attention)

        # Apply output linear transformation
        return self.concat_weights(concat_attention)
