import torch
from torch import nn
import itertools


class SelfAttention:
    """ """

    def __init__(self, input_matrix: torch.tensor, number_of_heads: int) -> None:
        self.input_matrix = input_matrix
        self.number_of_heads = number_of_heads

    def init_linear_weights(self):
        """ """
        return torch.rand(3, *self.input_matrix.shape)

    def mul_input(self, weights=None):
        """ """
        return torch.mul(self.input_matrix, weights)

    def compute_attention_weights(self):
        """"""
        queries, keys, values = self.mul_input(self.init_linear_weights())
        softmax = nn.Softmax(dim=-1)
        return torch.matmul(
            softmax(
                torch.div(
                    torch.matmul(queries, torch.transpose(keys, 0, 1)),
                    torch.tensor(queries.size(dim=1)),
                )
            ),
            values,
        )
