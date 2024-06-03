import torch
from torch import nn


class SelfAttention:
    """ """

    def __init__(self, input_matrix: torch.tensor, number_of_heads: int) -> None:
        self.input_matrix = input_matrix
        self.number_of_heads = number_of_heads

    def mul_input(self, weights=None):
        """ """
        if weights == None:
            weights = torch.rand(3, *self.input_matrix.shape)
        return torch.mul(self.input_matrix, weights)

    def scaled_dot_product_attention(self, weights=None):
        """"""
        querys, keys, values = self.mul_input()

        # Apply matmul, scaling factor
        query_key_scaled = torch.div(
            torch.matmul(keys, querys.T),
            torch.sqrt(torch.Tensor([querys.shape[1]])),
        )
        # Get Softmax values
        softmax = torch.nn.Softmax(dim=1)
        softmax_values = softmax(query_key_scaled)

        return torch.matmul(softmax_values, values)

    def multihead_attention(self):

        split_input_matrix = torch.split(self.input_matrix, self.number_of_heads, dim=1)
        for matrix in split_input_matrix:
            print(matrix)

        return None
