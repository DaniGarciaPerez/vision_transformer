"""
Author: Daniel Garcia

Description: This module implements the multilayer perceptron for image classification
"""

import torch
from torch import nn


class MLP(nn.Module):
    """
    A multilayer perceptron (MLP) class for image classification.

    This class implements an MLP with one hidden layer.

    attributes:
    -----------
        input_matrix: torch.Tensor -> The input matrix for the MLP.
        mlp_layer: nn.Linear -> The linear layer of the MLP.
    """

    def __init__(self, d_model, mlp_size, dropout_ratio):
        """
        Initializes the MLP with the given input matrix and output size.

        Args:
            input_matrix (torch.Tensor): The input matrix for the MLP.
            output_size (int): The number of output classes.
        """
        super(MLP, self).__init__()
        # Initialize the linear layer with the correct input and output sizes
        self.hidden_layer = nn.Linear(d_model, mlp_size)
        self.mlp_layer = nn.Linear(mlp_size, d_model)
        # Define the GELU function
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        returns:
        ---------
            torch.Tensor: The output of the GELU function after the MLP layer.
        """

        # Hidden layer
        x = self.gelu(self.hidden_layer(x))
        x = self.dropout(x)

        # Output layer
        return self.gelu(self.mlp_layer(x))
