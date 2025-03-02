"""
Author: Daniel Garcia

Description: This module implements the multilayer perceptron for image classification
"""

import torch
from torch import nn


class MLPHead(nn.Module):
    """
    A multilayer perceptron (MLP) class for image classification.

    This class implements an MLP with one hidden layer.

    attributes:
    -----------
        input_matrix: torch.Tensor -> The input matrix for the MLP.
        mlp_layer: nn.Linear -> The linear layer of the MLP.
    """

    def __init__(self, d_model, n_classes):
        """
        Initializes the MLP with the given input matrix and output size.

        Args:
            input_matrix (torch.Tensor): The input matrix for the MLP.
            output_size (int): The number of output classes.
        """
        super(MLPHead, self).__init__()
        self.classifier_layer = nn.Linear(d_model, n_classes)
        # Define the GELU function
        self.softmax = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Defines the forward pass through the network.

        returns:
        ---------
            torch.Tensor: The output of the softmax function for classification.
        """
        x = self.tanh(self.classifier_layer(x))

        # Output layer
        return self.softmax(x)
