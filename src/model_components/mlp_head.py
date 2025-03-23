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
            d_model (int): The input size of the MLP.
            n_classes (int): The number of output classes.
            hidden_dim (int, optional): The size of the hidden layer. Defaults to 128.
    """

    def __init__(self, d_model:int, n_classes:int, hidden_dim:int):
        """
        Initializes the MLP with the given input matrix and output size.

        """
        super(MLPHead, self).__init__()
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.classifier_layer = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        

    def forward(self, x:torch.Tensor):
        """
        Defines the forward pass through the network.

        returns:
        ---------
            torch.Tensor: The output of the softmax function for classification.
        """

        # This a mix between the MLP head implementation using a class token and using GAP.
        # (no hidden layer for GAP vs. hidden layer for class token)
        x = self.tanh(self.hidden_layer(x))
        x = self.classifier_layer(x)

        # Output layer
        return self.softmax(x)
