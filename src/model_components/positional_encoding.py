"""
Author: Daniel Garcia

Description: This script implements a Positional Encoding module for transformer-based models.

"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """

    A module to add positional encoding to input token embeddings.
    Positional encoding is computed using sine and cosine functions across the embedding dimensions.

    Attributes:
    -----------
        d_model: int -> The dimensionality of the token embeddings.


    """

    def __init__(self, d_model:int) -> None:
        """

        Initializes the ScaledDotProductAttention module.

        """
        # Instantiate the base class
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def generate_positional_encoding_values(
        self, batch_size:int, sequence_length:int
    ) -> torch.tensor:
        """

        Function to generate positional encoding for each of the tokens
        in a sequence using sine and cosine functions.

        params:
        --------
            batch_size: int -> The size of the batch to be processed.
            sequence_length: int -> The length of the input sequence.

        returns:
        ---------
            positional_encoding: torch.tensor -> Positional encoding matrix.


        """
        # Create a tensor of indices with float data type to create indices for the embeddings dimension.
        embedding_indices = torch.arange(self.d_model, dtype=torch.float)
        # Calculate an exponential factor for positional encoding using a base of 10000.
        exponential_block = torch.pow(
            10000, (2 * embedding_indices / self.d_model)
        ) 
        # Create a matrix of token indices along the sequence length.
        # Each row corresponds to a position in the sequence, and columns represent different embedding dimensions.
        tokens_index_matrix = (
            torch.arange(sequence_length)
            .unsqueeze(1)
            .expand(-1, self.d_model)
        )
        # Calculate the positional (element wise division).
        positional_encoding = tokens_index_matrix / exponential_block
        # Apply the sine function to even-indexed dimensions (0, 2, 4, ...).
        positional_encoding[0::2] = torch.sin(positional_encoding[0::2])
        # Apply the cosine function to odd-indexed dimensions (1, 3, 5, ...).
        positional_encoding[1::2] = torch.cos(positional_encoding[1::2])
        
        return positional_encoding

    def forward(self, x:torch.Tensor):
        """
        Forward pass of the PositionalEncoding module.
        Adds positional encoding to the input embeddings.

        returns:
        ---------
            positional_embeddings: torch.tensor -> The input tensor with positional encoding added.

        """
        batch_size, sequence_length, _ = x.shape

        # Generate positional encoding values
        positional_encoding_values = self.generate_positional_encoding_values(
            batch_size, sequence_length
        )
        # Add positional encoding values to the embeddings matrix
        return torch.add(x, positional_encoding_values)
