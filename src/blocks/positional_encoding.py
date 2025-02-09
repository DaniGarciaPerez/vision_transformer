""""TODO: include docstring"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """ """

    def __init__(
        self,
        input_matrix: torch.tensor,
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.input_matrix = input_matrix
        self.sequence_length = input_matrix.shape[0]
        self.d_model = input_matrix.shape[1]

    def generate_positional_encoding_values(self) -> torch.tensor:
        """
        Function to generate positional encoding for each of the tokens
        in a sequence using sine and cosine functions.

        Returns:
        ---------
            positional_encoding: torch.tensor -> Positional encoding matrix.

        """
        # Create a tensor of indices from 0 to 'vector_dim - 1' with float data type to capture embeddings length.
        embedding_indices = torch.arange(self.d_model, dtype=torch.float)
        # Calculate an exponential factor for positional encoding using a base of 10000.
        exponential_block = torch.pow(
            10000, (2 * (embedding_indices // 2)) / self.d_model
        )  # TODO: Review the exponential block
        # Create a matrix of token indices along the sequence length.
        # Each row corresponds to a position in the sequence, and columns represent different embedding dimensions.
        tokens_index_matrix = (
            torch.arange(self.sequence_length).unsqueeze(1).expand(-1, self.d_model)
        )
        # Calculate the positional encoding by dividing token indices by the exponential factor.
        positional_encoding = tokens_index_matrix / exponential_block
        # Apply the sine function to even-indexed dimensions (0, 2, 4, ...).
        positional_encoding[0::2] = torch.sin(positional_encoding[0::2])
        # Apply the cosine function to odd-indexed dimensions (1, 3, 5, ...).
        positional_encoding[1::2] = torch.cos(positional_encoding[1::2])
        # Return the computed positional encoding for further use in the model.
        return positional_encoding

    def generate_positional_encoding_matrix(self):
        """ """
        # Generate positional encoding values
        positional_encoding_values = self.generate_positional_encoding_values()
        # Add positional encoding values to the embeddings matrix
        return torch.add(self.input_matrix, positional_encoding_values)
