""""TODO: include docstring"""

import torch


def generate_positional_encoding(sequence_length: int, vector_dim: int) -> torch.tensor:
    """
    Function to generate positional encoding for each of the tokens
    in a sequence using sine and cosine functions.

    Params:
    ---------
        vector_dim: int -> Dimension of the input vector for each of the sequence tokens.
        sequence_length: int -> Max. length of the total sequence of tokens.

    Returns:
    ---------
        positional_encoding: torch.tensor -> Positional encoding matrix.

    """
    # Create a tensor of indices from 0 to 'vector_dim - 1' with float data type.
    embedding_indices = torch.arange(vector_dim, dtype=torch.float)
    # Calculate an exponential factor for positional encoding using a base of 10000.
    exponential_block = torch.pow(10000, 2 * (embedding_indices / vector_dim))
    # Create a matrix of token indices along the sequence length.
    # Each row corresponds to a position in the sequence, and columns represent different embedding dimensions.
    tokens_index_matrix = (
        torch.arange(sequence_length).unsqueeze(1).expand(-1, vector_dim)
    )
    # Calculate the positional encoding by dividing token indices by the exponential factor.
    positional_encoding = tokens_index_matrix / exponential_block
    # Apply the sine function to even-indexed dimensions (0, 2, 4, ...).
    positional_encoding[0::2] = torch.sin(positional_encoding[0::2])
    # Apply the cosine function to odd-indexed dimensions (1, 3, 5, ...).
    positional_encoding[1::2] = torch.cos(positional_encoding[1::2])
    # Return the computed positional encoding for further use in the model.
    return positional_encoding
