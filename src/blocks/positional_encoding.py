import torch


def generate_positional_encoding(vector_dim: int, sequence_length: int) -> torch.tensor:
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
    embedding_indices = torch.arange(vector_dim, dtype=torch.float)
    exponential_block = torch.pow(10000.0, 2.0 * (embedding_indices / vector_dim))
    tokens_index_matrix = (
        torch.arange(sequence_length).unsqueeze(1).expand(-1, vector_dim)
    )
    positional_encoding = tokens_index_matrix / exponential_block
    positional_encoding[0::2] = torch.sin(positional_encoding[0::2])
    positional_encoding[1::2] = torch.cos(positional_encoding[1::2])
    return positional_encoding
