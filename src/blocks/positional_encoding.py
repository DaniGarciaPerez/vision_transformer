import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def generate_positional_encoding(vector_dim: int, sequence_length: int):
    """ """
    embedding_indices = torch.arange(vector_dim, dtype=torch.float)
    exponential_block = torch.pow(10000.0, 2.0 * (embedding_indices / vector_dim))
    tokens_index_matrix = (
        torch.arange(sequence_length).unsqueeze(1).expand(-1, vector_dim)
    )
    positional_encoding = tokens_index_matrix / exponential_block
    positional_encoding[0::2] = torch.sin(positional_encoding[0::2])
    positional_encoding[1::2] = torch.cos(positional_encoding[1::2])
    print(positional_encoding)
    return positional_encoding
